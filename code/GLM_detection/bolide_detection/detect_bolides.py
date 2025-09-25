#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import bolide_filter_functions as bff
import bolide_support_functions as bsf
import glob
import sys
import os
import multiprocessing as mp
import time
from traceback import print_exc
import shutil
import netCDF4
import warnings
import copy

import time_utilities as timeUtil
import bolide_io as bio
import bolide_clustering as bCluster
import bolide_detections as bd
import glint_filter as glint
import plot_bolide_detections as plottingTool
import stereo_renavigation as bRenav
import bolide_dispositions as bDisp
import bolide_detection_performance as bPerformance
from io_utilities import copy_or_create_symlinks, create_path
import bolide_database as bolideDB
import bolide_features as bFeatures
from validator import BolideAssessment, bolideWebsiteBeliefSwitcher
from gen_goes_cutout import cutoutConfiguration, gen_goes_cutout, cutoutOutputClass

# Fortran renavigation code
import GLM_renavigation 

#*************************************************************************************************************
def detect_bolides (input_config, filePaths=[], output_dir=None, provenance=None):
    """
    Top level function to search a selection of GLM netCDF files for bolides.
   
    Parameters
    ----------
    input_config : bolide_io.input_config
    filePaths   : str list 
        list of files to search
        If empty then use the files in input_config.data_path
    output_dir : str
        Path where output will be saved
    provenance  : run_glm_pipeline_batch.Provenance 
        Contains the provenance for this run
   
    Returns
    -------
    num_bolides : int
        The total number of bolide candidates detected
    bolideDetectionList : list of BolideDetection
        The detected bolide candidates
    bolideRejectionList : list of BolideDetection
        The rejected bolide candidates

    """

    startTime = time.time()

    # Get the netCDF files to process
    if len(filePaths) == 0:
        filePaths = glob.glob(input_config.data_path + "*.nc")
        if (filePaths == []):
            if (input_config.verbosity):
                print('No files to process in directory {}'.format(input_config.data_path))
                return 0, None, None 

    filePaths = sorted(filePaths)

    if input_config.end_index == -1:
        filePaths = filePaths[input_config.start_index:]
    else:
        filePaths = filePaths[input_config.start_index:input_config.end_index]

    # Initialize the glint filter
    if (input_config.glint_G16_path != '' or input_config.glint_G17_path != ''):
        raise Exception('Glint filter is no longer supported.')
        if (input_config.verbosity): print('Initilizing glint filter...')
        glintFilter = glint.GlintFilter(input_config.glint_G16_path, input_config.glint_G17_path,
                input_config.glintRegionG16LatRadiusDegrees, input_config.glintRegionG16LonRadiusDegrees,
                input_config.glintRegionG17LatRadiusDegrees, input_config.glintRegionG17LonRadiusDegrees)
    else:
        glintFilter = None

    # Read in either the website data or the trained classifier
    if (input_config.ground_truth_path != '' and input_config.ground_truth_path != []):
        # Use the ground truth file to determine bolide candidate status
        input_config.bolidesFromWebsite = bd.unpickle_bolide_detections(input_config.ground_truth_path)
    elif (input_config.trained_classifier_path != '' and input_config.trained_classifier_path != []):
        # Read in the classifier
        with open(input_config.trained_classifier_path, 'rb') as fp:
            input_config.trainedClassifierDict = pickle.load(fp)
        fp.close()
        # When run on multiple cores, the classifier needs to have n_jobs = 1
        if hasattr(input_config.trainedClassifierDict['trainedClassifier'], 'n_jobs'):
            # This is for a plain random forest
            input_config.trainedClassifierDict['trainedClassifier'].n_jobs = 1

        elif hasattr(input_config.trainedClassifierDict['trainedClassifier'], 'base_estimator'):
            # This is for an AdaBoosted random forest
            if hasattr(input_config.trainedClassifierDict['trainedClassifier'].base_estimator, 'n_jobs'):
                input_config.trainedClassifierDict['trainedClassifier'].base_estimator.n_jobs = 1
                input_config.trainedClassifierDict['trainedClassifier'].estimators_[0].n_jobs = 1
            elif hasattr(input_config.trainedClassifierDict['trainedClassifier'], 'estimator_'):
                # This is for scikit-learn release 1.3
                if hasattr(input_config.trainedClassifierDict['trainedClassifier'].estimator_, 'n_jobs'):
                    input_config.trainedClassifierDict['trainedClassifier'].estimator_.n_jobs = 1
                    input_config.trainedClassifierDict['trainedClassifier'].estimators_[0].n_jobs = 1
                else:
                    raise Exception('Unsupported classifier')
            else:
                raise Exception('Unsupported classifier')

        else:
            raise Exception('Unsupported classifier')
    
    #*****************************
    # DETECTION
    detectionStartTime = time.time()

    # Either do multi- or single-core processing, depending on the value of 
    # n_cores. Note that single-core processing is required for profiling.
    if input_config.n_cores < 1:
        raise Exception('n_cores should an integer be >= 1')
    elif input_config.n_cores == 1:
        
        [bolideDetectionList, bolideRejectionList, oops_counter] = _detect_bolides_in_multiple_files(filePaths, input_config, glintFilter)
        
    else:
        # Do multi-core processing.
        #
        # Note that for input_config to be "pickleable" its class definition 
        # must not be in this file. It must be defined elsewhere and imported. 
        oops_counter = 0
        bolideDetectionList = []
        bolideRejectionList = []

        # Chunk files
        # Chunk factor decreases the number of jobs by this factor. 
        chunkFactor = 10
        chunkSize = int(np.ceil(len(filePaths) / (input_config.n_cores * chunkFactor)))
        filesChunked = [filePaths[i:i + chunkSize] for i in range(0, len(filePaths), chunkSize)]  
        
        with mp.Pool(processes=input_config.n_cores) as pool:
            results = [pool.apply_async(_detect_bolides_in_multiple_files, args=(file_paths, input_config, glintFilter)) for file_paths in filesChunked]
            outputs = [result.get() for result in results]
            for output in outputs:
                [detectedBolides, rejectedBolides, oops_counter_elemental] = output
                oops_counter += oops_counter_elemental
                bolideRejectionList.extend(rejectedBolides)
                bolideDetectionList.extend(detectedBolides)

    detectionEndTime = time.time()
    detectionTotalTime = detectionEndTime - detectionStartTime
    print("Detection processing time: {:.2f} seconds, {:.2f} minutes".format(detectionTotalTime, detectionTotalTime / 60))
    #*****************************

    # If creating explicit clusters then merge all the bolide candidates found above into single uber-bolide clusters
    # Each streak has a correpsonding index, use that to merge all the streaks from individual files that are a part of
    # each uber streak
    if input_config.spatiotemporal_box is not None and input_config.spatiotemporal_box[0].explicit_cluster:
        assert input_config.spatiotemporal_box[0].streak_width is not None, 'This only works for streaks with a streak_width'
        # Find the number of streak
        streakIndices = [b.streakIdx for b in bolideDetectionList]
        streakIndices = np.unique(streakIndices)
        streakMapping = {}
        for idx, streakIdx in enumerate(streakIndices):
            streakMapping[streakIdx] = idx
        uberBolides = [None for i in streakIndices]
        for bolide in bolideDetectionList:
            if uberBolides[streakMapping[bolide.streakIdx]] is None:
                uberBolides[streakMapping[bolide.streakIdx]] = copy.deepcopy(bolide)
            else:
                uberBolides[streakMapping[bolide.streakIdx]].add_data(bolide)
        bolideDetectionList = uberBolides
        

    #***
    # Remove duplicate detections and rejections
    # Groups are clustered including a small fraction of groups from neighboring files if close in time
    # to the main file. This means if a cluster is contained in the overlap region, the cluster (i.e. detection) will
    # be detected twice. We need to clean these duplicates from the list.
    bolideRejectionList = bd.remove_duplicates_from_bolideDetectionList(bolideRejectionList)
    bolideDetectionList = bd.remove_duplicates_from_bolideDetectionList(bolideDetectionList)

    #***
    # If only finding bolides within a very specific time window (spatiotemporal box), then remove all detections outside this window
    # NOTE: This should not be needed, but keeping it here commented out, just in case
    #bolideDetectionList = bd.keep_bolides_within_spatiotemporal_box(input_config.spatiotemporal_box, bolideDetectionList)


    #***
    if (input_config.verbosity): print("We had {:d} upses".format(oops_counter))

    #***

    # Only save up to input_config.max_num_detections, sorted by detection score
    # But only if using trained classifier. If using ground truth to generate traingin data set then save all we get.
    if input_config.max_num_detections > 0 and input_config.bolidesFromWebsite is None:
        bolideDetectionList.sort(key=lambda x: x.assessment.triage.score, reverse=True)
        bolideDetectionList = bolideDetectionList[0:np.min([len(bolideDetectionList), input_config.max_num_detections])]

    num_bolides = len(bolideDetectionList)    

    
    # Create list of all .nc files associated with detections
    raw_bolide_data_files = []
    for detection in bolideDetectionList:
        raw_bolide_data_files.extend(detection.filePathList)
    # There can be multiple detections per file. We only need one entry per file.
    raw_bolide_data_files = np.unique(raw_bolide_data_files).tolist()

    if num_bolides > 0:
        if (input_config.verbosity): print("We have {:d} bolide detections".format(num_bolides))

        # Create symbolic links or copy the file to the output directory
        # Note: If  copyNetCDFFiles = False, then createSymlinks = True and we need to temporarily copy over the netCDF
        # files. We delete the symlinks down below.
        # TODO: This is awkward, clean this up so we do not need to temporarily make these symlinks
        if (input_config.verbosity): 
            if input_config.createSymlinks: 
                print('Creating symbolic links to .nc files ...')
            else:
                print('Copying .nc files to output directory...')
        copy_or_create_symlinks(raw_bolide_data_files, output_dir, input_config.createSymlinks, input_config.verbosity)

        # Strip full path from .nc filenames and add in the output_dir to the path
        for bolide in bolideDetectionList:
            for idx in np.arange(len(bolide.filePathList)):
                bolide.filePathList[idx] = os.path.basename(bolide.filePathList[idx])
                bolide.filePathList[idx] = os.path.join(output_dir, bolide.filePathList[idx])
        
        #***
        # We need to generate IDs for the detections so that the stereo detection code can function
        # The IDs are generated in the database so that the IDs are unique wrt all other already stored detections and
        # rejections. Yet, we need to store extra StereoFeature data in the detections, which requires unique IDs. 
        # To get around this chicken-and-egg problem, here we just generate some ID, which may change once they are
        # added to the database. 
        for bolide in bolideDetectionList:
            bolide.ID = bolideDB.BolideDatabase._generate_ID(bolide)
        bolideDetectionList = bolideDB.BolideDatabase._check_duplicates(bolideDetectionList)

        # Copy over netCDF files from the other satellite that overlap in time with this detection
        otherSatelliteFilenames = copy_netCDF_files_from_other_satellite(input_config, bolideDetectionList, output_dir)

        # Stereo detections
        # Attempt to "detect" a bolide in the other satellite
        bolideDetectionList = detect_in_other_satellite(input_config, bolideDetectionList, otherSatelliteFilenames, 
                output_dir, provenance)
        
        # Perform stereo renavigation
        renavObj = bRenav.RenavigateClass(input_config, provenance, bolideDetectionList)
        renavObj.renavigate_all_detections()
        if input_config.generatePlots == True:
            # Stereo figures
            bolideDetectionList = renavObj.plot_all_stereo_detections(output_dir, bolideDetectionList)
        bolideDetectionList = renavObj.populate_stereo_features(bolideDetectionList)
        # Stereo renavigation can modify bolideDetectionList if stereoDetectionOnlyMode == True
        # So, recompute num_bolides 
        num_bolides = len(bolideDetectionList)    
        if num_bolides == 0:
            if (input_config.verbosity): print("There were no valid detections.")

        #***
        # Generate plots.
        if input_config.generatePlots == True and num_bolides > 0:

            #***
            # Now, finally, generate the detection figures
            if (input_config.verbosity): print('Generating plots from bolide detections ...')
            if input_config.pixel_boundary_plotting_enabled:
                latLon2Pix_table_path = input_config.latLon2Pix_table_path
                latLon2Pix_table_path_inverted = input_config.latLon2Pix_table_path_inverted
            else:
                latLon2Pix_table_path = None
                latLon2Pix_table_path_inverted = None
            plottingTool.plot_detections_parallel(bolideDetectionList, provenance, output_dir, 
                    latLon2Pix_table_path=latLon2Pix_table_path, 
                    latLon2Pix_table_path_inverted=latLon2Pix_table_path_inverted, n_cores=input_config.n_cores)
            plottingTool.plot_detections_parallel(bolideDetectionList, provenance, output_dir, 
                    latLon2Pix_table_path=latLon2Pix_table_path,
                    latLon2Pix_table_path_inverted=latLon2Pix_table_path_inverted, n_cores=input_config.n_cores, otherSatelliteFlag=True)

          # print('*********REMOVE ME*******************')
          # print('*********REMOVE ME*******************')
          # print('*********REMOVE ME*******************')
          # plottingTool.plot_detections_parallel(bolideDetectionList, provenance, output_dir, 
          #         latLon2Pix_table_path=latLon2Pix_table_path, 
          #         latLon2Pix_table_path_inverted=latLon2Pix_table_path_inverted, n_cores=1)
          # plottingTool.plot_detections_parallel(bolideDetectionList, provenance, output_dir, 
          #         latLon2Pix_table_path=latLon2Pix_table_path,
          #         latLon2Pix_table_path_inverted=latLon2Pix_table_path_inverted, n_cores=1, otherSatelliteFlag=True)
          # print('*********REMOVE ME*******************')
          # print('*********REMOVE ME*******************')
          # print('*********REMOVE ME*******************')

            #***
            # Generate the cutout figure
            if input_config.cutout_enabled:

                cutoutConfig = cutoutConfiguration()
                satellite = determine_satellite_from_file_path(input_config.data_path)
                if satellite == 'G16':
                    cutoutConfig.G16DIR = input_config.data_path
                    cutoutConfig.G17DIR = input_config.other_satellite_input_path 
                    cutoutConfig.G18DIR = input_config.other_satellite_input_path 
                elif satellite == 'G17':
                    cutoutConfig.G17DIR = input_config.data_path
                    cutoutConfig.G16DIR = input_config.other_satellite_input_path 
                    cutoutConfig.G19DIR = input_config.other_satellite_input_path 
                elif satellite == 'G18':
                    cutoutConfig.G18DIR = input_config.data_path
                    cutoutConfig.G16DIR = input_config.other_satellite_input_path 
                    cutoutConfig.G19DIR = input_config.other_satellite_input_path 
                elif satellite == 'G19':
                    cutoutConfig.G19DIR = input_config.data_path
                    cutoutConfig.G17DIR = input_config.other_satellite_input_path 
                    cutoutConfig.G18DIR = input_config.other_satellite_input_path 
                else:
                    raise Exception('Provenance Warning: Cannot determine if this is G16 or G17 data')
                cutoutConfig.data_source = input_config.data_source
                cutoutConfig.annotations = input_config.cutout_annotations
                cutoutConfig.G16ABICDIR = input_config.cutout_G16ABICDIR
                cutoutConfig.G16ABIFDIR = input_config.cutout_G16ABIFDIR
                cutoutConfig.G17ABICDIR = input_config.cutout_G17ABICDIR
                cutoutConfig.G17ABIFDIR = input_config.cutout_G17ABIFDIR
                cutoutConfig.G18ABICDIR = input_config.cutout_G18ABICDIR
                cutoutConfig.G18ABIFDIR = input_config.cutout_G18ABIFDIR
                cutoutConfig.G19ABICDIR = input_config.cutout_G19ABICDIR
                cutoutConfig.G19ABIFDIR = input_config.cutout_G19ABIFDIR
                cutoutConfig.coastlines_path = input_config.cutout_coastlines_path
                cutoutConfig.n_cores = input_config.cutout_n_cores
                if cutoutConfig.n_cores is None:
                    cutoutConfig.n_cores = input_config.n_cores
                cutoutConfig.plot_seperate_figures = input_config.cutout_plot_seperate_figures
                cutoutConfig.generate_multi_band_data = input_config.cutout_generate_multi_band_data
                cutoutConfig.SEPERATE_FIG_MARKERSCALE  = input_config.cutout_plot_glm_circle_size
                cutoutConfig.min_GB = input_config.min_GB_cutout_tool
                cutoutConfig.bands_to_read = input_config.cutout_bands_to_read
                # Try to run cutout tool. If fails throw last error
                try:
                    cutoutFeatures = gen_goes_cutout(bolideDetectionList, output_dir, cutoutConfig)
                except Exception as e:
                    # Fill cutoutFeatures with blank data
                    cutoutFeatures = {}
                    for detection in bolideDetectionList:
                        cutoutFeatures[detection.ID] = cutoutOutputClass(cutoutConfig, detection.ID)
                    print('\n****** Error generating cutout figures for {}'.format(databaseFilePath))
                    print_exc()
                    
                if (input_config.verbosity):
                    print('\n****** Successfully generated cutout figures')
            else:
                # Fill cutoutFeatures with blank data
                cutoutFeatures = {}
                cutoutConfig = cutoutConfiguration(cutout_enabled=False)
                for detection in bolideDetectionList:
                    cutoutFeatures[detection.ID] = cutoutOutputClass(cutoutConfig, detection.ID)
                
            # Add cutout features to bolideDetectionList
            for detection in bolideDetectionList:
                detection.add_cutout_features(cutoutFeatures[detection.ID])


        # Delete the temporary raw netCDF file symlinks
        # TODO: This is awkward, clean this up so we do not need to temporarily make these symlinks
        if not input_config.copyNetCDFFiles:
            for filename in raw_bolide_data_files:
                # Do not delete the original, only the symlink
                filename = os.path.join(output_dir, os.path.split(filename)[1])
                os.unlink(filename)
            for fileList in otherSatelliteFilenames.values():
                for filename in fileList:
                    filename = os.path.join(output_dir, filename)
                    if os.path.exists(filename):
                        os.unlink(filename)



        if (input_config.verbosity): print('Data saved to {}.'.format(databaseFilePath))
        

    else:
        if (input_config.verbosity): print("There were no valid detections.")

    endTime = time.time()
    totalTime = endTime - startTime
    if (input_config.verbosity):
        print("Total processing time: {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))

    return num_bolides, bolideDetectionList, bolideRejectionList 

#*************************************************************************************************************
def _detect_bolides_in_multiple_files(fileList, input_config, glintFilter):
    """
    This function takes a list of netCDF files and finds bolides in all of them sequentially
   
    calls _detect_bolides_in_file
   
    Parameters
    ----------
    fileList    : str list
        path to netCDF files to process
    input_config : bolide_io.input_config object
    glintFilter : GlintFilter object 
        Initialized glint filter
   
    Returns
    -------
    bolideDetectionList : list of BolideDetection objects 
        The detected bolides from all files
    bolideRejectionList : list of BolideDetection objects 
        The rejected bolides from all files
    oops_counter        : int 
        Number of netCDF files experiencing an error
    
    """

    oops_counter = 0

    bolideDetectionList = []
    bolideRejectionList = []

    # Do processing on a single core.
    # If we are doing multi-core processing then that is handled in detect_bolides 
    for file_path in fileList:
        [status, detectedBolides, rejectedBolides] = \
            _detect_bolides_in_file(file_path, input_config, glintFilter)
        bolideRejectionList.extend(rejectedBolides)
        if status == 0:
            continue
        elif status == -99:
            oops_counter += 1
            continue
        # Else, we have a detection, add it to the list
        bolideDetectionList.extend(detectedBolides)

    return [bolideDetectionList, bolideRejectionList, oops_counter]

#*************************************************************************************************************
def _detect_bolides_in_file(file_path, input_config, glintFilter):
    """
    This function loads in a single netCDF file and tries to detect bolides.
   
    Parameters
    ----------
    file_path       : str 
        path to netCDF file
    input_config    : bolide_io.input_config object
    glintFilter     : GlintFilter object 
        Initialized glint filter
   
    Returns
    -------
    status  : int
        0 => no bolides detected, 1 => bolides detected, -99 => An Ups occured
    bolideDetectionList : list of bolideDetection objects 
        The detected bolides
    bolideRejectionList : list of bolideDetection objects
        The rejected bolides
   
    """
    
    assert (not(input_config.bolidesFromWebsite is not None and 
            input_config.trained_classifier_path != '' and 
            input_config.trained_classifier_path != [])), \
                'Choose either ground_truth_path or trained_classifier_path, not both'

    # Use try so we capture and print exceptions errors but not stop program execution
    try: 
        bolideRejectionList = []
        if (input_config.bolidesFromWebsite is not None):
            [status, bolideDetectionList, bolideRejectionList] = \
                _detect_bolides_from_ground_truth (file_path, input_config, None)
        elif (input_config.trained_classifier_path != '' and input_config.trained_classifier_path != []):
            [status, bolideDetectionList] = \
                _detect_bolides_trained_classifier (file_path, input_config, None)
        else:
            [status, bolideDetectionList] = \
                _detect_bolides_sequential_filters (file_path, input_config, glintFilter)

    except Exception as e:
        print('\n****** Error processing file {}'.format(file_path))
        print_exc()

        return [-99, [], []]
        
    else:
        if (input_config.verbosity):
            print("Finished working on file: {0:s}".format(file_path))
            print(" ")

        return [status, bolideDetectionList, bolideRejectionList]

#*************************************************************************************************************
# function _detect_bolides_from_ground_truth (file_path, input_config, glintFilter, randomSeed=None)
#
# Uses the passed ground truth in input_config.bolidesFromWebsite to determine the which clusters in netCDF are bolides.
# Use bolide_dispositions.pull_dispositions_from_website to get the <bolideFromWebsite> pickle file.
#
# If a glintFilter object is passed then the glint filter will be applied after clustering and 
# before searching for bolides.
#
# Rejected clusters will also be saved as determined by input_config.rejectsToSaveFrac which gives the fraction of
# rejected clusters to save. Saving all will results in very large data files, so choose a small fraction This works by
# giving each rejection thsi fraction of a chance to be saved, so, the exact final ratio will be subject to poisson
# statistics.
#
# Inputs: 
#   file_path       -- [str] path to netCDF file
#   input_config    -- bolide_io.input_config object
#       .bolidesFromWebsite  -- [WebsiteBolideEvent list] The list of ground truth bolides
#   glintFilter     -- [GlintFilter object] Initilized glint filter
#                       pass None to NOT use the glint filter
#   randomSeed      -- [int] Random seed for rejectsToSaveFrac 
#
# Outputs:   
#   status              -- 0 => no bolides detected, 1 => bolides detected, -99 => An Ups occured
#   bolideDetectionList -- [List of BolideDetection objects] The detected bolides
#   bolideRejectionList -- [List of BolideDetection objects] The rejected bolides
#
#*************************************************************************************************************
def _detect_bolides_from_ground_truth (file_path, input_config, glintFilter, randomSeed=None):

    # Random number generator to determine if we save a rejected cluster
    np.random.seed(randomSeed)
            
    # We want the list of bolides to be part of a ZODB database, so use BTrees
    bolideDetectionList = []
    bolideRejectionList = []

    # How close does the beginning or end the clusters need to be to the truth bolides to be considered a hit
    distDeltaRange = 50.0 # kilometers (was: 50.0)

  # # Bolide confidence 
  # # neo-bolides website uses the confidence of {'low', 'medium', 'high'}
  # # Define confidence as low = 0.5, medium = 0.75, high = 0.95
  # scoreSwitcher = {
  #     'low': 0.5,
  #     'medium': 0.75,
  #     'high': 0.95,
  #     'unknown': -1
  #     }
  # scoreSwitcher = bDisp.bolideWebsiteBeliefSwitcher

    # Somewhat arbitrarily set rejection confidence to 0.25
   #rejection_confidence = 0.25

    # We first extract the groups from the netCDF file and form clusters
    # Extract the L2 Data from the .nc file
    BCObj = bCluster.BolideClustering(file_path,
                cluster_3D_enabled=input_config.cluster_3D_enabled, 
                numba_threads=input_config.cluster_numba_threads, 
                closeness_seconds=input_config.cluster_closeness_seconds, 
                closeness_km=input_config.cluster_closeness_km, 
                extractNeighboringFiles=input_config.cluster_extractNeighboringFiles, 
                min_num_groups_for_outliers=input_config.cluster_min_num_groups_for_outliers ,
                outlierSigmaThreshold=input_config.cluster_outlierSigmaThreshold, 
                outlierMinClipValueDegrees=input_config.cluster_outlierMinClipValueDegrees,
                spatiotemporal_box=input_config.spatiotemporal_box)
    goesSatellite = BCObj.goesSatellite

    # Check if this is an empty file with no groups to process
    if (len(BCObj.glmGroups) < 1):
        return [0, [], []]

    #%% Glint filter
    if glintFilter is not None:
        raise Exception('Are you sure you want to turn on the glint filter?')
        BCObj.glmGroups = bff.apply_glint_filter(glintFilter, BCObj.goesSatellite, BCObj.productTime, BCObj.glmGroups)
        # Check if this is now an empty file with no groups to process after glint removal
        if (len(BCObj.glmGroups) < 1):
            return [0, [], []]

    # Organize data, Cluster the GLM groups into bolide detections
    success = BCObj.cluster_glm_groups(
                sequential_clustering_enabled=input_config.cluster_sequential_enabled, 
                outlier_rejection_enabled=input_config.cluster_outlier_rejection_enabled)
    if (not success):
        return [0, [], []]

    #***
    # Get same extra data from the truth bolides
    bolideYear          = []
    bolideMonth         = []
    bolideDay           = []
    bolideConfidence    = []
    howFound            = []
    nBolides = len(input_config.bolidesFromWebsite)
   #counter = 0
    for bolide in input_config.bolidesFromWebsite:
        idx = np.nonzero(np.in1d(bolide.satellite, goesSatellite))[0]
        if len(idx) > 1:
            # We have multiple bolide light curves on the website for this one satellite.
            # Pick the one that is the longest in time
            idx = [np.argmax(np.diff(np.array(bolide.timeRange)[idx]))]
           #counter += 1
        if (len(idx) == 1):
            idx = idx[0]
            # Found a bolide for the same GOES Satelite data as in file_path
            bolideConfidence.append(bolide.confidence)
            howFound.append(bolide.howFound)
            bolideYear.append(bolide.timeRange[idx][0].year)
            bolideMonth.append(bolide.timeRange[idx][0].month)
            bolideDay.append(bolide.timeRange[idx][0].day)
        else:
            # No match for this bolide on this satellite
            bolideConfidence.append(np.nan)
            howFound.append(np.nan)
            bolideYear.append(np.nan)
            bolideMonth.append(np.nan)
            bolideDay.append(np.nan)

    # Only search if there are any bolides within the day of this file.
    # This helps to ensure the rejections are only generated for days that have already been vetted
    thisFileYear    = BCObj.productTime.year
    thisFileMonth   = BCObj.productTime.month
    thisFileDay     = BCObj.productTime.day
    if (not np.any(np.logical_and(np.logical_and(thisFileYear == np.array(bolideYear), thisFileMonth == np.array(bolideMonth)), 
                                    thisFileDay == np.array(bolideDay)))):
        return [0, [], []]


    #***
    # Search clusters for agreement with ground truth
    # Search for agreement in time, lat and lon ranges
    bolidesWithAClusterMatch = np.full((nBolides,1), False)
    bolideDetectionListThisMatchIndex = [] # This lines up each bolide detection with the index in the ground truth list

    # Create bolideDispositionProfileList from all clusters
    bolideDetectionListAllClusters = []
    for idx,_ in enumerate(BCObj.clusteredGroups):
        bolideDetectionListAllClusters.append(bd.BolideDetection(BCObj, idx))
    bolideDetectionListAllClusters = bolideDB.generate_random_IDs(bolideDetectionListAllClusters)

    # Compute the features
    # (Needed for find_bolide_match_from_website)
    # We don't need all features computed (They are recomputed below for the detection and rejections)
    # Disable the slowest (glint (spice_kernel_path=None) and neighborhood (neighborhood_feature_enabled=False), and
    # legacy filters (legacy_filters_enabled=False))
    bolideDetectionListAllClusters = bFeatures.compute_bolide_features(bolideDetectionListAllClusters, 
            bolideDatabase=None, glmGroupsWithNeighbors=BCObj.glmGroupsWithNeighbors, 
            multiProcessEnabled=False, spice_kernel_path=None, hotSpots=None,
            neighborhood_feature_enabled=False, legacy_filters_enabled=False,
            verbosity=input_config.verbosity)

    # Convert to bolide dispositions
    bolideDispositionProfileList = []
    for detection in bolideDetectionListAllClusters:
        bolideDispositionProfileList.append(bDisp.BolideDispositionProfile(detection.ID, bolideDetection=detection,
            features=detection.features))

    # Find cluster matches to ground truth bolides
    matchIndexArray = bPerformance.find_bolide_match_from_website(bolideDispositionProfileList, 
            input_config.bolidesFromWebsite, BCObj.goesSatellite)
    
    # Go through each cluster, If it's a match, either create a new detection or add to an already created detection for
    # this website bolide. If it is not a match then roll the dice to add it to the rejections list.
    for clusterIdx, matchIndex in enumerate(matchIndexArray):
        cluster = BCObj.clusteredGroups[clusterIdx]

        if (matchIndex >= 0):
            # We found a match! 
            
            # Confidence Score
            # TODO: Update the score by how well the cluster matches the ground truth bolide
            confidenceSelection = bolideWebsiteBeliefSwitcher.get(bolideConfidence[matchIndex], bolideWebsiteBeliefSwitcher['unknown'])
            if (confidenceSelection == bolideWebsiteBeliefSwitcher['unknown']):
                # We found a match to a website bolide with an unknown confidence. What does this mean?
                # For now, just skip this bolide match.
                # TODO: investigate what is happening here.
              # warnings.warn("Unknown website bolide belief confidence: {}".format(bolideConfidence[matchIndex]))
              # continue
                raise Exception("Unknown website bolide belief score: {}".format(bolideConfidence[matchIndex]))
            
            # First check if we matched up with an already matched bolide. If so, merge the clusters.
            # NOTE: Disable cluster merging. I believe we want to leave the clusters just as they will be generated in a
            # detection run
            if (False and bolidesWithAClusterMatch[matchIndex]):
                # Merge with already existant bolide
                existantBolideIndex = np.nonzero(bolideDetectionListThisMatchIndex == np.array(matchIndex))[0]
                if (len(existantBolideIndex) != 1):
                    raise Exception('Error bookeeping existant detected bolides')
            
                previousGroupIds = [g.id for g in bolideDetectionList[existantBolideIndex[0]].groupList]
                tempClusterGroupIds = np.union1d(previousGroupIds, cluster['groupIds'])
                assessment = BolideAssessment()
                assessment.human.assessment = bolideConfidence[matchIndex]
                assessment.human.source = 'website'
                tempBolideDetection = bd.BolideDetection(BCObj, clusterIdx,
                        assessment=assessment, 
                        howFound=howFound[matchIndex], retrieveEvents=False)
                bolideDetectionList[existantBolideIndex[0]].add_data(tempBolideDetection)
            
            elif(len(BCObj.clusteredGroups[clusterIdx]['groupIndices']) >= input_config.min_num_groups):
                # NOTE: Only add the detection if the number of groups is >= min_num_groups
                # No such bolide exists, create a new bolide candidate
                bolidesWithAClusterMatch[matchIndex] = True
                bolideDetectionListThisMatchIndex.append(matchIndex)
                assessment = BolideAssessment()
                assessment.human.assessment = bolideConfidence[matchIndex]
                assessment.human.source = 'website'
                bolideDetectionList.append(bd.BolideDetection(BCObj, clusterIdx,
                        assessment=assessment, 
                        howFound=howFound[matchIndex], retrieveEvents=False))
                

        elif (len(cluster['groupIds']) >= input_config.min_num_groups):
            # This is a rejection. Consider adding to the rejection list
            # But only if a minimum number of groups is in the cluster
            # Roll the dice to check if we save this rejection
            if (np.random.rand() <= input_config.rejectsToSaveFrac):
                # We don't have knowledge of the rejection confidence. Default to 0.25
                assessment = BolideAssessment()
                assessment.human.assessment = 'rejected'
                assessment.human.source = 'website'
                bolideRejectionList.append(bd.BolideDetection(BCObj, clusterIdx,
                        assessment=assessment, 
                        howFound='rejected', retrieveEvents=False))

    #***
    # Compute the features.
    # The new neighborhood feature uses the entire netCDF file, not just the detection. 
    # It would be very slow to compute that feature afterwards because we would have to un-tar and process each data file
    # again. So, we have to compute it now during training data set generation.
    # The detections are all within the same file (with some over-leak, so the glmGroupsWithNeighbors is the same for
    # all detections.
    # The hot spot feature object has to be constructed
    hotSpots = bFeatures.HotSpots(input_config.lon_peaks,input_config.lat_peaks)
    if (len(bolideDetectionList) > 0):
        bolideDetectionList = bolideDB.generate_random_IDs(bolideDetectionList)
        bolideDetectionList = bFeatures.compute_bolide_features(bolideDetectionList, 
                bolideDatabase=None, glmGroupsWithNeighbors=BCObj.glmGroupsWithNeighbors, 
                multiProcessEnabled=False, spice_kernel_path=input_config.spice_kernel_path, hotSpots=hotSpots,
                verbosity=input_config.verbosity)
        status = 1
    else:
        status = 0

    if (len(bolideRejectionList) > 0):
        bolideRejectionList = bolideDB.generate_random_IDs(bolideRejectionList)
        bolideRejectionList = bFeatures.compute_bolide_features(bolideRejectionList, 
                bolideDatabase=None, glmGroupsWithNeighbors=BCObj.glmGroupsWithNeighbors, 
                multiProcessEnabled=False, spice_kernel_path=input_config.spice_kernel_path, hotSpots=hotSpots,
                verbosity=input_config.verbosity)


    return [status, bolideDetectionList, bolideRejectionList]
    
#*************************************************************************************************************
# function _detect_bolides_trained_classifier (file_path, input_config, glintFilter)
#
# Uses a trained Scikit-Learn classifier to classify GLM clusters.
#
# If a glintFilter object is passed then the glint filter will be applied after clustering and 
# before searching for bolides.
#
# Inputs: 
#   file_path       -- [str] path to netCDF file
#   input_config    -- bolide_io.input_config object
#       .trainedClassifierDict  -- [dict] Contains the trained classifer info
#           'trainedClassifier' -- [sklearn classifier object] The trained classifier
#                                   Must have a 'predict' method
#           'featuresToUse'     -- [str list] Features used in classifier
#           'columnTransformer' -- [sklearn ColumnTransformer object] The already fit transformer to normalize the features
#       .classifier_threshold   -- [float] Bolide detection threshold
#       .min_num_groups         -- [int] minumum number of groups in cluster to consdier as bolide
#       .spice_kernel_path      -- [str] Path to spice kernels for glint feature
#       .max_num_detections     -- [int] Maximum number of detections over this unit of work
#                                   (ranked by detection score)
#   glintFilter     -- [GlintFilter object] Initilized glint filter
#                       pass None to NOT use the glint filter
#
# Outputs:   
#   status              -- 0 => no bolides detected, 1 => bolides detected, -99 => An Ups occured
#   bolideDetectionList -- [list of BolideDetection objects] The detected bolides
#
#*************************************************************************************************************
def _detect_bolides_trained_classifier (file_path, input_config, glintFilter):

    trainedClassifier   = input_config.trainedClassifierDict['trainedClassifier']
    featuresToUse       = input_config.trainedClassifierDict['featuresToUse']
    columnTransformer   = input_config.trainedClassifierDict['columnTransformer']

    # We first extract the groups from the netCDF file and form clusters
    # Extract the L2 Data from the .nc file
    BCObj = bCluster.BolideClustering(file_path,
            cluster_3D_enabled=input_config.cluster_3D_enabled, 
            numba_threads=input_config.cluster_numba_threads, 
            closeness_seconds=input_config.cluster_closeness_seconds, 
            closeness_km=input_config.cluster_closeness_km, 
            extractNeighboringFiles=input_config.cluster_extractNeighboringFiles, 
            min_num_groups_for_outliers=input_config.cluster_min_num_groups_for_outliers ,
            outlierSigmaThreshold=input_config.cluster_outlierSigmaThreshold, 
            outlierMinClipValueDegrees=input_config.cluster_outlierMinClipValueDegrees,
            spatiotemporal_box=input_config.spatiotemporal_box)
    goesSatellite = BCObj.goesSatellite

    # Check if this is an empty file with no groups to process
    if (len(BCObj.glmGroups) < 1):
        return [0, []]

    #%% Glint filter
    if glintFilter is not None:
        raise Exception('Are you sure you want to turn on the glint filter?')
        BCObj.glmGroups = bff.apply_glint_filter(glintFilter, BCObj.goesSatellite, BCObj.productTime, BCObj.glmGroups)
        # Check if this is now an empty file with no groups to process after glint removal
        if (len(BCObj.glmGroups) < 1):
            return [0, []]

    # Organize data, Cluster the GLM groups into bolide detections
    success = BCObj.cluster_glm_groups(
                sequential_clustering_enabled=input_config.cluster_sequential_enabled, 
                outlier_rejection_enabled=input_config.cluster_outlier_rejection_enabled)
    if (not success):
        return [0, []]

    #***
    # Classify the groups

    # Remove clusters with fewer than input_config.min_num_groups groups
    reducedClusteredGroupsIndices = [idx for idx,cluster in enumerate(BCObj.clusteredGroups) if 
                                        len(cluster['groupIds']) >= input_config.min_num_groups]
    # Check if there are no clusters left to process
    if (len(reducedClusteredGroupsIndices) < 1):
        return [0, []]

    #***
    # Create bolideDispositionProfileList from all remaining clusters
    bolideDetectionListAllClusters = []
    for idx in reducedClusteredGroupsIndices:
        bolideDetectionListAllClusters.append(bd.BolideDetection(BCObj, idx))

    # Convert to bolide dispositions
    # These are just clusters, not detections, so no detection ID. But for bookkeeping purposes we still need an ID
    # Use a temporary random uint32 (to distinguish it from a proper int64 in the database)
    bolideDetectionListAllClusters = bolideDB.generate_random_IDs(bolideDetectionListAllClusters)
    bolideDispositionProfileList = []
    for idx, detection in enumerate(bolideDetectionListAllClusters):
        bolideDispositionProfileList.append(bDisp.BolideDispositionProfile(detection.ID, bolideDetection=detection, 
            humanOpinions=[bDisp.HumanOpinion()]))

    # Compute the features
    # Can be parallelized but cannot if detect_bolides is already parallelized 
    # (python multiprocessing does not allow it!)
    # Generally speaking, detect_bolides is run in the pipeline, which is already parallelized, if not then we are
    # debugging, in which case, we don't want children so the debugger can track all the code, so disable
    # multiprocessing
    # The hot spot feature object has to be constructed
    hotSpots = bFeatures.HotSpots(input_config.lon_peaks,input_config.lat_peaks)
    bDispObj = bDisp.BolideDispositions.from_bolideDispositionProfileList(bolideDispositionProfileList, 
            spice_kernel_path=input_config.spice_kernel_path, glmGroupsWithNeighbors=BCObj.glmGroupsWithNeighbors, 
            multiProcessEnabled=False, compute_features=True, hotSpots=hotSpots)
    [X, _, _, _, _, responseLabels, _, _] = bDispObj.create_ML_data_set( 
        featuresToUse=featuresToUse, computeMachineOpinionsMode=True, columnTransformer=columnTransformer,
        beliefSource='triage')

    #***
    # Predict!
    if (X is not None):
        yPredict = trainedClassifier.predict_proba(X)[:,1]
    else:
        yPredict = []

    #**********************
    #**********************
    #**********************
    # TEST: ATAP GitHub Issue #15: Very large clusters in lat/lon
    # Save out found_distance for all clusters
    # Each netCDF file will have a seperate file
    # Keeping this code in just in case we want to perform this experiment again.
    if False:
        netCDF_filename = os.path.split(file_path)[1]
        ground_dist_filename = os.path.join(input_config.output_path, 'ground_distance', '{}_ground_dist.p'.format(os.path.splitext(netCDF_filename)[0]))
        ground_distance_array = [d.features.ground_distance for d in bDispObj.bolideDispositionProfileList]
        assert len(ground_distance_array) == len(yPredict), 'Lengths do not match!'
        with open(ground_dist_filename, 'ab') as fp :
            pickle.dump([ground_distance_array, yPredict], fp)
        fp.close()

    #**********************
    #**********************
    #**********************

    #***
    # Deemphasize hot spots
    for idx, predictVal in enumerate(yPredict):
        yPredict[idx] -= bDispObj.deemphasisFcn(input_config.deemph_radius, input_config.deemph_alpha,
                bDispObj.bolideDispositionProfileList[idx].features.hotSpot)
    # New bolide belief must be between 0.0 and 1.0
    yPredict = [np.max([0.0, b]) for b in yPredict]
    yPredict = [np.min([1.0, b]) for b in yPredict]

    #***
    # Force candidacy for minimum number of groups
    #print('*******************************')
    #print('Number pass triage = {}'.format(np.count_nonzero(np.greater_equal(yPredict, input_config.classifier_threshold))))
    candidacy_forced = np.full(np.shape(yPredict), False, dtype=bool)
    if input_config.min_num_groups_to_force_candidacy >= 0:
        for idx, predictVal in enumerate(yPredict):
            if bDispObj.bolideDispositionProfileList[idx].features.nGroups >= input_config.min_num_groups_to_force_candidacy:
                candidacy_forced[idx] = True
    #    print('Number of forced candidates = {}'.format(np.count_nonzero(candidacy_forced)))
    #print('*******************************')

    #***
    # Add clusters predicted to be bolides to bolideDetectionList
    # Or those that are forced
    bolideDetectionList = []
    for idx, predictVal in enumerate(yPredict):
        if (predictVal >= input_config.classifier_threshold or candidacy_forced[idx]):
            bolideDetectionFromCluster = bd.BolideDetection(BCObj, reducedClusteredGroupsIndices[idx],
                        howFound='forest', retrieveEvents=True, 
                        features=bDispObj.bolideDispositionProfileList[idx].features, 
                        latLon2Pix_table_path=input_config.latLon2Pix_table_path, 
                        latLon2Pix_table_path_inverted=input_config.latLon2Pix_table_path_inverted )

            # Set the Triage assessment
            bolideDetectionFromCluster.assessment = BolideAssessment(triage_threshold=input_config.classifier_threshold)
            bolideDetectionFromCluster.assessment.triage.score = predictVal
            bolideDetectionFromCluster.assessment.triage.method = os.path.basename(input_config.trained_classifier_path)
            bolideDetectionFromCluster.assessment.triage.candidacy_forced = candidacy_forced[idx]
            bolideDetectionList.append(bolideDetectionFromCluster)

    if (bolideDetectionList == []):
        # no detections
        return [0, []]
    else:
        return [1, bolideDetectionList]
    

#*************************************************************************************************************
# function _detect_bolides_sequential_filters (file_path, input_config, glintFilter):
# 
# This function will detect bolides based on Clemens' sequential filters as referenced in 
#
# note: some modification have been made since the paper.
#
# Inputs: 
#   file_path       -- [str] path to netCDF file
#   input_config    -- bolide_io.input_config object
#   glintFilter     -- [GlintFilter object] Initilized glint filter
#
# Outputs:   
#   status          -- 0 => no bolides detected, 1 => bolides detected, -99 => An Ups occured
#   bolideDetectionList -- [list of BolideDetection objects] The detected bolides
#
#*************************************************************************************************************
def _detect_bolides_sequential_filters (file_path, input_config, glintFilter):

    raise Exception('This code has not been run for a very long time. Do some tests before running it')

    try:

        # Extract the L2 Data from the .nc file
        BCObj = bCluster.BolideClustering(file_path,
                cluster_3D_enabled=input_config.cluster_3D_enabled, 
                numba_threads=input_config.cluster_numba_threads, 
                closeness_seconds=input_config.cluster_closeness_seconds, 
                closeness_km=input_config.cluster_closeness_km, 
                extractNeighboringFiles=input_config.cluster_extractNeighboringFiles, 
                min_num_groups_for_outliers=input_config.cluster_min_num_groups_for_outliers ,
                outlierSigmaThreshold=input_config.cluster_outlierSigmaThreshold, 
                outlierMinClipValueDegrees=input_config.cluster_outlierMinClipValueDegrees,
                spatiotemporal_box=input_config.spatiotemporal_box)

        # Check if this is an empty file with no groups to process
        if (len(BCObj.glmGroups) < 1):
            return [0, []]

        #%% Glint filter
        BCObj.glmGroups = bff.apply_glint_filter(glintFilter, BCObj.goesSatellite, BCObj.productTime, BCObj.glmGroups)
        # Check if this is now an empty file with no groups to process after glint removal
        if (len(BCObj.glmGroups) < 1):
            return [0, []]

        #%% organize data, Cluster the GLM groups into bolide detections
        success = BCObj.cluster_glm_groups()
        if (not success):
            return [0, []]

        #***
        # Form an old-style "bolide" class from the clustered groups
        # Define the bolideClusters
        bolideClusters = [ ]
        groupIdList = np.array([o.id for o in BCObj.glmGroups])
        for i in np.arange(len(BCObj.clusteredGroups)):
            # First create each bolide using the first group in each cluster
            groupIdsThisBolide = BCObj.clusteredGroups[i]['groupIds']
            firstGroupIndex = list(np.nonzero(np.in1d(groupIdList, groupIdsThisBolide[0]))[0])
            firstGroup = BCObj.glmGroups[firstGroupIndex[0]]
            dtime = firstGroup.time
            firstGroupOldStyle = bsf.Group(firstGroup.id, firstGroup.latitudeDegreesNorth, 
                    firstGroup.longitudeDegreesEast, dtime, firstGroup.energyJoules)
            bolideClusters.append(bsf.Bolide(firstGroupOldStyle)) # start a new bolide instance
    
            # Then add the rest of the groups to this bolide
            # Be sure to preserve the group time sort order
            timeArray  = BCObj.clusteredGroups[i]['time']
            theRestGroupIdsSorted = [x for _,x in sorted(zip(timeArray[1:],groupIdsThisBolide[1:]))]

            groupsArray = []
            for groupIndex in theRestGroupIdsSorted:
                group = BCObj.glmGroups[np.nonzero(np.in1d(groupIdList, groupIndex))[0][0]]
                dtime = group.time
                groupsArray.append(bsf.Group(group.id, group.latitudeDegreesNorth, 
                        group.longitudeDegreesEast, dtime, group.energyJoules))
            bolideClusters[-1].vector_add(groupsArray)
            bolideClusters[-1].counter = len(bolideClusters[-1].group)


        #***
        # Skip the current file if too many groups. In such a case, ExtractGroupsFromEvents will return the number of
        # groups along with an empty 'bolide' variable.
        if len(bolideClusters)>0 and not bolideClusters:
            print('Too many groups ({}) to cluster: Skipping file {}'.format(len(bolideClusters), file_path))
            return [0, []]

        #***
        #%% Group count filter

        [bolide_index_list, cum_bolide_probability_list] = bff.FilterBolidesByGroupCount(bolideClusters, input_config.bolide_probability_cutoff, \
                input_config.group_05)

        #***
        #%% Time filter

        [bolide_index_list, cum_bolide_probability_list] = bff.FilterBolidesByTime(bolideClusters, bolide_index_list, cum_bolide_probability_list, \
                input_config.bolide_probability_cutoff, input_config.time_05)

        #***
        #%% Apply linelet filter

        [bolide_index_list, cum_bolide_probability_list] = bff.FilterBolidesByLatLonLinelets(bolideClusters, bolide_index_list, cum_bolide_probability_list, \
                input_config.bolide_probability_cutoff, input_config.linelet_05)

        #***
        #%% Energy Deposition Profile Balance filter

        [bolide_index_list, cum_bolide_probability_list] = bff.FilterBolidesByEnergyRatio(bolideClusters, bolide_index_list, cum_bolide_probability_list, \
                input_config.bolide_probability_cutoff, input_config.energy_05)


        #***
        #%% Filter by fitting splinelets to a moving window over the energy deposition profile

        [bolide_index_list, cum_bolide_probability_list] = bff.FilterBolidesBySplinelets(bolideClusters, bolide_index_list, cum_bolide_probability_list, \
                input_config.bolide_probability_cutoff, input_config.spline_05)

        #***
        #%% Remove events that have large deviations from a ground track line fit

        [bolide_index_list, cum_bolide_probability_list] = bff.FilterBolidesByGroundTrackDeviation(bolideClusters, bolide_index_list, cum_bolide_probability_list, \
                input_config.bolide_probability_cutoff, input_config.dist_km_05)



        #%% Announce final detection outcome

        if len(bolide_index_list) < 1:
            if (input_config.verbosity):
                print("No suitable detection in this file.")
                print("Finished working on file: {0:s}".format(file_path))
                print(" ")
            return [0, []]

        bolide_list_index_ascending_order = np.argsort(cum_bolide_probability_list)
        bolide_list_index_descending_order = bolide_list_index_ascending_order[::-1]

        # Pick the highest ranked bolide
        bolide_list_index = bolide_list_index_descending_order[0]

        bolide_index = bolide_index_list[bolide_list_index]

        if (input_config.verbosity):
            print("There are {:d} detections in this file".format(len(bolide_index_list)))
            if len(bolide_index_list) > 1:
                print('The two highest bolide scores are {:.2f} and {:.2f} ... ' 
                        'index {:d} and {:d}... continuing with highest scoring one.'.format(
                            cum_bolide_probability_list[bolide_list_index], 
                            cum_bolide_probability_list[bolide_list_index_descending_order[1]], 
                            bolide_index_list[bolide_list_index], 
                            bolide_index_list[bolide_list_index_descending_order[1]]))

            if cum_bolide_probability_list[bolide_list_index] < input_config.bolide_probability_cutoff:
                print("FAILED BOLIDE ASSESSMENT WITH P = {:.2f}".format(cum_bolide_probability_list[bolide_list_index]))
                return [0, []]
            else:
                print("PASSED BOLIDE ASSESSMENT WITH P = {:.2f}, index is {:d}".format(cum_bolide_probability_list[bolide_list_index], bolide_index))

        # Convert to new bolide detection class
        if (isinstance(bolide_index, int)):
            numFiles = 1
        else:
            raise Exception ('This code right now only accepts one detection per file')
        bolideList = []
        for i in range(numFiles):
            bolideDetection = bolideClusters[bolide_index]
            bolideDetection.filePathStr = file_path
            bolideList.append(bolideDetection)

        bolideDetectionList = bd.convert_detection_representation(bolideList, 
                confidence=cum_bolide_probability_list[bolide_list_index], confidenceSource='sequential filters',
                howFound='sequential filters pipeline')
        score = cum_bolide_probability_list[bolide_list_index]

        # Success! Return bolide detections
        return [1, bolideDetectionList]

    except:
        print(" ")
        print("Ups, I jumped over file: {0:s}".format(file_path))
        print(" ")
        return [-99, []]


"""
#*************************************************************************************************************
#
# This takes the config dictionary created by bio.read_config_file(detectBolidesConfigFile) and converts the parameters
# to a input_config class for use by detect_bolides.
#
# TODO: simplify this process so it is not a two-step process. 
#
# Inputs:
#   config  -- [Dict] output of bio.read_config_file(detectBolidesConfigFile)
#
# Outputs:
#   input_config -- [input_config class] used to store the configuration information for detect_bolides
#
#*************************************************************************************************************
def config_to_input_config_object(config, verbosity=False):

    raise Exception('This function has been moved to bolide.io.py')

    verbosity = config['verbosity']

    #***
    # Make sure path strings are terminated with a '/' character.
    data_path = config['data_path']
    if not data_path[len(data_path) - 1] == '/':
        config['data_path'] = data_path + '/'

    other_satellite_input_path = config['other_satellite_input_path']
    if not other_satellite_input_path[len(other_satellite_input_path) - 1] == '/':
        config['other_satellite_input_path'] = other_satellite_input_path + '/'

    output_path = config['output_path']
    if not output_path[len(output_path) - 1] == '/':
        config['output_path'] = output_path + '/'

    #***
    # Check to see whether the output directory exists and create it if necessary.
    create_path(output_path, verbosity)

    # Create the input_config object
    input_config = bio.input_config(config)

    return input_config
"""

#*************************************************************************************************************
# In the stereo region copy over the netCDF files from the other satellite.
#
# To determine if we are in the stereo region we can use the following fields in each netCDF file:
#   lat_field_of_view_bounds (1) (north)
#   lat_field_of_view_bounds (2) (south)
#   lon_field_of_view_bounds (1) (west)
#   lon_field_of_view_bounds (2) (east)
#
# This requires us to extract the data from the netCDF files (and untar a tarball). That takes time but it doesn't take
# that much time, So, we might as well do the extraction and perform the search properly instead of relying on a 
# hard-coded list of field of view limits.
# 
# Inputs:
#   input_config    -- bolide_io.input_config object
#   bolideDetectionList -- [list of BolideDetection objects] The detected bolides from all files
#   output_dir          -- [str] Output path to copy or symlink files to
#
# Outputs:
#   filesToCopyDict     -- [Dict] list of files identified and copied over from other satellite
#                       Organized by detection ID in bolideDetectionList
#                       returns {} if no files were identified or copied.
#
#*************************************************************************************************************
def copy_netCDF_files_from_other_satellite(input_config, bolideDetectionList, output_dir):

    filesToCopyAll = [] # All files to copy in one big list
    filesToCopyDict = {}

    # If other satellite data path not given then do nothing
    if (input_config.other_satellite_input_path is None or input_config.other_satellite_input_path == ''):
        return filesToCopyDict

    #***
    # The pipeline unit of work is one day. Check that all detections are on the same day. If not then our assumption is
    # incorrect and more coding is necessary
    yearList    = []
    monthList   = []
    dayList     = []
    for detection in bolideDetectionList:
        yearList.append(detection.bolideTime.year)
        monthList.append(detection.bolideTime.month)
        dayList.append(detection.bolideTime.day)
    yearList    = np.array(yearList)
    monthList   = np.array(monthList)
    dayList     = np.array(dayList)

    if (not np.all(yearList == yearList[0]) or
            not np.all(monthList == monthList[0]) or
            not np.all(dayList == dayList[0])):
        raise Exception('Expecting all detections to be on the same day')

    #***
    # Untar the daily bundle if using daily tarballs
    # Get subdirectory from year,month and day for this detection
    yearStr = '%04d' % detection.bolideTime.year
    monthStr = '%02d' % detection.bolideTime.month
    dayStr = '%02d' % detection.bolideTime.day
    # Formulate directory path for other satellite
    if input_config.data_source == 'geonex':
        dayOfYear   = '{:03}'.format(detection.bolideTime.timetuple().tm_yday)
        other_satellite_path = os.path.join(input_config.other_satellite_input_path, yearStr, dayOfYear, '*')
    else:
        other_satellite_path = os.path.join(input_config.other_satellite_input_path, yearStr, monthStr+dayStr)

    # Check if the data is in a tarball
    availableFiles = glob.glob(os.path.join(other_satellite_path, '*.nc'), recursive=True)
    if input_config.data_source == 'daily_bundle':
        # This is a daily bundles data set
        other_satellite_path = os.path.join(input_config.other_satellite_input_path, yearStr)
        availableFiles = glob.glob(os.path.join(other_satellite_path, '*'+monthStr+dayStr+'.nc.tgz'), recursive=False)
        if (len(availableFiles) == 1):
            dailyBundleRun = True
            # On the NAS /tmp is a RAM disk, use that for temporary .nc file storage
            # On Jeff Smith's personal machine and tessdb1, /tmp/ramdisk is a ramdisk
            satellite = determine_satellite_from_file_path(availableFiles[0])
            dailyTmpFilePath = os.path.join('/tmp', 'ramdisk', 'glm_tmp', satellite, yearStr, monthStr+dayStr)
            if os.path.isdir(dailyTmpFilePath):
                # The directory already exists. Delete it and all its contents
                shutil.rmtree(dailyTmpFilePath, ignore_errors=True)
            
            # Create the directory
            create_path(dailyTmpFilePath, input_config.verbosity)
                
            os.system('tar -xzf {} -C {}'.format(availableFiles[0], dailyTmpFilePath))
            availableFiles = glob.glob(os.path.join(dailyTmpFilePath, '*.nc'))

    if len(availableFiles) == 0:
        # No data available, nothing to do.
        print('Other Satellite: directory {} does not contain GLM data'.format(other_satellite_path))
        return filesToCopyDict

    nc4Data = netCDF4.Dataset(availableFiles[0])
    # Make sure this is data for the other satellite
    if bolideDetectionList[0].goesSatellite == nc4Data.platform_ID:
        raise Exception ('The other satellite data appears to be from the same satellite!')
    # Collect the field of view bounds for the other satellite for this day
    # Assume the satellite did not move in position over the course of the day!
    other_satellite_lat_field_of_view_bounds = [nc4Data.variables['lat_field_of_view_bounds'][0].data.tolist(), 
                                         nc4Data.variables['lat_field_of_view_bounds'][1].data.tolist()]
    other_satellite_lon_field_of_view_bounds = [nc4Data.variables['lon_field_of_view_bounds'][0].data.tolist(), 
                                         nc4Data.variables['lon_field_of_view_bounds'][1].data.tolist()]

    # Remove full path from file names
    availableFiles = np.array(availableFiles)
    availableFilesNoDir = np.array([os.path.basename(f) for f in availableFiles])

    #***
    for detection in bolideDetectionList:

        # Check if in stereo region, if not then skip
        avgLat, avgLon = detection.average_group_lat_lon
        if not (avgLat <= other_satellite_lat_field_of_view_bounds[0] and
                avgLat >= other_satellite_lat_field_of_view_bounds[1] and
                avgLon >= other_satellite_lon_field_of_view_bounds[0] and
                avgLon <= other_satellite_lon_field_of_view_bounds[1]):
            detection.isInStereo = False
            continue

        # Record that this is in stereo region
        detection.isInStereo = True

        # Find data files around this detection
        startTime, endTime = detection.get_time_interval
        # Construct GLM datestamps associated with start and end times
        startTimeDet = timeUtil.generate_glm_timestamp(startTime)
        endTimeDet = timeUtil.generate_glm_timestamp(endTime)
        startTimeDetSec = timeUtil.extract_total_seconds_from_glm_timestamp(startTimeDet)
        endTimeDetSec = timeUtil.extract_total_seconds_from_glm_timestamp(endTimeDet)

        # Look through available other satellite files and find those about the bolide detection times
        # First find the other satellite file corresponding to the detection

        # Get StartTimes and EndTimes for all files
        startTimes  = np.array([fn[fn.index('_s')+2: fn.index('_e')] for fn in availableFilesNoDir], dtype=int)
        endTimes    = np.array([fn[fn.index('_e')+2: fn.index('_c')] for fn in availableFilesNoDir], dtype=int)
        startTimesSec = timeUtil.extract_total_seconds_from_glm_timestamp(startTimes)
        endTimesSec = timeUtil.extract_total_seconds_from_glm_timestamp(endTimes)

        # Keep all files with start or end times within 20 seconds of detection
        filesIndicesToCopy = np.nonzero(np.logical_or(np.abs(startTimesSec - startTimeDetSec) < 20, 
                                                            np.abs(endTimesSec - endTimeDetSec) < 20))

        filesToCopyAll.extend(availableFiles[filesIndicesToCopy])
        filesToCopyDict[detection.ID] = availableFilesNoDir[filesIndicesToCopy].tolist()

    # Copy or symlink the files
  # if input_config.data_source == 'daily_bundle':
  #     filesToCopyFullPath = [os.path.join(dailyTmpFilePath, filename) for filename in filesToCopyAll]
  # else:
  #     filesToCopyFullPath = [os.path.join(other_satellite_path, filename) for filename in filesToCopyAll]
  # copy_or_create_symlinks(filesToCopyFullPath, output_dir, input_config.createSymlinks, input_config.verbosity)
    copy_or_create_symlinks(filesToCopyAll, output_dir, input_config.createSymlinks, input_config.verbosity)

    # Delete the untarred data if running on a daily bundle
    if input_config.data_source == 'daily_bundle':
        shutil.rmtree(dailyTmpFilePath, ignore_errors=True)


    return filesToCopyDict

#*************************************************************************************************************
# Determines the satellite based on the file path
#*************************************************************************************************************
def determine_satellite_from_file_path(file_path):
    """ 
    Parameters
    ----------
    file_path : str
        The file path string. It should contain only one of: 
        'G16', 'G17', 'G18' or 'G19
        or
        'GOES16', 'GOES17', 'GOES18' or 'GOES19'

    Returns
    -------
    satellite : str
        The determined satellite {'G16', 'G17', 'G18', 'G19'}
    """

    nG16 = file_path.count('G16') + file_path.count('GOES16') 
    nG17 = file_path.count('G17') + file_path.count('GOES17') 
    nG18 = file_path.count('G18') + file_path.count('GOES18') 
    nG19 = file_path.count('G19') + file_path.count('GOES19') 

    if (((nG16 > 1) + (nG17 > 1) + (nG18 > 1) + (nG19 > 1)) > 1):
        raise Exception('Provenance Warning: Cannot determine satellite')
    elif nG16 >= 1:
        satellite = 'G16'
    elif nG17 >= 1:
        satellite = 'G17'
    elif nG18 >= 1:
        satellite = 'G18'
    elif nG19 >= 1:
        satellite = 'G19'
    else:
        raise Exception('Provenance Warning: Cannot determine satellite')

    return satellite

#*************************************************************************************************************
def detect_in_other_satellite(input_config, bolideDetectionList, otherSatelliteFilenames, output_dir, provenance):
    """Attempts to identify a bolide in the other satellite.

    This function will form a box in lat/lon and time. It then tries to find goups in the other satellite within the box.

    Parameters
    ----------
    input_config            : bolide_io.input_config object
    bolideDetectionList     : [list of BolideDetection objects] The detected bolides from all files
    otherSatelliteFilenames : [Dict] list of files identified and copied over from other satellite
                                Organized by detection ID in bolideDetectionList
    output_dir              : [str] Output path to copy or symlink files to
    provenance              : [bolide_database.Provenance] Configuration of pipeline that generated dataset

    Returns
    -------
      bolideDetectionList : [list of BolideDetection objects] The bolide detection objects with the other satellite data added
                                        No change to BolideDetection object if no stereo data.
    """

    if len(otherSatelliteFilenames) == 0:
        # no data, nothing to do
        return bolideDetectionList

    # If no detections in stereo region then nothing to do here
    if np.all(np.logical_not([d.isInStereo for d in bolideDetectionList])):
        return bolideDetectionList

    if (input_config.verbosity): print('Detecting bolides in other satellite ...')

    bolideDetectionListOtherSatellite = []
    if (input_config.verbosity): print('Generating figures from other satellite ...')
    for detection in bolideDetectionList:
        if detection.isInStereo is None:
            raise Exception('Bookkeeping Error: Detection candidates stereo status has not yet been determined.')
        elif not detection.isInStereo:
            # Not in stereo region, skip
            continue

        if len(otherSatelliteFilenames[detection.ID]) == 0:
            # No available files from other satellite. Nothing can be done.
            continue

        # Extract the groups from the other satellite
        otherSatelliteFilenamesFullPath = [os.path.join(output_dir,filename) for filename in otherSatelliteFilenames[detection.ID]]
        BCObj = bCluster.BolideClustering(otherSatelliteFilenamesFullPath,
                cluster_3D_enabled=input_config.cluster_3D_enabled, 
                numba_threads=input_config.cluster_numba_threads, 
                closeness_seconds=input_config.cluster_closeness_seconds, 
                closeness_km=input_config.cluster_closeness_km, 
                extractNeighboringFiles=False,
                min_num_groups_for_outliers=input_config.cluster_min_num_groups_for_outliers ,
                outlierSigmaThreshold=input_config.cluster_outlierSigmaThreshold, 
                outlierMinClipValueDegrees=input_config.cluster_outlierMinClipValueDegrees,
                spatiotemporal_box=input_config.spatiotemporal_box)


        # Assemble bounding box
        latArray, lonArray = detection.group_lat_lon
        minLat = np.nanmin(latArray)
        maxLat = np.nanmax(latArray)
        minLon = np.nanmin(lonArray)
        maxLon = np.nanmax(lonArray)
        minTime, maxTime = detection.get_time_interval

        # Expand all dimensions by expansion factor
        minLat -= input_config.otherSatelliteLatExpansionDegrees
        maxLat += input_config.otherSatelliteLatExpansionDegrees
        minLon -= input_config.otherSatelliteLonExpansionDegrees
        maxLon += input_config.otherSatelliteLonExpansionDegrees
        minTime -= datetime.timedelta(seconds= input_config.otherSatelliteTimeExpansionSeconds)
        maxTime += datetime.timedelta(seconds= input_config.otherSatelliteTimeExpansionSeconds)

        # Find all groups within this box
        clusterFound = BCObj.create_cluster_within_box(minLat, maxLat, minLon, maxLon, minTime, maxTime)

        if clusterFound:
            # Generate BolideDetection objects for the cluster
            # The most recent cluster is the final one in the clusteredGroups list
            # There is no confidence so leave assessment as None
            if detection.bolideDetectionOtherSatellite is not None:
                raise Exception('Detection already has an "Other Satellite" detection')
            detection.bolideDetectionOtherSatellite = bd.BolideDetection(BCObj, len(BCObj.clusteredGroups)-1,
                        howFound='other_satellite', retrieveEvents=True, 
                        features=None)
            
            # Set ID to that of the original detection
            detection.bolideDetectionOtherSatellite.ID = detection.ID
            detection.bolideDetectionOtherSatellite.isInStereo = True
            
    return bolideDetectionList



#*************************************************************************************************************
# Main function for command line calling
#
# Use this to call the GLM detector from the command line
#
# It will run the detection code then generate figures for the bolide detections.
#
#*************************************************************************************************************
if __name__ == "__main__":

    raise Exception('Command line execution of detect_bolides no longer supported')
    
    # Make sure we're running Python 3
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")


    # Parse command line arguments
    config = bio.parse_arguments( sys.argv[1:] )


    # Construct input_config object and run the detector.
    input_config = config_to_input_config_object(config)
    print('Configuration:')
    print(input_config)

    detect_bolides(input_config)
    
# ************************************ EOF ************************************
