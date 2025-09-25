import glob
from traceback import print_exc
import multiprocessing as mp
import time
import os
import numpy as np
import pickle

import io_utilities as ioUtil

import bolide_io as bIo
import bolide_clustering as bCluster

#*************************************************************************************************************
def detect_gigantic_jets (input_config, data_files=[], output_path=None, provenance=None):
    """
    Top level function to search a selection of GLM netCDF files for gigantic jets.

    This simple test code will just return all clusters over a minimum number of groups in the cluster using the bolide
    clustering module. For this simple test code, we just save out the detected clusters in a pickle file in each daily processing
    directory
   
   
    Parameters
    ----------
    input_config : gigantic_jet_io.input_config
        Configuration parameters for the elemental run.
        See gigantic_jet_io.input_config.configuration_template
    data_files   : str list 
        list of netCDF files to search
        This is nominally a day of GLM data.
        If empty then use the files in input_config.data_path
    output_path : str
        Path to save outputs
    provenance  : run_glm_pipeline_batch.Provenance object
        Contains the provenance for this run
   
    Returns
    -------
      None, other than saved data files.

    """

    startTime = time.time()

    # Get the netCDF files to process
    if len(data_files) == 0:
        data_files = glob.glob(input_config.data_path + "*.nc")
        if (data_files == []):
            if (input_config.verbosity):
                print('No files to process in directory {}'.format(input_config.data_path))
                return

    data_files = sorted(data_files)


    #***
    # Either do multi- or single-core processing, depending on the value of 
    # n_cores. Note that single-core processing is required for profiling.
    if input_config.n_cores < 1:
        raise Exception('n_cores should an integer >= 1')
    elif input_config.n_cores == 1:
        
        [giganticJetDetectionList, oops_counter] = _detect_gigantic_jets_in_multiple_files(data_files, input_config)
        
    else:
        # Do multi-core processing.
        #
        oops_counter = 0
        giganticJetDetectionList = []

        # Chunk files
        # Chunk factor decreases the number of jobs by this factor. 
        chunkFactor = 10
        chunkSize = int(np.ceil(len(data_files) / (input_config.n_cores * chunkFactor)))
        filesChunked = [data_files[i:i + chunkSize] for i in range(0, len(data_files), chunkSize)]  
        
        with mp.Pool(processes=input_config.n_cores) as pool:
            results = [pool.apply_async(_detect_gigantic_jets_in_multiple_files, args=(file_paths, input_config)) for file_paths in filesChunked]
            outputs = [result.get() for result in results]
            for output in outputs:
                [detectedGiganticJets, oops_counter_elemental] = output
                oops_counter += oops_counter_elemental
                giganticJetDetectionList.extend(detectedGiganticJets)

    if (input_config.verbosity): print("We had {:d} oopses".format(oops_counter))

    #***
    # Save results to output_path directory
    num_gigantic_jets = len(giganticJetDetectionList)
    if num_gigantic_jets > 0:
        print("We have {:d} gigantic jet detections".format(num_gigantic_jets))
        # for this simple test code, we just save out the detected clusters in a pickle file in each daily processing
        # directory
        filename = os.path.join(output_path, 'gigantic_jet_detections.p')
        with open(filename, 'wb') as fp :
            pickle.dump(giganticJetDetectionList, fp)
        fp.close()

    else:
        if (input_config.verbosity): print("There were no valid detections.")

    endTime = time.time()
    totalTime = endTime - startTime
    if (input_config.verbosity):
        print("Total processing time: {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))

    return

#*************************************************************************************************************
def _detect_gigantic_jets_in_multiple_files(fileList, input_config):
    """
    This function takes a list of netCDF files and finds gigantic jets in all of them sequentially
   
    Parameters
    ----------
    fileList    : str list
        path to netCDF files to process
    input_config : gigantic_jet_io.input_config object
   
    Returns
    -------
    giganticJetDetectionList : list of gigantic jet detection objects 
        The detected gigantic jets from all files
    oops_counter : int 
        Number of netCDF files experiencing an error
    
    """

    oops_counter = 0

    giganticJetDetectionList = []

    # Do processing on a single core.
    # If we are doing multi-core processing then that is handled in detect_gigantic_jets 
    for file_path in fileList:
        status, detectedGiganticJets = _detect_gigantic_jets_in_file(file_path, input_config)
        if status == 0:
            continue
        elif status == -99:
            oops_counter += 1
            continue
        # Else, we have a detection, add it to the list
        giganticJetDetectionList.extend(detectedGiganticJets)

    return [giganticJetDetectionList, oops_counter]

#*************************************************************************************************************
def _detect_gigantic_jets_in_file(file_path, input_config):
    """
    This function loads in a single netCDF file and tries to detect gigantic jets.
    
    This simple test code will just return all clusters over a minimum numbr of groups in the cluster using the bolide
    clustering module.
   
    Parameters
    ----------
    file_path       : str 
        path to netCDF file
    input_config    : gigantic_jet_io.input_config object
   
    Returns
    -------
    status  : int
        0 => no gigantic jets detected, 1 => gigantic jets detected, detected, -99 => An oops occured
    giganticJetDetectionList : list of gigantic jet detection objects 
        The detected gigantic jets
   
    """

    try:
        # This simple test code will just return all clusters over a minimum number of groups in the cluster

        # Create a default bolide input_config to run the clustering algorithm
        config = ioUtil.get_default_configuration(bIo.input_config.configuration_template())
        bolide_input_config = bIo.input_config(config)

        # We first extract the groups from the netCDF file and form clusters
        # Extract the L2 Data from the .nc file
        BCObj = bCluster.BolideClustering(file_path,
                cluster_3D_enabled=input_config.cluster_3D_enabled, 
                closeness_seconds=bolide_input_config.cluster_closeness_seconds, 
                closeness_km=bolide_input_config.cluster_closeness_km, 
                extractNeighboringFiles=bolide_input_config.cluster_extractNeighboringFiles, 
                min_num_groups_for_outliers=bolide_input_config.cluster_min_num_groups_for_outliers ,
                outlierSigmaThreshold=bolide_input_config.cluster_outlierSigmaThreshold, 
                outlierMinClipValueDegrees=bolide_input_config.cluster_outlierMinClipValueDegrees)
        goesSatellite = BCObj.goesSatellite
        
        # Check if this is an empty file with no groups to process
        if (len(BCObj.glmGroups) < 1):
            return 0, None

        # Organize data, Cluster the GLM groups into bolide detections
        success = BCObj.cluster_glm_groups(
                    sequential_clustering_enabled=bolide_input_config.cluster_sequential_enabled, 
                    outlier_rejection_enabled=bolide_input_config.cluster_outlier_rejection_enabled)
        if (not success):
            return 0, None

        # Create gigantic jets candidate list from remaining clusters
        # Remove all clusters with below min_num_groups
        giganticJetDetectionList = [cluster for cluster in BCObj.clusteredGroups if 
                len(cluster['groupIds']) >= input_config.min_num_groups]
        # Check if there are no clusters left to process
        if (len(giganticJetDetectionList) < 1):
            return 0, None

        # Add netCDF filenames list to each detection
        for detection in giganticJetDetectionList:
            detection['filenames'] = BCObj.netCDFFilenameList
    
    except Exception as e:
        print('\n****** Error processing file {}'.format(file_path))
        print_exc()

        return -99, None
        
    else:
        if (input_config.verbosity):
            print("Finished working on file: {0:s}".format(file_path))
            print(" ")

        return 1, giganticJetDetectionList


