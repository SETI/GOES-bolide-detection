#!/usr/bin/env python

# This module is to be called by each node processing a single chunck of GLM data

import pickle
import os
import sys
import time
import glob
import shutil
import psutil
from traceback import print_exc
import numpy as np
from subprocess import run

import io_utilities as ioUtil
from time_utilities import extract_ISO_date_from_directory
import detect_bolides as db
import detect_gigantic_jets as dgj
import detect_streaks as ds
from generate_detection_validation_report import generate_detection_validation_report
import validator as bValidator
import bolide_database as bolideDB

config_file_name = 'daily_GLM_processing.p'

verbosity = False

# ******************************************************************************
def run_glm_pipeline_elemental(outputDir, detection_input_config, validation_input_config,
                daySubDir, filesByDay, provenance):
    """
    This functions is called for each individual data chunk being processed. 
    I.e. this is the functionality called in each child node parallel batch job.
   
    Parameters
    ----------
    outputDir   : str
        The top level directory in which to write the output file.
    detection_input_config : bolide_io.input_config object
        The elemental detection configuration parameters
    validation_input_config : validation_io.input_config object
        The validation configuration parameters
    daySubDir   : 
        subdirectory under <outputDir> for this elemental process
    filesByDay  : list str 
        Full path to all files to process for this elemental process
    provenance  : run_glm_pipeline_batch.Provenance 
        Configuration of pipeline that generated dataset
   
    Returns
    -------

    """

  # # This is needed for PyCharm
  # import matplotlib
  # matplotlib.use('TkAgg')

    startTime = time.time()

    # Github #71: Set group ID for this process and all files it created.
    gid = os.getenv('GROUP', default='GID_NOT_SET')


    # Get satellite from input file path
    # This assumes elemental pipeline is running exclusively on a single satellite
    satellite = db.determine_satellite_from_file_path(filesByDay[0])

    print('\n****** run_glm_pipeline_elemental processing {} day: {}'.format(satellite, daySubDir))

    if not isinstance(filesByDay, list):
        raise Exception ('<filesByDay> should be a list of files')

    outputSubDir = os.path.join(outputDir, daySubDir)
    
    ioUtil.create_path(outputSubDir, verbosity)
    
    # Strip '/' from daySubDir
    slashLoc = daySubDir.find('/')
    yearString = daySubDir[0:slashLoc]
    dayString = daySubDir[slashLoc+1:]
    yearDayString = yearString + dayString

    # Copy input files to ramdisk
    if detection_input_config.use_ramdisk:
        dailyTmpFilePath = os.path.join('/tmp', 'ramdisk', 'glm_tmp', satellite, daySubDir)
        filesByDay = ioUtil.copy_or_create_symlinks(filesByDay, dailyTmpFilePath, createSymlinks=False, verbosity=False, return_new_paths=True)
        
    dailyOutDatabaseFilePath = os.path.join(outputSubDir, bolideDB.databaseFileBasename + '_' + yearDayString + '.fs')

    dailyProcessingHistoryPath = os.path.join(outputSubDir, 'processing_history' + '_' + yearDayString + '.txt')

    # If we are running on a daily bundle then untar the bundle into a temporary directory.
    if detection_input_config.data_source == 'daily_bundle':
        [_,ext] = os.path.splitext(filesByDay[0])
        if (ext == '.tgz'):
            # This is a daily bundle
            # On the NAS /tmp is a RAM disk, use that for temporary .nc file storage
            # On Jeff Smith's personal machine and tessdb1 /tmp/ramdisk is a ramdisk
            dailyTmpFilePath = os.path.join('/tmp', 'ramdisk', 'glm_tmp', satellite, daySubDir)
            if os.path.isdir(dailyTmpFilePath):
                # The directory already exists. Delete it and all its contents
                shutil.rmtree(dailyTmpFilePath, ignore_errors=True)

            # Create the directory
            ioUtil.create_path(dailyTmpFilePath, verbosity)
                
            os.system('tar -xzf {} -C {}'.format(filesByDay[0], dailyTmpFilePath))
            filesByDay = glob.glob(os.path.join(dailyTmpFilePath, '*.nc'))
            

    #***
    # Run the daily detector.
    try:
        # First delete the file notifying if processing was completed, if it already exists
        finishedFilename = os.path.join(outputSubDir,'PROCESSED_' + satellite)
        if os.path.exists(finishedFilename):
            os.remove(finishedFilename)

        if detection_input_config.detectionType == 'bolides':
            num_detections, bolideDetectionList, bolideRejectionList = db.detect_bolides(detection_input_config, filePaths=filesByDay,
                    output_dir=outputSubDir, provenance=provenance)
        elif detection_input_config.detectionType == 'gigantic_jets':
            # TODO: get Gigantic Jet code to return total number of detections
            num_detections = None
            dgj.detect_gigantic_jets(detection_input_config, data_files=filesByDay, output_path=outputSubDir, provenance=provenance)
        elif detection_input_config.detectionType == 'streaks':
            num_detections = None
            ds.detect_streaks(detection_input_config, data_files=filesByDay, output_path=outputSubDir, provenance=provenance)
    
        # Record the processing in the history log file
        # Log entry is 'YYYY-MM-DD PATH', where PATH points to the top level directory of the data for that day
        # The unit of work is one day, so we only need to check the first file of the day
        ISODate = extract_ISO_date_from_directory(filesByDay[0], detection_input_config.data_source)
        dirName = os.path.dirname(filesByDay[0])
        # If this is geonex data then strip off the hourOfDay subdirectory
        if detection_input_config.data_source == 'geonex':
            dirName = os.path.dirname(dirName)
        log_entry = ISODate + ' ' + dirName
        if not ioUtil.append_lines_to_file(dailyProcessingHistoryPath, [log_entry]):
            print('Error updating elemental processing history.')
    
    except Exception as e:
        print('\n****** Error processing {} day {} ******'.format(satellite, daySubDir))
        print_exc()
        print('\n*********************************************')
    
    else:
        #************
        # Perform validation
        bolideDetectionList = bValidator.evaluate_bolide_candidates(validation_input_config, bolideDetectionList)

        # Add validation assessment to the detection figure
        bValidator.add_validation_assessment_to_detection_figures(detection_input_config, validation_input_config, bolideDetectionList, outputSubDir)


        #************
        # Add detections and rejections to database
        # Create bolide database
        # We want to purge anything already in the database
        bolideDatabase = bolideDB.BolideDatabase(dailyOutDatabaseFilePath, purgeDatabase=True, provenance=provenance)
        #***
        # Now store the detections in the database
        bolideDatabase.add(bolideDetectionList, dataType='detections')

        # Create the CSV file for everyone who does not use ZODB
        detectionCSVFile = os.path.join(outputSubDir, bolideDB.databaseFileBasename + '_' + yearDayString + '_detections.csv')
        bolideDB.write_detection_list_to_csv_file(detectionCSVFile, bolideDetectionList, purgeCSVFile=True,
                GJ_data=detection_input_config.stereoDetectionOnlyMode)

        #***
        # Store the rejections (if any)
        if len(bolideRejectionList) > 0:
            # Strip full path from .nc filenames so that the database file names point to the current directory
            for bolide in bolideRejectionList:
                for idx in np.arange(len(bolide.filePathList)):
                    bolide.filePathList[idx] = os.path.basename(bolide.filePathList[idx])
            bolideDatabase.add(bolideRejectionList, dataType='rejections')
            bolideDB.write_detection_list_to_csv_file(os.path.splitext(dailyOutDatabaseFilePath)[0] + '_rejections.csv',
                    bolideRejectionList, purgeCSVFile=True,
                    GJ_data=detection_input_config.stereoDetectionOnlyMode)

        bolideDatabase.commit()
        bolideDatabase.close()

        #************
        # Post-processing and light curve generation
        if detection_input_config.post_process_enabled and num_detections > 0:
            post_process_startTime = time.time()
            # Divide up the detection jobs so that a minimum amount of memory is available for each task.
            mem = psutil.virtual_memory()
            availMemGiB = mem.available / 2**30 # 2**30 = 1 GiB
            maxThreads = int(np.floor(availMemGiB / detection_input_config.min_GB_post_processing))
            maxThreads = np.min([detection_input_config.n_cores, maxThreads])
            exeStr = '$ATAP_REPO_ROOT/code/GLM_bolide_analysis/system/sh/partition_and_run_bolide_analysis.sh'
            cmdstr = exeStr + ' -n ' + str(maxThreads) + ' ' + detection_input_config.post_process_config_file + ' ' + detectionCSVFile + ' ' + outputSubDir
            try:
                if (detection_input_config.verbosity):
                    print('Running post-processing and light curve generation process...')
                result = run(cmdstr, shell=True, check=True)
            except:
                print('\n****** Error post-processing detection file {}'.format(detectionCSVFile), file=sys.stderr)
                print_exc()
                print('\n********************************************************')
               #print("Execution failed:", e, file=sys.stderr)
            post_process_endTime = time.time()
            post_process_totalTime = post_process_endTime - post_process_startTime
            print('Post-processing time: {:.2f} seconds, {:.2f} minutes'.format(post_process_totalTime, post_process_totalTime/60))


        #************
        # Generate PDF Report
        # Generate the bolide detection validation report PDF
        if validation_input_config.reportEnabled and num_detections > 0:
            # Determine output directory for report
            if validation_input_config.report_output_path == '':
                report_output_path = outputSubDir
            else:
                if os.path.abspath(validation_input_config.report_output_path) != os.path.abspath(detection_input_config.output_path):
                    splitPath = os.path.split(outputSubDir)
                    monthDay = splitPath[1]
                    year = os.path.basename(splitPath[0])
                    satellite = os.path.basename(os.path.split(splitPath[0])[0])
                    report_output_path = os.path.join(validation_input_config.report_output_path,year,monthDay)
                else:
                    report_output_path = outputSubDir
            generate_detection_validation_report(bolideDetectionList, outputSubDir, report_output_path,
                    createSymlinks=validation_input_config.report_create_symlinks, 
                    verbosity=detection_input_config.verbosity, copy_indiv_files=validation_input_config.reportCopyIndivFiles)

        # Create file to flag users this day was processed
        os.system('touch ' + finishedFilename)

        # If generating reports then also copy the processing status file
        if validation_input_config.reportEnabled:
            # Only need to do this if the report output dir is not the output dir
            if os.path.abspath(validation_input_config.report_output_path) != os.path.abspath(validation_input_config.report_output_path):
                report_output_path = os.path.join(validation_input_config.report_output_path,yearString,dayString)
                if not os.path.exists(report_output_path):
                    # If the report directory does not yet exist create it now
                    ioUtil.create_path(report_output_path)
                _, fileName = os.path.split(finishedFilename)
                outFileStr = os.path.join(report_output_path, fileName)
                shutil.copyfile(finishedFilename, outFileStr)
                # Set read permission for group
                os.chmod(outFileStr, 0o640)
                # Set the group ID, if it was specified
                if gid != 'GID_NOT_SET':
                    try:
                        shutil.chown(outFileStr, group=gid)
                    except:
                        pass

        if num_detections is not None:
            print('\nSuccessfully processed {} day {}. Number detections: {}'.format(satellite, daySubDir, num_detections))
        else:
            print('\nSuccessfully processed {} day {}'.format(satellite, daySubDir))


    endTime = time.time()
    totalTime = endTime - startTime
    print("Total processing time for {} day {}: {:.2f} seconds, {:.2f} minutes".format(satellite, daySubDir, totalTime, totalTime / 60))
    print('\n******')


    if detection_input_config.use_ramdisk or detection_input_config.data_source == 'daily_bundle':
        # Delete temporary directory
        if os.path.exists(dailyTmpFilePath):
            shutil.rmtree(dailyTmpFilePath)
        else:
            raise Exception('Expected path {} to exist but it does not.'.format(dailyTmpFilePath))

    # Make sure all output directories and files are group readable
    # Note: os.chmod expects mode to be an octal number, so prepend with '0o'
    os.chmod(outputDir, 0o750)
    os.chmod(os.path.join(outputDir, yearString), 0o750)
    os.chmod(outputSubDir, 0o750)
    # All files in subdirectory
    for name in glob.glob(outputSubDir+'/*'):
        if gid != 'GID_NOT_SET':
            try:
                shutil.chown(name, group=gid)
            except:
                pass
        if os.path.isdir(name):
            os.chmod(name, 0o750)
        elif os.path.islink(name):
            # This will sometimes fail if the link is not valid, which happens in PyCharm debugging for some reason, not
            # sure why
            try:
                os.chmod(name, 0o750)
            except:
                pass
        elif os.path.exists(name):
            os.chmod(name, 0o640)
    


# ******************************************************************************
def pickle_elemental_data (outputDir, detection_input_config, validation_input_config, 
        daySubDir, filesByDay, provenance):
    """
    Generates the pickle file named
      run_glm_pipeline_elemental.config_file_name 
    which is read in when running the __main__ processs below
  
    Parameters
    ----------
      outputDir       -- The top level directory in which to write the output file.
      detection_input_config    -- bolide_io.input_config object
      validation_input_config    -- validation_io.input_config object
      daySubDir       -- subdirectory under <outputDir> for this elemental process
      filesByDay      -- Full path to all files to process for this elemental process
      provenance      -- [run_glm_pipeline_batch.Provenance] Configuration of pipeline that generated dataset
  
    Returns
    -------
      A file in the subdirectory called 
          os.path.join(outputDir, daySubDir, config_file_name)
    """

    outputSubDir = os.path.join(outputDir, daySubDir)
    filename = os.path.join(outputDir, daySubDir, config_file_name)

    # Check to see whether the output subdirectory exists and create it if necessary.
    ioUtil.create_path(outputSubDir, verbosity)
    
    # Create a dictionary with the desired data
    dataToStore =  {'outputDir': outputDir,
                    'detection_input_config': detection_input_config,
                    'validation_input_config': validation_input_config,
                    'daySubDir': daySubDir,
                    'filesByDay': filesByDay,
                    'provenance': provenance}

    # Write the data.
    try:
        with open(filename, 'wb') as fp :
            pickle.dump(dataToStore, fp)
    except:
        raise Exception('Could not write to file {}.'.format(filename))

    fp.close()


# ******************************************************************************
# Command line functionality to call run_glm_pipeline_elemental.
#
# Pass the path to the elemental GLM data output directory for this call.
#
# The executable searches the passed directory for a file called
#   run_glm_pipeline_elemental.config_file_name 
#
# ******************************************************************************
if __name__ == "__main__":

    # Make sure we're running Python 3
    if sys.version_info.major < 3:
        raise Exception("Python 3.0 or higher is required")

    # Import the Provenance class in order to read the pickle dict which contains a Provenance object
    # Do a local import here so we do not have a circular import
    from run_glm_pipeline_batch import Provenance

    # Path to the working output directory
    filename = os.path.join(sys.argv[1], config_file_name)

    # Unpickle the data we need
    if not os.path.isfile(filename):
        raise Exception('The file {} does not exist.'.format(filename))
    else:
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)

    # Call the elemental function
    run_glm_pipeline_elemental(data['outputDir'], data['detection_input_config'], data['validation_input_config'], 
            data['daySubDir'], data['filesByDay'], data['provenance'])

