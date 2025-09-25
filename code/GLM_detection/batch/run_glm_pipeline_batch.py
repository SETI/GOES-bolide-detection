#!/usr/bin/env python

import sys
import os
import glob
import difflib
import copy
import numpy as np
from traceback import print_exc
from datetime import datetime, date, timedelta, timezone
import time
import shutil
import subprocess
import pickle
import multiprocessing as mp
import warnings

import io_utilities as ioUtil
from time_utilities import extract_ISO_date_from_directory, extract_total_seconds_from_glm_timestamp

from _version import __version__
import detect_bolides as db
import bolide_detections as bd
import bolide_io as bIo
import run_glm_pipeline_elemental as runElem
import bolide_database as bolideDB
import validation_io as vIo

import gigantic_jet_io as gjIo
import streak_io as streakIo

aggregate_data_file_name = 'aggregation_data.p'

# *****************************************************************************
def get_configuration_template():
    """
    Retrieve a dictionary defining configuration parameter names, data types, and
    default values.
   
    Parameters
    ----------
        (none)
   
    Returns
    -------
    configTemplate : Dict
        <attribute, [type, value]>
        specifying parameter name, type, and a default value for all valid parameters.

    Initialize the configuration dictionary with parameter names and their
    associated types. Values read from the file will be cast to the specified
    type.
                      parameter                      type    default
    """
    configTemplate = {'inputRootDir':               ['str', '.'],
                      'inputRootDirL0':             ['str', '.'],
                      'otherSatelliteRootDir':      ['str', '.'],
                      'outputDir':                  ['str', './output'],
                      'reportOutputDir':            ['str', None],
                      'detectionType':              ['str', 'bolides'],
                      'elementalConfigFile':        ['str', None],
                      'validationConfigFile':       ['str', None],
                      'processingHistoryBaseName':  ['str', 'processing_history'],
                      'multiProcessEnabled':        ['bool', False], 
                      'forceReprocess':             ['bool', False],
                      'deleteDailyDirs':            ['bool', False],
                      'deleteDailyDatabases':       ['bool', False],
                      'deleteDailyNetCdfFiles':     ['bool', True],
                      'deleteDailyExtraFigures':    ['bool', False],
                      'startDate':                  ['str', ''],
                      'endDate':                    ['str', ''],
                      'doNotProcessCurrentDay':     ['bool', False],
                      'delayNDaysForL0Data':        ['int', 0],
                      'n_cores_aggregation':        ['int', 1],
                      'dirListFileName':            ['str', 'list_of_dirs_to_process.txt']}

    return configTemplate

def check_input_param_consistency(config):
    """ Check that the input configuration parameters are consistent and within appropriate ranges

    Parameters
    ----------
    config : dict
        Input configuration dictionary
        See get_configuration_template()

    Returns
    -------
    success : bool
        If true, then input parameters are correct and consistent

    """
    
    detectionTypeOptions = ('bolides', 'gigantic_jets', 'streaks')


    assert not (config['deleteDailyDirs'] and config['deleteDailyDatabases']), \
        'Cannot set both deleteDailyDirs and deleteDailyDatabases to True'

    assert not (config['deleteDailyDirs'] and config['deleteDailyNetCdfFiles']), \
        'Cannot set both deleteDailyDirs and deleteDailyNetCdfFiles to True'

    assert detectionTypeOptions.count(config['detectionType']) == 1, \
        'Valid detectionType options: {}'.format(detectionTypeOptions)

    # If we get to the end then we are successful
    return True
    

#******************************************************************************
# Generates a provenance information characterizing the pipeline run.
class Provenance:

    __version__ = __version__

    def __init__(self, configurationDict, elementalConfig, validationConfig):
        """
        Constructor. 

        Note that the elementalConfig object is converted to a dict when stored in the Provenance object. 
        See Github Issue 57. 
        
        Inputs:
        configurationDict : dict 
            Contains top level configuration parameter information
        elementalConfig   : bolide_io.input_config
            Contains the elemental detection configurion information
        validationConfig   : validation_io.input_config
            Contains the validation configurion information
     
        """

        assert isinstance(configurationDict, dict), 'configurationDict must be a dict'
        assert isinstance(elementalConfig, bIo.input_config), 'elementalConfig must be of type bolide_io.input_config'
        assert isinstance(validationConfig, vIo.input_config), 'validationConfig must be of type validation_io.input_config'

        now = datetime.now()
        scriptDir = os.path.dirname(__file__)
        scriptName = os.path.basename(__file__)

        # Git branch
        # Only works with git version 2.0 or above (why does NAS use older git version?!?!)
        if os.popen('git --version').read().strip().count('git version 2.') > 0:
            cmd = 'git -C {} branch --show-current'.format(scriptDir)
            self.gitBranch = os.popen(cmd).read().strip()
        else:
            # This is for the older git version on the NAS
            try:
                cmd = 'cd {}; git branch'.format(scriptDir)
                stdout = os.popen(cmd).read()
                self.gitBranch = stdout[stdout.index('*')+2:stdout.index('*')+stdout[stdout.index('*'):].index('\n')]
            except:
                self.gitBranch = None

        # Git commit number
        if os.popen('git --version').read().strip().count('git version 2.') > 0:
            cmd = 'git -C {} log -1 | head -1 | awk \'{{print $2}}\''.format(scriptDir)
            self.gitCommitNum = os.popen(cmd).read().strip()
        else:
            # This is for the older git version on the NAS
            try:
                cmd = 'cd {}; git log -1 | head -1 | awk \'{{print $2}}\''.format(scriptDir)
                self.gitCommitNum = os.popen(cmd).read().strip()
            except:
                self.gitCommitNum = None

        # Time of run and Run ID
        # Run ID is the POSIX timestamp * 1000 (i.e. nearest millisecond)
        self.runID = np.int64(np.rint(now.timestamp()*1000))
        self.date  = now.strftime("%B %d, %Y")
        self.time  = now.strftime("%H:%M:%S")

        # Host identifier is <user>@<hostname>
        hostname = os.uname().nodename
        # on a NAS compute node os.getlogin() will throw an error
        try:
            username = os.getlogin()
        except:
            username = os.getenv('USER')
        self.hostID = username + '@' + hostname

        # Determine the commit number of the currently running script and print it to the
        # standard output. This way we can refer to the log file to find out which version
        # of the code was run. Also include configuration parameter information.
        self.pythonVersion  = '{}.{}.{}'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
        self.scriptPath     = '{}/{}'.format(scriptDir, scriptName)
        self.configurationDict = copy.deepcopy(configurationDict)

        self.elementalConfigDict = elementalConfig.copy_to_dict()
        self.validationConfigDict = validationConfig.copy_to_dict()

        # Get the Conda environment
        try:
            cmd = 'conda info | grep \'active env location\' | awk \'{{print $5}}\''
            self.condaEnvPath = os.popen(cmd).read().strip()
        except Exception as e:
            print_exc()
            self.condaEnvPath = ''

    def copy_to_dict(self):
        """ Converts this Provenance object to a dict

        This is for storing in the BolideDatabase, where there are persistent problems with accessing the Provenenace
        object. 

        """

        return copy.deepcopy(self.__dict__)

    # Write or append a text record to a stream.
    def print(self):
        print('+--------------------------------------------------------------------------------')
        print('| PIPELINE PROVENANCE')
        print('|')
        print('| VERSION:\t{}'.format(self.__version__))
        print('| RUN ID:\t{}'.format(self.runID))
        print('| HOST ID:\t{}'.format(self.hostID))
        print('| DATE:\t\t{} : {}'.format(self.date, self.time))
        print('| PYTHON:\t{}'.format(self.pythonVersion))
        print('| EXECUTABLE PATH:\t{}'.format(self.scriptPath))
        print('| BRANCH:\t{}'.format(self.gitBranch))
        print('| MOST RECENT COMMIT#:\t{}'.format(self.gitCommitNum))
        print('| Conda Env:\t{}'.format(self.condaEnvPath))
        print('|')
        print('| Top Level Configuration:')
        ioUtil.print_dictionary(self.configurationDict, verbosity=True)
        print('|')
        print('| Elemental Configuration:')
        ioUtil.print_dictionary(self.elementalConfigDict, verbosity=True)
        print('|')
        print('| Validation Configuration:')
        ioUtil.print_dictionary(self.validationConfigDict, verbosity=True)
        print('+--------------------------------------------------------------------------------')

    # TODO: Why does adding a __repr__ cause constant dsiplaying of the print() information?
   #def __repr__(self): 
   #    return self.print()


#******************************************************************************
def set_diff(first, second):
    """
    Return elements in first set that are not in second.
    """
    return [item for item in first if item not in second]

# ******************************************************************************
# Generates the pickle file named
#   run_glm_pipeline_batch.aggregate_data_file_name 
# which is read in when performing the final aggregation step in final_aggregation_work.
#
# Inputs:
#   outputDir       -- [str] Path to save aggregation pickle file
#   filesByDayDict  -- [dict] Files to process per day
#   dirListFilePath -- [str list] gives the path to each directory for daily elemental processing
#   provenance      -- [Provenance Object] Contains the provenance for this run
#
# Outputs:
#   A file in the subdirectory called 
#       os.path.join(outputDir, daySubDir, config_file_name)
# ******************************************************************************
def pickle_aggregation_data (outputDir, filesByDayDict, dirListFilePath, provenance):

    filename = os.path.join(outputDir, aggregate_data_file_name)

    # The output directory should already exist. If it does not then somethine went wrong.
    if not os.path.isdir(outputDir):
        raise Exception('directory does not exist: {}'.format(outputDir))
    
    # Create a dictionary with the desired data
    dataToStore =  {'filesByDayDict': filesByDayDict,
                    'dirListFilePath': dirListFilePath,
                    'provenance': provenance}

    # Write the data.
    try:
        with open(filename, 'wb') as fp :
            pickle.dump(dataToStore, fp)
    except:
        raise Exception('Could not write to file {}.'.format(filename))

    fp.close()

#******************************************************************************
# Final aggregation steps
#
# This is to be run after all data has been processed. It aggregates all the data from each elemental run into summary
# files located at the top output directory level. The detections are stored in a ZODB database. The rejections are
# stord in a bolideDispisitionProfileList. 
#
# The rejections are used for training purposes and there are very many of them generated. Too many to efficiently store
# in a ZODB database, which can be quite slow when the number of entries gets into the millions.
#
# It also deletes unneeded temporary files.
#
# Inputs:
#   In provenance.configurationDict:
#       outputDir       -- The top level directory in which to write the output file.
#       processingHistoryBaseName -- name of processing history file (postpended with '.txt')
#       deleteDailyDirs -- [bool] If True, delete all daily directory data
#
#   filesByDayDict  -- [dict] Files to process per day
#   dirListFilePath -- [str list] gives the path to each directory for daily elemental processing
#   provenance      -- [Provenance] Contains the provenance for this run
# 
#******************************************************************************

def final_aggregation_work_parallel (filesByDayDict, dirListFilePath, provenance):

    from bolide_database import databaseFileBasename 

    outputDir = provenance.configurationDict['outputDir']
    processingHistoryBaseName = provenance.configurationDict['processingHistoryBaseName']
    deleteDailyDirs = provenance.configurationDict['deleteDailyDirs']

    # Determine satellite
    # This assumes all data files are from the same satellite
    firstFilename = filesByDayDict[list(filesByDayDict.keys())[0]][0]
    satellite = db.determine_satellite_from_file_path(firstFilename)
    outputDir = os.path.join(outputDir, satellite)

    #***
    # Combine the outputs from all the separate batch jobs
    print("")
    print('Aggregating detections and rejections...')
    print("")
    outFilePath = os.path.join(outputDir, databaseFileBasename + '.fs')
    outRejectionPath  = os.path.join(outputDir, 'bolide_rejections' + '.p')
    csvFilePathDetect = os.path.join(outputDir, 'bolide_detections' + '.csv')
    csvFilePathReject = os.path.join(outputDir, 'bolide_rejections' + '.csv')
    processingHistoryPath = os.path.join(outputDir, processingHistoryBaseName + '.txt')

    #***
    # The database is only for storing detections
    bolideDatabaseAggregate = bolideDB.BolideDatabase(outFilePath, provenance=provenance, purgeDatabase=False)
    # Rejection are stored in a BolideDispositionProfile to save file size
    allBolideRejectionPathList = []
    allYearDirs = []
    bolideDetectionList = []

    # Parallelize the parsing of each day's data
    print('Compiling detections and rejections from all processed days...')
    with mp.Pool(processes=provenance.configurationDict['n_cores_aggregation']) as pool:
        results = [pool.apply_async(_final_aggregation_work_elemental, 
            args=(outputDir, databaseFileBasename, processingHistoryBaseName, daySubDir,
                provenance.configurationDict['deleteDailyDatabases'], 
                provenance.configurationDict['deleteDailyNetCdfFiles'],
                provenance.configurationDict['deleteDailyExtraFigures'])) for daySubDir in filesByDayDict]
        outputs = [result.get() for result in results]
        for output in outputs:
            [bolideDetectionListDay, bolideRejectionPath, yearDir, daySubDir] = output
      
            if bolideRejectionPath is not None:
                allBolideRejectionPathList.append(bolideRejectionPath)

            # If this day was not processed then bolideDetectionListDay is None
            if bolideDetectionListDay is not None:
                bolideDetectionList.extend(bolideDetectionListDay)
      
                # Record the processing in the history log file
                # Log entry is 'YYYY-MM-DD PATH', where PATH points to the top level directory of the data for that day
                # The unit of work is one day, so we only need to check the first file of the day
                ISODate = extract_ISO_date_from_directory(filesByDayDict[daySubDir][0], provenance.elementalConfigDict['data_source'])
                dirName = os.path.dirname(filesByDayDict[daySubDir][0])
                # If this is geonex data then strip off the hourOfDay subdirectory
                if provenance.elementalConfigDict['data_source'] == 'geonex':
                    dirName = os.path.dirname(dirName)
                log_entry = ISODate + ' ' + dirName
                if not ioUtil.append_lines_to_file(processingHistoryPath, [log_entry]):
                    raise Exception('Error updating processing history file.')

            # If this year directory is not yet in allYearDirs then add it
            try: 
                allYearDirs.index(yearDir)
            except:
                allYearDirs.append(yearDir)
      
      
    #***
    print('Aggregating all data into final lists...')
    # Now add all the days' detections to the aggregate database
    # The rejections are stored differently
    bolideDatabaseAggregate.add(bolideDetectionList, dataType='detections')
    bolideDB.write_detection_list_to_csv_file(csvFilePathDetect, bolideDetectionList, purgeCSVFile=False,
            GJ_data=provenance.elementalConfigDict['stereoDetectionOnlyMode'])
    print('Total number of detections = {}'.format(bolideDatabaseAggregate.n_tot_detections))
    # If no bolides processed then we still have to issue a commit before closing
    bolideDatabaseAggregate.commit()
    bolideDatabaseAggregate.close()
    del bolideDetectionList

    #***
    # The rejections for each day needs to be combined together into a single very large list and saved
    bolideRejectionProfileList = []
    for bolideRejectionPath in allBolideRejectionPathList:
        bolideRejectionProfileListDaily = bd.unpickle_bolide_detections(bolideRejectionPath)
        bolideRejectionProfileList.extend(bolideRejectionProfileListDaily)

    print('Total number of rejections = {}'.format(len(bolideRejectionProfileList)))

    # Pickle the rejections list
    if len(bolideRejectionProfileList) > 0:
        with open(outRejectionPath, 'wb') as fp:
            pickle.dump(bolideRejectionProfileList, fp)

        fp.close()

    # Delete all daily files, if requested
    if deleteDailyDirs:
        for yearlyDir in allYearDirs:
            if os.path.exists(yearlyDir):
                shutil.rmtree(yearlyDir)

    if os.path.isfile(dirListFilePath):
        os.remove(dirListFilePath)

    aggregation_data = os.path.join(outputDir, aggregate_data_file_name)
    if os.path.isfile(aggregation_data):
        os.remove(aggregation_data)

    # Remove any remnants of the daily tarball temporary .nc file location
    dailyTmpFilePath = os.path.join('/tmp', 'ramdisk', 'glm_tmp', daySubDir)
    if os.path.exists(dailyTmpFilePath):
        shutil.rmtree(dailyTmpFilePath, ignore_errors=True)

    # Make sure aggregate database files are group readable
    # Note: os.chmod expects mode to be an octal number, so prepend with '0o'
    gid = os.getenv('GROUP', default='GID_NOT_SET')
    # All database files
    database_filenames = glob.glob(outputDir+'/*fs*')
    database_filenames.extend(glob.glob(outputDir+'/*.csv'))
    database_filenames.extend(glob.glob(outputDir+'/*.txt'))
    for name in database_filenames:
        if gid != 'GID_NOT_SET':
            shutil.chown(name, group=gid)
        os.chmod(name, 0o640)


#*************************************************************************************************************
def _final_aggregation_work_elemental(outputDir, outputBaseName, processingHistoryBaseName,
        daySubDir, deleteDailyDatabases, deleteDailyNetCdfFiles, deleteDailyExtraFigures):
    """ Subfunction that extracts data from each daily run to be aggregated.

    Also performs final maintenance work for each daily directory, such as deleting extra unneeded files.

    This code is parallelized in a pool in final_aggregation_work_parallel.
   
    Parameters
    ----------
        outputDir : str
            The top level output directory
        outputBaseName
        processingHistoryBaseName
        daySubDir : str
            The daily subdirectory
        deleteDailyDatabases : bool
          If true then delete the database files for the day
        deleteDailyNetCdfFiles : bool
            If True then delete the daily netCDF files from the daily directory
        deleteDailyExtraFigures : bool
            If True then the daily intermediate figures used to generate the main merged detection figure and validation reports are deleted.
            The combined detection and cutout figure is retained.
   
    Returns
    -------
        bolideDetectionListDay : list
            If there is no database file for this daya then return bolideDetectionListDay = None
        outRejectionPath
        yearDir
        daySubDir

    """

    # Local import so we do not create a circular dependency
    from bolide_dispositions import BolideDispositionProfile, HumanOpinion, MachineOpinion

    outputSubDir = os.path.join(outputDir, daySubDir)
    # Strip '/' from daySubDir
    slashLoc = daySubDir.find('/')
    dayString = daySubDir[0:slashLoc] + daySubDir[slashLoc+1:]
    yearString = daySubDir[0:slashLoc]
    yearDir = os.path.join(outputDir, yearString)
    
    dailyOutFilePath = os.path.join(outputSubDir, outputBaseName + '_' + dayString + '.fs')
    outRejectionPath = None
    
    # The filenames in bolideDetectionList and bolideRejectionList are relative to the daily database filepath
    # We need them relative to the bolideDatabaseAggregate filepath
    pathRelativeList = list(difflib.ndiff(outputDir, outputSubDir))
    pathRelative = '.'
    for e in pathRelativeList:
        if e[0] == '+':
            pathRelative = pathRelative + e[2:]
    
    # Update the bolide detections summary file only if there were any detections for this day
    # If there is no database file for this daya then return bolideDetectionListDay = None
    if (os.path.exists(dailyOutFilePath)):

        # TODO: Make this read-only by specifying time (at=...)
        bolideDatabaseDaily = bolideDB.BolideDatabase(dailyOutFilePath, purgeDatabase=False, wait_time_sec=60.0)
    
        #***
        # Detections
        bolideDetectionListDay = [bolideDatabaseDaily.detections[int(i)].copy() for i in bolideDatabaseDaily.detection_keys]
        # Set relative pathnames
        for bolide in bolideDetectionListDay:
            for idx in np.arange(len(bolide.filePathList)):
                bolide.filePathList[idx] = os.path.join(pathRelative, bolide.filePathList[idx])
            
        print('Day {}, Number of bolide detections = {}'.format(daySubDir, len(bolideDetectionListDay)))
        print("")
    
        #***
        # Rejections
        bolideRejectionListDay = [bolideDatabaseDaily.rejections[int(i)] for i in bolideDatabaseDaily.rejection_keys]
        if (len(bolideRejectionListDay) > 0):
            # Set relative pathnames
            for bolide in bolideRejectionListDay:
                for idx in np.arange(len(bolide.filePathList)):
                    bolide.filePathList[idx] = os.path.join(pathRelative, bolide.filePathList[idx])
            # Create a bolideDispositionProfileList from the rejection list
            # In order to save file size, do not store all data, just the features
            # The rejections are generated from the website, so these are human opinions
            bolideRejectionProfileListDaily = []
            for rejection in bolideRejectionListDay:
                if hasattr(rejection,'cutoutFeatures') and rejection.cutoutFeatures is not None:
                    cutoutFeatures = rejection.cutoutFeatures.copy()
                else:
                    cutoutFeatures = None

                if rejection.assessment.triage.score is not None:
                    machineOpinions=[MachineOpinion(
                        bolideBelief=copy.copy(rejection.assessment.triage.score),
                        method=copy.copy(rejection.assessment.triage.method),
                        source='triage',
                        comments='howFound: '.format(copy.copy(rejection.howFound))
                        )]
                    humanOpinions = None
                elif rejection.assessment.validation.score is not None:
                    machineOpinions=[MachineOpinion(
                        bolideBelief=copy.copy(rejection.assessment.validation.score),
                        method=copy.copy(rejection.assessment.validation.method),
                        source='validation',
                        comments='howFound: '.format(copy.copy(rejection.howFound))
                        )]
                    humanOpinions = None
                elif rejection.assessment.human.assessment is not None:
                    machineOpinions = None
                    humanOpinions=[HumanOpinion(
                       belief=copy.copy(rejection.assessment.human.score),
                       name=copy.copy(rejection.assessment.human.source),
                       comments='howFound: '.format(copy.copy(rejection.assessment.human.source))
                       )]
                else:
                    raise Exception('Unknown assessment source')
                bolideRejectionProfileListDaily.append(
                        BolideDispositionProfile(copy.copy(rejection.ID), 
                            detectionFlag=False, 
                            bolideDetection=None,
                            machineOpinions=machineOpinions,
                            humanOpinions=humanOpinions,
                            features=rejection.features.copy(),
                            stereoFeatures=rejection.stereoFeatures.copy(),
                            cutoutFeatures=cutoutFeatures))
            # Save bolideRejectionProfileListDaily in the local directory
            # There is data, so record the data in a file
            outRejectionPath = os.path.join(outputSubDir, 'bolide_rejections_' + dayString + '.p')
            with open(outRejectionPath, 'wb') as fp:
                pickle.dump(bolideRejectionProfileListDaily, fp)
            print('Day {}, Number of bolide rejections = {}'.format(daySubDir, len(bolideRejectionListDay)))
            print("")
    
    
        # Nothing should change in the database, so abort any changes
        bolideDatabaseDaily.abort()
        bolideDatabaseDaily.close()
    else:
        bolideDetectionListDay = None

    #***
    # Clean-up
    # Delete redundant daily files
    filename = os.path.join(outputSubDir, runElem.config_file_name)
    if os.path.isfile(filename):
        os.remove(filename)
    filename = os.path.join(outputSubDir, processingHistoryBaseName + '_' + dayString + '.txt')
    if os.path.isfile(filename):
        os.remove(filename)

    if deleteDailyDatabases:
        databaseFiles = glob.glob(os.path.join(outputSubDir, 'bolide_database*'))
        # But do not delete the CSV file
        databaseFiles = [file for file in databaseFiles if file.count('csv') == 0]
        for filename in databaseFiles:
            if os.path.isfile(filename):
                os.remove(filename)
    
    # detect_bolides always need to copy over the netCDF files so that the plotting functions can collect their data
    # TODO: use the original files so we don't have to copy then delete!
    # Here we have the option to now delete the netCDF file copies
    if deleteDailyNetCdfFiles:
        filesToRemove = glob.glob(os.path.join(outputSubDir, '*.nc'))
        for filename in filesToRemove:
            os.remove(filename)

    if deleteDailyExtraFigures:
        filesToRemove = glob.glob(os.path.join(outputSubDir, '*.png'))
        # Retain the combined detection and cutout figures
        filesToRemove = [f for f in filesToRemove if f.count('_with_cutout') == 0]
        for filename in filesToRemove:
            os.remove(filename)
        
    return bolideDetectionListDay, outRejectionPath, yearDir, daySubDir

#******************************************************************************
def run_glm_pipeline_batch(config):
    """
    Top level GLM glm detection batch function
   
    Run the detector on any previously unprocessed GLM data files.
   
    Performs the following steps:
    (1) Examines the processing history and the contents of the input data
        directory to identify any newly available .nc files to process.
    (2) Deterimine what we are detecting (i.e. bolides, gigantic jets, etc...)
    (3) Run the detector. Files are processed by day.
    (4) Log daily processing in a hisotry file.
   
    Parameters
    ----------
    config : Dict
        Top level configuration dictionary 
        see get_configuration_template
    """

    # Force multiprocessing to spawn new processes.
    # See https://pythonspeed.com/articles/python-multiprocessing/
   #mp.set_start_method("spawn")

    startTime = time.time()

    ATAP_REPO_ROOT = os.getenv('ATAP_REPO_ROOT', default='ERROR')
    if (ATAP_REPO_ROOT=='ERROR'):
        raise Exception('ATAP_REPO_ROOT not defined in shell environment')

    if not check_input_param_consistency(config):
        raise Exception('Error in processing top level configuration parameter')

    inputRootDir    = config['inputRootDir']
    # Temove trailing '/' from path
    inputRootDirL0  = config['inputRootDirL0']
    otherSatelliteRootDir = config['otherSatelliteRootDir']
    inputRootDir = inputRootDir.rstrip('/')
    if inputRootDirL0 is not None:
        inputRootDirL0 = inputRootDirL0.rstrip('/')
    if otherSatelliteRootDir is not None:
        otherSatelliteRootDir  = otherSatelliteRootDir .rstrip('/')

    outputDir       = config['outputDir']
    if config['reportOutputDir'] is None:
        reportOutputDir = outputDir
    else:
        reportOutputDir = config['reportOutputDir']
    outputBaseName  = bolideDB.databaseFileBasename
    processingHistoryBaseName = config['processingHistoryBaseName']
    detectionType = config['detectionType']
    elementalConfigFile = config['elementalConfigFile']
    validationConfigFile = config['validationConfigFile']


    #******************************************
    # Create the elemental configuration object
    if detectionType == 'bolides':
        # Create the detect_bolides configuration dictionary.
        if not os.path.isfile(elementalConfigFile):
            sys.exit('The elemental config file {} does not exist.'.format(elementalConfigFile))
        else:
            detectBolidesConfig = ioUtil.read_config_file(elementalConfigFile, bIo.input_config.configuration_template())
        
        # Override the input and output paths specified in the
        # elementalConfigFile. Use absolute paths for file locations.
        detectBolidesConfig['data_path']   = os.path.abspath(inputRootDir)
        if otherSatelliteRootDir is not None:
            detectBolidesConfig['other_satellite_input_path'] = os.path.abspath(otherSatelliteRootDir)
        else:
            # If not given, set to inputRootDir
            detectBolidesConfig['other_satellite_input_path'] = None
        detectBolidesConfig['output_path'] = os.path.abspath(outputDir)
        
        input_config = bIo.input_config._from_config_dict(detectBolidesConfig)
        del detectBolidesConfig
    elif detectionType == 'gigantic_jets':
        # Create the detect_gigantic_jets configuration dictionary.
        if not os.path.isfile(elementalConfigFile):
            sys.exit('The elemental config file {} does not exist.'.format(elementalConfigFile))
        else:
            detectGiganticJetsConfig = ioUtil.read_config_file(elementalConfigFile, gjIo.input_config.configuration_template())
        
        # Override the input and output paths specified in the
        # elementalConfigFile. Use absolute paths for file locations.
        detectGiganticJetsConfig['data_path']   = os.path.abspath(inputRootDir)
        detectGiganticJetsConfig['other_satellite_input_path']   = os.path.abspath(otherSatelliteRootDir)
        detectGiganticJetsConfig['output_path'] = os.path.abspath(outputDir)
        
        input_config = gjIo.input_config._from_config_dict(detectGiganticJetsConfig)
        del detectGiganticJetsConfig
    elif detectionType == 'streaks':
        # Create the detect_streaks configuration dictionary.
        if not os.path.isfile(elementalConfigFile):
            sys.exit('The elemental config file {} does not exist.'.format(elementalConfigFile))
        else:
            detectStreaksConfig = ioUtil.read_config_file(elementalConfigFile, streakIo.input_config.configuration_template())
        
        # Override the input and output paths specified in the
        # elementalConfigFile. Use absolute paths for file locations.
        detectStreaksConfig['data_path']   = os.path.abspath(inputRootDir)
        detectStreaksConfig['other_satellite_input_path']   = os.path.abspath(otherSatelliteRootDir)
        detectStreaksConfig['output_path'] = os.path.abspath(outputDir)
        
        input_config = streakIo.input_config._from_config_dict(detectStreaksConfig)
        del detectStreaksConfig

    #***
    # If this is a quick run (via run_glm_pipeline_quick) using a spatiotemporal box then change some configuration
    # parameters
    if 'spatiotemporal_box' in config:

        assert config['detectionType'] == 'bolides', 'Quick pipeline spatiotemporal box only works with bolide detection'

        input_config.spatiotemporal_box = config['spatiotemporal_box']
        # Set detection threshold to a very small number
        input_config.classifier_threshold = input_config.SMALL_CLASSIFIER_THRESHOLD
        # Set minimum number of groups per cluster to 1
        input_config.min_num_groups = 1

        # If finding an explicit cluster then turn off neighborhood searching
        # We combine all per file clusters in a single cluster and do not want the duplicates
        if input_config.spatiotemporal_box[0].explicit_cluster:
            input_config.cluster_extractNeighboringFiles = False

    #******************************************
    # Create the Validation configuration object
    validationConfig = ioUtil.read_config_file(validationConfigFile, vIo.input_config.configuration_template())
    
    # Override the input and output paths specified in the
    # validationConfigFile. Use absolute paths for file locations.
    validationConfig['report_output_path'] = os.path.abspath(reportOutputDir)
    
    validation_input_config = vIo.input_config(validationConfig, input_config)
    del validationConfig

    #--------------------------------------------------------------------------
    # Construct a list of files to process. There are three options:
    # 
    # 1) GeoNEX individual files (subdirectories by hour of day)
    # 2) Daily bundles
    # 3) Individual netCDF files in one directory
    #--------------------------------------------------------------------------

    # First check if there are individual files in this directory
    indivFiles = glob.glob(os.path.join(inputRootDir, '*.nc'), recursive=False)
    if os.path.splitext(inputRootDir)[1] == '.nc':
        input_config.data_source = 'files'
        satellite = db.determine_satellite_from_file_path(inputRootDir)
        outputDir = os.path.join(outputDir, satellite)
        indivFiles = [inputRootDir]
    elif len(indivFiles) > 0:
        input_config.data_source = 'files'
        satellite = db.determine_satellite_from_file_path(indivFiles[0])
        outputDir = os.path.join(outputDir, satellite)
    else:
        # Get list of daily directories
        availableDays = glob.glob(os.path.join(inputRootDir, '*/*/'), recursive=False)
        if len(availableDays) == 0:
            raise Exception('No data at inputRootDir available to process')
        
        # Check if the last directory level is a 3 digit Day Of Year (DOY) or a 4-digit MMDD format
        # DOY year means this is GeoNex data, MMDD means individual netCDF files
        lastDirNameLength = len(os.path.split(os.path.split(availableDays[0])[0])[1])
        if lastDirNameLength == 3:
            input_config.data_source = 'geonex'
        elif lastDirNameLength == 4:
            input_config.data_source = 'files'
        else:
            # This might be a daily bundles data set
            availableFiles = glob.glob(os.path.join(inputRootDir, '*/*.tgz'), recursive=True)
            if (len(availableFiles) > 0):
                input_config.data_source = 'daily_bundle'
            else:
                raise Exception('inputRootDir does not appear to be a GLM data archive.')

        # Get the days of data available from the directory names
        availableDaysList = []
        for file in availableDays:
            ISODate = extract_ISO_date_from_directory(file, input_config.data_source)
            if ISODate is not None:
                availableDaysList.append(ISODate)
        availableDaysList = np.unique(availableDaysList).tolist()
        
        # outputDir does not contain the satellite. Add that now.
        satellite = db.determine_satellite_from_file_path(availableDays[0])
        outputDir = os.path.join(outputDir, satellite)

    # We now know everything we need for the Provenance object. Create that now and display the configurations.
    provenanceObject = Provenance(config, input_config, validation_input_config)
    provenanceObject.elementalConfigDict['data_source'] = copy.copy(input_config.data_source)
    provenanceObject.print()

    if input_config.data_source == 'geonex':
        print("")
        print("****** Processing GLM data from the GeoNEX data stream...")
        print("")
    elif input_config.data_source == 'files':
        print("")
        print("****** Processing GLM data by individual files...")
        print("")
    elif input_config.data_source == 'daily_bundle':
        print("")
        print("****** Processing GLM data by daily bundles...")
        print("")


    #***
    # If forcing reprocess then delete all files located at outputDir subdirectory for satellite
    if (config['forceReprocess']):
        if os.path.exists(outputDir):
            shutil.rmtree(outputDir)
        # Also delete the validation reports
        if os.path.exists(reportOutputDir):
            satelliteFiles = glob.glob(os.path.join(reportOutputDir, '*/*/*{}*.pdf'.format(satellite)), recursive=True)
            if len(satelliteFiles) > 0:
                [os.remove(filename) for filename in satelliteFiles]

    #***
    # Set range of date and times to process
    if config['startDate'] == '' or config['startDate'] is None:
        startDatetime = datetime.min
        startDate = date.min
    else:
        # We only want the day, not the time of day
        startDatetime = datetime.fromisoformat(config['startDate'])
        startDate = startDatetime.date()
    
    if config['endDate'] == '' or config['endDate'] is None:
        endDatetime = datetime.max
        endDate = date.max
    else:
        # We only want the day, not the time of day
        endDatetime = datetime.fromisoformat(config['endDate'])
        endDate = endDatetime.date()
    
    nowDatetime = datetime.now(timezone.utc)
    nowDate = date(nowDatetime.year, nowDatetime.month, nowDatetime.day)
    if config['doNotProcessCurrentDay']:
        yesterdayDate = nowDate - timedelta(days=1)
        if endDate > yesterdayDate:
            endDate = yesterdayDate
    
    print('Only processing data between {} and {}'.format(startDate, endDate))
        
    if input_config.data_source != 'files':
        # ***
        # Find what days have been processed
        # The main unit of work is a day of data.
        # The history file records what days have been processed already,
        # Log entry is 'YYYY-MM-DD PATH', where PATH points to the top level directory of the data for that day
        # Get list of processed days
        processingHistoryPath = os.path.join(outputDir, processingHistoryBaseName + '.txt')
        processedDaysList = []
        if os.path.isfile(processingHistoryPath):
            for line in open(processingHistoryPath):
            
                # Strip out comments and lines containing only whitespace.
                s = line.split(sep='#', maxsplit=2)[0]
                s = ''.join(s.split())  # Remove all whitespace.
                # Only keep the date stamp
                s = line.split(sep=' ', maxsplit=2)[0]
                if len(s.strip()) <= 0:  # if s is empty, get the next line.
                    continue
            
                processedDaysList.append(s)
        
        # Identify files to process.
        availableSet   = set(availableDaysList)
        processedSet   = set(processedDaysList)
        daysToProcess = set_diff(availableSet, processedSet)
        
        print('Available days: {}'.format(len(availableSet)))
        print('Processed days: {}'.format(len(processedSet)))
        
        # Only include files within the requested processing dates
        # If we are not to process the current day then set the endDate to yesturday at a maximum
        # This only works if not using a path to individual files
        if (config['doNotProcessCurrentDay'] or
                (config['startDate'] != '' and config['startDate'] is not None) or
                (config['endDate']   != '' and config['endDate']   is not None)):
        
            if input_config.data_source == 'geonex':
                daysToProcessSave = daysToProcess.copy()
                for day in daysToProcessSave:
                    thisDate = date.fromisoformat(day)
                    if not (thisDate >= startDate and thisDate <= endDate):
                        daysToProcess.remove(day)
            elif input_config.data_source == 'daily_bundle':
                filesToProcess = availableFiles
                # Get dates from tar filename
                # Daily bundle files have names like: OR_GLM-L2-LCFA_G16_s20191220.nc.tgz
                # So, search for _s*.tgz
                fileIndicesToKeep = []
                for fileListIdx, filename in enumerate(filesToProcess):
                    dateIdx = filename.find('_s')
                    dateEndIdx = filename.find('.nc.tgz')
                    if dateIdx == -1 or dateEndIdx == -1:
                        raise Exception('Daily bundle filename not as expected')
                    # Add 2 to index to get to beginning of date string
                    dateIdx += 2
                    dateStr = filename[dateIdx:dateEndIdx]
                    # First 4-digit year, then 2-digit month then 2-digit day
                    thisFileDate = date(int(dateStr[0:4]), int(dateStr[4:6]), int(dateStr[6:8]))
                    if thisFileDate >= startDate and thisFileDate <= endDate:
                        # Within date range, add to list to keep
                        fileIndicesToKeep.append(fileListIdx)
        
                filesToProcess = [filesToProcess[idx] for idx in fileIndicesToKeep]
            else:
                raise Exception('Using specific dates not compatible with this data source')
        
            # Check if Level 0 data is available
            if config['delayNDaysForL0Data'] > 0:
                print('Not processing days if no Level 0 data is present')
                # For each day up to delayNDaysForL0Data check if the Level 0 data is available
                firstDayToCheck = nowDate - timedelta(days=config['delayNDaysForL0Data']-1)
                firstDayToCheck = np.max([np.datetime64(startDate), np.datetime64(firstDayToCheck)])
        
                daysToCheck = np.arange(firstDayToCheck, endDate+timedelta(days=1), timedelta(days=1)).astype(date)
                daysToCheck = [day.date() for day in daysToCheck]
                for day in daysToCheck:
                    MMDD    = str(day.month).zfill(2) + str(day.day).zfill(2)
                    availableL0Files = glob.glob(os.path.join(inputRootDirL0, str(day.year), MMDD, '*.nc'), recursive=False)
                    # The L0 files are 5 minute long, so 288 files per day
                    # We want atleast 95% of those (to allow for some missing files)
                    # So, we want atleast 288*0.95 = 273 files
                    if len(availableL0Files) < 273:
                        # Sometimes the day to remove is not in daysToProcess
                        if day.isoformat() in daysToProcess:
                            daysToProcess.remove(day.isoformat())
        
        elif input_config.data_source == 'daily_bundle':
            filesToProcess = availableFiles

    elif input_config.data_source == 'files':
        filesToProcess = indivFiles


    if input_config.data_source != 'geonex':
        print('****** Processing {} files...'.format(len(filesToProcess)))
    else:
        # geonex data
        if len(daysToProcess) == 0:
            sys.exit('****** No new files to process. Exiting ******')
        print('****** Processing {} days...'.format(len(daysToProcess)))

    #--------------------------------------------------------------------------
    # Create list of individual files to process for each day of processing.
    # Call it filesByDayDict 
    #--------------------------------------------------------------------------
    if input_config.data_source == 'geonex':
        # Create a list of distinct daily subdirectories from the list of files to process.
        filesByDayDict = {}
        for day in daysToProcess:
            # day is in ISO format: 'YYYY-MM-DD'
            dayObj      = date.fromisoformat(day)
            daySubDir   = str(dayObj.year) + '/' + '{:02}'.format(dayObj.month) + '{:02}'.format(dayObj.day)
            dayOfYear   = '{:03}'.format(dayObj.timetuple().tm_yday)
            # Group files-to-process by day.
            # Find files for this day
            filesForThisDay = glob.glob(os.path.join(inputRootDir, '{}/{}/*/*.nc'.format(dayObj.year, dayOfYear)), recursive=True)
            filesForThisDay.sort()
            filesByDayDict[daySubDir] = filesForThisDay

        
    elif input_config.data_source == 'daily_bundle':
        # We cannot create symlinks if a daily Bundle run because the .nc files are in a temporary directory.
        if (input_config.createSymlinks):
            print('Warning: Runing on daily bundles. We need to copy the .nc files. Setting createSymlinks = False.')
            input_config.createSymlinks = False
        # Create a list of distinct daily subdirectories in the list of files to process.
        daySubDirList = []
        filesByDayDict = {}
        for file in filesToProcess:
            [_, yearSubDir] = os.path.split(os.path.dirname(file))
            [_, filename] = os.path.split(file)
            # Filename is like: OR_GLM-L2-LCFA_G16_s20170619.nc.tgz
            # First strip off both extensions
            [filename,ext] = os.path.splitext(filename)
            if ext != '.tgz':
                raise Exception ('Unexpected filename extension: {}'.format(ext))
            [filename,ext] = os.path.splitext(filename)
            if ext != '.nc':
                raise Exception ('Unexpected filename extension: {}'.format(ext))
            monthDayStr = filename[-4:]
            yearStr = filename[-8:-4]
            if (yearStr != yearSubDir):
                raise Exception ('Data file year does not match directory name')
            daySubDir = yearSubDir + '/' + monthDayStr
            # For daily bundles there should be only one file per day
            if daySubDir in daySubDirList:
                raise Exception ('There appears to be two files with the same year month and day.')
            daySubDirList.append(daySubDir)
        
            # Group files-to-process by day.
            filesByDayDict[daySubDir] = [file]

    elif input_config.data_source == 'files':
        # Create a list of distinct daily subdirectories in the list of files to process.
        daySubDirList = []
        for file in filesToProcess:
            [parentDir, monthDaySubDir] = os.path.split(os.path.dirname(file))
            [parentDir, yearSubDir] = os.path.split(parentDir)
            daySubDir = yearSubDir + '/' + monthDaySubDir
            if daySubDir not in daySubDirList:
                daySubDirList.append(daySubDir)
        
        # Group files-to-process by day.
        filesByDayDict = { k: [] for k in daySubDirList }
        for file in filesToProcess:
            [parentDir, monthDaySubDir] = os.path.split(os.path.dirname(file))
            [parentDir, yearSubDir] = os.path.split(parentDir)
            daySubDir = yearSubDir + '/' + monthDaySubDir
            filesByDayDict[daySubDir].append(file)

    #***
    # If the start and end dates specify times of day then only keep data files within these specific times
    # But also add in the 50-second offset to ensure we keep enough neighboring files for when we search three 20-second files at a time
    # Intitialize the time part of the datetime to midnight (0,0,0)
    if  ( (startDatetime > datetime.combine(startDate, datetime.min.time()) and startDatetime != datetime.min) or
             (endDatetime > datetime.combine(endDate, datetime.min.time())  and endDatetime != datetime.max)):
        minProcessTime = (startDatetime - timedelta(seconds=50) - bd.BolideDetection.epoch).total_seconds()
        maxProcessTime = (endDatetime + timedelta(seconds=50) - bd.BolideDetection.epoch).total_seconds()
        for daySubDir, filesThisDay in filesByDayDict.items():
            idxToKeep = []
            for idx, filename in enumerate(filesThisDay):
                # Get the time stamp of the file under question.
                # just get the filename without the path so that the '_s', '_e' and '_c' searches do not find a directory name
                filenameNoPath = os.path.basename(filename)
                fileStartTimeStr   = int(filenameNoPath[filenameNoPath.index('_s')+2: filenameNoPath.index('_e')])
                fileEndTimeStr     = int(filenameNoPath[filenameNoPath.index('_e')+2: filenameNoPath.index('_c')])
                
                fileStartTime = extract_total_seconds_from_glm_timestamp(fileStartTimeStr)
                fileEndTime = extract_total_seconds_from_glm_timestamp(fileEndTimeStr)

                if fileStartTime >= minProcessTime and fileEndTime <= maxProcessTime:
                    idxToKeep.append(idx)

            # Create new list of just the files we want
            filesByDayDict[daySubDir] = [filesByDayDict[daySubDir][i] for i in idxToKeep]

    # Remove days with no files
    keysToRemove = []
    for day, files in filesByDayDict.items():
        if len(files) == 0:
            keysToRemove.append(day)
    if len(keysToRemove) > 0:
        warnings.warn('Some days to process have no files available. These days cannot be processed.')
        for key in keysToRemove:
            filesByDayDict.pop(key)

    # If no days available then terminate now. Nothing to do
    if len(filesByDayDict.keys()) == 0:
        sys.exit('****** No new files to process. Exiting ******')


    #--------------------------------------------------------------------------
    # Process files by day.
    # If batch processing, collect all information we need for each batch 
    # Create a text file (dirListFilePath) which is a list of the paths to each directory for daily elemental processing
    # We aggregate the data later.
    #--------------------------------------------------------------------------
    dirListFilePath = os.path.join(outputDir, config['dirListFileName'])

    for daySubDir in filesByDayDict:

        if (config['multiProcessEnabled']):

            # For each day, create a file with the stored information
            runElem.pickle_elemental_data(outputDir,  
                    input_config, validation_input_config, daySubDir, filesByDayDict[daySubDir], provenanceObject)
            
            # Create file with paths to all elemental (day) process jobs to call
            outputSubDir = os.path.join(outputDir, daySubDir)
            ioUtil.append_lines_to_file(dirListFilePath, [outputSubDir])

        else:
            
            # No parallel processing
            #  Just call the elemental function directly
            runElem.run_glm_pipeline_elemental(outputDir, input_config, validation_input_config,
                    daySubDir, filesByDayDict[daySubDir], provenanceObject)


    # Call GNU parallel if multiProcessEnabled
    # Call the elemental function with GNU parallel
    if (config['multiProcessEnabled']):
        print('Starting parallel batch processing...')

        # If on the NAS then use qsub, otherwise, call parallel directly
        if (os.getenv('isOnNas', default='False') == 'False'):
            # This is running on a local machine
            print('Running on a local machine, using GNU Parallel...')
            # Use the maxumum number of CPUs available
            os.system('parallel -a {} run_glm_pipeline_elemental.py'.format(dirListFilePath))

        else:
            # We are on the NAS.
            # Submit the job then exit
            print('Running on the NAS, submitting jobs with qsub and GNU Parallel...')
            # Submit the PBS jobs
            gnu_parallel_script_path = os.path.normpath(ATAP_REPO_ROOT + '/system/bin/gnu_parallel_script.pbs')
            mainProcessOut = subprocess.run(
                    ['qsub', '-N', 'GlmPipeline', '-v', 
                        'ATAP_PROC_FILE="{}",ATAP_REPO_ROOT="{}"'.format(dirListFilePath,ATAP_REPO_ROOT),
                        gnu_parallel_script_path],
                    capture_output=True, text=True)
            jid = mainProcessOut.stdout[0:-1] # There's a cairrage return at the end of stdout!
            print('Parallel batch job ID: {}'.format(jid))
            
            # process submited.
            # Save the data necessary for the aggregation step
            pickle_aggregation_data (outputDir, filesByDayDict, dirListFilePath, provenanceObject)

            # Submit the aggregation step job
            # Tell PBS to run this job after the main job
            aggregator_script_path = os.path.normpath(ATAP_REPO_ROOT + '/system/bin/aggregator_batch_script.pbs')
            aggregation_data = os.path.join(outputDir, aggregate_data_file_name)
            aggregatorProcessOut = subprocess.run(
                    ['qsub', '-N', 'GlmAggregator', '-W', 'depend=afterany:{}'.format(jid), '-v', 
                        'ATAP_AGGRE_FILE="{}",ATAP_REPO_ROOT="{}"'.format(aggregation_data,ATAP_REPO_ROOT),
                        aggregator_script_path],
                    capture_output=True, text=True)
            jid = aggregatorProcessOut.stdout[0:-1] # There's a cairrage return at the end of stdout!
            print('Aggregation job ID: {}, paused until parallel batch job finishes'.format(jid))
            
            # now exit
            print('pbs batch jobs submitted. Exiting. ')
            sys.exit()
            
        print('****')
        print('Finished parallel batch processing')
        print('****')

    # Final aggregation steps Only perform if NOT on NAS
    # and only for bolides detections for now...
    if detectionType == 'bolides':
        final_aggregation_work_parallel (filesByDayDict, dirListFilePath, provenanceObject)

    endTime = time.time()
    totalTime = endTime - startTime
    print("")
    print("****** run_glm_pipeline_batch successfuly finished")
    print("")
    print("Total batch processing time: {:.2f} minutes, {:.2f} hours".format(totalTime/60, totalTime/60/60))
    print("")

#******************************************************************************
if __name__ == "__main__":
    """ Main top level program


    Arguments
    ---------
    top_level_config_file : str 
        The path to the top level configuration file to set up the batch processing.

    """

    # Create the configure dictionary using the passed top level configuration file
    config = ioUtil.read_config_file(sys.argv[1], configTemplate=get_configuration_template() )

    run_glm_pipeline_batch(config)
