#*************************************************************************************************************
# This module will take labeled GLM detections created via glm_labeling_gui fomr the pipeline and compile the data into
# a data set to be used to train a machine learning classifier, or anything else you want to use the compiled data set
# for!
#
# This module creates a data set to be used with the bolide_dispositions module.
#

import sys
import os
import argparse
import pickle
import numpy as np
import bolide_detections as bd
import bolide_dispositions as bDisposition

# *****************************************************************************
# read_bolide_disposition_csv_records(csvFileName, ncFilePath=''):
#
# Read a bolide detections CSV file and return a list of BolideDispositionProfile objects 
# which contains bolide detections and disposition labels.
#
# INPUTS
#   csvFileName     -- [str] A CSV file in which each line contains the following
#                       information for a single bolide disposition:
#                       CONFIDENCE
#                       START_TIME
#                       END_TIME
#                       LAT_DEGREES
#                       LON_DEGREES
#                       JOULES
#                       COMMENTS
#                       NAME
#                       EXPERTISE
#                       FILE_NAMES
#                       ID_TYPE
#                       ID_LIST    
#   ncFilePath      -- [str] If the path in <FILE_NAMES> for the .nc file above is not correct, for example, because the data was
#                       processed on a different machine then use this path instead (use '' to use what's in the csv
#                       file)
#
# OUTPUTS:
#     bolideDispositionProfileList -- [list] of bolide_dispositions.BolideDispositionProfile objects, one per line of the CSV file.
#
# NOTES
#     If the first row of the CSV file contains column labels, this function
#     will attempt to find the column name order and will
#     read from those columns. Otherwise it will assume the column order
#     listed above
#
# *****************************************************************************
def read_bolide_disposition_csv_records(csvFileName, ncFilePath=''):

    raise Exception("Fix this for use with new time field in CSV file")

    isoDateFormat = "%Y-%m-%d %H:%M:%S.%f"

    bolideDispositionProfileList = []

    # Set default column indices.
    confidenceIndex = 0
    startTimeIndex  = 1
    endTimeIndex    = 2
    latDegreesIndex = 3
    lonDegreesIndex = 4
    joulesIndex     = 5
    commentsIndex   = 6
    nameIndex       = 7
    expertiseIndex  = 8
    fileNamesIndex  = 9
    idTypeIndex     = 10
    idListIndex     = 11

    with open(csvFileName, 'r') as fp:
        reader = csv.reader(fp)
        try:
            for row in reader:
                # Check first row of CSV file to set column order.
                if row.__contains__('CONFIDENCE') :
                    confidenceIndex = row.index('CONFIDENCE')
                    startTimeIndex  = row.index('START_TIME')
                    endTimeIndex    = row.index('END_TIME')
                    latDegreesIndex = row.index('LAT_DEGREES')
                    lonDegreesIndex = row.index('LON_DEGREES')
                    joulesIndex     = row.index('JOULES')
                    commentsIndex   = row.index('COMMENTS')
                    nameIndex       = row.index('NAME')
                    expertiseIndex  = row.index('EXPERTISE')
                    fileNamesIndex  = row.index('FILE_NAMES')
                    idTypeIndex     = row.index('ID_TYPE')
                    idListIndex     = row.index('ID_LIST')

                else:
                    # If nothing is selected, the default confidence is '', skip such entries
                    if (row[confidenceIndex] == ''):
                        continue
                    confidence      = eval(row[confidenceIndex])
                    start_time      = datetime.datetime.strptime(row[startTimeIndex], isoDateFormat)
                    end_time        = datetime.datetime.strptime(row[endTimeIndex], isoDateFormat)
                    lat_degrees     = eval(row[latDegreesIndex])
                    lon_degrees     = eval(row[lonDegreesIndex])
                    joules          = eval(row[joulesIndex])
                    comments        = row[commentsIndex]
                    name            = row[nameIndex]
                    expertise       = eval(row[expertiseIndex])
                    fileNames       = eval(row[fileNamesIndex])
                    id_type         = row[idTypeIndex]
                    id_list         = eval(row[idListIndex]) 

                    # Check if we need to change the .nc file path
                    if (ncFilePath != ''):
                        for i in np.arange(np.size(fileNames)):
                            fileName = fileNames[i]
                            # Retrieve the file name from the path string
                            fileName = fileName[fileName.rfind('/')+1:]
                            # Prepend with the correct path
                            fileName = ncFilePath + '/' + fileName
                            fileNames[i] = fileName

                    #***
                    # Construct a bolide disposition candidate profile

                    # Create a bolideDetection object from the .nc files
                    glmRecord = bd.glmDataSubsetRecord(fileNames, id_type, id_list)
                    # Create a bolide detection object from the record
                    bolideDetectionInstance = bd.bolideDetection.fromGlmDataFiles(glmRecord)
                    if (bolideDetectionInstance == []):
                        # Error Creating bolideDetection object, skipping this detection
                        continue
                    # Now create the disposition profile and apend to the list
                    bolideDispositionProfileList.append(bDisposition.BolideDispositionProfile(bolideDetectionInstance,
                        humanOpinions=[HumanOpinion(confidence, name, expertise, comments)]))

        except csv.Error as e:
            sys.exit('file %s, line %d: %s' % (csvFileName, reader.line_num, e))

    return bolideDispositionProfileList

# *****************************************************************************
# compile_labeled_data_set (inPath, readCsv=True, ncFilePath='')
#
# Reads in a collection of dispositions generated by glm_labeling_gui and creates a data set containing all the
# dispositions, ready for analysis.
#
# Inputs:
#   inPath      -- [str] Path to the data files to be read in (either CSV files or pickle files)
#                   Does not recursively search subdirectories.
#   readCsv     -- [bool] If true then read in the CSV (*.csv) files, otherwise read in the pickle (*.p) files
#   ncFilePath  -- [str] If the path the the *.nc files containing the raw GLM data is not correct in the labeled
#                   disposition files then search this path for the data. Only needed is readCsv = True, otherwise, 
#                   all the data is already in the pickle files.
#
# Outputs:
#   bolideDispositionProfileList   -- [list] a list of bolide_dispositions.BolideDispositionProfile, one for each disposition
#
# *****************************************************************************
def compile_labeled_data_set (inPath, readCsv=True, ncFilePath=''):

    if (readCsv):
        print('Constructing bolideDisposition objects from CSV records ...')
        extensionStr = '.csv'
    else:
        print('Constructing bolideDisposition objects from pickle records ...')
        extensionStr = '.p'

    # Read in the disposition files.

    fileNames = os.listdir(path=inPath)

    bolideDispositionProfileList = []
    nFiles = len(fileNames)
    fileCount = 0
    for fileName in fileNames:

        fileCount += 1
        print("Working on file {} of {}".format(fileCount, nFiles))

        # If not a correct file extension then skip
        if (fileName.endswith(extensionStr)):
            if readCsv:
                bolideDispositionProfileList = bolideDispositionProfileList + read_bolide_disposition_csv_records(fileName, ncFilePath)
            else:
                bolideDispositionProfileList = bolideDispositionProfileList + bd.unpickle_bolide_detections(fileName)

    return bolideDispositionProfileList


# *****************************************************************************
# pickle_bolide_bolideDispositionProfileList (filename, bolideDispositionProfileList):
#
# Write a list of bolide disposition objects to a binary pickle file.
#
# INPUTS
#   filename                     -- [str] path and name fo file to write to
#   bolideDispositionProfileList -- [list] a list of bolide_disposition.BolideDispositionProfile, one for each disposition
#
# OUTPUTS
#     (none)
#
# NOTES
#
# *****************************************************************************
def pickle_bolide_bolideDispositionProfileList (filename, bolideDispositionProfileList):

    # Write the data.
    try:
        # If the file exists, overwriting the disposition list.
        with open(filename, 'wb') as fp :
            pickle.dump(bolideDispositionProfileList, fp)
    except:
        sys.exit('Could not write to file {}.'.format(filename))

    fp.close()

# *****************************************************************************
# unpickle_bolide_bolideDispositionProfileList (filename)
#
# Unpickle bolide disposition objects and return them as a list. The pickle file may
# contain either a list of objects or a series of individual objects that are
# not part of a list.
#
# INPUTS
#     filename              -- A string designating the the path and name of
#                              the pickle file. The file may contain serial
#                              objects or one or more lists of objects.
# OUTPUTS
#   bolideDispositionProfileList   -- [list] a list of bolide_disposition.BolideDispositionProfile, one for each disposition
# *****************************************************************************
def unpickle_bolide_bolideDispositionProfileList (filename):

    bolideDispositionProfileList = []

    with open(filename, 'rb') as fp:
        try:
            obj = pickle.load(fp)
            if isinstance(obj, list) :
                bolideDispositionProfileList = obj
                while True:
                    bolideDispositionProfileList.extend(pickle.load(fp))
            else :
                bolideDispositionProfileList = [obj]
                while True:
                    bolideDispositionProfileList.append(pickle.load(fp))
        except EOFError:
            pass # We've reached the end of the file.

    fp.close()

    return bolideDispositionProfileList


# *****************************************************************************
# Parse an argument list.
#
# INPUTS
#     arg_list : A list of strings, each containing a command line argument.
#                NOTE that the first element of this list should NOT be the
#                program file name. Instead of passing sys.argv, pass
#                arg_list = sys.argv[1:]
#
# OUTPUTS
#     args     : A Namespace containing the extracted arguments.
#
# *****************************************************************************
def parse_arguments(arg_list):

    parser = argparse.ArgumentParser(description='Compile Labeled GLM Data Set from many seperate disposition files into' \
                'a single data strcuture. Give the path the disposition files in <inPath> which will be searched' \
                'recursively into subdirectories.')
    parser.add_argument('inPath', metavar='inPath', type=str, nargs=1,
                        help='Input file path containing the disposition files.')
    parser.add_argument('outFile', metavar='outFile', type=str, nargs=1,
                        help='File name to save compiled data')
    parser.add_argument('--csv', '-c', dest='csv', action='store_true',
                        help='Indicates in files are CSV files (default: False)')
    parser.add_argument('--ncFilePath', '-ncp', dest='ncFilePath', type=str, default='',
                        help='Path to the .nc files (if not what is given in the .csv file)')

    args = parser.parse_args(arg_list)

    return args


#*************************************************************************************************************
if __name__ == "__main__":

    # Make sure we're running Python 3
    if sys.version_info[0] < 3:
        raise Exception("Python 3.0 or higher is required")

    args = parse_arguments(sys.argv[1:])

    inPath  = args.inPath[0]
    outFile  = args.outFile[0]
    readCsv = args.csv
    ncFilePath = args.ncFilePath

    bolideDispositionProfileList = compile_labeled_data_set (inPath, readCsv, ncFilePath)
    print('{} dispositioned bolide detections loaded'.format(len(bolideDispositionProfileList)))

    pickle_bolide_bolideDispositionProfileList(outFile, bolideDispositionProfileList)

    print('Labeled bolide detections written to file {}'.format(outFile))

    pass
