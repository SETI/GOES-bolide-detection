#!/usr/bin/env python
import sys
import argparse
import bolide_detections as bd

# *****************************************************************************
# Convert Bolide objects (defined in bolide_support_functions.py) to
# bolideDetection objects (defined in bolide_detections.py).
#
# INPUTS
#   bolideList          : A list of Bolide objects. See
#                           bolide_support_functions.py for details.
#   confidence          : Confidence score for this detection
#                           low=> 0.0, high=> 1.0, unknown=> -1.0
#   confidenceSource    : str, Where confidence came from (I.e. neo-bolides website)
#   howFound            : str, How the detection was found
#
# OUTPUTS
#     bolideDetectionList   : A list of bolideDetection objects. See
#                             bolide_detections.py for details.
#
# NOTES
#   - For a given Bolide object, the conversion is done by first constructing
#     a glmDataSubsetRecord object, which specifies the original data file
#     along with the event or group IDs comprising the detection. The detection
#     data is then extracted from the original netCDF4 data file to construct
#     the bolideDetection object.
# *****************************************************************************
def convert_detection_representation(bolideList, confidence=-1.0, 
        confidenceSource='unknown', howFound='unknown'):

    raise Exception('This function is no longer used')

    bolideDetectionList= []

    for iBolide in range( len(bolideList) ) :
        print('Converting {} of {} detections ...'.format(iBolide+1, len(bolideList)))

        detection = bolideList[iBolide]

        # Get the group ID list for this detection.
        idList = []
        for grp in detection.group:
            idList.append(grp.id)

        detectionRecord = bd.glmDataSubsetRecord([detection.filePathStr], 'group_id', idList)
        bolideDetectionList.append(bd.bolideDetection.fromGlmDataFiles(detectionRecord, 
            confidence=confidence, confidenceSource=confidenceSource, howFound=howFound))

    return bolideDetectionList

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
# *****************************************************************************
def parse_arguments(arg_list):
    parser = argparse.ArgumentParser(description='Convert pickled Bolide objects to bolideDetection objects and re-pickle.')
    parser.add_argument('infile', metavar='infile', type=str, nargs=1,
                        help='A pickle file containing Bolide objects.')
    parser.add_argument('outfile', metavar='outfile', type=str, nargs=1,
                        help='Output pickle file to which bolideDetection objects will be written.')
    parser.add_argument('-n', dest='maxNumToConvert', type=int, default=-1,
                        help='The maximum number of detections to convert.')

    args = parser.parse_args(arg_list)

    return args


# *****************************************************************************
# convert_detections 
#
# The actual workhorse function used in __main__ to perform the data conversion.
#
# Inputs:
#   infile  -- [str] The input pickled file with the original data format
#   outfile -- [str] The output pickled file with the new data format
#   maxNumToConvert -- [int] Max number of detections to convert (default = -1 => convert all)
#
# Outputs:
#   NONE, just the new file saved to outfile
#
# *****************************************************************************
def convert_detections (infile, outfile, maxNumToConvert=-1):

    print('Converting detection data format...')

    bolideList = bd.unpickle_bolide_detections(infile)

    if maxNumToConvert > 0 and maxNumToConvert < len(bolideList) :
        bolideList = bolideList[:maxNumToConvert]

    bolideDetectionList = bd.convert_detection_representation(bolideList)
    bd.pickle_bolide_detections(outfile, bolideDetectionList)

    print('Conversion complete.')

# *****************************************************************************
# Convert Bolide objects in an output file from detect_bolides() to
# bolideDetection objects, as defined in bolide_detections.py.
# This tool requires the data to be saved as a pickle file.
# *****************************************************************************
if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])

    infile = args.infile[0]
    outfile = args.outfile[0]
    maxNumToConvert = args.maxNumToConvert

    convert_detections (infile, outfile, maxNumToConvert)

# ************************************ EOF ************************************
