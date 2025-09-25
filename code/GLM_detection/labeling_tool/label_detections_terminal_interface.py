# *****************************************************************************
# label_detections_terminal_based
#
# This is a terminal based tool to label GLM bolide detections.
#

import matplotlib.pyplot as plt
import sys
import argparse
import bolide_detections as bd
import bolide_dispositions as bDisposition
import plot_bolide_detections as pbd
import pickle

# *****************************************************************************
# Parse an argument list.
#
# INPUTS
#     arg_list   : A list of strings, each containing a command line argument.
#                  NOTE that the first element of this list should NOT be the
#                  program file name. Instead of passing sys.argv, pass
#                  arg_list = sys.argv[1:]
#
# OUTPUTS
#     args :
# *****************************************************************************
def parse_arguments(arg_list):
    parser = argparse.ArgumentParser(description='Label bolide detections.')
    parser.add_argument('infile', metavar='infile', type=str, nargs=1,
                        help='Input file name')
    parser.add_argument('outfile', metavar='outfile', type=str, nargs=1,
                        help='Output file name')
    parser.add_argument('--name', '-n', dest='name', type=str, default='',
                        help='The name of the human user (default: None)')
    parser.add_argument('--expertise', '-e', dest='expertise', type=float, default='',
                        help='The user\'s level of expertise in the interval [0,1] (default: None)')

    args = parser.parse_args(arg_list)

    return args


# *****************************************************************************
#
# *****************************************************************************
if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])

    infile      = args.infile[0]
    outfile     = args.outfile[0]
    userName    = args.name
    expertise   = args.expertise

    # Read the detection file and create the list of detection summary objects.
    objList = bd.unpickle_bolide_detections(infile)
    if not isinstance(objList, list) :
        sys.exit('Unable to construct object list from file {}. '.format(infile))

    #newOpinion = bd.HumanOpinion(0, userName, expertise)

    if isinstance(objList[0], bd.bolideDetection) :
        summaryList = []
        for detection in objList:
            summaryList.append(bDisposition.BolideDispositionProfile(detection, humanOpinions=[bDisposition.HumanOpinion(0, userName, expertise)]))
    elif isinstance(objList[0], bDisposition.BolideDispositionProfile) :
        summaryList = objList
        for summary in summaryList:
            summary.humanOpinions.append(bDisposition.HumanOpinion(0, userName, expertise))
    else:
        sys.exit('Unknown object type in file {}. '.format(infile))

    numDetections = len(summaryList)
    i = 0
    while i < numDetections :

        detection = summaryList[i].bolideDetection
        fig = pbd.plot_detections( detection )
        fig.show()

        validInputReceived = False
        while not validInputReceived :
            messageStr = '--- Detection {} of {}: Enter confidence this is a bolide siganture ---'.format(i+1, numDetections)
            print(messageStr)
            s = input('\tEnter confidence on the interval [0,1] (default={}) : '.format(
                summaryList[i].humanOpinions[-1].confidence))

            if not s :
                validInputReceived = True
                i += 1  # Increment the index
                break
            elif s == 'f':
                validInputReceived = True
                i += 1
                break
            elif s == 'b' :
                validInputReceived = True
                i -= 1
                break
            elif s == 'w':
                validInputReceived = True
                try:
                    with open(outfile, 'ab') as fp:
                        pickle.dump(summaryList, fp)
                        print('Saved current opinions to file {}'.format(outfile))
                except:
                    print('Error writing file {}'.format(outfile))
            elif s == 'q':
                sys.exit('Quitting ...')
            else :
                try :
                    confidence = float(s)
                    if confidence >= 0 and confidence <= 1.0 :
                        validInputReceived = True
                        summaryList[i].humanOpinions[-1].confidence = confidence
                        i += 1 # Increment the index
                except :
                    continue

        plt.close(fig)

    with open(outfile, 'ab') as fp :
        pickle.dump(summaryList, fp)

# ************************************ EOF ************************************
