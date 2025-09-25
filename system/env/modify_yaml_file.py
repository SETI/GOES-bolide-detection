###############################################################################
# modify_yaml_file.py
###############################################################################
# Modify an environment.yml file as follows:
#
# (1) Remove OS-specific tags that follow version numbers of conda packages
#     (pip packages are not affected).
# (2) Remove version numbers from packages specified in the optional file
#     specified with "-v". Any blank lines are ignored.
# (3) Remove packages specified in the optional file specified with "-p". Any
#     blank lines are ignored.
# (4) Remove lines containing an optional substring (e.g., "torch") specified
#     with the -e option
#
###############################################################################

import os
import sys
import argparse

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
    parser = argparse.ArgumentParser(description='Insert ancillary data files into MongoDB with gridFS.')
    parser.add_argument('yamlFile', metavar='yamlFile', type=str, nargs=1,
                        help='Path to a YAML file specifying a linux python environment')
    parser.add_argument('--package', '-p', dest='removePackageFile', type=str, default=None,
                        help='Optional path to a file containing a list of package names that should be removed.')
    parser.add_argument('--version', '-v', dest='removeVersionFile', type=str, default=None,
                        help='Optional path to a file containing a list of package names that should have their version numbers removed.')
    parser.add_argument('--string', '-s', dest='excludeString', type=str, default=None,
                        help='Remove lines containing the specified string')

    args = parser.parse_args(arg_list)

    return args

# *****************************************************************************
# Note that we need to sort the indices in reversed order to ensure that
# the shift of indices induced by the deletion of elements at lower indices
# won’t invalidate the index specifications of elements at larger indices.
# *****************************************************************************
def remove_list_elements_by_index(lst, indices):
    for i in sorted(list(indices), reverse=True):
        del lst[i]
    return lst

# *****************************************************************************
# Run the main program.
# *****************************************************************************
if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])

    yamlFile  = args.yamlFile[0]
    removeVersionFile  = args.removeVersionFile
    removePackageFile  = args.removePackageFile
    excludeStr = args.excludeString

    removePackageList = []
    removeVersionList = []

    # ------------------------------------------------------------------------
    # Read contents of the input files into lists
    # ------------------------------------------------------------------------
    if removePackageFile:
        if not os.path.isfile(removePackageFile):
            raise(Exception('File not found: {}'.format(removePackageFile)))
        else:
            with open(removePackageFile) as file:
                removePackageList = []
                for line in file:
                    if not line.isspace(): # Skip blank lines
                        removePackageList.append(line.rstrip('\n'))

    if removeVersionFile:
        if not os.path.isfile(removeVersionFile):
            raise(Exception('File not found: {}'.format(removeVersionFile)))
        else:
            with open(removeVersionFile) as file:
                removeVersionList = []
                for line in file:
                    if not line.isspace(): # Skip blank lines
                        removeVersionList.append(line.rstrip('\n'))

    if not os.path.isfile(yamlFile):
        raise(Exception('File not found: {}'.format(yamlFile)))
    else:
        with open(yamlFile) as file:
            yamlLineList = []
            for line in file:
                yamlLineList.append(line.rstrip('\n'))

    # ------------------------------------------------------------------------
    # Remove packages specified in removePackageFile and lines containing
    # excludeStr.
    # ------------------------------------------------------------------------
    removeIndices = []
    for i in range(0, len(yamlLineList) - 1):
        line = yamlLineList[i]
        if excludeStr in line or any(x in line for x in removePackageList):
            removeIndices.append(i)
    yamlLineList = remove_list_elements_by_index(yamlLineList, removeIndices)

    # ------------------------------------------------------------------------
    # Strip OS-specific tags. This applies to conda packages only. Packages
    # listed in the pip section are ignored.
    # ------------------------------------------------------------------------
    tagsRemovedList = []
    for line in yamlLineList:
        # Skip lines containing '=='
        if ("==" not in line) and (line.count("=") > 1):
            groups = line.split('=')
            line = '='.join(groups[:2])
        tagsRemovedList.append(line)

    # ------------------------------------------------------------------------
    # Strip version numbers from packages named in removeVersionList. Al other
    # lines are left as they are. Results are printed to the standard output.
    # ------------------------------------------------------------------------
    for line in tagsRemovedList:
        if "=" in line and any([x in line for x in removeVersionList]):
            unversionedLine = line.split("=")[0]
            print(unversionedLine)
        else:
            print(line)

# ************************************ EOF ************************************
