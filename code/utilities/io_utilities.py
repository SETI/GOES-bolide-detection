# This module contains basic iput/output utilities

import os
import copy
from shutil import copyfile

# *****************************************************************************
# Print dictionary name-value pairs to the standard output.
#
# This function can be used to quickly view the attributes of an object.
#
# It is very useful to use in the __repr__ or __str__ to view the attributes of an object.
#    
# INPUTS
#   dictionary  : [dict or object] A dictionary object of <attribute, value> pairs specifying the 
#                configuration parameters. If an object is passed then the object is first converted to a dict
#   verbosity   : [bool] If True then print the dictionary. Otherwise, just return as a string
#
# OUTPUTS
#   Prints a summary to the standard output, if request
#   dict_string : [str] contains the formated dictionary attributes    
# *****************************************************************************
def print_dictionary(dictionary, verbosity=False):

    if (not isinstance(dictionary, dict) and isinstance(dictionary, object)):
        dictionary = dictionary.__dict__

    if not isinstance(dictionary, dict):
        raise Exception ('Passed dictionary does not appear to be of type dict')

    dict_string = '\n'
    for key in dictionary.keys():
        dict_string += '\t{} = {}\n'.format(key, dictionary[key])

    if verbosity:
        print(dict_string)
 
    return dict_string

#******************************************************************************
def read_config_file(fileName, configTemplate):
    """
    Read configuration parameters from a text file. Any parameters not found in 
    the file will be left in their default states. Any parameters in the config 
    file but not in the template will be ignored.
   
    If the value is set to None then so will be the configuration parameter.
       
    Parameters
    ----------
    fileName : A string specifying the complete path to a valid configuration 
               file (ASCII text).    
    configTemplate : Dictionary specifying a configuration template (see
                      bolide_io.input_config.configuration_template() for an example).

    Returns
    -------
    config   : A dictionary object of <attribute, value> pairs specifying the 
               configuration parameters.
    """

    if not configTemplate:
        raise Exception('configTemplate must be passed to get_default_configuration')
    
    config = get_default_configuration(configTemplate)

    # If no fileName is passed then just reuturn the default configuration
    if fileName is None:
        return config

    assert os.path.isfile(fileName), 'fileName {} is not an existant file'.format(fileName)

    for line in open(fileName):
        
        # Strip out comments and lines containing only whitespace.
        s = line.split(sep='#', maxsplit=2)[0]
        s = ''.join(s.split()) # Remove all whilespace.
        if len(s.strip()) <= 0: # if s is empty, get the next line.
            continue
    
        # Parse parameters
        s = s.split(sep='=', maxsplit=2)
        s0 = s[0].strip()            
        for key in configTemplate.keys():
            if key == s0:
                s1 = s[1].strip()
                if len(s1) > 0:
                    if s1 == 'None':
                        value = None
                    elif len(configTemplate[key]) > 0:
                        if (configTemplate[key][0] == 'str'):
                            # Account for empty string or empty set when string is not passed
                            if (s1 == "''" or s1 == '""'):
                                s1 = ''
                            elif (s1 == '[]'):
                                s1 = ''
                        value = cast_to_type(configTemplate[key][0], s1)
                    else:
                        value = s1
                        
                    config[key] = value

    return config

# *****************************************************************************
# Retrieve the default configuration defined in configTemplate.
#    
# INPUTS
#     configTemplate : Dictionary specifying a configuration template (see
#                       bolide_io.get_configuration_template() for an example).
# OUTPUTS
#     configDict   : A dictionary object of <attribute, value> pairs specifying the 
#                       default configuration parameters.
# *****************************************************************************
def get_default_configuration(configTemplate=None):

    # If a configTemplate was passed we need to explicitly copy it. This is
    # because python passes mutable objects, such as dictionaries, by reference.
    if not configTemplate:
        raise Exception('configTemplate must be passed to get_default_configuration')
    configDict = copy.deepcopy(configTemplate)

    for key in configDict.keys():
        configDict[key] = cast_to_type(configDict[key][0], configDict[key][1])
        
    return configDict


#******************************************************************************
#******************************************************************************
def cast_to_type(typeStr, value) :
    """Converts a value to a specific type <typeStr>

    If <value> is None then the castValue is None
 
    Parameter
    ---------
    typeStr : str
        The type of cast the value to
    value   : the value to cast from
    
    Returns
    -------
    castValue : type typeStr
        The newly cast value
    """

    if value is None:
        return None

    s = '\'' if typeStr == 'str' else ''
    castValue = eval('{}({}{}{})'.format(typeStr, s, value, s))

    return castValue

# *****************************************************************************
# Return the number of lines in a text file.
#    
# INPUTS
#     filepath  :
#    
# OUTPUTS
#     nLines    : 
# *****************************************************************************
def get_num_lines(filepath):
    
    cmd = 'wc -l ' + filepath
    outputStr = os.popen(cmd).read()
    nLines = int(outputStr.split()[0])
    
    return nLines

#*************************************************************************************************************
# function copy_or_create_symlinks()
#
#
#*************************************************************************************************************
def copy_or_create_symlinks(targetFileList, destDir, createSymlinks=False, verbosity=False, return_new_paths=False):
    """
    Either copies a file or creates a symlink to a list of files.
   
    If the file does not exist, then do nothing.
   
    Parameters
    ----------
    targetFileList  -- [list of str] List of full path of files to copy or symlink
    destDir         -- [str] The destination path to copy or create symlink
    createSymlinks  -- [bool] If True, then create a symlink instead of copying the files, Default=False
    verbosity       -- [bool] If True then be verbose, default = False
    return_new_paths  -- [bool] If True then return the new paths to the files in a list

    Returns
    -------
    targetFileList : [list of str]
        The new target file list paths

    """

    if targetFileList is None:
        return

    if isinstance(targetFileList, str):
        targetFileList = [targetFileList]

    # Check to see whether the output directory exists and create it if necessary.
    create_path(destDir, verbosity)

    outFileList = []
    for targetFileStr in targetFileList:
        if os.path.exists(targetFileStr):
            dirName, fileName = os.path.split(targetFileStr)
            outFileStr = os.path.join(destDir, fileName)
            outFileList.append(outFileStr)
            if (not os.path.isfile(outFileStr) and not os.path.islink(outFileStr)):
                if createSymlinks: 
                    # Use the absolute path so that the symlink will always work correctly
                    os.symlink(os.path.abspath(targetFileStr), outFileStr)
                else:
                    copyfile(targetFileStr, outFileStr)

    if return_new_paths:
        return outFileList
    else:
        return

#******************************************************************************
# Append each element of strList as a single line in the file pointed to by
# filePath.
#
# Inputs:
#   filePath    -- [str] path and filename to file to append to
#   strList     -- [string list]  Appends each str in the list to the file as a new line
#
# Outputs:
#   status  -- [bool] return False if there was an error
#
#******************************************************************************
def append_lines_to_file(filePath, strList):
    try:
        with open(filePath, 'a') as f:
            for item in strList:
                f.write(item + '\n')
        status = True
    except:
        status = False

    return status

#******************************************************************************
# Creates a direcotry path if it does not already exist
#
# Inputs:
#   path -- [str] directory path to create
#   verbosity   -- [bool] If True then be verbose
#
# Outputs:
#
#******************************************************************************
def create_path(path, verbosity=False):

    if not os.path.isdir(path):
        if (verbosity): print('The directory {} does not exist. Creating it now ...'.format(path))
        try:
            os.makedirs(path)
        except OSError:
            raise Exception('Creation of the directory {} failed'.format(path))
        else:
            if (verbosity): print('Successfully created the directory {} '.format(path))

    return
