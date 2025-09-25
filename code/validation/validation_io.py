# This module contains tools for setting up the input and output configuration for the validation step

import copy

import io_utilities as ioUtil

class input_config:

    verbosity = False

    report_output_path = []
    reportEnabled = False
    reportCopyIndivFiles = False
    report_create_symlinks = None
    delete_image_pickle_files = None

    validation_model_path = None

    def __init__(self, config, detection_input_config):
        """ Import config dict into input_config object

        Check that configuration parameters are consistent and correct

        Parameters
        ----------
        config : dict
            A dict contatining the validation configuration parameters
        detection_input_config : bolide_io.input_config
            The detection step input_config object

        """
        verbosity = config['verbosity']


        # Transfer configuration parameters
        for key in config.keys():
            self.__setattr__(key, config[key])

        # If report_output_path is not set then set it to detection_input_config.output_path
        if self.report_output_path == '':
            self.report_output_path = detection_input_config.output_path
        # Make sure path strings are terminated with a '/' character.
        if not self.report_output_path[len(self.report_output_path) - 1] == '/':
            self.report_output_path = self.report_output_path + '/'

        # If we are creating the indiv-files folder then copyNetCDFFiles must be True
        if self.reportCopyIndivFiles:
            assert detection_input_config.copyNetCDFFiles, 'If report reportCopyIndivFiles is True then detection copyNetCDFFiles must be True'

        if self.reportEnabled:
            assert detection_input_config.generatePlots, 'If reportEnabled is True then detection generatePlots must be True'

        # If we are runnign the validator then the cutout tool should be generating multi-band data
        if self.validation_model_path is not None:
            assert detection_input_config.cutout_generate_multi_band_data, 'If performing validation then cutout_generate_multi_band_data must be True'

        # Check to see whether the output directory exists and create it if necessary.
        ioUtil.create_path(self.report_output_path, verbosity)

        # If min_num_groups_to_force_candidacy is None then set to the value in detection_input_config
        if self.min_num_groups_to_force_candidacy is None:
            self.min_num_groups_to_force_candidacy = detection_input_config.min_num_groups_to_force_candidacy

    def copy_to_dict(self):
        """ Converts this object to a dict.

        """

        validationConfigDict = self.__dict__

        return copy.deepcopy(validationConfigDict)

    #******************************************************************************
    @staticmethod
    def configuration_template():
        """
        Retrieve a dictionary defining configuration parameter names, data types, and
        default values.
           
        Parameters
        ----------
        (none)
           
        Returns
        -------
        configTemplate : dict
            specifying <attribute, [type, value]> parameter name, type, and a default value for 
            all valid parameters.
                           
        """
        
        # Initialize the configuration dictionary with parameter names and their 
        # associated types. Values read from the file will be cast to the specified 
        # type.
        #                   parameter                      type    default
        configTemplate = {  'verbosity'                         : ['bool',   False],
                            'reportEnabled'                     : ['bool',   False],
                            'reportCopyIndivFiles'              : ['bool',   False],
                            'report_output_path'                : ['str',    None],
                            'report_create_symlinks'            : ['bool',   False],
                            'validation_model_path'             : ['str',    None],
                            'validation_image_cache_path'       : ['str',    '/tmp/ramdisk/cnn_image_cache'],
                            'validation_low_threshold'          : ['float',  0.1],
                            'validation_high_threshold'         : ['float',  0.9],
                            'gpu_index'                         : ['int',    None], 
                            'delete_image_pickle_files'         : ['bool',   True],
                            'min_num_groups_to_force_candidacy' : ['int',    None],
                            }
        
        return configTemplate

        
    def __repr__(self):
        return ioUtil.print_dictionary(self)


