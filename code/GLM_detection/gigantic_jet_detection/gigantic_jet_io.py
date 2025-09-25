# Gigantic Jet detection specific configuration paramaters
import os
import io_utilities as ioUtil

class input_config:

    detectionType = 'gigantic_jets'

    verbosity   = False
    data_path   = None
    other_satellite_input_path = None
    output_path = None

    min_num_groups = None

    n_cores = None

    def __init__(self, config):
        """ Import config dict into input_config object

        Check that configuration parameters are consistent and correct

        """

        if self.n_cores == -1:
            # Use os.sched_getaffinity becaise it gives you the number of CPU cores available to this process, 
            # not the number of physical cores.
            self.n_cores = len(os.sched_getaffinity(0))

        # Transfer configuration parameters
        for key in config.keys():
            self.__setattr__(key, config[key])

    @classmethod
    def _from_config_dict(cls, config, verbosity=False):
        """
        classmethod to take the config dictionary created by ioUtil.read_config_file(elementalConfigFile) and converts the parameters
        to a input_config class for use by detect_gigantic_jets.
       
        Parameters
        ----------
          config  : dict 
          Output of ioUtil.read_config_file(elementalConfigFile)
       
        Returns
        input_config -- input_config object 
            used to store the configuration information for detect_gigantic_jets

        """
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
        ioUtil.create_path(output_path, verbosity)
        
        # Create the input_config object
        input_config = cls(config)
        
        return input_config


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
                            'data_path'                         : ['str',      '.'], 
                            'other_satellite_input_path'        : ['str',       ''], 
                            'output_path'                       : ['str',      '.'], 
                            'min_num_groups'                    : ['int',       20], 
                            'n_cores'                           : ['int',        1], 
                         }

        return configTemplate

        
    def __repr__(self):
        return ioUtil.print_dictionary(self)

