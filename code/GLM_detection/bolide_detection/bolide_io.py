import numpy as np
import datetime
import sys
import os
import argparse
import csv
import pickle
import copy
import io_utilities as ioUtil

# *****************************************************************************
# CLASS : input_config
#    
# A container for configuration parameters used by detect_bolides. 
# 
# *****************************************************************************
class input_config:
    
    detectionType = 'bolides'

    verbosity   = False

    # File paths
    data_path   = []
    other_satellite_input_path = None
    output_path = []
    latLon2Pix_table_path = None
    latLon2Pix_table_path_inverted = None
    pixel_boundary_plotting_enabled = None
    # Used by the glint feature
    spice_kernel_path = None

    generatePlots = False
    copyNetCDFFiles = True
    createSymlinks = True
    start_index = []
    end_index   = [] 
    n_cores     = [] 
    min_GB_post_processing = [] 
    min_GB_cutout_tool = [] 
    ground_truth_path = [] 
    rejectsToSaveFrac = [] 
    trained_classifier_path = None

    # SMALL_CLASSIFIER_THRESHOLD is used to force all detection to propagete in a small spatiotemporal box
    SMALL_CLASSIFIER_THRESHOLD = 0.0
    classifier_threshold = None

    max_num_detections = None
    min_num_groups = None
    min_num_groups_to_force_candidacy = None

    # Clustering
    cluster_sequential_enabled = False
    cluster_outlier_rejection_enabled = True
    cluster_closeness_seconds   = None 
    cluster_closeness_km        = None
    cluster_extractNeighboringFiles = True 
    cluster_min_num_groups_for_outliers = None
    cluster_outlierSigmaThreshold   = None 
    cluster_outlierMinClipValueDegrees  = None

    # Bolide detection hot spots
    # This is a list of lat and lon for each hot spot
    lon_peaks = None
    lat_peaks = None
    # This is the list of cylinder deemphasis readius and alpha terms
    deemph_radius = None
    deemph_alpha = None

    # Searching other satellite for detection
    otherSatelliteLatExpansionDegrees   = None
    otherSatelliteLonExpansionDegrees   = None
    otherSatelliteTimeExpansionSeconds  = None

    # Stereo re-navigation figure generation
    stereoDetectionOnlyMode = False
    minAltToGenerateFigureKm = None
    maxAltToGenerateFigureKm = None

    # ABI cutout figures
    cutout_enabled = False
    cutout_annotations = True
    cutout_G16ABICDIR = None
    cutout_G16ABIFDIR = None
    cutout_G17ABICDIR = None
    cutout_G17ABIFDIR = None
    cutout_G18ABICDIR = None
    cutout_G18ABIFDIR = None
    cutout_G19ABICDIR = None
    cutout_G19ABIFDIR = None
    cutout_coastlines_path = None
    cutout_n_cores = None # If None, use n_cores above
    cutout_plot_seperate_figures = False
    cutout_generate_multi_band_data = False
    cutout_plot_glm_circle_size = None # This is the size of the marker for the GLM data in the cutouts
    cutout_bands_to_read = None # The ABI bands to read in (1-based!)


    # Post-processing and light curve generation
    post_process_enabled = None
    post_process_config_file = None

    # Packager
    packagerEnabled = None
    packagerOutputDir = None

    # The Spatiotemporal box is for search for bolides only within the box
    spatiotemporal_box = None

    #***
    # These are dependent configuration parameters determined at beginning of run
    # This gives the type of input data file configuration
    valid_data_sources = ('files', 'daily_bundle', 'geonex')
    __data_source = None

    #***
    # These are the legacy filter functions params
    glint_G16_path = []
    glint_G17_path = []
    glint_G18_path = []
    glint_G19_path = []
    glintRegionG16LatRadiusDegrees = []
    glintRegionG16LonRadiusDegrees = []
    glintRegionG17LatRadiusDegrees = []
    glintRegionG17LonRadiusDegrees = []
    glintRegionG18LatRadiusDegrees = []
    glintRegionG18LonRadiusDegrees = []
    glintRegionG19LatRadiusDegrees = []
    glintRegionG19LonRadiusDegrees = []
    bolide_probability_cutoff = []
    group_05    = []
    energy_05   = []
    spline_05   = []
    time_05     = []
    linelet_05  = []
    dist_km_05  = []

    def __init__(self, config):
        """ Import config dict into input_config object

        Check that configuration parameters are consistent and correct

        """

        if (config['ground_truth_path'] != '' and config['ground_truth_path'] != []):
            if (config['trained_classifier_path'] != '' and config['trained_classifier_path'] != []):
                raise Exception('trained_classifier_path and ground_truth_path cannot both be set.')

        # Check that the hot spot parameters are consistent
        # all array must be the same length
        n_hot_spots = len(config['lon_peaks'])
        assert len(config['lat_peaks']) == n_hot_spots and len(config['deemph_radius']) == n_hot_spots and \
            len(config['deemph_alpha']) == n_hot_spots, 'All hot spot deemphasis parameters must be the same length.'

        # Transfer configuration parameters
        for key in config.keys():
            self.__setattr__(key, config[key])

        if self.n_cores == -1:
            # Use os.sched_getaffinity because it gives you the number of CPU cores available to this process, 
            # not the number of physical cores.
            self.n_cores = len(os.sched_getaffinity(0))

        self.bolidesFromWebsite = None
        self.trainedClassifierDict = None

        # If copyNetCDFFiles then set createSymlinks to True
        # This is for tmeporarily copying over the other satellite raw data files
        if not self.copyNetCDFFiles:
            self.createSymlinks = True

    @classmethod
    def _from_config_dict(cls, config, verbosity=False):
        """
        classmethod to take the config dictionary created by ioUtil.read_config_file(elementalConfigFile) and converts the parameters
        to a input_config class for use by detect_bolides.
       
        TODO: simplify this process so it is not a two-step process. 
       
        Parameters
        ----------
          config  : dict 
          Output of ioUtil.read_config_file(detectBolidesConfigFile)
       
        Returns
        input_config -- input_config object 
            used to store the configuration information for detect_bolides

        """
        verbosity = config['verbosity']
        
        #***
        # Make sure path strings are terminated with a '/' character.
        data_path = config['data_path']
        if not data_path[len(data_path) - 1] == '/':
            config['data_path'] = data_path + '/'
        
        other_satellite_input_path = config['other_satellite_input_path']
        if other_satellite_input_path is not None and not other_satellite_input_path[len(other_satellite_input_path) - 1] == '/':
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

    def copy_to_dict(self):
        """ Converts this object to a dict. This is not entirely trivial because of the data_source property.

        """

        elementalConfigDict = self.__dict__

        # Take care of the data_source property
        elementalConfigDict['data_source'] = self.data_source

        return copy.deepcopy(elementalConfigDict)

    #***
    @property
    def data_source(self):
        return self.__data_source

    @data_source.setter
    def data_source(self, data_source):
        if data_source is None:
            self.__data_source = None
        elif (self.valid_data_sources.count(data_source) == 1):
            self.__data_source = data_source
        else:
            raise Exception('Unknown data source')

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
                            'data_path'                         : ['str',      '.'], 
                            'other_satellite_input_path'        : ['str',       ''], 
                            'use_ramdisk'                       : ['bool',   False], 
                            'output_path'                       : ['str',      '.'], 
                            'latLon2Pix_table_path'             : ['str',       ''], 
                            'latLon2Pix_table_path_inverted'    : ['str',       ''], 
                            'pixel_boundary_plotting_enabled'   : ['bool',   False], 
                            'generatePlots'                     : ['bool',   False],
                            'copyNetCDFFiles'                   : ['bool',   False],
                            'createSymlinks'                    : ['bool',    True],
                            'start_index'                       : ['int',        0],
                            'end_index'                         : ['int',       -1], 
                            'n_cores'                           : ['int',        1], 
                            'min_GB_post_processing'            : ['float',   15.0], 
                            'min_GB_cutout_tool'                : ['float',   15.0], 
                            'ground_truth_path'                 : ['str',       ''], 
                            'spice_kernel_path'                 : ['str',       ''], 
                            'rejectsToSaveFrac'                 : ['float','0.001'], 
                            'trained_classifier_path'           : ['str',       ''], 
                            'classifier_threshold'              : ['float',    0.5], 
                            'max_num_detections'                : ['int',       -1], 
                            'min_num_groups'                    : ['int',       25],
                            'min_num_groups_to_force_candidacy' : ['int',       -1],
                            'cluster_3D_enabled'                : ['bool',    True],
                            'cluster_numba_threads'             : ['int',       16],
                            'cluster_sequential_enabled'        : ['bool',   False],
                            'cluster_outlier_rejection_enabled' : ['bool',    True],
                            'cluster_closeness_seconds'         : ['float',    0.2],
                            'cluster_closeness_km'              : ['float',   14.0],
                            'cluster_extractNeighboringFiles'   : ['bool',    True], 
                            'cluster_min_num_groups_for_outliers': ['int',      25],
                            'cluster_outlierSigmaThreshold'     : ['float',   10.0], 
                            'cluster_outlierMinClipValueDegrees': ['float',   0.15],
                            'lon_peaks'                         : ['list',      []],
                            'lat_peaks'                         : ['list',      []],
                            'deemph_radius'                     : ['list',      []],
                            'deemph_alpha'                      : ['list',      []],
                            'otherSatelliteLatExpansionDegrees' : ['float',    0.2],
                            'otherSatelliteLonExpansionDegrees' : ['float',    3.0],
                            'otherSatelliteTimeExpansionSeconds': ['float',    0.1],
                            'stereoDetectionOnlyMode'           : ['bool',   False],
                            'minAltToGenerateFigureKm'          : ['float',   -1.0],
                            'maxAltToGenerateFigureKm'          : ['float',   -1.0],
                            'glint_G16_path'                    : ['str',       ''], 
                            'glint_G17_path'                    : ['str',       ''], 
                            'glint_G18_path'                    : ['str',       ''], 
                            'glint_G19_path'                    : ['str',       ''], 
                            'glintRegionG16LatRadiusDegrees'    : ['float',    5.0], 
                            'glintRegionG16LonRadiusDegrees'    : ['float',    5.0], 
                            'glintRegionG17LatRadiusDegrees'    : ['float',    5.0], 
                            'glintRegionG17LonRadiusDegrees'    : ['float',    5.0], 
                            'glintRegionG18LatRadiusDegrees'    : ['float',    5.0], 
                            'glintRegionG18LonRadiusDegrees'    : ['float',    5.0], 
                            'glintRegionG19LatRadiusDegrees'    : ['float',    5.0], 
                            'glintRegionG19LonRadiusDegrees'    : ['float',    5.0], 
                            'bolide_probability_cutoff'         : ['float',    0.5], 
                            'cutout_enabled'                    : ['bool',    False],
                            'cutout_annotations'                : ['bool',    True],
                            'cutout_G16ABICDIR'                 : ['str',       ''],
                            'cutout_G16ABIFDIR'                 : ['str',       ''],
                            'cutout_G17ABICDIR'                 : ['str',       ''],
                            'cutout_G17ABIFDIR'                 : ['str',       ''],
                            'cutout_G18ABICDIR'                 : ['str',       ''],
                            'cutout_G18ABIFDIR'                 : ['str',       ''],
                            'cutout_G19ABICDIR'                 : ['str',       ''],
                            'cutout_G19ABIFDIR'                 : ['str',       ''],
                            'cutout_coastlines_path'            : ['str',       ''],
                            'cutout_n_cores'                    : ['int',     None],
                            'cutout_plot_seperate_figures'      : ['bool',   False],
                            'cutout_plot_glm_circle_size'       : ['float',  100.0],
                            'cutout_generate_multi_band_data'   : ['bool',   False],
                            'cutout_bands_to_read'              : ['tuple',  (1,2,3,7,11,13)],
                            'post_process_enabled'              : ['bool',   False],
                            'post_process_config_file'          : ['str',       ''],
                            'group_05'                          : ['float',   25.0], 
                            'energy_05'                         : ['float',    0.3], 
                            'spline_05'                         : ['float',   -2.0], 
                            'time_05'                           : ['float',    6.0], 
                            'linelet_05'                        : ['float',   -5.0], 
                            'dist_km_05'                        : ['float',    1.0]  }
        
        return configTemplate

        
    def __repr__(self):
        return ioUtil.print_dictionary(self)



# *****************************************************************************
# Parses the argument list for detect_bolides. 
#
# Also reads in the configuration file
#    
# INPUTS
#     arg_list   : A list of strings, each containing a command line argument. 
#                  NOTE that the first element of this list should NOT be the 
#                  program file name. Instead of passing sys.argv, pass 
#                  arg_list = sys.argv[1:]
#    
# OUTPUTS
#     config : A dictionary object of <attribute, value> pairs specifying the 
#              configuration parameters.
# *****************************************************************************
def parse_arguments(arg_list):
 
    parser = argparse.ArgumentParser(description=
        'Detect bolide signatures in GLM L2 data.')
    parser.add_argument('--config_file', '-f', dest='config_file',     
        type=str, default='',
        help='the path to a file containing systems configuration parameters')
    parser.add_argument('--indir', '-i', dest='indir', type=str, default='',
        help = 'The directory containing input files (default: the '
        + 'directory defined in the config file, if specified)')
    parser.add_argument('--outdir', '-o', dest='outdir', type=str,  default='',
        help = 'The directory to which output files are written (default: the '
        + 'directory defined in the config file, if specified)')
    parser.add_argument('--verbosity', '-v', dest='verbosity', type=bool,  default=False,
        help = 'If True then be verbose while detecting')


    args = parser.parse_args(arg_list)
    
    if len(args.config_file.strip()) > 0:
        if not os.path.isfile(args.config_file):
            sys.exit('The file {} does not exist.'.format(args.config_file))      
        else:
            config = ioUtil.read_config_file(args.config_file, input_config.configuration_template())
    else:
        config = ioUtil.get_default_configuration(input_config.configuration_template())
        
    # If input and/or output directories were specified on the command line, 
    # then they should override any values in the config file.
    if len(args.indir.strip()) > 0:
        config['data_path'] = args.indir
    if len(args.outdir.strip()) > 0:
        config['output_path'] = args.outdir

    config['verbosity'] = args.verbosity
        
    # Use absolute paths for file locations
    config['data_path']   = os.path.abspath(config['data_path'])
    config['other_satellite_input_path'] = os.path.abspath(config['other_satellite_input_path'])
    config['output_path'] = os.path.abspath(config['output_path'])
    
    return config
