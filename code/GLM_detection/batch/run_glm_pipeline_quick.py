#!/usr/bin/env python

# This tool is used to rapdily run the detection pipeline on a small specific set of data. 

import sys
import argparse
from datetime import datetime, time
import pickle
import numpy as np

import io_utilities as ioUtil
import run_glm_pipeline_batch as glmBatch

class SpatiotemporalBox:

    streak_file = None
    explicit_cluster = None
    streak_width = None

    satellite = None
    minDate = None
    maxDate = None
    minLat = None
    maxLat = None
    minLon = None
    maxLon = None
    r2 = None

    def __init__ (self):

        return

    def __repr__(self): 

        return ioUtil.print_dictionary(self)



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
        specifying <attribute, [type, value]>
        specifying parameter name, type, and a default value for all valid parameters.

    Initialize the configuration dictionary with parameter names and their
    associated types. Values read from the file will be cast to the specified
    type.
                      parameter                      type    default
    """
    configTemplate =   {'inputFilePath':              ['str', '.'],
                        'otherSatelliteFilePath':     ['str', '.'],
                        'outputDir':                  ['str', './output'],
                        'elementalConfigFile':        ['str', ''],
                        'validationConfigFile':       ['str', ''],
                        'detectionType':              ['str', 'bolides'],
                        'minDate':                    ['str', None],
                        'maxDate':                    ['str', None],
                        'minLat':                     ['float', None],
                        'maxLat':                     ['float', None],
                        'minLon':                     ['float', None],
                        'maxLon':                     ['float', None],
                        'explicit_cluster':           ['bool', False],
                        'streak_width':               ['float', 20],
                        'streak_file':                ['str', None],
                        'deleteDailyDirs':            ['bool', False],
                        'deleteDailyDatabases':       ['bool', False],
                        'deleteDailyNetCdfFiles':     ['bool', True],
                        'deleteDailyExtraFigures':    ['bool', True]}

    return configTemplate

if __name__ == "__main__":
    """ Command line tool for quick running the detection pipeline.

    This function is run by passing the path to the configuration file.


    """

    config = ioUtil.read_config_file(sys.argv[1], configTemplate=get_configuration_template() )
    inputRootDir    = config['inputFilePath']
    detectionType = config['detectionType']

    # Create a top level configuration Dict and set default parameters for a quick run
    top_level_config = ioUtil.get_default_configuration(glmBatch.get_configuration_template())
    top_level_config['inputRootDir'] = config['inputFilePath']
    top_level_config['otherSatelliteRootDir'] = config['otherSatelliteFilePath']
    top_level_config['outputDir'] = config['outputDir']
    top_level_config['elementalConfigFile'] = config['elementalConfigFile']
    top_level_config['validationConfigFile'] = config['validationConfigFile']
    top_level_config['detectionType'] = config['detectionType']
    top_level_config['startDate'] = config['minDate']
    top_level_config['endDate'] = config['maxDate']

    # Force these parameters for a quick run
    top_level_config['multiProcessEnabled'] = False
    top_level_config['forceReprocess'] = True
    top_level_config['doNotProcessCurrentDay'] = False
    top_level_config['delayNDaysForL0Data'] = 0

    # If we are searching for a very specific spatiotemporal box, then set the detection threshold to a very small value
    # spatiotemporal_box is a list of all boxes
    spatiotemporal_box = []


    # If reading from a streak file then create a spatiotemporal_box for each streak,
    # Otherwise, we are creatign just a single spatiotemporal_box
    if config['streak_file'] is not None:

        # Read in streak_file and create a spatiotemporal_box object for each
        with open(config['streak_file'], 'rb') as fp:
            streak_data = pickle.load(fp)
        fp.close()

        assert len(np.unique(streak_data['satellite'])) == 1, 'streak_data only works if all are from the same satellite'

        for idx in range(len(streak_data['start_lat'])):
            spatiotemporal_box.append(SpatiotemporalBox())

            spatiotemporal_box[-1].streak_file = config['streak_file']
            spatiotemporal_box[-1].streak_width = config['streak_width']
            # If a streak_file is passed then set explicit cluster to True
            spatiotemporal_box[-1].explicit_cluster = True
            
            spatiotemporal_box[-1].satellite = streak_data['satellite'][idx]
            spatiotemporal_box[-1].minDate = streak_data['start_time'][idx].to_pydatetime()
            spatiotemporal_box[-1].maxDate = streak_data['end_time'][idx].to_pydatetime()
            spatiotemporal_box[-1].minLat = streak_data['start_lat'][idx]
            spatiotemporal_box[-1].maxLat = streak_data['end_lat'][idx]
            spatiotemporal_box[-1].minLon = streak_data['start_long'][idx]
            spatiotemporal_box[-1].maxLon = streak_data['end_long'][idx]
            spatiotemporal_box[-1].r2 = streak_data['r2'][idx]


    else:
        spatiotemporal_box.append(SpatiotemporalBox())
        spatiotemporal_box[-1].streak_file = config['streak_file']
        spatiotemporal_box[-1].streak_width = config['streak_width']
        spatiotemporal_box[-1].explicit_cluster = config['explicit_cluster']
        spatiotemporal_box[-1].minDate = datetime.fromisoformat(config['minDate'])
        spatiotemporal_box[-1].maxDate = datetime.fromisoformat(config['maxDate'])
        spatiotemporal_box[-1].minLat = config['minLat']
        spatiotemporal_box[-1].maxLat = config['maxLat']
        spatiotemporal_box[-1].minLon = config['minLon']
        spatiotemporal_box[-1].maxLon = config['maxLon']


    #***
    # Check that the spatiotemporal box is a box
    # MinDate and MaxDate can be the same day, if so, the box is the entire day in length
    # Otherwise, make sure max Date is greater than min Date
    for box in spatiotemporal_box:
        if not isinstance(box.minDate, list):
            dayTimestamp = datetime.combine(box.minDate.date(), time()).timestamp()
            if not ((box.minDate.timestamp() - dayTimestamp) == 0.0 and \
                    box.minDate == box.maxDate):
                assert box.minDate < box.maxDate, 'Spatiotemporal box minDate must be <= maxDate'
            # Streaks can have an end lat/lon smaller than the start lat/lon
  #         if box.minLat is not None and  box.maxLat is not None:
  #             assert box.minLat <= box.maxLat, 'Spatiotemporal box minLat must be <= maxLat'
  #         if box.minLon is not None and  box.maxLon is not None:
  #             assert box.minLon <= box.maxLon, 'Spatiotemporal box minLon must be <= maxLon'

    if np.any([box.explicit_cluster for box in spatiotemporal_box]):
        assert np.all([box.explicit_cluster for box in spatiotemporal_box]), 'All spatiotemporal boxes must be explicit or none are'

    top_level_config['spatiotemporal_box'] = spatiotemporal_box 

    glmBatch.run_glm_pipeline_batch(top_level_config)

