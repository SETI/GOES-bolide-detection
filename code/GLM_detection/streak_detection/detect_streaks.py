import glob
from traceback import print_exc
import multiprocessing as mp
import time
import os
import numpy as np
import pickle

import io_utilities as ioUtil

import bolide_io as bIo
import bolide_clustering as bCluster

from streakFinder import find_streaks

#*************************************************************************************************************
def detect_streaks (input_config, data_files=[], output_path=None, provenance=None):
    """
    Top level function to search for streaks (Near-Field Glints).

   
    Parameters
    ----------
    input_config : gigantic_jet_io.input_config
        Configuration parameters for the elemental run.
        See gigantic_jet_io.input_config.configuration_template
    data_files   : str list 
        list of netCDF files to search
        This is nominally a day of GLM data.
        If empty then use the files in input_config.data_path
    output_path : str
        Path to save outputs
    provenance  : run_glm_pipeline_batch.Provenance object
        Contains the provenance for this run
   
    Returns
    -------
      None, other than saved data files.

    """

    startTime = time.time()

    # Get the netCDF files to process
    if len(data_files) == 0:
        data_files = glob.glob(input_config.data_path + "*.nc")
        if (data_files == []):
            if (input_config.verbosity):
                print('No files to process in directory {}'.format(input_config.data_path))
                return

    data_files = sorted(data_files)


    #***
    # Either do multi- or single-core processing, depending on the value of 
    # n_cores. Note that single-core processing is required for profiling.
    if input_config.n_cores < 1:
        raise Exception('n_cores should an integer >= 1')
    elif True or input_config.n_cores == 1:
        oops_counter = 0
        
        #TEST: truncate the file list for testing speed
      # print('TESTING: REMOVE ME!***************')
      # print('TESTING: REMOVE ME!***************')
      # print('TESTING: REMOVE ME!***************')
      # data_files = data_files[0:100]
        success = find_streaks(data_files, output_path, input_config.config_dict)
        
    else:
        # Do multi-core processing.
        #

        raise Exception('multi-core processing not implemented')
        oops_counter = 0
        giganticJetDetectionList = []

        # Chunk files
        # Chunk factor decreases the number of jobs by this factor. 
        chunkFactor = 10
        chunkSize = int(np.ceil(len(data_files) / (input_config.n_cores * chunkFactor)))
        filesChunked = [data_files[i:i + chunkSize] for i in range(0, len(data_files), chunkSize)]  
        
        with mp.Pool(processes=input_config.n_cores) as pool:
            results = [pool.apply_async(_detect_gigantic_jets_in_multiple_files, args=(file_paths, input_config)) for file_paths in filesChunked]
            outputs = [result.get() for result in results]
            for output in outputs:
                [detectedGiganticJets, oops_counter_elemental] = output
                oops_counter += oops_counter_elemental
                giganticJetDetectionList.extend(detectedGiganticJets)

    if (input_config.verbosity): print("We had {:d} oopses".format(oops_counter))

    endTime = time.time()
    totalTime = endTime - startTime
    if (input_config.verbosity):
        print("Total processing time: {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))

    return

