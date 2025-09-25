#!/usr/bin/env python

import sys
import os
import pickle
import time

from run_glm_pipeline_batch import final_aggregation_work_parallel, Provenance

#*******************************************************************************
# 
# This program will perform the final aggregation steps after the parallel 
# jobs are run, which process all the daily data.
#
# All that is in here is a wrapper to call the function in run_glm_pipeline_batch 
# This is needed to run the function from a batch job.
#
# Pass to this program the path to the file containing the information to process the aggregation step
#
# Inputs:
#   aggregate_data_file_name    -- full path to the data file 
#
#*******************************************************************************

if __name__ == "__main__":

    startTime = time.time()

    # Make sure we're running Python 3
    if sys.version_info.major < 3:
        raise Exception("Python 3.0 or higher is required")

    # Unpickle the data we need
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        raise Exception('The file {} does not exist.'.format(filename))
    else:
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)

    # Perform aggregation steps
    final_aggregation_work_parallel(data['filesByDayDict'], data['dirListFilePath'], data['provenance'])

    endTime = time.time()
    totalTime = endTime - startTime
    print("Total aggregation processing time: {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))
