#!/bin/sh

############################################################################# 
# This script is to be called on a single node to aggregate the data that was 
# processed in run_glm_pipeline_elemental for each chunck of data (nominally, 1 day)
#
# Pass the file path to the file containing the aggregation data. 
# This string is passed as the input argument to 
# run_bolide_aggregation.
#
# Typically this script will be submitted to a single Pleiades node.
#
# NOTES:
# - Assumes the ATAP git repository is checked out under $HOME/git/atap
# - Relies on atap_bash_startup.sh to make run_bolide_aggregation.py 
#   visible and executable.
############################################################################# 

DATE=`date +%d-%m-%y`

AGGREGATION_DATAFILE=$1

#***
# Force matplotlib to use the Agg backend for non-interactive plot generation. 
# We do this by way of an environment variable because the plotting tool is 
# pricipally used for interactive plotting.
export MPLBACKEND=agg

#***
# Set up the ATAP environment
source ${HOME}/.atap_env_startup.sh

#***
# Run the aggregation job
# The '2>&1' sends the output to both stdout and stderr
run_bolide_aggregation.py ${AGGREGATION_DATAFILE} 2>&1

#***
# Make sure all output files are world readable.



