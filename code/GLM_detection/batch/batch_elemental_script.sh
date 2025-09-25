#!/bin/sh

############################################################################# 
# This script is to be called on each node to process each 
# day's GLM data in parallel. (I.e. elemental processing)
#
# Pass the file path to the directory for run_glm_pipeline_elemental is to 
# be run from. This string is passed as the input argument to 
# run_glm_pipeline_elemental.
# A log file also records all stdout from the call.
#
# Typically this script will be submitted to a single Pleiades node.
#
# NOTES:
# - Assumes the ATAP git repository is checked out under $HOME/git/atap
# - Relies on atap_bash_startup.sh to make run_glm_pipeline_elemental.py 
#   visible and executable.
############################################################################# 

DATE=`date +%d-%m-%y`

PROCESSING_DIR=$1

#***
# Force matplotlib to use the Agg backend for non-interactive plot generation. 
# We do this by way of an environment variable because the plotting tool is 
# pricipally used for interactive plotting.
export MPLBACKEND=agg

#***
# Set up the ATAP environment
source ${HOME}/.atap_env_startup.sh

#***
# Run the elemental job
LOG_FILE=${PROCESSING_DIR}/run_glm_pipeline_elemental-${DATE}.log
# The '2>&1' sends the output to both stdout and stderr
run_glm_pipeline_elemental.py ${PROCESSING_DIR} 2>&1 | tee $LOG_FILE

#***
# Make sure all output files are world readable.
#find $PROCESSING_DIR -type f -exec chmod 644 {} \;


