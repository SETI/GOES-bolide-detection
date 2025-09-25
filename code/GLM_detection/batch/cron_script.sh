#!/bin/sh
#PBS -W group_list=s1488
#PBS -l select=1:ncpus=128:model=rom_ait
#PBS -l walltime=2:00:00
#PBS -q atap
#PBS -N GLMPipeCron
#PBS -o pbs_out_files
#PBS -e pbs_out_files


############################################################################# 
# This script can be submitted to a single NAS node on a
# daily schedule by adding a crontab entry like the following:
#
# $ crontab -e
# 0 23 * * * /PBS/bin/qsub cron_script.sh
#
# This will submit a serial job to the normal queue every night at 11:00 PM 
# with a 2 hour wall time limit.
# We pick 11:00PM because this is several hours after the cron job which downloaded the daily bundle (at 7PM).
#
# NOTES:
# - The environment variable  GLM_PIPELINE_DIR must point to the path to store the pipeline runs.
# - Relies on the shell script ~/.atap_env_startup_release.sh to set up the GLM pipeline envrionment.
############################################################################# 

DATE=`date +%y-%m-%d`

GLM_PIPELINE_DIR=pipeline_dir
cd $GLM_PIPELINE_DIR

LOG_FILE_DIR=$GLM_PIPELINE_DIR/logs

CFG_FILE_G16=$GLM_PIPELINE_DIR/cron_config/top_level_config_G16.txt
CFG_FILE_G17=$GLM_PIPELINE_DIR/cron_config/top_level_config_G17.txt

#***
# Force matplotlib to use the Agg backend for non-interactive plot generation. 
# We do this by way of an environment variable because the plotting tool is 
# pricipally used for interactive plotting.
export MPLBACKEND=agg

#***
# Set up the ATAP environment
source ${HOME}/.atap_env_startup_release.sh

#***
# Run the detector
LOG_FILE_G16=${LOG_FILE_DIR}/detect-bolides-G16-${DATE}.log
LOG_FILE_G17=${LOG_FILE_DIR}/detect-bolides-G17-${DATE}.log

run_glm_pipeline_batch.py $CFG_FILE_G16 2>&1 | tee $LOG_FILE_G16
run_glm_pipeline_batch.py $CFG_FILE_G17 2>&1 | tee $LOG_FILE_G17

#***
# Make sure all output files and directories are world readable.
# This is really slow when there a lot of files
#find $GLM_PIPELINE_DIR/output -type f -exec chmod 644 {} \;
#find $GLM_PIPELINE_DIR/output -type d -exec chmod 755 {} \;

