#!/bin/sh
#PBS -W group_list=s1488
#PBS -l select=1:ncpus=128:model=rom_ait
#PBS -l walltime=1:00:00
#PBS -q atap
#PBS -N GLMPackCron
#PBS -o /pbs_out_files
#PBS -e /pbs_out_files


############################################################################# 
# This script can be submitted to a single Pleiades node on a
# daily schedule by adding a crontab entry like the following:
#
# $ crontab -e
# 0 1 * * * /PBS/bin/qsub packaging_cron_script.sh
#
# This will submit a serial job to the normal queue every night at 1:00 AM 
# with a 1 hour wall time limit.
# We pick 01:00 AM because this is an hour after the cutout tool runs at 12:00 AM
#
# NOTES:
# - The environment variable  GLM_PACKAGER_DIR must point to the path to store the packager runs.
# - Relies on the shell script ~/.atap_env_startup.sh to set up the GLM pipeline envrionment.
############################################################################# 

DATE=`date +%y-%m-%d`

GLM_PACKAGER_DIR=packager
cd $GLM_PACKAGER_DIR

LOG_FILE_DIR=$GLM_PACKAGER_DIR/logs

CFG_FILE_G16=$GLM_PACKAGER_DIR/packaging_config_file_G16.txt
CFG_FILE_G17=$GLM_PACKAGER_DIR/packaging_config_file_G17.txt

#***
# Set up the ATAP environment
source ${HOME}/.atap_env_startup_release.sh

#***
# Run the packager
LOG_FILE_G16=${LOG_FILE_DIR}/packager-G16-${DATE}.log
LOG_FILE_G17=${LOG_FILE_DIR}/packager-G17-${DATE}.log

package_bolide_detections.py $CFG_FILE_G16 2>&1 | tee $LOG_FILE_G16
package_bolide_detections.py $CFG_FILE_G17 2>&1 | tee $LOG_FILE_G17


