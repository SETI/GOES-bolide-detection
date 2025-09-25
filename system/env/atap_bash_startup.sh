#!/bin/bash
##################################################################
# atap_bash_startup
#
# This BASH script will set up an environment for ATAP development
#
##################################################################

#***
# Configure for work in the NAS environment.
# 'hostname -d' is not available on MAC OS so only do this query if not on a darwin $OSTYPE
export isOnNas=False
if ! [[ $OSTYPE =~ darwin* ]] ; then
    if hostname -d | grep -q 'nas.nasa.gov' ; then
        # If this is a noninteractive shell, then we need to do some 
        # additional setup.
        if [[ $- == *i* ]]; then
            source /usr/local/lib/global.profile
        fi
        export isOnNas=True
    fi
fi

#***
# The environment relies on symbolic links to the executables, so core.symlinks must be set to true.
isSymLinks=`git config --get core.symlinks`
if [ "${isSymLinks}" != "true" ]; then
    echo "For ATAP development git core.symlinks must be set to true, setting now..."
    git config --global core.symlinks true
fi

#***
# All executables should be symbolic links in the following directory
# Make sure you use relative paths when creating the links so this works for all users
export PATH=${ATAP_REPO_ROOT}/system/bin:${PATH}

#***
# All supporting library files (such as compiled Fortran code) goes in a lib directory, 
# which must be added to the PYTHONPATH
export PYTHONPATH=${ATAP_REPO_ROOT}/system/lib:${PYTHONPATH}

#***
# Add GLM detect bolide code to PYTHONPATH and make executable. 

#***
# Add batch automation code to PYTHONPATH and make the top-level script run_glm_pipeline_*.py executable.
export PYTHONPATH=${ATAP_REPO_ROOT}/code/GLM_detection/batch:${PYTHONPATH}
# Make run_glm_pipeline_* executable if it isn't already
permStat=`ls -l ${ATAP_REPO_ROOT}/code/GLM_detection/batch/run_glm_pipeline_batch.py`
if [ "${permStat:3:1}" != "x" ]; then
    chmod +x ${ATAP_REPO_ROOT}/code/GLM_detection/batch/run_glm_pipeline_batch.py
fi
permStat=`ls -l ${ATAP_REPO_ROOT}/code/GLM_detection/batch/run_glm_pipeline_quick.py`
if [ "${permStat:3:1}" != "x" ]; then
    chmod +x ${ATAP_REPO_ROOT}/code/GLM_detection/batch/run_glm_pipeline_quick.py
fi
permStat=`ls -l ${ATAP_REPO_ROOT}/code/GLM_detection/batch/run_glm_pipeline_elemental.py`
if [ "${permStat:3:1}" != "x" ]; then
    chmod +x ${ATAP_REPO_ROOT}/code/GLM_detection/batch/run_glm_pipeline_elemental.py
fi
permStat=`ls -l ${ATAP_REPO_ROOT}/code/GLM_detection/batch/run_bolide_aggregation.py`
if [ "${permStat:3:1}" != "x" ]; then
    chmod +x ${ATAP_REPO_ROOT}/code/GLM_detection/batch/run_bolide_aggregation.py
fi
permStat=`ls -l ${ATAP_REPO_ROOT}/code/GLM_detection/batch/gnu_parallel_script.pbs`
if [ "${permStat:3:1}" != "x" ]; then
    chmod +x ${ATAP_REPO_ROOT}/code/GLM_detection/batch/gnu_parallel_script.pbs
fi
permStat=`ls -l ${ATAP_REPO_ROOT}/code/GLM_detection/batch/aggregator_batch_script.pbs`
if [ "${permStat:3:1}" != "x" ]; then
    chmod +x ${ATAP_REPO_ROOT}/code/GLM_detection/batch/aggregator_batch_script.pbs
fi

#***
# Add validation code to PYTHONPATH and make the top-level scripts executable.
export PYTHONPATH=${ATAP_REPO_ROOT}/code/validation:${PYTHONPATH}
# Make package_bolide_detections executable if it isn't already
permStat=`ls -l ${ATAP_REPO_ROOT}/code/validation/package_bolide_detections.py`
if [ "${permStat:3:1}" != "x" ]; then
    chmod +x ${ATAP_REPO_ROOT}/code/validation/package_bolide_detections.py
fi
#***
# Add bolide detection tools to PYTHONPATH
export PYTHONPATH=$ATAP_REPO_ROOT/code/GLM_detection/bolide_detection:${PYTHONPATH}

#***
# Add gigantic jet detection tools to PYTHONPATH
export PYTHONPATH=$ATAP_REPO_ROOT/code/GLM_detection/gigantic_jet_detection:${PYTHONPATH}

#***
# Add streak detection tools to PYTHONPATH
export PYTHONPATH=$ATAP_REPO_ROOT/code/GLM_detection/streak_detection:${PYTHONPATH}

#***
# Utilities
export PYTHONPATH=${ATAP_REPO_ROOT}/code/utilities:${PYTHONPATH}

#***
# Visualization (Plotting Tools)
export PYTHONPATH=${ATAP_REPO_ROOT}/code/GLM_detection/visualization:${PYTHONPATH}
permStat=`ls -l ${ATAP_REPO_ROOT}/code/GLM_detection/visualization/plot_bolide_detections.py`
if [ "${permStat:3:1}" != "x" ]; then
    chmod +x ${ATAP_REPO_ROOT}/code/GLM_detection/visualization/plot_bolide_detections.py
fi

#***
# Glint filter
export PYTHONPATH=${ATAP_REPO_ROOT}/code/GLM_detection/glint:${PYTHONPATH}

#***
# Renavigation
export PYTHONPATH=${ATAP_REPO_ROOT}/code/GLM_detection/renavigation/python:${PYTHONPATH}

#***
# Cutout tool
export PYTHONPATH=${ATAP_REPO_ROOT}/code/GLM_detection/ABI/cutout:${PYTHONPATH}
