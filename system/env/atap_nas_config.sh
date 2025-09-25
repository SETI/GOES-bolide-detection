#!/bin/bash
##################################################################
# Activates a conda environment for GLM pipeline development
################################################################## 

# If this is a noninteractive shell, then we need to do some 
# additional setup.
if [[ $- == *i* ]]; then
    source /usr/local/lib/global.profile
fi

conda activate glm-env

