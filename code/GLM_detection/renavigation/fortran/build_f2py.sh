#!/bin/bash
# This shell script will compile the extension modules for the FORTRAN code used by the GLM pipeline

# We want the compiled fortran module file to be placed in a lib directory 
# Create the lib directory if it does not exist
if [ ! -d "/path/to/dir" ] ; then
    mkdir ${ATAP_REPO_ROOT}/system/lib
fi
pushd ${ATAP_REPO_ROOT}/system/lib
python -m numpy.f2py -c -m GLM_renavigation ${ATAP_REPO_ROOT}/code/GLM_detection/renavigation/fortran/renavigate.f
popd
