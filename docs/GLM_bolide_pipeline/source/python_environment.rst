.. _python_environment:

.. sectionauthor:: Jeffrey C. Smith <jsmith@seti.org>

Setting up a Python Environment for the GLM Bolide Pipeline
===========================================================

This document assumes the reader is already familiar with Python evironments and Linux operating systems. 

Whether you are developing ATAP software or just using it, you'll need to configure your Python environment so your system can locate and 
execute the various software components. The following instructions have been tested on linux machines and on the NASA Advanced 
Supercomputer Facility (NAS) machines. They should work on macOS (but please give me feedback). 
These installation instructions have not been tested on Windows, nor does the author use Windows. If you use Windows, you're in your own.

There are numerous ways to install a Python Environment. One of these methods is using Anaconda, or just 'conda'. The
following instructions assumes you will use conda. 

Also note that the ATAP environment uses BASH. Make sure you are using BASH for the following scripts to function.

Installing Conda
----------------

The ATAP GLM pipeline python packages utilize a saved conda environment. Check if conda
is installed and ensure you have version 4.8.2 or above.::

    $ conda --version
    conda 4.8.2

If conda is not installed then you can either install a full conda distribution (I.e. Anaconda) or a miniconda3
distribution. I recommend miniconda. Installation instructions are `here <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_

Note that a single Python environment can be many gigabytes and many hundreds of throusands of files. This can be a strain on small and slower filesystems.
If working on the NAS, consider placing your conda installation somewhere other than in your home directory (which has a
8GB quota!). It is
suggested to place in in your nobackup directory, for example, ``/nobackupp##/<username>/miniconda3``.

Here is an example script to install miniconda::

    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash

Then restart your current BASH shell for the installation to complete. If miniconda3 successfully installed, your bash
prompt should look something like::

    (base) <username>@<machine>:~

where the "(base)" at the beginning means conda is active and you are currently in the base environment. 

In theory, you do not need a conda installation. You could install all the python packages in a different way, but the
following instructions assume you are using miniconda3.

Setting up the GLM Python Environment
-------------------------------------

We must now set up a conda python environment for use when running this pipeline. You should by now have an active
conda distribution. You must tell conda the location to place your local environments with the variable
"CONDA_ENVS_PATH". If working on the NAS then consider NOT placing your environment in your home directory because of
the limited disk quota there. Conda environments can be many gigabytes large and many, many thousands of files.
The following example uses the NAS nobackup Lustre filesystem, which was adapted from the
`NAS Knowledge-base <https://www.nas.nasa.gov/hecc/support/kb/managing-and-installing-python-packages-for-machine-learning_627.html>`_.
In any case, you must identify the path where miniconda was installed and point CONDA_ENVS_PATH to it.
Perform the following commands at your BASH prompt ('#' Means a comment)::

    # Specify your local conda environment path
    export CONDA_ENVS_PATH=/nobackupp<num>/<username>/miniconda3/envs

We must now set up the GLM bolide pipeline environment. Start by setting the ATAP_REPO_ROOT environment variable::

    # Set the GIT repository path variable to point to your ATAP source code repo path
    export ATAP_REPO_ROOT=<path> # i.e. $HOME/dev_atap/atap

A conda `environment.yml` file is available in the code repo to aid in creating the new environment. If running Mac OS, a few extra steps are needed 
(see :ref:`macos_supplemental_steps` below). In either case, we must then make a slight modification to the environment
yml file. `PyTorch <https://pytorch.org/>`_ will not install correctly from the environment file. So, first remove all
lines with `torch` on the line (as of now, 4 lines total). Save the `environment.yml` file and then 
do the following::

    # Create the pipeline environment:
    conda env create -f $ATAP_REPO_ROOT/system/env/environment.yml

    # Activate your newly created environment
    conda activate glm-env

    # Install PyTorch manually
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia


Do not commit the change to the `environment.yml` file. Revert the file to what is on the branch once you are done installing.

Note that the created environment might have a name other than `glm-env` (for example, a release branch environment). 
If this is the case then either rename the environment or replace the name of the environment above with the created one.
There will be a comment returned when you create the environment giving the environment name. To rename::
    
    conda create --name new_name --clone old_name
    conda remove --name old_name --all # Optional if you want to also keep the old envorinment

Now we must set up a script to activate the ATAP pipeline environment when we wish to do ATAP work or run the pipeline. Create a new file called
``$HOME/.atap_env_startup.sh`` and add the following lines. Three environment variables need to be specified:
``ATAP_REPO_ROOT``, ``CONDA_ENVS_PATH`` and ``CONDA_ROOT_PATH``. Modify the script to set these paths::

    # This sets up an ATAP development environment.

    # Specify the GID (Group ID) to use on the NAS (only needed if working on the NAS)
    # Note: The below GID is for ATAP, use the GID you have access to.
    # If you are an intern working with Jeff Smith, then you should have access to this GID.
    # If you are on a local machine and not on the NAS, then either do not set this or set it to your current group ID.
    export GROUP=s1488

    # Set the GIT repository path root
    export ATAP_REPO_ROOT=<path> # i.e. $HOME/dev_atap/atap

    # ATAP setup
    source $ATAP_REPO_ROOT/system/env/atap_bash_startup.sh

    # conda path
    export CONDA_ROOT_PATH=<path> # i.e. /nobackupp##/<username>/miniconda3

    # Conda environments
    export CONDA_ENVS_PATH=<path> # i.e. /nobackupp##/<username>/miniconda3/envs

    # >>> conda initialize >>>
    __conda_setup="$('$CONDA_ROOT_PATH/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$CONDA_ROOT_PATH/etc/profile.d/conda.sh" ]; then
            . "$CONDA_ROOT_PATH/etc/profile.d/conda.sh"
        else
            export PATH="$CONDA_ROOT_PATH/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<

    # Activate glm-env conda environment
    conda activate glm-env


These lines should be placed in ``$HOME/.atap_env_startup.sh`` and not in ``$HOME/.bashrc``. This is important if you work on multiple projects 
using different python environments you do not want to auto-initialize the ATAP enviornment. Also, when submitting pipeline
jobs on the NAS the node will be appropriately setup without also needing to execute everything in a person's .bashrc.
The lines within >>> conda initialize >>> are autogenerated with the miniconda3 distribution and placed in ``.bashrc``. 
We then manually move them
from ``.bashrc`` to here. If you have a different conda distribution then these lines might not be needed, or different. For example, a
full Anaconda distribution on MacOS does not require these lines.

The above script calls ``$ATAP_REPO_ROOT/system/env/atap_bash_startup.sh``, which does the following when sourced:
    1. Checks whether git core.symlinks==true, if not then sets it
    2. Adds $ATAP_REPO_ROOT/system/bin to PATH
    3. Adds each tool's source path to PYTHONPATH
    4. Checks whether the program's main files are executable, if not, uses 'chmod +x' to make them so.

You may be asking right now, why are we not using a python package and Kubernetes or some other comtainerization manager? Good question! The
answer is simply a lack of people and resources to set up and maintain such a package and container. The above method
has worked well. So... moving on...

Building the FORTRAN Library
----------------------------
The stereo re-navigation code from Lockheed Martin was originally Matlab code, it was conerted to FORTRAN 77 code. It's called by the pipeline to measure the altitude of
stereo detections. This code must be compiled so that it is callable in Python. The compiled library is placed in the
directory `${ATAP_REPO_ROOT}/system/lib`. After the environment is activated above, the code can be compiled with::

    bash $ATAP_REPO_ROOT/code/GLM_detection/renavigation/fortran/build_f2py.sh

This compilation has to be performed only *once*.  It will continue to operate until either the Fortran code is updated, the Python version is updated or
possibly other aspects of the OS is updated.


.. _installing_the_bolides_package:

Installing the bolides package
------------------------------
There is also a public bolides package that allows the public to utilize our data sets: 
`github.com/SETI/bolides <https://github.com/SETI/bolides>`_. It is utilized in the pipeline, so it must be installed.  
The above conda installation yml script might install a
release version of this pachage. You might need a development version. To install the development version::
  
    pip uninstall bolides # First uninstall the PyPi version
    # Install the bolides package *parallel* to $ATAP_REPO_ROOT
    cd $ATAP_REPO_ROOT
    cd ..
    git clone https://github.com/SETI/bolides.git
    pip install -e .

Note: for some reason, `pip uninstall bolides` might not work. If so, you may need to manually delete the directory in
the python environment located at `miniconda3/envs/glm-env/lib/python3.11/site-packages/bolides`.


Building the Post-Analysis Software
-----------------------------------
The Post-Analysis software will take each pipeline detection and generate calibrated pixel and light curve data. This code is written in Matlab. The Matlab
code must be compiled using the Matlab Compiler to be run in the pipeline. See :ref:`running the post-detection analysis<running_post_detection_analysis>` 
for instructions. 


.. _macos_supplemental_steps:

Creating a conda env on MacOS
-----------------------------

The `environment.py` stored conda environment configuration was created on a Linux machine. To install on MacOS, you
need to make some changes to the `environment.py` file and install pytorch differently. The procedure I have successfully used is this:

1. Make a copy of the YAML file (call it macos-atap-env.yml)
2. Edit macos-atap-env.yml as follows:
	a. Change the name field at the top to be whatever you prefer.
	b. Strip off all OS-specific tags from the listed conda packages
	c. Remove the bolides package from the list of conda packages
	d. Remove all lines containing the string “torch”
3. Attempt to create a conda environment.
4. If the environment was NOT successfully created, conda will print a message like "Could not solve for environment specs. The following packages are incompatible" followed by a list of packages. Do the following:
	a. Edit macos-atap-env.yml to strip off version numbers from the incompatible packages.
	b. Try again to create the environment.
5. If the environment was still NOT successfully created, compile a list of the packages that are still not compatible after having version numbers removed, do the following:
	a. Edit macos-atap-env.yml to remove lines containing the incompatible packages.
	b. Try again to create the environment.
6. At this point you should have been able to create the environment. If so, proceed to the next step. If not, try to figure out what’s wrong (sorry, but you’re on your own!).
7. Install pytorch as follows::
	``$conda install pytorch torchvision -c pytorch``
8. Install the bolides package in editable mode, as described above in :ref:`installing_the_bolides_package`.

Step 2 can be accomplised using ``sed`` as follows::
 
	$ cat $ATAP_REPO_ROOT/system/env/environment.yml | sed 's/\(.*[[:alnum:]]\)=[[:alnum:]][[:alnum:].-_]*/\1/' | sed '/bolides/d' | sed '/torch/d' > macos-atap-env.yml
	
There is a python script (``$ATAP_REPO_ROOT/system/env/modify_yaml_file.py``) available to assist with steps 4 and 5.
 
The above steps are becuase the primary OS used for the pipeline is Linux. If someone who actively develops on MacOS has
the ambition, they could create and maintain an ``environment_macos.yml`` specific for Mac OS. 
Then the above steps would not be needed.


.bashrc Setup
-------------

Very little must now be placed in your ``.bashrc`` because ``$HOME/.atap_env_startup.sh`` takes care of everything.
Simply add the following to your ``.bashrc``::

    ##################################################################
    # ATAP Setup

    # ATAP environment setup script
    alias atap-init='source ~/.atap_env_startup.sh'

If you are workling on the NAS, you can also add the following NAS node aliases in your ``.bashrc`` as a convenience to help you quickly request interactive compute nodes on the NAS.::
    
    ##################################################################
    # Some NAS node aliases
    alias devel_node_bro='qsub -I -W group_list=$GROUP -q devel -X -l select=1:model=bro'
    alias devel_node_sky='qsub -I -W group_list=$GROUP -q devel -X -l select=1:model=sky_ele'
    alias devel_node_rome='qsub -I -W group_list=$GROUP -q devel -X -l select=1:model=rom_ait'
    alias normal_node_bro='qsub -I -W group_list=$GROUP -q normal -X -l select=1:model=bro,walltime=8:00:00'
    alias normal_node_ivy='qsub -I -W group_list=$GROUP -q normal -X -l select=1:model=ivy,walltime=8:00:00'
    alias normal_node_has='qsub -I -W group_list=$GROUP -q normal -X -l select=1:model=has,walltime=8:00:00'
    alias normal_node_sky='qsub -I -W group_list=$GROUP -q normal -X -l select=1:model=sky_ele,walltime=8:00:00'
    alias normal_node_cas='qsub -I -W group_list=$GROUP -q normal -X -l select=1:model=cas_ait,walltime=8:00:00'
    alias normal_node_rome='qsub -I -W group_list=$GROUP -q normal -X -l select=1:model=rom_ait:aoe=toss3,walltime=8:00:00'
    alias long_node_bro='qsub -I -W group_list=$GROUP -q long -X -l select=1:model=bro,walltime=120:00:00'
    alias long_node_sky='qsub -I -W group_list=$GROUP -q long -X -l select=1:model=sky_ele,walltime=120:00:00'
    alias long_node_rome='qsub -I -W group_list=$GROUP -q long -X -l select=1:model=rom_ait:walltime=120:00:00'

    # Interactive GPU nodes
    # This is one v100 on a skylake GPU node, it uses 1/4 of the resources on the node
    alias gpu_node_sky_1='qsub -I -W group_list=$GROUP -q v100@pbspl4 -X -l select=1:model=sky_gpu:ngpus=1:ncpus=9:mem=96g,place=vscatter:shared,walltime=8:00:00'
    # This uses 1/2 of the resources (2 v100 GPUS) of a skylake GPU node
    alias gpu_node_sky_2='qsub -I -W group_list=$GROUP -q v100@pbspl4 -X -l select=1:model=sky_gpu:ngpus=2:ncpus=18:mem=192g,place=pack,walltime=8:00:00'
    alias gpu_node_sky_4='qsub -I -W group_list=$GROUP -q v100@pbspl4 -X -l select=1:model=sky_gpu:ngpus=4:ncpus=36:mem=376g,place=pack,walltime=8:00:00'
    # 1/4 of a cascadelake GPU node (1 v100 GPU)
    alias gpu_node_cas_1='qsub -I -W group_list=$GROUP -q v100@pbspl4 -X -l select=1:model=cas_gpu:ngpus=1:ncpus=12:mem=96g,place=vscatter:shared,walltime=8:00:00'
    # 1/2 of a cascade loake node (2 v100 GPUs)
    alias gpu_node_cas_2='qsub -I -W group_list=$GROUP -q v100@pbspl4 -X -l select=1:model=cas_gpu:ngpus=2:ncpus=24:mem=192g,place=pack,walltime=8:00:00'


    ##################################################################

Some systems are set up to not automatically source your ``.bashrc`` when you log into a terminal.
If so, you can source your ``.bashrc`` from inside ``~/.profile`` by adding the following line to
``~/.profile``::

    source ~/.bashrc

In either case, you will want to manual run `source ~/.bashrc` right now to re-read your bashrc file in your active bash
shell.

Activating the Environment
--------------------------

Whenever you wish to do ATAP development work or run the pipeline software, simply issue the following command::

    atap-init

To deactivate the ``glm-env`` environment type::

    conda deactivate

If you wish to reserve an interactive compute node on the NAS then call the appropriate alias, for example::
    
    normal_node_bro

This will request a broadwell compute node on the normal queue (8 hour max wall time).

Running the Pipeline
--------------------
Once the environment is all set up, you can run the pipeline. See the next section:
:ref:`running the pipeline<running_the_pipeline>`.

Comments
--------

* Once you have completed the above steps. You should be able to run the executables located in
  ``$ATAP_REPO_ROOT/system/bin`` from any working directory with no need to give full paths.  The executables in this bin
  directory are symbolic links to the actual main functions in the their respective code directories.
* `$ATAP_REPO_ROOT/system/env/atap_bash_startup.sh` utilizes the BASH shell. If your default shell is something else,
  then be sure to switch to the BASH shell before calling the script.
* Generally speaking, unless you are developing new code, you should be running the code off of either main or a release branch.
