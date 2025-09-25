.. _software_releases:

Software Release Procedure and Release History
==============================================

Here we document what software is available in the ATAP Git repository how to perform releases and pipeline release history.

Release Procedure
-----------------
This is a brief description of the procedure to generate a software release.

A release comprises both a git branch *and* a conda environment. The conda environment is critical because the pipeline
relies on many Python packages and we want the environment to be stable. 

As of now, a deployed release requires some configuration files to have custom changes when in a release versus the
development branch. This is not ideal because this means one cannot cut a branch directly from the release and then merges the
changes to both main and the release. One needs to cheery-pick. **TODO: resolve this!** However, the work-around for
now is to not commit the changes to the branch in the working clone for operations. 

Create a pipeline release
^^^^^^^^^^^^^^^^^^^^^^^^^

#. **TEST!**
    Perform a system test on the main branch and conda environment with a standard set of GLM data. Confirm
    appropriate operation (I hope this goes without saying!).

#. **Ancillary Data Files**
    Move the needed ancillary data files to the location where the shared database reads the ancillary data files. In
    the pipeline configuration files, point to these copies of the ancillary files.
    As of now, the location of the shared ancillary files is::
        
        /nobackupp17/rlmorri5/GLM/data/ancillary/detection

#. **Create Conda environment yml file***
    The conda environment is a critical part of the pipeline setup. If new conda packages were added or updates made to the installed packages then we need to
    record this information in a new conda .yml file. This can then be used to set up the conda release envrionment.::

        # First make sure the ATAP conda environment is active
        atap-init # or conda activate glm-env
        cd $ATAP_REPO_ROOT/system/env
        conda env export > environment.yml

    Then, of course, commit the change to the main branch (or whatever branch we are creating the release branch off of).

#. **Release Branch**
    **First, make sure you update the version number in `atap/code/GLM_detection/batch/_version.py` before cutting the release branch.**

    Create a new git branch off of main (could use a different branch if needed, but that would be an exception). Name the
    branch `releases/#.#.#/base`::

        git checkout -b releases/#.#.#/base

    and push to origin::

        git push -u origin releases/#.#.#/base

#. **Conda Environment**
    Create a new conda environment off the main `glm-env` environment. Name the environment `glm-release-#.#.#-env` to
    correspond to the Git release::

        conda create --name glm-release-#.#.#-env --clone glm-env    

    Note that one could also use the just created yml file.
    Then activate the newely create release conda environment.

#. **Code changes in release branch**
    There should be a seperate git clone used just for operations. Give it the name 'atap-ops'::

        git clone https://<username>@github.com/SETI/atap.git atap-ops

    Then 'cd' in to this git clone and checkout the release branch::
        
        git checkout -b releases/#.#.#/base origin/releases/#.#.#/base

    This protects the pipeline from development work.
    In the git clone location used for operations, modify the following files in `atap/code/GLM_detection/batch`:

    * `batch_elemental_script.sh`
    * `aggregation_script.sh`

    to call `~/.atap_env_startup_release.sh` (See below for setting up this script). Do not commit these minor changes to
    the branch so that the release branch does not fork with main. This is only intended for the checked out repo used
    for operations.

Deploy a pipeline release
^^^^^^^^^^^^^^^^^^^^^^^^^

There are two ways to process the data: 1) as a batch job in Pleiades, and 2) as a daily processing run on a single machine.

#. **Directory**
    Create a directory to house the run scripts and output, such as:::

        ATAP/GLM/ops_pipeline_runs/20220223_0.5.3_release

    on the nobackup lustre filesystem. 

    There is a symlink in `ATAP/GLM/ops_pipeline_runs/current_release`. Point the
    link to this new directory.
#. **Configuration Files**
    There are five configuration files in total. These all must be set up in the output directory:

    * top_level_config_G16.txt
    * top_level_config_G18.txt
    * elemental_config_G16.txt
    * elemental_config_G18.txt
    * post_process_config.txt
    * validation_config.txt

    Modify these as is appropriate for the release pipeline. Change all the reference directories to point to within
    the release run top level path. Point ancillary data files to the ancillary data file location. For production runs,
    the ancillary data location is `/nobackupp17/rlmorri5/GLM/data/ancillary/`.
    Create all the output directories needed for the batch jobs.
#. **cron_script.sh**
    The cron script is for the daily processing. Copy `cron_script.sh` from `code/GLM_detection/batch` to the 
    directory to run the pipeline. Modify it to point to the appropriate directories.
#. **.atap_env_startup_release.sh**
    Modify `~/.atap_env_startup_release.sh` to activate the new release conda environment.
#. **Permissions**
    We want the files to be readable by the ATAP team (group s1488).
    In the top level directory just created above, do:::

        chown <username>:s1488 <path_created_above>
        cd <path_created_above>
        find . -type d -exec chmod 750 {} +
        find . -type f -exec chmod 644 {} +
        chown -R <username>:s1488 .


#. **Compile code**
    Some code in the pipeline needs to be compiled:::
    
        source ~/.atap_env_startup_release.sh 
        bash $ATAP_REPO_ROOT/code/GLM_detection/renavigation/fortran/build_f2py.sh
        pushd $ATAP_REPO_ROOT/code/utilities/L0_reader
        make
        popd
        module load matlab/2021a
        matlab -batch "run $ATAP_REPO_ROOT/code/GLM_bolide_analysis/system/matlab/compile_bolide_analysis_code"

    Note that the last two lines are to compile the Matlab post-processing code. This requires a Matlab compiler
    license.

#. **Run batch job**
    If you wish to process historical data then submit a batch job to the NAS. Check the PBS configuration files in the
    `code/GLM_bolide_detection/L2/batch` directory to confirm the correct node requests.
#. **Cron job**
    On the machine you wish to perform daily processing or to issue the PBS submission, set up a cron job with the command:::

        crontab -e

    The cron job will then call the `cron_script.sh`. The following example will run the daily processing at 11 PM *on
    the machine with the cron job*:::

        0 23 * * * /home5/jcsmit20/nobackupp2/ATAP/GLM/ops_pipeline_runs/current_release/cron_config/cron_script.sh

    Or this command will submit a job to PBS:::

        0 23 * * * /PBS/bin/qsub /home5/jcsmit20/nobackupp2/ATAP/GLM/ops_pipeline_runs/current_release/cron_config/cron_script.sh
        
    Note that if you had updated the symlink in the step above to point to the new release directory then the crontab
    might not need to be updated.

Software Releases
-----------------

Initial Prototype
^^^^^^^^^^^^^^^^^

This is the initial prototype of the GLM L2 bolide detector and plotter. Here is the reference paper: https://www.ncbi.nlm.nih.gov/pubmed/30818807
A release branch was taken to preserve this initial prototype:::

    releases/0.1/base

The code in this branch is here:::

    atap/code/experimental/GLM/L2/glm_detect_bolides

Subsequent Releases
^^^^^^^^^^^^^^^^^^^

We will use the following nomenclature to name releases:::

    releases/#.#.#

Major releases is the first `#`.

Minor releases is the second `#`

Simple parameter changes is the third `#`. 

Changes in a parameter release will be pushed up. For example, say we are on release 0.2.1 and we wish to make a small
parameter change. We would increment the third `#` to 0.2.2 and then merge the changes to releases/0.2. The automated
pipeline code will log which release branch it is running on. This way, we should be able to reproduce any results by
simply checking out the correct release branch.

The following table gives a summary of the subsequent releases. Early on, the release engineer was not entirely
dedicated to documenting the release version, hence the missing information.

=======     ==========  ====================    ========
Release     Date        Branch                  Comments
=======     ==========  ====================    ========
0.1                     releases/0.1/base       **Summary:**

                                                The initial prototype of the GLM L2 bolide detector and plotter as
                                                coded by Clemens Rumpf.
                                                
0.2                     releases/0.2/base       **Summary:**

                                                Attempt to mainly reproduce the 0.1 prototype but with a couple changes.
                                                
                                                Change log:
                                                
                                                * Glint filter
                                                * Use of hierarchical clustering of groups for speed
                                                * Added labeling tool to classify detections
                                                
0.3         2019/09/25  releases/0.3/base       **Summary:**
                                                
                                                * Functionality to automatically run the detector on Pleiades.
                                                * Functionality to support automated backups from Pleiades to Lou.
                                                * Directory reorganization.
                                                * Updated symbols to conform with current use of terminology  within the ATAP.
                                                
0.4                     releases/0.4/base       **Summary**

                                                This release was mainly created because the release engineer messed up a
                                                branch and wanted to just start with a clean branch.
                                                
                                                Change Log:
                                                
                                                Details not recorded. We should do a better job logging our work.
                                                
                                                * ATAP-70: fixed hierarchical clustering for too many groups
                                                
0.5         2020/08/21  releases/0.5/base       **Summary**

                                                This was a major release, switching over to a Random Forest classifier
                                                
                                                Change Log:
                                                
                                                * GLM pipeline parallel batch operation mode. Farms out GLM data to Pleiades node by day.
                                                * GLM pipeline operation mode to generate training data set using bolide website data as ground truth.
                                                * GLM pipeline operation mode to use a trained classifier using Scikit-Learn's API.
                                                * Added new features for use with trained classifier.
                                                * Switched Python environment to use a custom miniconda
                                                * installation allowing for complete control of python
                                                * environment.
                                                * Fixed a lot of small bugs causing individual .nc data files to error.
                                                * Speed improvements
                                                
0.5.1       2021/05/24  releases/0.5.1/base     **Summary**

                                                This was a minor release where we greatly improved the random forest
                                                classifier but many other improvements were also made. Main Epic ticket:
                                                ATAP-105
                                                
                                                Changes Log:
                                                
                                                * Switched to a ZODB object oriented database for storing detections and
                                                  rejections
                                                * New Feature: Latitude and longitude
                                                * New Feature: Neighborhood group activity
                                                * Now searching for bolides that span multiple netCDF files
                                                * Clusterer outlier rejection

0.5.2       2021/09/16  releases/0.5.2/base     **Summary**

                                                This minor release is mainly to get the packager running so that we can
                                                get vetting figures to the human vetters. Main Epic ticket: ATAPJ-26

                                                Change Log:

                                                * Packager to combine detection and cutout figures and copy netCDF files
                                                  to vetting directory
                                                * Include NetCDF files from other satellite in output directory for
                                                  stereo detections
                                                * Generate "detection" figure for group data in other satellite in
                                                  stereo region
                                                * Generate "re-navigation" figures to measure group altitude in stereo
                                                  region
0.5.3       2022/02/23  releases/0.5.3/base     **Summary**

                                                This is a minor release that makes a couple small improvements. 
                                                Main Epic ticket: ATAPJ-55

                                                Change Log:

                                                * Can now use geonex data sets (ATAPJ-35)
                                                * Plots corresponding pixels on detection figures.
                                                * Plot event energies in addition to group energies on figures
                                                * Option to set minimum re-navigated altitude to generate stereo
                                                  figures.
                                                * Aggregator runs significantly faster
                                                * New classifiers trained using the same features as 0.5.1.
0.5.4       2022/04/21  releases/0.5.4/base     **Summary**

                                                This is a minor release that merges the cutout tool and packager into the main pipeline.
                                                Main Epic ticket: ATAPJ-90
                
                                                Change Log:

                                                * Cutout tool and packager now run in the main pipeline (ATAPJ-88)
                                                * Some minor fixes to plotting figures and logging information.
                                                * Option to delete daily database files.
                                                * Hot Spot de-emphasis added. ATAPJ-89
0.6.0       2022/11/02  releases/0.6.0/base     **Summary**
    
                                                This is a major release mainly to incorporate the post-processing Matlab code and the validation report
                                                generator. Main release Epic ticket: ATAPJ-109

                                                Change Log:

                                                * Call post-processing and light curve generation code in pipeline on detections.
                                                * Generate bolide detection candidate validation PDF report.
                                                * Sped up neighborhood feature computation by factor ~20.
                                                * Some minor other tweaks to data handling and putput figures.
                                                * ATAPJ-125: File created in output and public directories signifying if processing was performed (versus just
                                                  no detection for that day).
0.6.1       2023/01/19  releases/0.6.1/base     **Summary**
    
                                                This is a minor release mainly to make changes to allow for processing
                                                on GOES-18 data. Main release Epic ticket: ATAPJ-114.

                                                Change Log:

                                                * Modifications to detection, cutout and post-processing code to run on
                                                  G18. Cutout tool required signficant modifications.
                                                * Detection figure now notes when detection is stereo with other
                                                  satellite data but renavigated altitude is below altitude threshold.
0.7.0       2024/01/17  releases/0.7.0/base     **Summary**

                                                This is a major release mainly to update the pipeline for use with the
                                                new unified MongoDB database (https://github.com/SETI/atap-bolide-db).
                                                There have also been an accumulation of other small bugs and design
                                                changes. Main release Epic ticket: Github Issue #46 (https://github.com/SETI/atap/issues/46).

                                                Change Log:

                                                * Fixed several bugs and fragile aspects of the cutout tool.
                                                * Added extra fields to and format of Provenenace class for import to
                                                  MongoDB database.
                                                * Changed output data format to reflect changes in the new validation
                                                  process.
                                                * Fixed some issues with the detection figures.
                                                * New Feature: ground_distance.
                                                * Fixed bug in applying median absolute deviation scaling factor twice
                                                  for the chop feature.
                                                * Sped up initial steps to determine what days to process.
                                                * New trained Random Forest classifiers.
1.0.0       2025/03/31  releases/1.0.0/base     **Summary**

                                                This is a major release to incorporate the CNN-based auto-validator.
                                                Main release Epic ticket: Github Issue #103 (https://github.com/SETI/atap/issues/103).

                                                Change Log:

                                                * Added CNN-based auto-validator 
                                                * New BolideAssessment class to store both triage and validation
                                                  assessments
                                                * 3D hierarchical clustering
                                                * Parallelized (sped up) bolide detection figure generation
                                                * Parallelized the clustering step
1.0.1       2025/04/22  releases/1.0.1/base     **Summary**
                                
                                                This is a minor release to allow the pipeline to process GOES-19 data.
                                                Main release Epic Ticket: Github Issue #154: https://github.com/SETI/atap/issues/154

1.0.2       2025/09/24  releases/1.0.2          **Summary**
                                            
                                                This release is specifically to create a public release to be provided
                                                with the publication of the auto-validator paper.
                                                There are no algorithmic changes in this release, just documentation and
                                                some code reorganization.
                                                Main release Epic Ticket: Github Issue #197: https://github.com/SETI/atap/issues/197
=======     ==========  ====================    ========

