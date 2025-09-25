.. _running_the_pipeline:

.. sectionauthor:: Jeffrey C. Smith <jsmith@seti.org>

Running the GLM Detection Pipeline
===============================

The pipeline was originally built to detect bolides. The instructions below are therefore relevant to this mode of
operations. Subsequent development has allowed for a second mode of operation to detect Gigantic Jets, a type of
transient luminous events, which is a type of lightning. This work is
in development, but to activate this mode set ``detectionType = gigantic_jets``. 

After making sure your environment is correctly set up (see :ref:`python_environment`), the executable
``run_glm_pipeline_batch.py`` should be accessible in your path. If you wish to also run the post-processing code then
follow the instructions in :ref:`running the post-detection analysis<running_post_detection_analysis>` to set that up.
You will need a Matlab license, or a compiled version of the post-analysis Matlab code.
Note that if the post-processing is enabled and run in the pipeline then there is no need to manually specify the
configuration file, detection CSV file or output directory in it's configuration file. 
Those are set automatically when the post-processing is run in the pipeline.

There is also a "quick run" mode for the pipeline. This is used to rapidly process a specific set of data, instead of in
tpyical pipelines operations mode. see 
:ref:`running the quick pipeline<running_the_quick_pipeline>` for details.

The pipeline's main "unit of work" is a day's worth of GLM data. The data can stored in three ways:

1. In the "NASA GeoNEX" archive with a directory structure such as `/G16/YYYY/DOY/HOD/\*.nc`, where "YYYY" is the year, "DOY"
   is the three-digit day of the year and "HOD" is the two-digit hour of the day (i.e. 00 - 23).
   See on the NAS Lustre filesystem for an example: `/nobackupp10/geonex/public/GOES16/NOAA-L2/GLM-L2-LCFA`
2. A single daily tarball archive ("Daily Dundles")
   See on the NAS Lustre filesystem for an example: 
   `/nobackupp2/ldm/archive16/goes-r.nsstc.nasa.gov/restricted/goes-16/GLM-L2-LCFA/daily-bundles`
3. Individual netCDF files
   in a directory structure such as `./G16/YYYY/MMDD/\*.nc`

Note that only option 1, using GeoNex data, is actively used. The other two data formats might fail to run.

The operator can specify the range of days to be processed and how they are processed using the configuration files below.

Three configuration files are necessary:

1. top_level_pipeline_config_file.txt
2. elemental_config_file.txt
3. validation_config_file.txt

Examples of these two files is located in the ``$ATAP_REPO_ROOT/code/GLM_bolide_detection/batch`` directory.

.. _top_level_config_file:

top_level_pipeline_config_file.txt
----------------------------------

This file tells the batch processor how to run and call the ``run_glm_pipeline_elemental.py`` function which operates on each day of data.
The configuration options to consider are below.  See
``$ATAP_REPO_ROOT/code/GLM_bolide_detection/batch/example_top_level_config_file.txt`` for an example.

* **inputRootDir** : str
    Path to the data to process, for example ``/nobackupp2/ldm/archive16/goes-r.nsstc.nasa.gov/restricted/goes-16/GLM-L2-LCFA/daily-bundles``.
    See the three data store options above.
* **inputRootDirL0** : str
    Path to the Level 0 data files. This is only used if running post-detection analysis.
* **otherSatelliteRootDir** : str
    Path to the data for the other satellite when performing stereo detection analysis. For example, If `inputRootDir`
    points to GOES-16 data then this could point to the GOES-18 data store directory.
* **outputDir** : str
    Path to where to store the output
* **reportOutputDir** : str
    Path to where to store the bolide detection validation report PDFs
    Set to None to use **outputDir**
* **detectionType** : str
    What are we detecting in the GLM data?
    (e.g. 'bolides', 'gigantic_jets'
* **elementalConfigFile** : str
    Path and name of file for the daily elemental configuration file (e.g. file 2. above, ``elemental_config_file.txt``)
* **validationConfigFile** : str
    Path and name of file for the validation configuration file (see :ref:`validation of detections<validation_of_detections>`)
* **forceReprocess** : bool
    If true then the processing history log file is deleted, ``outputDir`` and ``reportOutputDir`` are wiped clean and the full data set is reprocessed. 
    **WARNING: This deletes all data in outputDir and reportOutputDir!**
* **multiProcessEnabled** : bool
    If parallel processing is enabled, distributed as a batch job across multiple machines, otherwise, all processing occurs on the current machine.
* **n_cores_aggregation** : int
    Number of multiprocessing cores to use for the aggregation step
* **deleteDailyDirs** : bool
    If True then delete all the daily processing directories (greatly dimishes the disk space for the run).
* **deleteDailyDatabases** : bool
    If True then delete all the daily database files but keep the other files in the daily directories (daily database files not needed because the 
    aggregate database contains all the information).
    **Note**: deleteDailyDirs and deleteDailyDatabases cannot both be set to True.
* **deleteDailyNetCdfFiles** : bool
    If True then the individual netCDF files for each detection copied to each daily directory will be deleted. 
    This will typically be 3 files (the main file and the 20-second files on either side). 
* **deleteDailyExtraFigures** : bool
    If True then the daily intermediate figures used to generate the main merged detection figure and validation reports are deleted.
    The combined detection and cutout figure is retained.
* **startDate** : str
    The start date for processing in ISO datetime string format: ``YYYY-MM-DD[*HH[:MM[:SS[.fff[fff]]]][+HH:MM[:SS[.ffffff]]]]``

    Example: ``2011-11-04 00:05:23.283``

    Use ``''`` if no startDate is set. 
* **endDate**
    The end date for processing in ISO datetime string format.

    Use ``''`` if no endDate is set.     
* **doNotProcessCurrentDay** : bool
    If True then do not process any data from the current day.
    This is used during daily processing where we do not want to process the partial day available if using an active data
    stream (such as GeoNEX).
* **delayNDaysForL0Data** : int
    Do not process the daily data if the Level 0 data is not present. But only check up to this number of days back in time. This means for all days older than this
    many N days old, process the data irrespective of if the Level 0 data is available


If multi-processing is enabled and running on the NAS then `qsub` and gnu parallel are used to farm out each day's processing.
If not running on the NAS then gnu parallel is used to farm out the runs on the local machine.

If multi-processing is NOT enabled then all days are run serially on a single machine. However, *within* each day
parallel processing can occur via the ``elemental_config_file.txt`` configuration. 

.. _elemental_config_file:

elemental_config_file.txt
-------------------------
**Note:** this section assumes you are detecting bolides. If `detectionType == gigantic_jets` then you are running the
experimental gigantic jet detection code. This is in active development and will have its own elemental configuration
file.

This file tells the daily processor (the elemental "unit of work" process) how to run. There are three basic modes of operation for the
daily processing:

1.  Use the trained classifier. This is the normal mode of operation.
2.  Use the ground truth data file, based on the neo-bolides website data, to find known bolides in the data.
    This will save out both bolide detections and rejections.
    You can generate the ground truth data file with ``bolide_dispositions.pull_dispositions_from_website``.
    This mode of processing is used to generate the training data set to train the classifier.
3.  Use the classic sequential filters. This is the old, original pipeline before the ML classifier was trained. This
    option is not maintained and  might not work anymore.

The configuration options to consider are below. There are *a lot* of configuration parameter below. As the pipeline
has evolved, we add new features. Every new feature has its own set of parameters. This would be a very long document if
we described every feature in minute detail. 
See ``code/GLM_bolide_detection/L2/batch/example_elemental_config_file.txt`` for a more complete list.

* **verbosity** : bool
    Whether to be verbose and print a ton of diagnostic information
* **use_ramdisk** : bool
    If True then copy daily raw netCDF files to a ramdisk (localed at `/tmp/ramdisk/glm_tmp`) before processing the
    data. This should speed up processing. Data at ramdisk is deleted after run.
* **generatePlots** : bool
    If true, plots and corresponding symbolic links to
    .nc files are generated and placed in the output directory specified by
    `outputDir` above.
* **copyNetCDFFiles** : bool
    If True then copy (or symlink) over the raw .nc files for each detection 
    to the outputDir. This will typically be 3 files (the main file and 
    the 20-second files on either side).
* **createSymlinks** : bool
    If True then create a symlink to each .nc file associated with a detection.
    If False then make a copy of the .nc file.
* **n_cores** : int
    Number of processors to be used in multiprocessing. The 20-second netCDF files will then be processed in parallel
    across a single machine. (See `multiProcessEnabled` in the top level directory for processing over multiple compute
    nodes, which is different than this.)
    (default := 1), (to use all available cores := -1)
* **min_GB_post_processing** : float
    The minimum about of memory in GB per post-processing thread. This is to ensure out of memory doesnot occur if too many jobs run in parallel.
* **min_GB_cutout_tool** : float
    The minimum about of memory in GB per cutout tool thread. This is to ensure out of memory does not occur if too many jobs run in parallel.
* **trained_classifier_path** : str
    Path the to the trained classifier to use. If set then processing mode 1. is used (see above). 
    Empty string or [] means do not use, if so, use the filters or ground truth instead.
* **classifier_threshold*** : float
    The detection threshold for the Scikit-Learn predict_proba method of the classifier 
* **max_num_detections** : int
    The maximum number of detections to save per day
    (-1 => save all)
* **ground_truth_path** : str
    Use the ground truth data file to determine if a cluster is a detection. 
    If set then processing mode 2. is used (see above). 
    Empty string or [] means do not use, use the filters or trained classifier instead.
* **rejectsToSaveFrac** : float Range:[0,1]
    Gives the fraction of all rejected clusters to save in bolide_rejections.p if using model 2.
* **spice_kernel_path** : str
    Gives the path to the spice kernels needed to compute the glint point feature.
* **latLon2Pix_table_path** : str
* **latLon2Pix_table_path_inverted** : str
    latLon2Pix converts the L2 latitude and longitude to raw detector pixels.
    Lookup tables have been created to perform the conversion. Pixels are computed and stored in detection objects. The
    pixel corresponding to each group lat/lon will also be plotted on the detection figures.
* **pixel_boundary_plotting_enabled** : bool
    If true then the pixel boundaries are plotted on the detection figures. 
    Uses `latLon2Pix_table_path` and `latLon2Pix_table_path_inverted`.
* **min_num_groups** : int Range:[0:inf]
    The minimum group count a cluster must have to be considered a bolide candidate and pass triage.
* **min_num_groups_to_force_candidacy** : int Range:[0:inf]
    The minimum group count a cluster must have to be forced to pass triage.
* **cluster_3D_enabled**                : bool
    See `bolide_detection.bolide_clustering.BolideClustering` class for details of the clustering parameters.
* **cluster_numba_threads**             : bool
* **cluster_sequential_enabled**        : bool
* **cluster_outlier_rejection_enabled** : bool
* **cluster_closeness_seconds**         : float
* **cluster_closeness_km**              : float
* **cluster_extractNeighboringFiles**   : bool
* **cluster_min_num_groups_for_outliers** : int
* **cluster_outlierSigmaThreshold**     : float
* **cluster_outlierMinClipValueDegrees** : float
* **stereoDetectionOnlyMode** : bool
    If True then only keep detections in the stereo region and between the altitude limits (below)
* **minAltToGenerateFigureKm** : float
    The minimum re-navigated altitude in order to generate the stereo detection figures. Set to -1 to disable.
* **maxAltToGenerateFigureKm** : float
    The maximum re-navigated altitude in order to generate the stereo detection figures. Set to large number to disable.
    Only used if stereoDetectionOnlyMode is True
* **otherSatelliteLatExpansionDegrees** : float
* **otherSatelliteLonExpansionDegrees** : float
* **otherSatelliteTimeExpansionSeconds** : float
    These parameters are used when finding data in the other satellite when a detection is in the stereo region.
    Gives the amount to expand a box about each detection when searching for groups in
    the other satellite. The expansion is plus and minus by each amount.
* **lon_peaks** : float array
* **lat_peaks** : float array
    These parameters are used to de-emphasize "hot spots" of high density of false positive detections.
    Gives the center longitude and latitude of the hot spots in the detection heat map
* **deemph_radius** : float array
* **deemph_alpha** : float array
    Gives the radius and height of the de-emphasis cylinders used to de-emphasize the above hot spots.
    Should be the same length as the lat/lon peaks arrays above (one cylinder definition per hot spot).
    Set the alpha term to 0.0 to disable the de-emphasis for that hot spot.
* **post_process_enabled** : bool
    If True then run the post-processing code, see :ref:`running_post_detection_analysis` for configuring the post-analysis processing.
* **post_process_config_file** : str
    Path to the post-processing analysis configuration file, see the post-processing :ref:`post-processing_configuration_file` documentation for parameter settings.
* **cutout_enabled** : bool
    If True then run the ABI cutput tool
* **cutout_annotations** : bool
    If True then annotate the ABI cutout figures
* **cutout_G16ABICDIR** : str
* **cutout_G16ABIFDIR** : str
* **cutout_G17ABICDIR** : str
* **cutout_G17ABIFDIR** : str
* **cutout_G18ABICDIR** : str
* **cutout_G18ABIFDIR** : str
    The path to the various ABI data files used by the cutout tool.
* **cutout_coastlines_path** : str
    Path to the coastlines data file for outlining the coasts in the cutout figures.
* **cutout_n_cores** : int
    Number of cores to use to parallelize the cutout figures process. Parallelizes over detection candidates, 
    uses `min_GB_cutout_tool` to determine how many parallel jobs to run.
    Set to None to use `n_cores` above. 
* **cutout_generate_multi_band_data** : bool
    If True, then generate multi-band data arrays and stored to file. THis is used for CNN training.
    These files are very large, so only enable if they will be used.
* **cutout_plot_seperate_figures** : bool
    If True then do not just generate the summary single figure. Instead generate separate figures in addition to the summary single figure.
    This is used to generate the figures for automated ML validation (not human vetting).
* **cutout_plot_glm_circle_size** : float
    Radius of circle used to mark GLM data in GLM and ABI composite figure. Only applicable if `cutout_plot_seperate_figures == True`.
    100.00 makes a decent size circle. 2.5 makes a dot.
* **cutout_bands_to_read** : tuple
    List of ABi bands to read out and use when generating the multi_band_data dict containing ABI cutouts with GLM data
    superimposed. This data is not used for the cutout figures by humans, but by machine learning algorithms to identify
    bolides.


Running
-------
If everything above has been a boring slog then feel happy that after setting up the configuration files you can now run the pipeline! 
The pipeline is run with the following command::

    run_glm_pipeline_batch.py top_level_pipeline_config_file.txt

The pipeline is designed to run on each satellite GLM separately (GOES-16, GOES-17 and GOES-18). 
To process data from all satellites, run the pipeline
multiple times, with separate top level configuration files. The `elemental_config_file.txt` configuration file can be
the same or different for multiple runs, however, typically a different trained classifier is used for each satellite.
In the ``outputDir`` a file named ``processing_history.txt`` is created or
appended after each run.  This file is used to keep track of which days have been processed. The pipeline will
only process days that have not yet been listed in this file. If you want to reprocess a day, simply remove
the corresponding entry from ``processing_history.txt``, or, set ``forceReprocess = True``.

**Note:** For daily processing make sure **doNotProcessCurrentDay** is set to True so that the partially
downloaded current day is not processed. Otherwise, the pipeline will log the day complete and never go back to process
the rest of the day!

Parallel Processing
-------------------

When ``multiProcessEnabled = True`` then `run_glm_pipeline_batch` first checks to see what data needs to be processed and
sets up the directory structure for each day's processing. The program then uses `qsub` and GNU parallel to farm out each day's
processing on the NAS. The final aggregation step combines all days' processing into a summary database file located at outputDir. 
This last aggregation job
is also submitted with `qsub`, but told to `halt` until the main batch job is finished.
The NAS PBS configuration for the batch jobs are in the files located at ``$ATAP_REPO_ROOT/code/GLM_bolide_detection/batch/``:

1. **gnu_parallel_script.pbs**
    How to farm out each day's data on the NAS
2. **aggregator_batch_script.pbs**
    How to run the final aggregation job.

The aggregation step is very fast so it can use a single lesser Pleiades node. 
The GNU parallel processing script
file is generally set up by default to process all historical data over many compute nodes. 
If doing short runs, or testing, then it is recommended to limit the number
of requested nodes, CPU model and wall time to shorten the wait time in the queue.
More information about submitted batch jobs using PBS on the NAS can be found
`here <https://www.nas.nasa.gov/hecc/support/kb/running-jobs-with-pbs-121/>`_.

Pipeline Results
----------------
After the pipeline has completed, see the section :ref:`examining_the_results<examining_the_results>` for how to interpet and utilize the results.

