.. _examining_the_results:

.. sectionauthor:: Jeffrey C. Smith <jsmith@seti.org>

Examining the Results of the Pipeline
=====================================

Once a pipeline run has finished the output directory (``outputDir`` path in the `top_level_pipeline_config_file.txt` file) will contain a series 
of files and folders. The top level will be one or more directories like `G**` which contains the out for each
satellite. Within each of these top level directories is:

* **bolide_database.fs**
    The ZODB database of all detections (and rejections if a ground truth run)
* **bolide_database.fs.index**
    Auxillary ZODB database file
* **bolide_database.fs.lock**
    ZODB lock file (to control access to the database)
* **bolide_database.fs.old**
    Old database file copy used by ZODB
* **bolide_database.fs.tmp**
    Temp file copy used by ZODB
* **bolide_detections.csv**
    Human readible CSV file of all detections in the database
* **bolide_rejections.csv**
    Human readible CSV file of all rejection in the database. This file is only created if generating a ground truth
    data set (I.e. when ``ground_truth_path`` is set in the configuration file).
* **processing_history.txt**
    List of days that were processed in this pipeline run. This is a cimuulatative tally. If the pieplein is run
    multipel times with `forceReprocess = False` then this file is not reset, nut accumulates the days processed.
* **<year>**
    One or more directories containing daily processing data and figures from the run. 
    If ``deleteDailyDirs==True`` then these do not exist.

For much analysis the only important file is ``bolide_database.fs``. This contains all the information needed to
analyse the detections. 

Other Figures in Output Directory
---------------------------------
If ``deleteDailyDirs!=True`` then there is an output directory created for each day of processing. In these daily directories there are numerous figures.

A file with no content is created signifying if processing was performed for that day with a name such as `PROCESSED_G16`, meaning the G16 pipeline was run for
that day. This is important because if processing occurs but there just happens to not be any detection candidates for that day, the directory would be empty,
so, it is not immediately clear if there were no detections or the processing has not happened yet. 

The main detection figures have names like:

* **`OR_GLM-L2-LCFA_G##_s<date>_e<date>_c<date>_<ID>_detection.png`** 
    Contains the main information used in the bolide detection.

Where <ID> is a unique Identification int64 number for each detection.
There are two other figures generated. One or more of these are created and always correspond to a single "main" figure
above.

* **`OR_GLM-L2-LCFA_G##_s<date>_e<date>_c<date>_<ID>_otherSatellite_detection.png`**
    Contains a "detection"-like figure for whatever group data is in the other satellite in the neighborhood of the
    detection.
* **`OR_GLM-L2-LCFA_G##_s<date>_e<date>_c<date>_<ID>_stereo.png`**
    Contains a "re-navigation" figure attempting to measure the altitude of the event in the stereo region.
* **<ID>_OR_GLM-L2-LCFA_G##_s<date>_e<date>_c<date>_<lat>_<lon>_ABI_forest.png**
    Contains the ABI cutout figure.

There are also a collection of post-processing figures. TODO for the author of this code: Provide descriptions of these figures.

Public Validation Reports
-------------------------

All figures associated for each detection candidate are combined into a single bolide detection validation report.
This allows for a rapid analysis of all relevent figures and data.
See the section on :ref:`validation of detections<validation_of_detections>` for details about these reports and how the
validation is performed.

Plotting Detections
-------------------

**Note:** the plotting tool below will not provide all information in the generated figures. Not all information is
stored in the ZODB database `.fs` file. The tool below will only plot what is avalable in the database. If you want to
view the full extent of all data available for plotting then refer to the validation reports.
See the section on :ref:`validation_of_detections` for details about these reports.

Individual detections can be seen either in the saved .png files in each day's processing directory, or generated
afterwards with the executable ``plot_bolide_detections.py``. You can get an overview of its syntax with the following
command:::

    $ plot_bolide_detections.py -h
    usage: plot_bolide_detections.py [-h] [--outdir OUTDIR] [--interactive] [--number-random NUMRANDOM] [--startDate STARTDATE] [--endDate ENDDATE]
                                 [--confidence-threshold CONFIDENCETHRESHOLD]
                                 inFiles [inFiles ...]

    Plot bolide detections.

    positional arguments:
      inFiles               One or more input file names
    
    optional arguments:
      -h, --help            show this help message and exit
      --outdir OUTDIR, -o OUTDIR
                            Output directory name (default is None (do not save))
      --interactive, -i     Display and Pause for each figure (default: False)
      --number-random NUMRANDOM, -n NUMRANDOM
                            Number of random detections to plot
      --startDate STARTDATE, -s STARTDATE
                            Start date to plot (ISO format)
      --endDate ENDDATE, -e ENDDATE
                            End date to plot (ISO format)
    --confidence-threshold CONFIDENCETHRESHOLD, -c CONFIDENCETHRESHOLD
                            Mininum confidence threshold


The inFile would be ``bolide_database.fs`` file. So, to plot all detections from 2020-11-12 through 2020-11-24 with a
detection confidence of 0.7 or greater and pause when viewing each detection (interactive mode) issue the following
command:::

    $ plot_bolide_detections.py -i -s 2020-11-12 -e 2020-11-24 -c 0.7 bolide_database.fs

If you wish to save the generated figures then specify an output directory to save the figures to.

Further Analysis of the Data
----------------------------

The data can be extensively analyzed using the functions and classes in the bolide_dispositions module located at 
``$ATAP_REPO_ROOT/code/GLM_bolide_detection/L2/bolide_detection/bolide_dispositions.py``. Discussing all the analysis
that can be performed is beyond the scope of this short document, but please peruse the module to see all the great
tools already available.

You can also use the piblic Bolide package avialable at https://github.com/SETI/bolides.
    
