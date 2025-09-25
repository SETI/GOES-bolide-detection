.. _running_the_quick_pipeline:

.. sectionauthor::  Jeffrey C. Smith <jsmith@seti.org>

Running the Quick Pipeline for Rapid Results
============================================

The standard pipeline (See: :ref:`Running the GLM Detection Pipeline<running_the_pipeline>`) is a sophisticated and
versitile pipeline. But it is not ideal for a fast run to examine very specific data. One needs to set up the
configuration files and directories for each run. There are some scenarioes, however, where the standard pipeline is not
ideal. Here are three scenarios:

1. There is a known event and the pipeline did not generate a detection candidate. You wish to force the pipeline to
   generate a candidate for a very specific group of data and see the analysis and detection score. The alternative,
   without this quick pipeline tool, is to process an entire day and use a very low detection threshold, which will
   probably generate a very large number of candidates. The user would then have to be peruse this set for the very
   specific event under question. With the quick pipeline, the specific group of data to search is specified by a spatiotemporal box.

2. Process a single raw netCDF file, or a small number of files. This is similar to use case #1 above but where we point
   to very specific files to process, instead of a spatiotemporal box. 

3. We want to very quickly process a day of data without having to set up the configuration files and processing
   directory. For rapid analysis, we want to be able to use a set of configuration files, perhaps kept in a master
   location, and tell the pipeline to run on a day's worth, or some other set, of data.  The point is to very rapidly
   process some data without needing to set up the directories and configuration files.

To execute these scenarios, a quick pipeline tool has been created called `run_glm_pipeline_quick`. Running this tool
is very similar to the standard pipeline, but the confguration has been simplifed. The elemental configuration file is
the same and must
still be set up. Otherwise, the detection algorithm does not know how to operate. For bolide
detection see the section :ref:`on the elemental configuration file<elemental_config_file>`. But instead of using the
standard top-level configuration file (as discussed :ref:`here<top_level_config_file>`) we use a simplified quick
confgiuration file. An example quick pipeline configuration file is located at 
``$ATAP_REPO_ROOT/code/GLM_bolide_detection/batch/example_quick_run_config_file.txt``

This tool can also be used to force to identify very specific features in the data. The currently implemented method is
to identify streaks. Here, we specify the end points (in time, lat and lon) for a geodesic arc in a **streak_file** data
file. We also identify the
**streak_width** which is the width of the streak in kilometers.

If a **streak_file** is specified then the minimum and maximum lats and lon (minLat, maxLat, minLon, maxLon) are not
used, instead, read form the data file. However, the **minDat** and **maxDate** are still used to specify whcih days,
and times to search the data for the streaks.

When a spatiotemporal box is specified then only GLM group data within the box is considered.

Here is a descripton of the configuration parameters:

* **inputFilePath** : str
    Specify the input netCDF files for the detection satellite
    [str or list of str] You can specify:
    1. An entire directory path
    2. A single NetCDF file
    3. A list of individual files.
* **otherSatelliteFilePath** : str
    Specify the input netCDF files for the other satellite
    (Same options as **inputFilePath**)
* **outputDir** : str
    Path to where to store the output
* **elementalConfigFile** : str
    Path and name of file for the daily elemental configuration file
* **detectionType** : str
    What are we detecting in the GLM data?
    (e.g. 'bolides', 'gigantic_jets', 'streaks)


* *BEGIN Spatiotemporal Box*
* **minDate** : str
    Minimum date and time to search in ISO format
    E.g. 2022-11-05T02:26:12 (DO NOT USE A SPACE BETWEEN THE DATE AND TIME STRING)
* **maxDate** : str
* **minLat** : float
    Minimum Latitude to search in degrees.
* **maxLat** : float
* **minLon** : float
* **maxLon** : float
* *END Spatiotemporal Box*

* **explicit_cluster** : bool
    If True then make a single explcit cluster for each spatiotemporal box.
    Otherwise, use the standard clustering algorithm to find as many clisters as would normally be foudn in the
    spatiotemporal box.
* **streak_width** : float
    If finding streaks then thsi defines the width of the streak as a geodesic arc
    Specify in kilometers.
* **streak_file** : str
    Path to a pickel fiel containing a set of streaks (geodesic arcs) to plot find.

Running
=======

The quick pipeline does not allow for batch jobs (it sets `multiProcessEnabled = False`). It also forces a reprocessing (sets `forceReprocess = True`).

If the spatiotemporal box is specified then `classifier_threshold = 0.0`.

The pipeline is run with the following command::

    run_glm_pipeline_quick.py quick_run_config_file.txt


The output to the quick pipeline will be the same as the standard pipeline, as specified by the elemental configuration
file. After the quick pipeline has completed, see the section :ref:`examining_the_results<examining_the_results>` for how to interpret and utilize the results.
