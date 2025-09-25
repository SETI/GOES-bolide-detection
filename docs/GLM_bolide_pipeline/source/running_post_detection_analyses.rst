.. _running_post_detection_analysis:

.. sectionauthor:: Robert L. Morris <robert.l.morris@nasa.gov>

Running Post-Detection Analysis
===============================
Follow-up analysis include clustering to identify impact-triggered pixel events, registration of L0 and L2 events, reconstruction or bounding of 14-bit pixel values, recalibration, and light curve and ground track estimation. For each detection some or all of these analyses may be run, depending on the configuration (see :ref:`post-processing_configuration_file` below). Results are saved in either Matlab or JSON format and a complement of diagnostic figures is generated. Results are saved in a directory hierarchy - organized by GOES number, year and day - under the specified output directory.

The post-detection analysis can be run in the detection pipeline if `post_process_enabled == True` in the elemental configuration file. See :ref:`elemental_config_file` for details.

.. _compiling_post_detection_software:

Compiling and Running the Code
------------------------------
Assuming you're working interactively on a NAS Pleiades compute node or PFE, :ref:`check out the ATAP git repository <source_code_repo>` under ``${HOME}/git/atap`` and compile the code by issuing the following commands from the bash prompt: ::

    $ export ATAP_REPO_ROOT=${HOME}/git/atap
    $ module load matlab/2021a
    $ matlab -batch "run $ATAP_REPO_ROOT/code/GLM_bolide_analysis/system/matlab/compile_bolide_analysis_code"

Note that if not on the NAS then you will need a Matlab compiler license and all needed tool box licenses. Also note that when compiling Matlab code, the
`startup.m` file is integrated with the compiled executable. If, for example, you set `dbstop if error` in your `startup.m` then you will get a warning about
debugging not available every time you call the executable.

You must also compile the level-0 reader before running the analysis: ::

    $ pushd $ATAP_REPO_ROOT/code/utilities/L0_reader
    $ make
    $ popd

To run the code, ::

    $ export CFG_FILE=/path/to/configuration/file
    $ export DET_FILE=/path/to/detection/CSV/file
    $ export OUT_DIR=/path/to/output/directory
    $ $ATAP_REPO_ROOT/code/GLM_bolide_analysis/system/sh/run_bolide_analysis.sh $CFG_FILE $DET_FILE $OUT_DIR

where ``DET_FILE`` is a pipeline detection CSV file produced by the :ref:`ATAP detection pipeline<running_the_pipeline>`, and ``CFG_FILE`` is an ASCII text file described in detail below. The output directory ``OUT_DIR`` can be any directory for which you have write permissions. Specific detections can be processed by appending a list of detection IDs to the ``run_bolide_analysis.sh`` command. If desired, text output can be suppressed with a ``-q`` option. Run ``run_bolide_analysis.sh`` without arguments to see a usage synopsis.

**Exit Status** : The run_bolide_analysis.sh script will initiate a subprocess for each detection in the detection file. When all subprocesses have completed, the script terminates with exit code equal to the number of failed subprocesses, up to 255. The exit status is stil 255 if the number of failed subprocesses is greater than 255.

Avoiding Out-of-Memory Errors
------------------------------
When processing detection files containing many detections there is a risk of one or more processes failing due to insufficient memory. When processing on Pleiades you may have as little as 32 GB or as much as 512 GB, depending on the node type. An additional script (``partition_and_run_bolide_analysis.sh``) is provided that will ensure you are processing no more than ``n`` detections at a time. Use it in place of ``run_bolide_analysis.sh`` if out-of-memory errors are a concern. To ensure that no more than 10 detections at a time are processed (a safe limit on broadwell nodes sporting 128 GB), run the code as follows: ::

    $ $ATAP_REPO_ROOT/code/GLM_bolide_analysis/system/sh/partition_and_run_bolide_analysis.sh -n 10 $CFG_FILE $DET_FILE $OUT_DIR

where ``CFG_FILE``, ``DET_FILE`` and ``OUT_DIR`` are the same as in the previous section. As with ``run_bolide_analysis.sh``, the exit code reflects the number of failed subprocesses, up to 255.

.. _post-processing_configuration_file:

Configuration File
------------------
The configuration file specified by ``CFG_FILE`` is an ASCII text file containing the definition of a configuration data structure. Each line in the file must contain a comment, a field definition, or whitespace. Lines beginning with ``#`` are interpreted as comments. Lines containing field definitions must have the format ``<struct_field> = <value>``, where ``<struct_field>`` is a series of dot-separated names starting with the name ``configStruct``, as in the following example: ::

    # -----------------------------------------------------------------------------
    # GLM data directories
    # -----------------------------------------------------------------------------
    configStruct.g16L0DataDir = /nex/datapool/geonex/internal/GOES16/NOAA-L0/GLM-L0
    configStruct.g16L2DataDir = /nex/datapool/geonex/public/GOES16/NOAA-L2/GLM-L2-LCFA
    configStruct.g17L0DataDir = /nex/datapool/geonex/internal/GOES17/NOAA-L0/GLM-L0
    configStruct.g17L2DataDir = /nex/datapool/geonex/public/GOES17/NOAA-L2/GLM-L2-LCFA
    configStruct.g18L0DataDir = /nex/datapool/geonex/internal/GOES18/NOAA-L0/GLM-L0
    configStruct.g18L2DataDir = /nex/datapool/geonex/public/GOES18/NOAA-L2/GLM-L2-LCFA
    configStruct.g19L0DataDir = /nex/datapool/geonex/internal/GOES19/NOAA-L0/GLM-L0
    configStruct.g19L2DataDir = /nex/datapool/geonex/public/GOES19/NOAA-L2/GLM-L2-LCFA
    
    configStruct.L0SubdirFormat = MMDD
    configStruct.L2SubdirFormat = DOY

    # -----------------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------------
    configStruct.bashScriptPath = ATAP_REPO_ROOT/code/navigation/sh/locate_L0_file.sh
    configStruct.L0ReaderExec   = ATAP_REPO_ROOT/system/build/read_GLM_L0
    configStruct.h5dumpPath     = 
    configStruct.ncgenPath      = 

    # -----------------------------------------------------------------------------
    # Supporting data
    # -----------------------------------------------------------------------------
    configStruct.navigationStruct.g16UprightFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/navigation_lookup_tables/2023-11-16-000000_v0_G16_nav_LUT.mat
    configStruct.navigationStruct.g17UprightFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/navigation_lookup_tables/2023-11-16-000000_v0_G17_nav_LUT.mat
    configStruct.navigationStruct.g18UprightFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/navigation_lookup_tables/2023-11-16-000000_v0_G18_nav_LUT.mat
    configStruct.navigationStruct.g19UprightFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/navigation_lookup_tables/2023-11-16-000000_v0_G16_nav_LUT.mat
    configStruct.navigationStruct.g16InvertedFile = 
    configStruct.navigationStruct.g17InvertedFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/navigation_lookup_tables/2023-11-16-000000_v0_G17_nav_LUT_inverted.mat
    configStruct.navigationStruct.g18InvertedFile = 
    configStruct.navigationStruct.g19InvertedFile = 

    configStruct.calibrationParams.bbCalibrationTableFile   = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/calibration/2025-04-16-000000_v0_bb_calibration_LUT.mat
    configStruct.calibrationParams.lineCalibrationTableFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/calibration/2025-04-16-000000_v0_lightning_calibration_LUT.mat
    configStruct.calibrationParams.g16DarkFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/calibration/2023-11-16-000000_v0_dark_image_struct_G16.mat
    configStruct.calibrationParams.g17DarkFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/calibration/2023-11-16-000000_v0_dark_image_struct_G17.mat
    configStruct.calibrationParams.g18DarkFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/calibration/2023-11-16-000000_v0_dark_image_struct_G18.mat
    configStruct.calibrationParams.g19DarkFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/calibration/2025-04-17-000000_v0_dark_image_struct_G19.mat
    configStruct.dataBooks.directory = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/calibration
    configStruct.dataBooks.g16File = 2023-11-16-000000_v0_GLM_Data_Book_FM1.h5
    configStruct.dataBooks.g17File = 2023-11-16-000000_v0_GLM_Data_Book_FM2.h5
    configStruct.dataBooks.g18File = 2023-11-16-000000_v0_GLM_Data_Book_FM3.h5
    configStruct.dataBooks.g19File = 2025-04-16-000000_v0_GLM_Data_Book_FM4.h5

    configStruct.qualityAssessmentParams.assessmentModelFile = /nobackupp17/rlmorri5/GLM/data/ancillary/photometry/validation/2023-11-16-000000_v0_assessment_model.mat
    configStruct.qualityAssessmentParams.modelName = assessmentModel

    # -----------------------------------------------------------------------------
    # Process control flags
    # -----------------------------------------------------------------------------
    configStruct.controlFlags.gpaProcessingEnabled          = 0
    configStruct.controlFlags.pixelReconstructionEnabled    = 1
    configStruct.controlFlags.pruneSaturatedEventsEnabled   = 1
    configStruct.controlFlags.bgEstimationEnabled           = 1
    configStruct.controlFlags.bgSubtractUncalibratedEnabled = 1
    configStruct.controlFlags.calibrationEnabled            = 1
    configStruct.controlFlags.lightCurveEstimationEnabled   = 0
    configStruct.controlFlags.qualityAssessmentEnabled      = 0
    configStruct.controlFlags.groundTrackEstimationEnabled  = 0 

    # -----------------------------------------------------------------------------
    # Output control parameters
    # -----------------------------------------------------------------------------
    configStruct.outputControlStruct.saveFiguresEnabled = 1
    configStruct.outputControlStruct.saveDataConfThresh = 0.4
    configStruct.outputControlStruct.outputFileFormat   = json
    configStruct.outputControlStruct.useSubDirs         = 0
    configStruct.outputControlStruct.warningsEnabled    = 1
    configStruct.outputControlStruct.verbosity          = 2

    # -----------------------------------------------------------------------------
    # Known errors
    # -----------------------------------------------------------------------------
    configStruct.knownErrors.a = No L0 events within the plausible region
    configStruct.knownErrors.b = Dot indexing is not supported for variables of this type

    # -----------------------------------------------------------------------------
    # Additional parameters
    # -----------------------------------------------------------------------------

This example contains all the required fields. Any additional fields specified in the configuration file will be added to the ``configStruct`` and will be ignored if they are not recognized. Any instances of ATAP_REPO_ROOT will be replaced with the value of the ATAP_REPO_ROOT environment variable during processing. Note that in this example the G16 navigation table is used for G19. This is not an error. Lockheed was unable to produce a table of noinal pixel locations due to bugetary constraints and we don't currently have sufficient G19 data to generate a high-quality one ourselves. The G16 table should be close enough for our purposes.


Data Directories
----------------
These are the directories under which L0 and L2 GLM data can be found for each of the GOES GLM instruments. Different naming conventions in the daily subdirectories are accommodated by the fields ``L0SubdirFormat`` and ``L2SubdirFormat``, which must be specified as either ``MMDD`` (4-digit month and day) or ``DOY`` (3-digit day of year).

**configStruct.g16L0DataDir**
    The directory under which L0 files for G16 can be found. Files are assumed to be organized in subdirectories by year and day in the format ``g16L0DataDir/YYYY/L0SubdirFormat``.

**configStruct.g16L2DataDir**
    The directory under which L2 files for G16 can be found. Files are assumed to be organized in subdirectories by year and day in the format ``g16L2DataDir/YYYY/L2SubdirFormat``.

**configStruct.g17L0DataDir**
    The directory under which L0 files for G17 can be found. Files are assumed to be organized in subdirectories by year and day in the format ``g17L0DataDir/YYYY/L0SubdirFormat``.

**configStruct.g17L2DataDir**
    The directory under which L2 files for G17 can be found. Files are assumed to be organized in subdirectories by year and day in the format ``g17L0DataDir/YYYY/L2SubdirFormat``.

**configStruct.g18L0DataDir**
    The directory under which L0 files for G18 can be found. Files are assumed to be organized in subdirectories by year and day in the format ``g18L0DataDir/YYYY/L0SubdirFormat``.

**configStruct.g18L2DataDir**
    The directory under which L2 files for G18 can be found. Files are assumed to be organized in subdirectories by year and day in the format ``g18L0DataDir/YYYY/L2SubdirFormat``.
    
**configStruct.g19L0DataDir**
    The directory under which L0 files for G19 can be found. Files are assumed to be organized in subdirectories by year and day in the format ``g19L0DataDir/YYYY/L0SubdirFormat``.

**configStruct.g19L2DataDir**
    The directory under which L2 files for G19 can be found. Files are assumed to be organized in subdirectories by year and day in the format ``g19L0DataDir/YYYY/L2SubdirFormat``.
    
**configStruct.L0SubdirFormat**
    The format of the L0 daily directories (either ``MMDD`` or ``DOY``).

**configStruct.L2SubdirFormat**
    The format of the L2 daily directories (either ``MMDD`` or ``DOY``).

Dependencies
------------
These are the shell scripts and binary executables that the software requires to perform its analyses:

**configStruct.bashScriptPath**
    The full path to the shell script used to search the L0 data directory for files containing data within the specified time range.

**configStruct.L0ReaderExec**
    The full path to the executable L0 reader (see compilation notes above).

**configStruct.h5dumpPath**
    The full path to the ``h5dump`` utility. If left blank, as in the example above, the software will attempt to locate the utility automatically.

**configStruct.ncgenPath**
    The full path to the ``ncgen`` utility. If left blank, as in the example above, the software will attempt to locate the utility automatically.


Data Files
----------
**configStruct.navigationStruct.g16UprightFile**

    The file containing an approximate navigation lookup table for G16 when in the upright orientation.

**configStruct.navigationStruct.g17UprightFile**

    The file containing an approximate navigation lookup table for G17 when in the upright orientation.

**configStruct.navigationStruct.g18UprightFile**

    The file containing an approximate navigation lookup table for G18 when in the upright orientation.

**configStruct.navigationStruct.g19UprightFile**

    The file containing an approximate navigation lookup table for G19 when in the upright orientation.

**configStruct.navigationStruct.g16InvertedFile**

    This is currently a placeholder parameter, since G16 has never collected data in the inverted orientation.

**configStruct.navigationStruct.g17InvertedFile**

    The file containing an approximate navigation lookup table for G17 when in the inverted orientation.

**configStruct.navigationStruct.g18InvertedFile**

    This is currently a placeholder parameter, since G18 has never collected data in the inverted orientation.

**configStruct.navigationStruct.g19InvertedFile**

    This is currently a placeholder parameter, since G19 has never collected data in the inverted orientation.

**configStruct.calibrationParams.bbCalibrationTableFile**

    The full path to the file containing calibration gain tables for G16, G17, G18, and G19 in units of Joules/DN. 

**configStruct.calibrationParams.lineCalibrationTableFile**

    The full path to the file containing lightning gain tables for G16, G17, G18, and G19 in units of Joules/DN. 

**configStruct.calibrationParams.g16DarkFile**

    The full path to a file containing a dark model for G16. 

**configStruct.calibrationParams.g17DarkFile**

    The full path to a file containing a dark model for G17. 

**configStruct.calibrationParams.g18DarkFile**

    The full path to a file containing a dark model for G18.

**configStruct.calibrationParams.g19DarkFile**

    The full path to a file containing a dark model for G19.

**configStruct.dataBooks.directory**

    The directory containing FM1, FM2, FM3, and FM4 databooks.

**configStruct.dataBooks.g16File**

    The name of the file containing the Flight Model 1 (G16) databooks. 

**configStruct.dataBooks.g17File**

    The name of the file containing the Flight Model 2 (G17) databooks. 

**configStruct.dataBooks.g18File**

    The name of the file containing the Flight Model 3 (G18) databooks.

**configStruct.dataBooks.g19File**

    The name of the file containing the Flight Model 4 (G19) databooks.

**configStruct.qualityAssessmentParams.assessmentModelFile**

    The full path to a file containing a quality assessment model, used for validating the results of light curve estimation.
 


Process Control Flags
---------------------
These binary flags control the processing performed and the contents of the output files. Some options depend on others, as detailed below. Effectively, the value of each flag is logically OR-ed with its prerequisites before processing begins.

**configStruct.controlFlags.gpaProcessingEnabled**

    If true (1), apply the overshoot and crosstalk ground processing algorithms to the L0 data.

**configStruct.controlFlags.pixelReconstructionEnabled**

    If true (1), perform the 14-bit background reconstruction and bounding for each pixel in the L0 data set.

**configStruct.controlFlags.pruneSaturatedEventsEnabled**

    If true (1), saturating events are pruned from the results. Requires that ``pixelReconstructionEnabled = true``.

**configStruct.controlFlags.bgEstimationEnabled**

    If true (1), background levels are reestiamted. Requires that ``pixelReconstructionEnabled = true``.

**configStruct.controlFlags.bgSubtractUncalibratedEnabled**

    If true (1), the re-estiamted background levels are subtracted from the reconstructed 14-bit pixel values. 

**configStruct.controlFlags.calibrationEnabled**

    If true (1), compute the calibrated bounds of each observation in units of Joules. Requires that ``bgEstimationEnabled = true``.

**configStruct.controlFlags.lightCurveEstimationEnabled**

    If true (1), estimate the impact event's light curve. Requires that ``calibrationEnabled = true``.

**configStruct.controlFlags.qualityAssessmentEnabled**

    If true (1), use the specified (by configStruct.qualityAssessmentParams) quality assessment model to validate the resulting light curve. Requires that ``lightCurveEstimationEnabled = true``.

**configStruct.controlFlags.groundTrackEstimationEnabled**

    If true (1), produce a ground track estimate with propagated uncertainties. Requires that ``lightCurveEstimationEnabled = true``.

Output Control Parameters
-------------------------
These parameters control aspects of the system output such as data file formats, figure generation, and the the verbosity of messages delivered to ``stdout`` and ``stderr``.

**configStruct.outputControlStruct.saveFiguresEnabled**

    If true (1), generate and save diagnostic figures.

**configStruct.outputControlStruct.saveDataConfThresh**

    Save the data to a file only if the detection confidence is greater than this value (all data will be saved if set to 0).

**configStruct.outputControlStruct.outputFileFormat**

    Specifies the output file format (either ``mat`` or ``json``).

**configStruct.outputControlStruct.useSubDirs**

    If false (0), all outputs (data files, figures, and error logs) will be written to $OUT_DIR. If true (1), figures and data files will be written to daily subdirectories under $OUT_DIR. In either case error logs are written to $OUT_DIR.

**configStruct.outputControlStruct.warningsEnabled**

    If false (0), warning messages will be suppresed.

**configStruct.outputControlStruct.verbosity**

    A numeric value specifying the degree of verbosity in messages printed to ``stdout`` and ``stderr``. 

        =====  =======================================================================
        Value  Meaning
        =====  =======================================================================
        0      print only error messages and warnings (if enabled)                    
        1      print alerts and any messages that may indicate off-nominal behavior   
        2      print event handler messages (e.g., when object properties are updated)
        3      print messages reporting on progress of the analyses                  
        4      display data that may be useful in debugging                         
        =====  =======================================================================

Known Errors
------------
The user may add any number of fields to the struct ``knownErrors``. Field names and values are user-defined. The value of each field should contain a string that uniquely identifies an error message produced by a known error (e.g., ``No L0 events within the plausible region``). If an error message contains any of these strings, the process that produced it will exit normally with exit code 0. Otherwise, it will produce a non-zero exit code.
 
**configStruct.knownErrors**

    A structure containing an arbitrary number of user-defined fields.

Additional Parameters
---------------------
There are many additional parameters that can be passed through the configuration file, but these are best left at their default settings.


