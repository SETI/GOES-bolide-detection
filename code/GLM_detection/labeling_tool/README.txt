06/20/2019

Before proceeding, take a moment to set up your ATAP environment if you haven't
already done so. Details can be found here:

https://babelfish.arc.nasa.gov/confluence/display/ATAP/ATAP+Development+Environment+Setup

Tools for format conversion and data labeling
---------------------------------------------
This directory contains two executable tools to be used in conjunction with 
the current (as of this date) python bolide detector. The first is 
convert_detection_file.py, which takes an output file from detect_bolides() and
converts the data structures to a new format. The second, glm_labeling_gui.py,
is a graphical user interface (GUI) that facilitates labeling of detection 
results by a human expert.

From the shell prompt, you can get a usage summary for either tool with the 
-h option:

[shell prompt] $ convert_detection_file.py -h
[shell prompt] $ glm_labeling_gui.py -h


-------------------------------------------------------------------------------
Usage summary for convert_detection_file.py
-------------------------------------------------------------------------------
usage: convert_detection_file.py [-h] [-n MAXNUMTOCONVERT] infile outfile

Convert pickled Bolide objects to bolideDetection objects and re-pickle.

positional arguments:
  infile              Input pickle file containing Bolide objects.
  outfile             Output pickle file to which bolideDetection objects will
                      be written.

optional arguments:
  -h, --help          show this help message and exit
  -n MAXNUMTOCONVERT  The maximum number of object to convert.


NOTE that when running the converter it will append bolide detections to 
outfile. If you repeat the operation and want to start fresh, be sure to delete 
the existing file first.


-------------------------------------------------------------------------------
Usage summary for glm_labeling_gui.py
-------------------------------------------------------------------------------
usage: glm_labeling_gui.py [-h] [--name NAME] [--expertise EXPERTISE] [--csv]
                           [--scale_levels LEVELS] [--slider]
                           infile outbase

Label bolide detections.

positional arguments:
  infile                Input file name
  outbase               Output file base name (extensions are ignored)

optional arguments:
  -h, --help            show this help message and exit
  --name NAME, -n NAME  The name of the human user (default: None)
  --expertise EXPERTISE, -e EXPERTISE
                        The user's level of expertise in the interval [0,1]
                        (default: None)
  --csv, -c             Indicates infile is a CSV file (default: False)
  --scale_levels LEVELS, -l LEVELS
                        The integer number of confidence levels (default: 3)
  --slider, -s          Use a slider instead of radio buttons (default: False)

In addition to the command-line usage above, you may want to make use of the 
following key bindings when operating the GUI:

Key		Function
---		--------
<Left Arrow>	Go to the previous detection.
<Right Arrow>	Go to the next detection.
'e'		Edit the comment for the current detection.
's'		Save
'q'		Quit without saving.

-------------------------------------------------------------------------------
Processing a Data Set
-------------------------------------------------------------------------------

(0) Set environment variables and current working directory. ATAP_REPO_ROOT
    should already be set. You'll probably want to just type explicit path
    names, but we use variables L2_NC_FILE_DIR and OUTPUT_DIR here for
    generality.

    [shell prompt] $ export L2_NC_FILE_DIR=/directory/containing/nc/files/
    [shell prompt] $ export OUTPUT_DIR=/directory/where/results/go/

    [shell prompt] $ cd ${ATAP_REPO_ROOT}/atap/code/experimental/GLM/L2/labeling_tool

(1) Process all .nc files in the directory L2_NC_FILE_DIR using 
    parameters in my_config.txt and write the results to a pickle file named
    bolide-detection-results-2019-04-18.p in OUTPUT_DIR. If the configuration 
    file specifies 'generatePlots=True', then a diagnostic figure for each 
    detection is written to a PNG file in the specified output directory.:

    [shell prompt] $ detect_bolides.py -i $L2_NC_FILE_DIR \
		     -o ${OUTPUT_DIR}/bolide-detection-results-2019-04-18.p \
                     -f my_config.txt

(2) Convert bolide detection objects to the new format :

    [shell prompt] $ convert_detection_file.py \
                     ${OUTPUT_DIR}/bolide-detection-results-2019-04-18.p \
                     ${OUTPUT_DIR}/converted_detections.p

(3) Label the bolide detections with the opinions of a human expert. Write the
    results to bolide_detection_profiles.p and bolide_detection_profiles.csv:

    [shell prompt] $ glm_labeling_gui.py --name "Randy" --expertise 0.9
                     ${OUTPUT_DIR}/converted_detections.p ./bolide_detection_profiles
