# This module is used to generate the bolide detection validation report used by final vetters before declaring a bolide.

import os
import signal
import glob
from PIL import Image
from subprocess import run, Popen
import numpy as np
from traceback import print_exc

import io_utilities as ioUtil
from generate_pdf_report import generate_pdf_report

def generate_detection_validation_report(
        bolideDetectionList, 
        detectionFigurePath, 
        outputPath, 
        createSymlinks=True, 
        verbosity=True,
        copy_indiv_files=True):
    """ Generates a bolide detection Validation PDF Report.

    Comnbines the detection, cutout and post-processing figures into a single PDF document per bolide detection candidate.

    Parameters
    ----------
    bolideDetectionList : list of BolideDetection objects 
        The detected bolides to generate reports for
    detectionFigurePath : str
        Path to all the detection, cutout and post-processing figures in bolideDetectionList
        This is the full path to the individual files, not a top level path with subdirectories
    outputPath : str
        Path to save the generated PDF reports
    createSymlinks : bool
        When copying over the netCDF files and stereo detection figures, make a copy or create symlink
    verbosity : bool
        To be or not to be verbose
    copy_indiv_files : bool
        If True, then create a sudirectory called 'indiv_files' to copy over the individual files

    Returns
    -------
    PDF_filenames : str list
        List of the filenames, with full path, to the generated PDFs

    """

    if outputPath != detectionFigurePath:
        ioUtil.create_path(outputPath, verbosity)

    PDF_filenames = []
    for detection in bolideDetectionList:

        ID = detection.ID

        # Find all files associated with this detection
        filesThisID = glob.glob(os.path.join(detectionFigurePath, '*{}*.png'.format(ID)))
        
        # Get the detection figure
        detectionFileThisID = os.path.join(detectionFigurePath, detection.figureFilename)
        assert os.path.exists(detectionFileThisID), 'Found no detection figure for detection ID: {}, This should not be.'.format(ID)

        # Find the other satellite figure, if it exists
        if detection.figureFilenameOtherSatellite is not None:
            detectionOtherFileThisID = os.path.join(detectionFigurePath, detection.figureFilenameOtherSatellite)
            if not os.path.exists(detectionOtherFileThisID):
                detectionOtherFileThisID = None
        else:
            detectionOtherFileThisID = None


        # Get the cutput figure, if it exists
        cutoutFileThisID = glob.glob(os.path.join(detectionFigurePath, '*{}*ABI*.png'.format(ID)))
        if (len(cutoutFileThisID) > 1): 
            raise Exception('Found multiple cutout figures with same detection ID: {}'.format(ID))
        elif (len(cutoutFileThisID) == 1): 
            cutoutFileThisID = cutoutFileThisID[0]
        else:
            cutoutFileThisID = None

        # Generate the combined detection and cutout figure
        merged_filename = os.path.split(detectionFileThisID)[1]
        split_filename = os.path.splitext(merged_filename)
        merged_filename = os.path.join(detectionFigurePath,split_filename[0] + '_with_cutout' + split_filename[1])
        merge_images(detectionFileThisID, cutoutFileThisID, merged_filename)

        # Generate other satellite combined detection and cutout figure
        if detectionOtherFileThisID is not None:
            merged_filename_other = os.path.split(detectionOtherFileThisID)[1]
            split_filename = os.path.splitext(merged_filename_other)
            merged_filename_other = os.path.join(detectionFigurePath,split_filename[0] + '_with_cutout' + split_filename[1])
            merge_images(detectionOtherFileThisID, cutoutFileThisID, merged_filename_other)
        else:
            merged_filename_other = None

        # Generate detection validation report PDF
        # Filename includes the ID, lat, lon and time, and satellite
        lat, lon = detection.average_group_lat_lon
        # We want the time in ISO format, but not the colons because Mac OS finder GUI will replace the colons ':' with forward slashes '/'
        # So, replace 05:34:12.543 with 05h34m12.543
        time_in_ISO = detection.bolideTime.isoformat(sep='_', timespec='milliseconds')
        time_in_ISO = time_in_ISO.replace(':', 'h', 1)
        time_in_ISO = time_in_ISO.replace(':', 'm', 1)

        report_filename = detection.goesSatellite + '_' + time_in_ISO + '_' + \
                '{:.3f}'.format(lat) + '_' + '{:.3f}'.format(lon) + '_' + '{}'.format(ID) + '.pdf'
        report_full_filename = os.path.join(outputPath, report_filename)
        generate_pdf_report(detectionFigurePath, ID, report_full_filename)
        # Make report world-readable
        # Note: os.chmod expects mode to be an octal number, so prepend with '0o'
        os.chmod(report_full_filename, 0o644)

        # If outputPath is not the same as detectionFigurePath then copy or symlink over the .nc files.
        if copy_indiv_files:
            # Copy or symlink the netCDF files to different subdirectories for each detection
            netCDFFilesThisDetection = detection.filePathList
            if detection.bolideDetectionOtherSatellite is not None:
                netCDFFilesThisDetection.extend([os.path.basename(file) for file in detection.bolideDetectionOtherSatellite.filePathList])
            # Add in full path to files
            netCDFFilesThisDetection = [os.path.join(detectionFigurePath, file) for file in netCDFFilesThisDetection]
        
            # Copy all individual files that Randy likes to see into <outputPath>/indiv_files
            indiv_files_outputPath = os.path.join(outputPath, 'indiv_files')
            ioUtil.copy_or_create_symlinks(netCDFFilesThisDetection, indiv_files_outputPath, createSymlinks, verbosity)
            figureFilenameStereo = os.path.join(detectionFigurePath, detection.figureFilenameStereo)
            ioUtil.copy_or_create_symlinks(merged_filename, indiv_files_outputPath, createSymlinks, verbosity)
            ioUtil.copy_or_create_symlinks(merged_filename_other, indiv_files_outputPath, createSymlinks, verbosity)
            ioUtil.copy_or_create_symlinks(figureFilenameStereo, indiv_files_outputPath, createSymlinks, verbosity)

        PDF_filenames.append(report_full_filename)


    # Make sure all output directories and files are world readable
    if outputPath != detectionFigurePath:
        # Note: os.chmod expects mode to be an octal number, so prepend with '0o'
        if copy_indiv_files:
            os.chmod(indiv_files_outputPath , 0o755)
            for name in glob.glob(indiv_files_outputPath +'/*'):
                os.chmod(name, 0o644)
        try:
            os.chmod(outputPath , 0o755)
            os.chmod(os.path.dirname(outputPath), 0o755)
        except:
            pass

    if verbosity:
        print('Detection validation reports created and saved to {}'.format(outputPath))

    return PDF_filenames


#*************************************************************************************************************
def merge_images(fileLeft, fileRight, saveFilename):
    """ Merges two images into a single image and saves to file.

    Image2 is resized to that of image1.

    Parameters
    ----------
    fileLeft   :   str
        Filename for the file to be placed on the left
    fileRight   :   str
        Filename for the file to be placed on the right
        This image can be None, if so then a blank image is merged
    saveFilename : str
        Filename to save the merged figure

    Returns
    -------
    Just a saved file

    """

    # Read the two images
    image1 = Image.open(fileLeft)
    image1_size = image1.size

    if (fileRight is not None):
        image2 = Image.open(fileRight)
        # Resize image2 to be equal to image1
        image2 = image2.resize(image1_size)
        image2_size = image2.size
    else:
        image2 = None

    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    if image2 is not None:
        new_image.paste(image2,(image1_size[0],0))

    new_image.save(saveFilename,"PNG")

#*************************************************************************************************************
def display_report_for_these_IDs(
        bolideDetectionList,
        detectionFigureTopPath, 
        IDs='all',
        scores=None,
        tempOutputPath='/tmp/bolide_figures', 
        outputPath='/tmp/bolide_figures', 
        purge_temp_figs_at_exit=True,
        disposition_options=('TP', 'TN', 'FP', 'FN')
        ):
    """

    Displays a bolide validation report for a given ID. It will display the report via evince. Then ask for the user's disposition. 
    It will then copy the report to a folder bases on its disposition.
    Then close the window and move to the next.

    This must first generate the PDF, so a location must be given to store the file, this can be a temporary directory.

    Parameters
    ----------
    bolideDetectionList : list of BolideDetection objects 
        The detected bolides to generate reports for
    detectionFigureTopPath : str
        Path to all the detection, cutout and post-processing figures in bolideDetectionList
        This is the top level path with subdirectories like /G17/2022/1115/
    IDs : int list or str
        List of IDs to display
        'all' means display all in bolideDetectionList
    scores : np.array
        The detection classifer score to print for each report
    tempOutputPath : str
        Path to temporarily save all the generated PDF reports
    outputPath : str
        Path to save the generated PDF reports for those use requested to save
    purge_temp_figs_at_exit : bool
        If true, then purge the generated figures at tempOutputPath
    disposition_options : list of str
        List of possible dispositions entered by user

    Returns
    -------
    bolide_opinions : HumanOpinion list
        The dispositions of the human viewing the validation reports

    """
    # Placing this import here so we do not have a circular import
    from bolide_dispositions import HumanOpinion, humanDispositionError
    
    if isinstance(IDs, str) and IDs == 'all':
        IDs = [d.ID for d in bolideDetectionList]
        assert scores is None, "If IDs=='all' then cannot pass scores"


    all_pdf_filenames = []
    bolide_opinions = []
    display_counter = 0
    for bolide in bolideDetectionList:
        idx = np.nonzero(bolide.ID == IDs)[0]
        if np.isin(bolide.ID, IDs):
            display_counter += 1
            print('*****************')
            print('Showing candidate {} of {}'.format(display_counter, len(IDs)))
            print('Classifier score: {}'.format(scores[idx]))

            bolide_opinions.append(HumanOpinion(ID=bolide.ID, valid_dispositions=disposition_options))
            
            # Generate the PDF report
            MMDD    = str(bolide.bolideTime.month).zfill(2) + str(bolide.bolideTime.day).zfill(2)
            detectionFigurePath = os.path.join(detectionFigureTopPath, bolide.goesSatellite, str(bolide.bolideTime.year), MMDD)

            PDF_filename = generate_detection_validation_report([bolide], detectionFigurePath, 
                tempOutputPath, verbosity=False, copy_indiv_files=False)

            all_pdf_filenames.append(PDF_filename[0])

            # Display the report
            # Spawn a process without waiting for it to exit
            result = Popen(['evince', PDF_filename[0]])

            # Record the disposition (TP, FP, TN, FN)
            # Keep tryign until a correct disposition is passed
            keep_trying = True
            while keep_trying:
                try:
                    usr_disposition = input('Disposition {}:'.format(bolide_opinions[-1].valid_dispositions))
                    bolide_opinions[-1].disposition = usr_disposition
                except humanDispositionError:
                    print('Error entering human opinion disposition. Try again...')
                   #print_exc()
                else:
                    keep_trying=False
                    # Copy to folder bases on disposition
                    bin_output_path = os.path.join(outputPath, bolide_opinions[-1].disposition)
                    ioUtil.copy_or_create_symlinks(PDF_filename[0], bin_output_path, createSymlinks=False, verbosity=True)
                   #usr_requests_save = input('Save this figure [S] or just move to next [enter]')
                   #if usr_requests_save == 'S' and outputPath is not None:
                   #    # Save to outputPath
                   #    ioUtil.copy_or_create_symlinks(PDF_filename[0], outputPath, createSymlinks=False, verbosity=True)
            # Close the PDF viewer before moving to the next
            os.kill(result.pid, signal.SIGTERM)


    if purge_temp_figs_at_exit:
        for filename in all_pdf_filenames:
            os.remove(filename)



    return bolide_opinions

