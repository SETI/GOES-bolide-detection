#!/usr/bin/env python

# This module is used to combine the bolide detection and post-processing images into a single PDF report.
# It is to be used in conjunction with generate_detecton_validation_report.py

from fpdf import FPDF
from PIL import Image
import sys
import glob
import os
import ntpath # This should work on all platforms.
from itertools import compress



class PDF(FPDF):

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')


# ******************************************************************************
# Generate a PDF report
# ******************************************************************************
def generate_pdf_report(imageDirectory, detectionId, outputPath):
    """ Generates a PDF detection validation report.

    Parameters
    ----------
    imageDirectory : str
        Path to the individual files to combine into the PDF report
    detectionId : int64
        Detection ID to generate report for. Searches for relevent figures in imageDirectory.
    outputPath : str
        Path and filename to save PDF detection validation report to

    Returns
    -------
    PDF reports saved to outputPath

    """

    extensions = ('*.jpg','*.png','*.gif')
    textHeight  = 40
    spaceHeight = 0
    fontSize    = 9
        
    pdf = PDF('P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False, margin = 0.0)
    pdf.set_font('Courier', 'B', fontSize) # Arial, bold, 12 pt.

    imageList=[]
    for ext in extensions:
        imageList.extend(glob.glob(os.path.join(imageDirectory, ext)))

    # The substrings list determines which figures to add to the document and the order. The first file listed is first,
    # etc... Give the search string to use to find each figure. If a figure file is nto found. it is skipped.
    substrings = [str(detectionId) + '_detection_with_cutout',                   \
                  str(detectionId) + '_otherSatellite_detection_with_cutout',    \
                  str(detectionId) + '_stereo',                        \
                  'vetting_panel_' + str(detectionId),                 \
                  'L2_L0_registration_' + str(detectionId),            \
                  'max_energy_pixel_with_context_' + str(detectionId), \
                  'highest_energy_pixels_' + str(detectionId),         \
                  'onboard_14bit_' + str(detectionId),                 \
                  'detected_L2_vs_extracted_L0_' + str(detectionId)]
    captions = ['Left:  Pipeline detection results.\nRight: ABI cutouts.', \
                'Left:  Other Satellite pipeline detection results.\nRight: ABI cutouts (from detection).', \
                'Stereo detection analysis.', \
                'Top Left:                  Sum of reconstructed and background-subtracted 14-bit pixel values.\nTop Right:                 CCD locations and times of pixel events.\nBottom Left (if present):  Estimated light curve.\nBottom Right (if present): Estimated ground track.', \
                'Top Left:    Sum of L2 event energies at each timestep (frame).\nBottom Left: Sum of registered L0 event intensities at each timestep (frame).\nRight:       Results of <lat, lon, time> point cloud registration, projected onto Lat/Lon plane.', \
                'Top Left:     CCD location of the detection.\nBottom Left:  Active pixels for this detection.\nTop Right:    Corrected (uncalibrated) intensities for the maximum energy pixel.\nBottom Right: Calibrated energies for the maximum energy pixel.', \
                'Results for pixels recording the most energy in descending order.', \
                'Results of onboard 14-bit intensity reconstruction for pixels recording the most energy in descending order. Red line represents the reconstructed 14-bit background. Blue line represents the sum of the reconstructed background (or the mean value, if exact reconstruction was not possible) and the returned 14-bit L0 intensity ', \
                'Pixel events determined to have been caused by the impact. Pixel events included in the pipeline\'s detection are circled in red. ']

    # Search for images and associate captions with each one found.
    imageList,indicatorList = _order_list_by_substrings(imageList, substrings)
    captions = list(compress(captions, indicatorList))

    pdfSize = {'P': {'w': 210, 'h': 297}, 'L': {'w': 297, 'h': 210}}
    
    n = 0
    for imageFile in imageList:
        
        imageFrameSize = {'P': {'w': 210, 'h': 297 - textHeight - spaceHeight}, 'L': {'w': 297, 'h': 210 - textHeight - spaceHeight}}

        orientation, x, y, width, height = _get_image_position_and_size(imageFile, imageFrameSize)
            
        # Add a new page and place the image
        pdf.add_page(orientation=orientation)
        pdf.image(imageFile, x, y, width, height)
        
        # Print the image file name.
        pdf.set_font('Courier', 'B', fontSize) # Arial, bold, 12 pt.
        pdf.set_y(-textHeight)
        pdf.cell(w=imageFrameSize[orientation]['w'] - 20, h=textHeight/4, txt=ntpath.basename(imageFile), ln=1, align='l', border='T')
 
        # Add space between iamge name and caption
        pdf.ln(0) 
    
        # Print the corresponding caption.
        pdf.set_font('Courier', '', fontSize) # Arial, bold, 12 pt.
        pdf.multi_cell(w=imageFrameSize[orientation]['w'] - 20, h=textHeight/8, txt=captions[n], border=None)
        n = n + 1
                      
    pdf.output(outputPath, 'F')

# ******************************************************************************
# Sort a list of names in the order indicated by 'substrings'. 
# Base file names (excluding directory names) are searched for matching 
# substrings and ordered accordingly. 
# ******************************************************************************
def _order_list_by_substrings(nameList, substrings):

    reorderedNameList = []
    indicatorList = [False] * len(substrings)
    n = 0
    for s in substrings :
        matching = [name for name in nameList if s in ntpath.basename(name)]
        reorderedNameList.extend(matching)
        if matching : 
            indicatorList[n] = True
        n = n + 1

    return reorderedNameList, indicatorList

# ******************************************************************************
# Determine position and sizing parameters for image display.
# Assumes a text box is at the bottom of the page.
# ******************************************************************************
def _get_image_position_and_size(imageFile, cellSize=0):
    cover = Image.open(imageFile)
    width, height = cover.size

    # convert pixel in mm with 1px=0.264583 mm
    imWidth, imHeight = float(width * 0.264583), float(height * 0.264583)

    # given we are working with A4 format size 
    if cellSize == 0:
        cellSize = {'P': {'w': 210, 'h': 297}, 'L': {'w': 297, 'h': 210}}

    # get page orientation from image size 
    orientation = 'P' if imWidth < imHeight else 'L'

    #  make sure image size is not greater than the pdf format size
    if imWidth / cellSize[orientation]['w'] > imHeight / cellSize[orientation]['h'] :
        width = cellSize[orientation]['w']
        height = 0 # Auto-adjust height, preserving aspect ratio.
        scaleFactor = width / imWidth
    else:
        width = 0 # Auto-adjust width, preserving aspect ratio.
        height = cellSize[orientation]['h']
        scaleFactor = height / imHeight
        
    # Position the images so they are approximately centered.
    if width == 0 : 
        x = (cellSize[orientation]['w'] - scaleFactor * imWidth) / 2.0
        y = 0
    else :
        x = 0
        y = (cellSize[orientation]['h'] - scaleFactor * imHeight) / 2.0

    return orientation, x, y, width, height


# ******************************************************************************
# Print help.
# ******************************************************************************
def _print_usage(programName):
    print('USAGE: python {} <image_directory> <detection_id> <output_file>'.format(programName))


# ******************************************************************************
# Command line functionality to call generate_pdf_report.
#
# ******************************************************************************
if __name__ == "__main__":

    # Make sure we're running Python 3.
    if sys.version_info.major < 3:
        raise Exception("Python 3.0 or higher is required")

    # Check arguments.
    if len(sys.argv[1:]) < 3 :
        _print_usage(sys.argv[0])
        sys.exit()

    # Get command line arguments.
    imageDirectory = sys.argv[1]
    detectionId    = sys.argv[2]
    outputFile     = sys.argv[3]

    # Check existence of directories.
    if not os.path.isdir(imageDirectory):
        print('Directory not found: {}'.format(imageDirectory))
        _print_usage(sys.argv[0])
        sys.exit()

    if not os.path.isdir( os.path.dirname(outputFile) ):
        print('Directory not found: {}'.format(os.path.dirname(outputFile)))
        _print_usage(sys.argv[0])
        sys.exit()

    # Call the report generating function.
    generate_pdf_report(imageDirectory, detectionId, outputFile)

# ************************************ EOF *************************************
