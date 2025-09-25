#!/usr/bin/env python

import sys
#if running on headless nodes, e.g. devel queu
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#import matplotlib.image as mpimg
#from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as mticker
import numpy as np
import time
from datetime import datetime
#import cv2
import csv
#import math
#import glob
import fnmatch
from copy import deepcopy
import argparse
import os
import multiprocessing as mp
import warnings
from shapely.errors import ShapelyDeprecationWarning
import psutil
from traceback import print_exc
import cartopy.crs as ccrs
import pickle
import bz2

import io_utilities as ioUtil
import bolide_detections as bd

#cwd = os.getcwd()
#sys.path.append(cwd)
#from cpt_convert import loadCPT

import ABI_image_processing_functions as ipf
import file_functions as ff
import overlays

class cutoutConfiguration:
    """ This class contains the configuration and methods to generate the cutout figures.

    """

    # This is the date GOES-18 officially became GOES-WEST: 2023-01-04
    G18_IS_WEST_DATE = datetime(2023, 1, 4)

    # This is the date GOES-16 officially became GOES-East: 2025-04-04
    G19_IS_EAST_DATE = datetime(2024, 4, 4)

    # Default parameters

    # Data source options: {'geonex', 'daily_bundle', 'files'}
    data_source = 'geonex'

    # This is the ABI bands to read
    bands_to_read = None

    # Do we annotate the cutout figures with lat/lon text
    annotations = True
    # Number of cores to parallelize the cutout generation
    n_cores = 1
    # Minimum amount of memory in GB per thread
    min_GB = 15.0

    # Target ABI band to use for vetting image or seperate figures.
    # Note that the band index is the index for the quasi-GeoColor image. So, for daytime, that is ABI Band 2, for
    # nighttime, it is ABI Band 7.
    # This is the band to show if only showing a single band
    # Use 'all' to show all
    # Note: this is 0-base indexing for [Red,Green,Blue], Red band is index 0
    # single_fig_band is the band for the single figure validation file.
    single_fig_band = 2
    single_fig_band_idx = 0
    seperate_fig_band_idx = 'ALL'

    # There are four sets of ABI images created. This gives the names of them and their column positions (in big_loop_idx)
    ABI_image_set_names = {0:'detection_satellite_before', 1:'detection_satellite_after', 2:'other_satellite_before', 3:'other_satellite_after'}
        
    # Are we generating raw multi-band data arrays?
    # If True, then save out to pickle files multi-band data arrays
    # (see: create_multi_band_data)
    generate_multi_band_data = False
    # Are we generating a single figure or seperate figures for each image?
    # If True, the plot seperate figures for each image, otherwise we generate a single figure of all images
    plot_seperate_figures = False
    # This specifies which individual figures to plot if not plotting the combined figure
    # Individual figure are only plotted for the satellite that generated the detection
   #indiv_images_to_plot = ('detection', '60_sec_integ', '60_sec_integ_zoom', 'GLM_projection', 'GLM_projection_zoomed_out')
    indiv_images_to_plot = ('60_sec_integ', '60_sec_integ_zoom', 'GLM_projection', 'GLM_projection_zoomed_out')
    # This gives which ABI multi-band image data to save to file. 
    # Storing this data consumes a lot of disk space
   #multi_band_data_to_save = ('detection_satellite_before', 'detection_satellite_after')
    multi_band_data_to_save = ('detection_satellite_before')

    # These are the paths to the data files. They must be entered in your code after you create the cutoutConfiguration object
    G16DIR = None
    G17DIR = None
    G18DIR = None
    G19DIR = None

    G16ABIFDIR = None
    G16ABICDIR = None

    G17ABIFDIR = None
    G17ABICDIR = None

    G18ABIFDIR = None
    G18ABICDIR = None

    G19ABIFDIR = None
    G19ABICDIR = None

    coastlines_path = None

    #**********************
    # These are internal configuration parameters

    # Set size of patch +/-5 degrees north/south (approx since projected grid)
    dlat = 5.0
    dlon = 5.0
    # Limits (+/- degrees) for a zoomed-in image
    dlat_zoom = 1.0
    dlon_zoom = 1.0

    # The size of the zoomed-out and zoomed-in GLM 1-hr integrations
    dlat_1hr_zoomed_out = 50.0
    dlon_1hr_zoomed_out = 50.0
    dlat_1hr_zoomed_in = 10.0
    dlon_1hr_zoomed_in = 10.0

    assert dlat_1hr_zoomed_out == dlon_1hr_zoomed_out and dlat_1hr_zoomed_in == dlon_1hr_zoomed_in, \
        "The Lat and Lon GLM 1-hr integration windows must be equal"
        
    # Beginning and end index for glob string
    str_beg = 0
    str_end = None # This is set in the __init__ function
       
    #***
    # Figure parameters
    #set image size for combined figure
    imgszx=3000
    imgszy=3000

    # Set image size for seperate figures
    # Creates a square figure
    # Also for the GLM projection multi-band data array
    image_size_seperate = 512
    dpi_seperate = 150 # Dots per inch, This parameter is not that important. The image_size_seperate is more important.
    image_60_sec_pixel_size = 1056
    image_60_sec_zoom_pixel_size = 210
    
    fsz=24
    rot=90
    imgname = ""

    # Lats part of the pickle filename to store the multi-band image data
    # bzip2 the file. It's a bunch of very sparse matrices
    pickleName = 'multi_band.pbz2'

    # We want different GLM scatter dots sizes/colors depending on if we are generating a single figure or seperate figures for
    # each image
    SINGLE_FIG_MARKERSCALE = 30.0
    SEPERATE_FIG_MARKERSCALE = 100.0

    SINGLE_FIG_MARKERALPHA = 0.5
    SEPERATE_FIG_MARKERALPHA = 1.0

    SINGLE_FIG_PROJ_MARKERSCALE = 2.0
    SEPERATE_FIG_PROJ_MARKERSCALE = 10.0

    SINGLE_FIG_FACECOLORS_DETECTION = 'lime'
    SEPERATE_FIG_FACECOLORS_DETECTION = 'none'

    SINGLE_FIG_FACECOLORS_20_SEC = 'red'
    SEPERATE_FIG_FACECOLORS_20_SEC = 'none'

    SINGLE_FIG_FACECOLORS_60_SEC = 'aqua'
    SEPERATE_FIG_FACECOLORS_60_SEC = 'none'

    # These are the actual variables used when plotting
    # The code will determine which of the constants above to use for the variables below
    facecolors_detection = None
    edgecolors_detection = 'lime'
    facecolors_20_sec = None
    edgecolors_20_sec = 'red'
    facecolors_60_sec = None
    edgecolors_60_sec = 'aqua'
    edgecolors_1_hr = 'fuchsia'

    markerScale = None
    markerProjectionScale = None
    markerAlpha = None

    cutout_enabled = None

    #**********************
    
    def __init__ (self, cutout_enabled=True):
        """ Just set to default parameters
        
        If you want to change configuration parameters then you have to change by hand.

        Parameters
        ----------
        cutout_enabled : bool
            If False then the cutout tool was disabled in the pipeline, but we create a blank cutoutConfiguration for
            bookeeping purposes.
        """
        
        self.cutout_enabled = cutout_enabled

        if self.cutout_enabled:
            # This assumes the path for all satellites is the same in length.
            # Should be true if only the 'G**' (3 characters) changes between the folders.
            if self.G16ABIFDIR is not None:
                goes_dir_len = len(self.G16ABIFDIR)
            else:
                # This else statement if for when creating this cutoutConfiguration object for bookkeeping purposes but
                # not really running the cutout tool.
                goes_dir_len = 0

            str_end = goes_dir_len + 62
            

        pass

    def create_single_figure(self, glmDetectionFilename, bolideLat, bolideLon):
        """ Creates a single figure to place all images on in a grid

        The input parameters are used to define the file name for the figure.

        Parameters
        ----------
        glmDetectionFilename : str
            The name of the main GLM data file associated with the detection
        bolideLat : float
            Average latitude of the bolide detection
        bolideLon : float
            Average longitude of the bolide detection

        Returns:
        --------
        self.fig : 
            A single figure object
        self.axs : 
            A single axis object list
        """
        plt.style.use('dark_background')
        self.fig, self.axs = plt.subplots(4,4,figsize=(self.imgszx/100.0,self.imgszy/100.0),dpi=50)
        plt.subplots_adjust(hspace=0.05,wspace=0.05)
        plt.box(on=None)
        self.axs = self.axs.ravel()

        self.band_to_show_idx = self.single_fig_band_idx

        #set axs parameters - likely a cleaner way
        for i in range(len(self.axs)):
            self.axs[i].tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
            self.axs[i].set_aspect(aspect='equal', adjustable='box')
            self.axs[i].set_frame_on(False)
        
        #initialize position on grid
        pos = self.axs[0].get_position()
                
        self.fig.suptitle(glmDetectionFilename, fontsize=24)
        self.imgname = glmDetectionFilename+"_"+str(bolideLat)+"_"+str(bolideLon)+"_ABI_forest"+".png"


    def create_seperate_figures(self):
        """ Creates a seperate figure for each image

        Returns
        -------
        self.fig : dict
            A dict of figure objects. Each key is the name of the figure
        self.axs : dict
            A single axis object list

        """

        self.band_to_show_idx = self.seperate_fig_band_idx

        self.annotations = False

        self.fig = {}
        self.axs = None
        for figName in self.indiv_images_to_plot:
           #self.fig[figName] = plt.figure(figsize=(self.imgszx/100.0,self.imgszy/100.0),dpi=50)
            self.fig[figName] = plt.figure(figsize=(self.image_size_seperate/self.dpi_seperate,self.image_size_seperate/self.dpi_seperate),dpi=self.dpi_seperate)

        
        
    def retrieve_axes(self, figure_label, index_in_axes_array, detection_satellite):
        """ This will return an axes to plot an image. 
        It will determine if we are plotting all images on a single figure or individual figures.

        If we are plotting individual figures and this is not a desired figure, then return None

        Parameters
        ----------
        figure_label : str
            The name of the figure, see self.indiv_images_to_plot for which figures to plot if indivual figures are
            being plotted
        index_in_axes_array : int
            This is the index in the self.axs array to return.
            For plotting on a combined figure, this gives the index of the axes to plot onto
        detection_satellite : bool
            If True then this data is for the satellite the detection was made on

        Returns
        -------
        axes : matplotlib.axes._axes.Axes
        fig : matplotlib.figure
        """

        if self.plot_seperate_figures and self.indiv_images_to_plot.count(figure_label) < 1:
            # Only plot this figure if plotting individual figures and this is one to plot
            return None, None
        elif self.plot_seperate_figures and self.indiv_images_to_plot.count(figure_label) == 1: 
            # We only want to plot the first image using the ABI data right before the detection
            # This is the case if index_in_axes_array is divisible by 2 and detection_satellite == True
            # The exception is the zoomed-out GLM 1 hr projection, which is in the second column
            if detection_satellite and (index_in_axes_array%2==0 or figure_label == 'GLM_projection_zoomed_out'):
                # Plotting an individual figure
                axes = self.fig[figure_label].add_axes((0,0,1,1))
                fig = self.fig[figure_label]
            else:
                return None, None
        else:
            # Combined figure
            axes = self.axs[index_in_axes_array]
            fig = self.fig

        return axes, fig

    def save_figures(self, outputdir, detection_ID):
        """ Saves the figures.

        Parameters
        outputdir : str
            Path to save output figures
        detection_ID : int64
            Detection candidate ID
        """

        figure_filenames = []
        with warnings.catch_warnings():
            # cartopy causes a deprecation warning in shapely. No clue about any of this, but outputs looks good, so ignore!
            warnings.simplefilter("ignore", ShapelyDeprecationWarning)
            if self.plot_seperate_figures:
                for key in self.fig:
                    filename = os.path.join(outputdir,str(detection_ID)+"_"+self.imgname+key+'.png')
                    self.fig[key].savefig(filename)
                    plt.close(self.fig[key])
                    # Pull out the basename, remove the path and add png to the end
                    filename = os.path.basename(filename)
                    figure_filenames.append(filename)
            
            else:
                filename = os.path.join(outputdir,str(detection_ID)+"_"+self.imgname)
                self.fig.savefig(filename)
                plt.close(self.fig)
                # Pull out the basename, remove the path and add png to the end
                filename = os.path.basename(filename)
                figure_filenames.append(filename)

        return figure_filenames

    def __repr__(self):
        return ioUtil.print_dictionary(self)

def gen_cutout(detection,outputdir,config):
    """ Generate cutout figure for a bolide detection

    Parameters
    ----------
    detection : BolideDetection
        The bolide detection candidate to generate a cutout figure for
    outputdir : str
        The output path to save the generated figures
    config : cutoutConfiguration class
        Contains all the configuration information for generating the cutout figures

    Returns
    -------
        detection_ID : int64
            The detection ID associated with these cutout figures
        success : bool
            True if the the tool ran correctly
        figure_filenames : list
            Names of the figures generated

    """

    # Do not plot annotations when plotting individual figures
    # Also set the GLM marker scale and alpha
    if config.plot_seperate_figures:
        config.markerScale = config.SEPERATE_FIG_MARKERSCALE 
        config.markerProjectionScale = config.SEPERATE_FIG_PROJ_MARKERSCALE 
        config.markerAlpha = config.SEPERATE_FIG_MARKERALPHA

        config.facecolors_detection = config.SEPERATE_FIG_FACECOLORS_DETECTION
        config.facecolors_20_sec    = config.SEPERATE_FIG_FACECOLORS_20_SEC
        config.facecolors_60_sec    = config.SEPERATE_FIG_FACECOLORS_60_SEC
    else:
        config.markerScale = config.SINGLE_FIG_MARKERSCALE 
        config.markerProjectionScale = config.SINGLE_FIG_PROJ_MARKERSCALE 
        config.markerAlpha = config.SINGLE_FIG_MARKERALPHA

        config.facecolors_detection = config.SINGLE_FIG_FACECOLORS_DETECTION 
        config.facecolors_20_sec    = config.SINGLE_FIG_FACECOLORS_20_SEC
        config.facecolors_60_sec    = config.SINGLE_FIG_FACECOLORS_60_SEC


    try:

        # If we get to the end of the function then change success to True and return
        success = False
        
        # Retrieve detection candidate variables
        bolideLat, bolideLon = detection.average_group_lat_lon
        # The "main" data file is the first in the list
        glmDetectionFilename = os.path.basename(detection.filePathList[0])
        
        # Extract the start time of GLM netCDF file containing the bolide detection
        glmDataFileStartTime = glmDetectionFilename.split("_")
        glmDataFileStartTime = glmDataFileStartTime[len(glmDataFileStartTime)-3].replace("s","")
        
        #retrieve abi data files - files right before and after eventime.
        #4 files in within the stereo region, 2 otherwise
        closest_abi_files = ff.get_closest_abi(glmDataFileStartTime,bolideLat,bolideLon,0, config)

        # Create the figure
        if config.plot_seperate_figures:
            config.create_seperate_figures()
        else:
            config.create_single_figure(glmDetectionFilename, bolideLat, bolideLon)

        #***
        # The code below works in a for-loop where it expects either 2 or 4 elements in a List. 
        # It wants 2 to be for GOES-East and 2 for GOES-West.
        # TODO: recode this to be more robust and easier to expand to more satellites
        # Now that there are multiple satellites, we have to set which is GOES-West and GOES-East
        # Get the satellite that detected the event

        abiclosests = [None, None, None, None]

        # Figure out which satellite is GOES-East (abiclosests[2:4])
        # If the detection satellite (detection.goesSatellite) is G17 or G18 then we need to decide if we use G16 or G19 as the other satellite. 
        # Preferentially choose G19 if after the transition date, fallback to G16 if G19 is not available
        if detection.goesSatellite in ['G17', 'G18']:
            if closest_abi_files['G19'] is not None and detection.bolideTime >= config.G19_IS_EAST_DATE:
                abiclosests[2:4] = closest_abi_files['G19']
            elif closest_abi_files['G16'] is not None:
                abiclosests[2:4] = closest_abi_files['G16']
            else:
                abiclosests[2:4] = ['NA', 'NA']
        elif detection.goesSatellite == 'G16':
            if closest_abi_files['G16'] is None:
                abiclosests[2:4] = ['NA', 'NA']
            else:
                abiclosests[2:4] = closest_abi_files['G16']
        elif detection.goesSatellite == 'G19':
            if closest_abi_files['G19'] is None:
                abiclosests[2:4] = ['NA', 'NA']
            else:
                abiclosests[2:4] = closest_abi_files['G19']
        else:
            raise Exception('Unknown satellite')

        # Now figure out which satellite is GOES-West (abiclosests[2:4])
        # If the detection satellite (detection.goesSatellite) is [G16, G19] then we need to decide if we use G17 or G18 as the other satellite. 
        # Preferentially choose G18 if after the transition date, fallback to G17 if G18 is not available
        if detection.goesSatellite in ['G16', 'G19']:
            if closest_abi_files['G18'] is not None and detection.bolideTime >= config.G18_IS_WEST_DATE:
                abiclosests[0:2] = closest_abi_files['G18']
            elif closest_abi_files['G17'] is not None:
                abiclosests[0:2] = closest_abi_files['G17']
            else:
                abiclosests[0:2] = ['NA', 'NA']
        elif detection.goesSatellite == 'G17':
            if closest_abi_files['G17'] is None:
                abiclosests[0:2] = ['NA', 'NA']
            else:
                abiclosests[0:2] = closest_abi_files['G17']
        elif detection.goesSatellite == 'G18':
            if closest_abi_files['G18'] is None:
                abiclosests[0:2] = ['NA', 'NA']
            else:
                abiclosests[0:2] = closest_abi_files['G18']
        else:
            raise Exception('Unknown satellite')

        # If ABI files are not available for the detection satellite then raise exception (resulting in success = False)
        if closest_abi_files[detection.goesSatellite] is None:
            raise ipf.ABIDataAvailabilityError('****WARNING**** ABI files not available for detection satellite for ID {}'.format(detection.ID))
        
        #*****************************************************************************************************
        # Retrieve GLM events in 20-sec, 60-sec and 1-hr windows
        # For both the detection satellite and the other satellite
        # Get the correct GLM data directory for this detection
        glm_dir_options = {'G16': 'config.G16DIR', 'G17': 'config.G17DIR', 'G18': 'config.G18DIR', 'G19': 'config.G19DIR'}
        glm_data_path = eval(glm_dir_options[detection.goesSatellite])
        if detection.goesSatellite in ['G16', 'G19']:
            if detection.bolideTime >= config.G18_IS_WEST_DATE:
                otherSatellite = 'G18'
            else:
                otherSatellite = 'G17'
        elif detection.goesSatellite in ['G17', 'G18']:
            if detection.bolideTime >= config.G19_IS_EAST_DATE:
                otherSatellite = 'G19'
            else:
                otherSatellite = 'G16'
        else:
            raise Exception('Unknown satellite')
        glm_data_path_otherSatellite = eval(glm_dir_options[otherSatellite])

        # Exclude all events from the smaller time window so each list of events is mutually exclusive
        GLM_event_list_20_sec = bd.extract_events_from_all_files(glm_data_path, detection.bolideTime, 20, 
                exclude_list=detection.eventList)
        GLM_event_list_60_sec = bd.extract_events_from_all_files(glm_data_path, detection.bolideTime, 60,
                exclude_list=detection.eventList+GLM_event_list_20_sec)
        GLM_event_list_1_hr   = bd.extract_events_from_all_files(glm_data_path, detection.bolideTime, 60*60,
                exclude_list=detection.eventList+GLM_event_list_20_sec+GLM_event_list_60_sec)
        
        # If there is other satellite stereo event data then exclude that
        if detection.bolideDetectionOtherSatellite is not None:
            otherSatelliteEventList = detection.bolideDetectionOtherSatellite.eventList
        else:
            otherSatelliteEventList = []
        GLM_event_list_20_sec_otherSatellite = bd.extract_events_from_all_files(glm_data_path_otherSatellite, detection.bolideTime, 20,
                exclude_list=otherSatelliteEventList)
        GLM_event_list_60_sec_otherSatellite = bd.extract_events_from_all_files(glm_data_path_otherSatellite, detection.bolideTime, 60,
                exclude_list=otherSatelliteEventList+GLM_event_list_20_sec_otherSatellite)
        GLM_event_list_1_hr_otherSatellite   = bd.extract_events_from_all_files(glm_data_path_otherSatellite, detection.bolideTime, 60*60,
                exclude_list=otherSatelliteEventList+GLM_event_list_20_sec_otherSatellite+GLM_event_list_60_sec_otherSatellite)
        
        #*****************************************************************************************************
        #*****************************************************************************************************
        # Big for-loop that generates all the figures

        # Do a handful of things only in the first column with images (meaning, big_loop_idx%2==0)
        # This first column has some labels and other extra things performed
        first_column_idx=0
        multi_band_data_all_images = {'detection_satellite_before':None,
                                        'detection_satellite_after':None,
                                        'other_satellite_before':None,
                                        'other_satellite_after':None}
        # Loop through ABI files (4 for stereo regions, 2 otherwise)
        for big_loop_idx in range(len(abiclosests)):
            if abiclosests[big_loop_idx]=="NA":
                first_column_idx=big_loop_idx+1
                continue
        
            #extract info from filename
            fname = abiclosests[big_loop_idx].split("/")
            fname = fname[len(fname)-1]
            info = fname.split("_")
            fileTime = info[3].replace("s","")
            GOES = info[2]

            # Which ABI image "view" is used
            # F := Full Frame
            # C := CONUS (Continental U.S.)
            view = info[1].split("-")[2].replace("Rad","")
        
            #only draw labels preceeding first column
            if big_loop_idx==first_column_idx and not config.plot_seperate_figures:
                ylabels=True
            else:
                ylabels=False
        
            if GOES == detection.goesSatellite:
                detection_satellite = True
            else:
                detection_satellite = False

            [multi_band_composite, RGB_composite, multi_band_composite_zoom, RGB_composite_zoom, bounds, raw_band_1_data] = \
                    ipf.generate_ABI_composites(config, detection.ID, abiclosests[big_loop_idx], bolideLat, bolideLon, detection_satellite)
            if multi_band_composite is None:
                if detection_satellite:
                    # This is the detection satellite, the ABI must be loaded, but it did not, so throw the error stored as RGB_composite
                    # re-raise the previous error
                    raise RGB_composite
                # There was a problem pulling the ABI data, Skip this column
                # But first post a notice
                axes, _ = config.retrieve_axes('detection', big_loop_idx, detection_satellite)
                if axes is not None:
                    axes.annotate('ABI data not available', xy=(0.0, 0.0), xycoords='axes fraction', color='Red', fontsize=config.fsz) 
                continue


            # Project the GLM data onto the ABi cutout
            if detection_satellite:
                # This is for the ABI image where the bolide candidate was detected
                # Use the larger 1 hour GLM integration degree span of +/- 50 degrees so that all GLM events are shown properly for all images.
                glmxydata_detection = ipf.project_glm_data(raw_band_1_data, detection.eventList, bolideLat, bolideLon, config.dlat_1hr_zoomed_out , config.dlon_1hr_zoomed_out)
                glmxydata_20_sec = ipf.project_glm_data(raw_band_1_data, GLM_event_list_20_sec, bolideLat, bolideLon, config.dlat_1hr_zoomed_out , config.dlon_1hr_zoomed_out)
                glmxydata_60_sec = ipf.project_glm_data(raw_band_1_data, GLM_event_list_60_sec, bolideLat, bolideLon, config.dlat_1hr_zoomed_out , config.dlon_1hr_zoomed_out)
                glmxydata_1_hr = ipf.project_glm_data(raw_band_1_data, GLM_event_list_1_hr, bolideLat, bolideLon, config.dlat_1hr_zoomed_out , config.dlon_1hr_zoomed_out)
            else:
                # This is for the other ABI satellite image and GLM data
                if detection.bolideDetectionOtherSatellite is not None:
                    # This is when a stereo detection and GLM from the other satellite is available
                    glmxydata_detection = ipf.project_glm_data(raw_band_1_data, detection.bolideDetectionOtherSatellite.eventList, bolideLat, bolideLon, config.dlat_1hr_zoomed_out , config.dlon_1hr_zoomed_out)
                else:
                    glmxydata_detection = {'x':np.array([]), 'y':np.array([]), 'lon':np.array([]), 'lat':np.array([]), 'energyJoules':np.array([])}
                glmxydata_20_sec = ipf.project_glm_data(raw_band_1_data, GLM_event_list_20_sec_otherSatellite, bolideLat, bolideLon, config.dlat_1hr_zoomed_out , config.dlon_1hr_zoomed_out)
                glmxydata_60_sec = ipf.project_glm_data(raw_band_1_data, GLM_event_list_60_sec_otherSatellite, bolideLat, bolideLon, config.dlat_1hr_zoomed_out , config.dlon_1hr_zoomed_out)
                glmxydata_1_hr = ipf.project_glm_data(raw_band_1_data, GLM_event_list_1_hr_otherSatellite, bolideLat, bolideLon, config.dlat_1hr_zoomed_out , config.dlon_1hr_zoomed_out)
            raw_band_1_data.close()
    
            ##############################################################
            ####################### DRAW CUTOUTS #########################
            # draw 5 rows:
            #1. zoom-in on event
            #2. event data
            #3. 20-second integration
            #4. 1 minute integration
            #5. lat/lon projected 1hr integration with zooms
        
            ###################################################################
            #1. event zoom-in
            plot_event_zoom_in(config, big_loop_idx, glmxydata_detection, RGB_composite, detection_satellite, bounds)
        
            ###################################################################
            ### 2. bolide detection RGB_composite
            plot_bolide_detection(config, big_loop_idx, glmxydata_detection, view, GOES, fileTime, ylabels, RGB_composite,
                    detection_satellite, bounds)
        
            ###################################################################
            ### 3. 20-second integration
            plot_20_second_integration(config, big_loop_idx, glmxydata_detection, RGB_composite, glmxydata_20_sec,
                    ylabels, detection_satellite, bounds)
        
            ###################################################################
            ### 4. 60 second integration
            plot_60_second_integration(config, big_loop_idx, glmxydata_detection, glmxydata_20_sec, glmxydata_60_sec, ylabels, 
                    RGB_composite, detection_satellite, bounds)
            # If plotting seperate figures then also plot zoomed in 60-sec integration figure
            if config.plot_seperate_figures:
                plot_60_second_integration(config, big_loop_idx, glmxydata_detection, glmxydata_20_sec, glmxydata_60_sec, ylabels, 
                    RGB_composite_zoom, detection_satellite, bounds, zoomed_in_flag=True)
        
            ########################################################################
            ### overlay lat lon grid and coastlines on all patches
            if big_loop_idx%2==0 and config.annotations:
                plot_annotations(config, big_loop_idx, bounds, GOES, bolideLat, bolideLon, RGB_composite, view)
        
            #######################################################################
            ### 5. 1 hr integration
            plot_1_hour_GLM_projection(config, big_loop_idx, detection_satellite, ylabels, bolideLon, bolideLat, 
                    glmxydata_detection, glmxydata_20_sec, glmxydata_60_sec, glmxydata_1_hr)

            ##############################################################
            ##################### multi-band images ######################
            # We do not need to redo generating the multi-band data if generating individual figures
            if config.generate_multi_band_data  and not config.plot_seperate_figures:
                multi_band_data = create_multi_band_data(config, multi_band_composite, multi_band_composite_zoom,
                    glmxydata_detection, glmxydata_20_sec, glmxydata_60_sec, glmxydata_1_hr,
                    bounds)

                if detection_satellite and big_loop_idx%2 == 0:
                    if config.ABI_image_set_names[0] in config.multi_band_data_to_save:
                        assert multi_band_data is not None, 'multi_band_data is None, this should not be'
                        # This is the before ABI image for the detection satellite
                        multi_band_data_all_images['detection_satellite_before'] = multi_band_data
                elif detection_satellite and big_loop_idx%2 == 1:
                    if config.ABI_image_set_names[1] in config.multi_band_data_to_save:
                        assert multi_band_data is not None, 'multi_band_data is None, this should not be'
                        # This is the after ABI image for the detection satellite
                        multi_band_data_all_images['detection_satellite_after'] = multi_band_data
                elif not detection_satellite and big_loop_idx%2 == 0:
                    if config.ABI_image_set_names[2] in config.multi_band_data_to_save:
                        if multi_band_data is None: continue
                        # This is the before ABI image for the other satellite
                        multi_band_data_all_images['other_satellite_before'] = multi_band_data
                elif not detection_satellite and big_loop_idx%2 == 1:
                    if config.ABI_image_set_names[3] in config.multi_band_data_to_save:
                        if multi_band_data is None: continue
                        # This is the after ABI image for the other satellite
                        multi_band_data_all_images['other_satellite_after'] = multi_band_data
            
            else:
                multi_band_data_all_images = None

        
        # End giant for-loop that generates all the figures
        #*****************************************************************************************************
        #*****************************************************************************************************
        
        # Write cutout to image
        figure_filenames = config.save_figures(outputdir, detection.ID)

        # Store multi-band data in pickle file
        if multi_band_data_all_images is not None:
            image_pickle_filename = os.path.join(outputdir,str(detection.ID)+"_"+config.pickleName)
            with bz2.open(image_pickle_filename, "wb") as fp:
                pickle.dump(multi_band_data_all_images, fp)
            fp.close()

        else:
            image_pickle_filename = None

    except ipf.ABIDataAvailabilityError as e:
       #print('\n****** Known error generating cutout for detection ID {}'.format(detection.ID))
        success = False
        figure_filenames = None
        image_pickle_filename = None

    except Exception as e:
        # Unexpected exception
        print('\n****** Unexpected Error generating cutout for detection ID {}'.format(detection.ID))
        print_exc()
        success = False
        figure_filenames = None
        image_pickle_filename = None
        
    else:
        # We got to the end of the function! Report success.
        success = True


    return detection.ID, success, figure_filenames, image_pickle_filename

def plot_event_zoom_in(config, big_loop_idx, glmxydata_detection, RGB_composite, detection_satellite, bounds):
    """ Plot the small zoomed-in image at the top of the cutout figure.

    This always plots the full GeoColor image.
    """
    figure_label = 'detection_zoomed_in'

    # Check if this is the other satellite and there is no detection data avilable
    if not detection_satellite and len(glmxydata_detection['x']) == 0:
        return
    elif detection_satellite and len(glmxydata_detection['x']) == 0:
        raise Exception('If this is the detection satellite then there should be event data available!')

    # Get the correct axes, depending on what we are plotting
    axes, fig = config.retrieve_axes(figure_label, big_loop_idx, detection_satellite)
    if axes is None:
        return False

    # Get the x and y coordinates shifted to the ABI image bounding box
    ex = glmxydata_detection['x']-bounds['RGB'][2]
    ey = glmxydata_detection['y']-bounds['RGB'][0]
    
    # Looks like this plots 10 pixels on either side of the detection event. 
    # This means the box size is dynamic, based on the number of pixels the event crosses.
    exmin=int(min(ex))-10
    exmax=int(max(ex))+10
    eymin=int(min(ey))-10
    eymax=int(max(ey))+10
    
    if exmin > 0 and exmin < RGB_composite[...,0].shape[1] and eymin > 0 and eymin < RGB_composite[...,0].shape[0]:
        pos = config.axs[big_loop_idx].get_position()
        groundtrack=fig.add_axes([pos.x0+pos.width/3.0,pos.y1+0.02,pos.width/3.0,pos.height/3.0],frameon=False)
        groundtrack.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
        groundtrack.imshow(RGB_composite)
        groundtrack.scatter(ex,ey,marker='o',s=config.markerScale,alpha=config.markerAlpha,
                facecolors=config.facecolors_detection, edgecolors=config.edgecolors_detection)
        groundtrack.plot(ex,ey,linewidth=0.5,color='lime')
        groundtrack.set_xlim((exmin,exmax))
        groundtrack.set_ylim((eymax,eymin))

    return True

def plot_bolide_detection(config, big_loop_idx, glmxydata_detection, view, GOES, fileTime, ylabels, RGB_composite,
        detection_satellite, bounds):
    """ Plot the bolide detection only overlaied on ABI image. 

    This always plots the full GeoColor image.
    """

    figure_label = 'detection'

    # Get the correct axes, depending on what we are plotting
    axes, _ = config.retrieve_axes(figure_label, big_loop_idx, detection_satellite)
    if axes is None:
        return False

    ### draw title and label
    # but only if on combined figure
    if not config.plot_seperate_figures:
        title=view+" "+GOES+" "+fileTime
        axes.set_title(title,fontsize=config.fsz)
        if ylabels:
            axes.set_ylabel("EVENT DATA",rotation=config.rot,labelpad=50,fontsize=config.fsz)
    
    ### plot ABI data
    image = axes.imshow(RGB_composite)
    axes.set_xlim((0,RGB_composite[...,0].shape[1]))
    axes.set_ylim((RGB_composite[...,0].shape[0],0))

    ### scatter bolide detection data
    axes.scatter(glmxydata_detection['x']-bounds['RGB'][2],glmxydata_detection['y']-bounds['RGB'][0],marker='o',s=config.markerScale,alpha=config.markerAlpha, 
            facecolors=config.facecolors_detection, edgecolors=config.edgecolors_detection)

    return True

def plot_20_second_integration(config, big_loop_idx, glmxydata_detection, RGB_composite, glmxydata_20_sec,
        ylabels, detection_satellite, bounds):
    """ Plots the bolide event plus 20 seconds of data about the event

    The ABI image plotted is dependent on config.band_to_show_idx. It could be just one band or all bands.
    """

    figure_label = '20_sec_integ'

    # Get the correct axes, depending on what we are plotting
    # This is the second row so add 4 to the big_loop_idx
    axes, _ = config.retrieve_axes(figure_label, big_loop_idx+4, detection_satellite)
    if axes is None:
        return False

    # Determine which ABI image bands to plot
    if config.band_to_show_idx == 'ALL':
        # This is RGB data
        image2 = axes.imshow(RGB_composite)
    else:
        # vmin/vmax based on self.single_fig_band data (band 2) (and Python is 0-based indexing!)
        # Set range for colormap
        _median = np.median(RGB_composite[...,config.band_to_show_idx])
        tst = deepcopy(RGB_composite[...,config.band_to_show_idx])
        tst[tst==0] = _median
        _vmin = np.amin(tst)
        _vmax = np.amax(RGB_composite[...,config.band_to_show_idx])
        image2 = axes.imshow(RGB_composite[...,config.band_to_show_idx],cmap='Greys_r',vmin=_vmin,vmax=_vmax)
        

    # plot 20-second glm data
    axes.scatter(glmxydata_20_sec['x']-bounds['RGB'][2], glmxydata_20_sec['y']-bounds['RGB'][0], marker='o',s=config.markerScale,alpha=config.markerAlpha, 
            facecolors=config.facecolors_20_sec, edgecolors=config.edgecolors_20_sec)
    
    # plot bolide detection data
    axes.scatter(glmxydata_detection['x']-bounds['RGB'][2],glmxydata_detection['y']-bounds['RGB'][0],marker='o',s=config.markerScale,alpha=config.markerAlpha, 
            facecolors=config.facecolors_detection, edgecolors=config.edgecolors_detection)
    
    # add labels if preceeding first column
    if ylabels:
        axes.set_ylabel("20 SEC INTEGRATION\n(Band 2 or 7)",rotation=config.rot,labelpad=50,fontsize=config.fsz)
    
    #set plot limits
    axes.set_xlim((0,RGB_composite[...,0].shape[1]))
    axes.set_ylim((RGB_composite[...,0].shape[0],0))

def plot_60_second_integration(config, big_loop_idx, glmxydata_detection, glmxydata_20_sec, glmxydata_60_sec, ylabels, 
        RGB_composite, detection_satellite, bounds, zoomed_in_flag=False):
    """ Plots the bolide event plus 60 seconds of data about the event

    Parameters
    ----------
    zoomed_in_flag : bool
        If True then plot a figure with zoomed in ranges set by config.dlat_zoom and config.dlon_zoom
    """

    if zoomed_in_flag:
        figure_label = '60_sec_integ_zoom'
        bounds_x_2 = bounds['RGB_zoom'][2]
        bounds_y_0 = bounds['RGB_zoom'][0]
    else:
        figure_label = '60_sec_integ'
        bounds_x_2 = bounds['RGB'][2]
        bounds_y_0 = bounds['RGB'][0]

    # Get the correct axes, depending on what we are plotting
    # This is the third row so add 8 to the big_loop_idx
    axes, fig = config.retrieve_axes(figure_label, big_loop_idx+8, detection_satellite)
    if axes is None:
        return False

    # Determine which ABI image bands to plot
    if config.band_to_show_idx == 'ALL':
        # This is RGB data
        image3 = axes.imshow(RGB_composite)
    else:
        # vmin/vmax based on self.single_fig_band data (band 2)
        # Set range for colormap
        _median = np.median(RGB_composite[...,config.band_to_show_idx])
        tst = deepcopy(RGB_composite[...,config.band_to_show_idx])
        tst[tst==0] = _median
        _vmin = np.amin(tst)
        _vmax = np.amax(RGB_composite[...,config.band_to_show_idx])
        image3 = axes.imshow(RGB_composite[...,config.band_to_show_idx],cmap='Greys_r',vmin=_vmin,vmax=_vmax)
        
    # plot 60-second integration
    axes.scatter(glmxydata_60_sec['x']-bounds_x_2,glmxydata_60_sec['y']-bounds_y_0,marker='o',s=config.markerScale,alpha=config.markerAlpha, 
            facecolors=config.facecolors_60_sec, edgecolors=config.edgecolors_60_sec)

    # plot 20-second integration
    axes.scatter(glmxydata_20_sec['x']-bounds_x_2,glmxydata_20_sec['y']-bounds_y_0,marker='o',s=config.markerScale,alpha=config.markerAlpha, 
        facecolors=config.facecolors_20_sec, edgecolors=config.edgecolors_20_sec)
        
    # Plot bolide detection data
    axes.scatter(glmxydata_detection['x']-bounds_x_2, glmxydata_detection['y']-bounds_y_0,marker='o',s=config.markerScale,alpha=config.markerAlpha, 
            facecolors=config.facecolors_detection, edgecolors=config.edgecolors_detection)
    
    if ylabels:
        axes.set_ylabel("1 MIN INTEGRATION\n(Band 2 or 7)",rotation=config.rot,labelpad=50,fontsize=config.fsz)
    
    axes.set_xlim((0,RGB_composite[...,0].shape[1]))
    axes.set_ylim((RGB_composite[...,0].shape[0],0))

def plot_1_hour_GLM_projection(config, big_loop_idx, detection_satellite, ylabels, bolideLon, bolideLat, 
        glmxydata_detection, glmxydata_20_sec, glmxydata_60_sec, glmxydata_1_hr):
    """ Plots the 1-Hour integration projection GLM data over the zoomed out Earth images
    """

    xlons = [100,120,140,160,-180,-160,-140,-120,-100,-80,-60,-40,-20,0,10]
    xlons_10 = [100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,-180,-175,-160,-165,-160,-155,-150,-145,-140,-135,-130,-125,-120,-115,-110,-105,-100,-95,-90,-85,-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10]
    
    #central_longitude=cc
    cc = -103.0
    
    # The bottom left figure is the standard projection at 20 degree plotting view
    # The bottom right figure is the zoomed-out  projection at 100 degree plotting view
    if big_loop_idx%2==0:
        figure_label = 'GLM_projection'
        diff = config.dlat_1hr_zoomed_in 
    else:
        figure_label = 'GLM_projection_zoomed_out'
        diff = config.dlat_1hr_zoomed_out 

    # Get the correct axes, depending on what we are plotting
    # This is the fourth row so add 12 to the big_loop_idx
    axes, fig = config.retrieve_axes(figure_label, big_loop_idx+12, detection_satellite)
    if axes is None:
        return False

    if ylabels:
        axes.set_ylabel("1 HR INTEGRATION",rotation=config.rot,labelpad=50,fontsize=config.fsz)
    
    #initialize position on grid
    pos = axes.get_position()
    hrInt = fig.add_axes(axes.get_position(),frameon=True,projection=ccrs.PlateCarree(central_longitude=cc))

    hrInt.set_extent([bolideLon-diff, bolideLon+diff, bolideLat-diff, bolideLat+diff], crs=ccrs.PlateCarree())
    
    if config.annotations:
        hrInt.coastlines(resolution='10m', color='white', linewidth=1)
        hrInt.add_feature(ccrs.cartopy.feature.COASTLINE)
    
        gl = hrInt.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.5)
        
        gl.top_labels = False
        gl.right_labels = False
        
        gl.xlabel_style = {'size':18}
        gl.ylabel_style = {'size':18}
        
        if diff==config.dlat_1hr_zoomed_out :
            gl.xlocator = mticker.FixedLocator(xlons)
        else:
            gl.xlocator = mticker.FixedLocator(xlons_10)
    
    
    # plot all events
    hrInt.scatter(glmxydata_1_hr['lon']-cc,glmxydata_1_hr['lat'],marker='o',s=config.markerProjectionScale,c=config.edgecolors_1_hr)
        
    # plot 1-min integration
    hrInt.scatter(glmxydata_60_sec['lon']-cc,glmxydata_60_sec['lat'],marker='o',s=config.markerProjectionScale,c=config.edgecolors_60_sec)
    
    # plot 20-second integration (target event file)
    hrInt.scatter(glmxydata_20_sec['lon']-cc,glmxydata_20_sec['lat'],marker='o',s=config.markerProjectionScale,c=config.edgecolors_20_sec)
    
    # plot detected events, if the detection was on this satellite
    hrInt.scatter(glmxydata_detection['lon']-cc,glmxydata_detection['lat'],marker='o',s=config.markerProjectionScale,c=config.edgecolors_detection)

    return True

def plot_annotations(config, i, bounds, GOES, bolideLat, bolideLon, data, view):
    """ Plots annotations on the cutout figures.
    This is lat/long lines and labels, and coastlines

    The lat/lon overlays uses a "brute force style," whatever that means

    This function only works for a combined figure.
    """

    if config.plot_seperate_figures:
        raise Exception('Do not plot annotations for seperate figures')

    ########################################################################
    ### overlay lat lon grid on all patches - brute force style

    bounds_2 = [bounds['RGB'][0]*2,bounds['RGB'][1]*2,bounds['RGB'][2]*2,bounds['RGB'][3]*2]
    
    #everything done based on full disk image, so need to apply offset if working with conus data
    if view=='C':
        #apply offset
        G16x=3609
        G16y=1689
        G17x=5849
        G17y=1688
        # Use G17 numbers for G18?
        G18x=5849
        G18y=1688
        # Use G16 numbers for G19?
        G19x=3609
        G19y=1689
    
        if GOES == 'G16':
            offsetx = G16x
            offsety = G16y
        elif GOES == 'G17':
            offsetx = G17x
            offsety = G17y
        elif GOES == 'G18':
            offsetx = G18x
            offsety = G18y
        elif GOES == 'G19':
            offsetx = G19x
            offsety = G19y
        else:
            raise Exception('Unknown GOES satellite')
    
        bounds_2 = [bounds_2[0]+offsety,bounds_2[1]+offsety,bounds_2[2]+offsetx,bounds_2[3]+offsetx]
        


    llxmin=bounds_2[2]
    llxmax=bounds_2[3]
    llymin=(10848*2)-bounds_2[1]
    llymax=(10848*2)-bounds_2[0]
    
    
    if GOES=='G16':
        overlay_latsy=overlays.G16_latsy
        overlay_latsx=overlays.G16_latsx
        overlay_lonsy=overlays.G16_lonsy
        overlay_lonsx=overlays.G16_lonsx
    elif GOES=='G17':
        overlay_latsy=overlays.G17_latsy
        overlay_latsx=overlays.G17_latsx
        overlay_lonsy=overlays.G17_lonsy
        overlay_lonsx=overlays.G17_lonsx
    elif GOES=='G18':
        overlay_latsy=overlays.G18_latsy
        overlay_latsx=overlays.G18_latsx
        overlay_lonsy=overlays.G18_lonsy
        overlay_lonsx=overlays.G18_lonsx
    elif GOES=='G19':
        overlay_latsy=overlays.G19_latsy
        overlay_latsx=overlays.G19_latsx
        overlay_lonsy=overlays.G19_lonsy
        overlay_lonsx=overlays.G19_lonsx
    else:
        raise Exception('Unknown GOES satellite')
    
    #create overlay axes
    gridaxMaster = [config.fig.add_axes(config.axs[i].get_position(), frameon=False),
                    config.fig.add_axes(config.axs[i+4].get_position(), frameon=False), 
                    config.fig.add_axes(config.axs[i+8].get_position(), frameon=False),
                    config.fig.add_axes(config.axs[i+1].get_position(), frameon=False),
                    config.fig.add_axes(config.axs[i+5].get_position(), frameon=False),
                    config.fig.add_axes(config.axs[i+9].get_position(), frameon=False)]
    gridax=[]
    gridax.append(gridaxMaster)
    index=0
    if GOES=='G16':
        index=0
    if GOES=='G19':
        index=0
    if GOES=='G17':
        index=len(gridax)-1
    if GOES=='G18':
        index=len(gridax)-1
    
    step=5
    
    for k in range(len(gridax[index])):
        gridax[index][k].tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
    
    for line in range(0,len(overlay_latsy),2):
        for k in range(len(gridax[index])):
            gridax[index][k].plot(overlay_latsx[line],overlay_latsy[line],linestyle='dashed',dashes=[12,4],color='white',linewidth=1.0,alpha=0.5)
        for j in range(0,len(overlay_latsx[line]),2):
            if overlay_latsx[line][j]>llxmin and overlay_latsx[line][j]<llxmax-50 and overlay_latsy[line][j]>llymin and overlay_latsy[line][j]<llymax-50:
                for k in range(len(gridax[index])):
                    gridax[index][k].text(overlay_latsx[line][j],overlay_latsy[line][j],70-(line*step),fontsize=config.fsz,color='aqua',alpha=1.0,verticalalignment='bottom',horizontalalignment='left')
    
    for line in range(0,len(overlay_lonsy),2):
        for k in range(len(gridax[index])):
            gridax[index][k].plot(overlay_lonsx[line],overlay_lonsy[line],linestyle='dashed',dashes=[12,4],color='white',linewidth=1.0,alpha=0.5)
        for j in range(0,len(overlay_lonsx[line]),2):
            if overlay_lonsx[line][j]>llxmin+30 and overlay_lonsx[line][j]<llxmax-100 and overlay_lonsy[line][j]>llymin+50 and overlay_lonsy[line][j]<llymax:
                if GOES=='G16':
                    lonval=-150+(line*step)
                elif GOES=='G17':
                    lonval=-210+(line*step) if line > 2 else 150+(line*step)
                elif GOES=='G18':
                    # Set G18 to the same as G17
                    lonval=-210+(line*step) if line > 2 else 150+(line*step)
                elif GOES=='G19':
                    # Set G19 to the same as G16
                    lonval=-150+(line*step)
                for k in range(len(gridax[index])):
                    gridax[index][k].text(overlay_lonsx[line][j],overlay_lonsy[line][j]-5,lonval,fontsize=config.fsz,color='lime',alpha=1.0,verticalalignment='top',horizontalalignment='right')
    
    ####################################################################
    ###### overlay coastlines on all patches
    binfile = open(config.coastlines_path,'rb')
    data = np.fromfile(binfile,'>i4')
    
    # TODO: figure out what is this list of unlabeled numbers
    G16 = [6356752.31414,6378137.0000,42142775.31414,-1.30899694,-0.151858,2.8e-05,0.151858,-2.8e-05,0.0818191910435]
    G17 = [6356752.31414,6378137.0000,42142775.31414,-2.39110108,-0.151858,2.8e-05,0.151858,-2.8e-05,0.0818191910435]
    G18 = [6356752.31414,6378137.0000,42142775.31414,-2.39110108,-0.151858,2.8e-05,0.151858,-2.8e-05,0.0818191910435]
    G19 = [6356752.31414,6378137.0000,42142775.31414,-1.30899694,-0.151858,2.8e-05,0.151858,-2.8e-05,0.0818191910435]
    
    if GOES == 'G16':
        G = G16
    elif GOES == 'G17':
        G = G17
    elif GOES == 'G18':
        G = G18
    if GOES == 'G19':
        G = G19
    
    if bolideLon < 0.0:
        abslon = 360.0 + bolideLon
    else:
        abslon = bolideLon
    
    abslat=bolideLat
    
    ii=0
    count=0
    while ii < len(data):
        id=data[ii]
        n=data[ii+1]
        flag=data[ii+2]
        west = data[ii+3]
        east = data[ii+4]
        south = data[ii+5]
        north = data[ii+6]
        ii+=11
    
        div=1000000.0
        if abslat<(south/div) or abslat>(north/div) or abslon<(west/div) or abslon>(east/div):
            ii+=(n*2)
            continue
        count+=1
        #have to break up continents
        x1 = data[ii:ii+(n*2):2]/1000000.0
        #x1[x1>180]-=360
        y1 = data[ii+1:ii+(n*2):2]/1000000.0
        [x2,y2]=ipf.latlon_to_grid_specs(y1,x1,G)
        for k in range(len(gridax[index])):
            gridax[index][k].plot(x2,y2,color='w',linewidth=1,alpha=0.5)
    
        ii+=(n*2)
    
    if config.annotations:
        #crop plots
        for k in range(len(gridax[index])):
            gridax[index][k].set_xlim((llxmin,llxmax))
            gridax[index][k].set_ylim((llymin,llymax))

    return True
    
def create_multi_band_data(config, multi_band_composite, multi_band_composite_zoom, 
        glmxydata_detection, glmxydata_20_sec, glmxydata_60_sec, glmxydata_1_hr,
        bounds):
    """ This function will generate a np.array containing the ABI cutout multi-band images
    With the GLM data projected onto the same field of view

    The images generated are listed in cutoutConfiguration.indiv_images_to_plot 

    The GLM data is added in as flux values where GLM data is present.


    Parameters
    ----------
    multi_band_composite : np.array of size (dy, dx, len(bands_to_read))
        The ABI data for all bands desired and clipped to the desired viewing box
        Note: GOES ABI bands are 1-based! So, Band 1 is index 0 in this array.
    multi_band_composite_zoom : np.array of size (dy, dx, len(bands_to_read))
        The ABI data for all bands desired and clipped to the desired viewing box
        Note: GOES ABI bands are 1-based! So, Band 1 is index 0 in this array.
    glmxydata_detection : dict
        Keys: 'x', 'y', 'lon', 'lat', 'energyJoules'
    glmxydata_20_sec : dict
    glmxydata_60_sec  : dict
    glmxydata_1_hr : dict
    bounds : dict 
        The viewing box limits in ABI pixels [y0,y1,x0,x1]
        keys = ('RGB', 'RGB_zoom', 'multi_band', 'multi_band_zoom')

    Returns
    -------
    multi_band_data : dict
        where each key is from config.indiv_images_to_plot
        Each ele,ment contains a np.array of size (dy, dx, len(config.bands_to_read)+n_glm_bands)
        The multi-band data plus the GLM data projected on the ABI data
        The specific dimensions depends on which image is being generated
    """

    # If no detection data then return None
    # TODO: Get this to work even if there was no detection data. 
    # We should be able to generate the figure of the neighboring data.
    if len(glmxydata_detection['lat']) == 0:
        return None

    # Get lat/lon range about detection
    avgLat = np.nanmedian(glmxydata_detection['lat'])
    avgLon = np.nanmedian(glmxydata_detection['lon'])

    # Remove the bands with empty data from the composite
    multi_band_composite_reduced = multi_band_composite[...,np.array(config.bands_to_read, dtype=int)-1]
    multi_band_composite_zoom_reduced = multi_band_composite_zoom[...,np.array(config.bands_to_read, dtype=int)-1]

    multi_band_composite_reduced_shape = multi_band_composite_reduced.shape
    multi_band_composite_zoom_reduced_shape = multi_band_composite_zoom_reduced.shape

    #('detection', '60_sec_integ', '60_sec_integ_zoom', 'GLM_projection', 'GLM_projection_zoomed_out')

    # The glmxydata is in units of coordinates of the ABI cutout pixel box after applying the bounds.
    # However, the glmxydata will span past the edges of the ABI cutout pixel box. If we try to align the GLM data to
    # the ABI cutout matrix then we will get out of bounds indexing. We need to clip the glmxydata to the bounds of the
    # ABI cutout image matrix.

    
    multi_band_data = {}
    for key in config.indiv_images_to_plot:

        if key == 'detection':
            # For detection figure, just add the GLM data for the detection, so, just one extra dimension
            # The GLM data is sparse, so most of it is zero
            GLM_data = np.zeros((multi_band_composite_reduced_shape[0], multi_band_composite_reduced_shape[1], 1), dtype=np.float32)
            # Fill in the GLM event data with the energy values
            # Round GLM data to the nearest composite image pixel
            # GLM data will round to the same pixel, integrate all GLM data on each pixel
            for value,x,y in zip(glmxydata_detection['energyJoules'], 
                    np.round(glmxydata_detection['x']-bounds['multi_band'][2]).astype(int),np.round(glmxydata_detection['y']-bounds['multi_band'][0]).astype(int)):
                if is_within_pixel_box(x,y,multi_band_composite_reduced_shape):
                    GLM_data[y,x,0] += value

           #multi_band_data[key] = np.concatenate((multi_band_composite_reduced, GLM_data), axis=2)
            multi_band_data[key] = (multi_band_composite_reduced, GLM_data)

        elif key == '60_sec_integ':
            
            # The 60-second integration data has three components (detection, 20-second, 60-second)
            GLM_data = np.zeros((multi_band_composite_reduced_shape[0], multi_band_composite_reduced_shape[1], 3), dtype=np.float32)

            # Detection data
            for value,x,y in zip(glmxydata_detection['energyJoules'], 
                    np.round(glmxydata_detection['x']-bounds['multi_band'][2]).astype(int),np.round(glmxydata_detection['y']-bounds['multi_band'][0]).astype(int)):
                if is_within_pixel_box(x,y,multi_band_composite_reduced_shape):
                    GLM_data[y,x,0] += value

            # 20-second data
            for value,x,y in zip(glmxydata_20_sec['energyJoules'],
                    np.round(glmxydata_20_sec['x']-bounds['multi_band'][2]).astype(int),np.round(glmxydata_20_sec['y']-bounds['multi_band'][0]).astype(int)):
                if is_within_pixel_box(x,y,multi_band_composite_reduced_shape):
                    GLM_data[y,x,1] += value

            # 60-second data
            for value,x,y in zip(glmxydata_60_sec['energyJoules'],
                    np.round(glmxydata_60_sec['x']-bounds['multi_band'][2]).astype(int),np.round(glmxydata_60_sec['y']-bounds['multi_band'][0]).astype(int)):
                if is_within_pixel_box(x,y,multi_band_composite_reduced_shape):
                    GLM_data[y,x,2] += value

           #multi_band_data[key] = np.concatenate((multi_band_composite_reduced, GLM_data), axis=2)
            multi_band_data[key] = (multi_band_composite_reduced, GLM_data)

        elif key == '60_sec_integ_zoom':
            
            # The 60-second integration data has three components (detection, 20-second, 60-second)
            GLM_data = np.zeros((multi_band_composite_zoom_reduced_shape[0], multi_band_composite_zoom_reduced_shape[1], 3), dtype=np.float32)

            # Detection data
            for value,x,y in zip(glmxydata_detection['energyJoules'], 
                    np.round(glmxydata_detection['x']-bounds['multi_band_zoom'][2]).astype(int),np.round(glmxydata_detection['y']-bounds['multi_band_zoom'][0]).astype(int)):
                if is_within_pixel_box(x,y,multi_band_composite_zoom_reduced_shape):
                    GLM_data[y,x,0] += value

            # 20-second data
            for value,x,y in zip(glmxydata_20_sec['energyJoules'],
                    np.round(glmxydata_20_sec['x']-bounds['multi_band_zoom'][2]).astype(int),np.round(glmxydata_20_sec['y']-bounds['multi_band_zoom'][0]).astype(int)):
                if is_within_pixel_box(x,y,multi_band_composite_zoom_reduced_shape):
                    GLM_data[y,x,1] += value

            # 60-second data
            for value,x,y in zip(glmxydata_60_sec['energyJoules'],
                    np.round(glmxydata_60_sec['x']-bounds['multi_band_zoom'][2]).astype(int),np.round(glmxydata_60_sec['y']-bounds['multi_band_zoom'][0]).astype(int)):
                if is_within_pixel_box(x,y,multi_band_composite_zoom_reduced_shape):
                    GLM_data[y,x,2] += value

           #multi_band_data[key] = np.concatenate((multi_band_composite_zoom_reduced, GLM_data), axis=2)
            multi_band_data[key] = (multi_band_composite_zoom_reduced, GLM_data)

        elif key == 'GLM_projection' or key == 'GLM_projection_zoomed_out':
            # This just shows the 1-hour integration GLM data zoomed-in

            # Get the lat/lon window box about the detection
            if key == 'GLM_projection':
                minLat = avgLat - config.dlat_1hr_zoomed_in
                maxLat = avgLat + config.dlat_1hr_zoomed_in
                minLon = avgLon - config.dlon_1hr_zoomed_in
                maxLon = avgLon + config.dlon_1hr_zoomed_in
            elif key == 'GLM_projection_zoomed_out':
                minLat = avgLat - config.dlat_1hr_zoomed_out
                maxLat = avgLat + config.dlat_1hr_zoomed_out
                minLon = avgLon - config.dlon_1hr_zoomed_out
                maxLon = avgLon + config.dlon_1hr_zoomed_out
            
            # Convert the lat/lon box into a pixel matrix
            nPixelsLat = config.image_size_seperate
            nPixelsLon = config.image_size_seperate
            box_shape = [nPixelsLon, nPixelsLat]
            degrees_per_pixel_lon = (maxLon-minLon)/nPixelsLon
            degrees_per_pixel_lat = (maxLat-minLat)/nPixelsLat
            
            # The 1-hour integration data has four components (detection, 20-second, 60-second, 1-hour)
            GLM_data = np.zeros((nPixelsLon, nPixelsLat, 4), dtype=np.float32)

            # Detection data
            for value,x,y in zip(glmxydata_detection['energyJoules'], 
                    np.round((glmxydata_detection['lon']-minLon) / degrees_per_pixel_lon).astype(int),
                    np.round((glmxydata_detection['lat']-minLat) / degrees_per_pixel_lat).astype(int)):
                if is_within_pixel_box(x,y,box_shape):
                    GLM_data[x,y,0] += value

            # 20-second data
            for value,x,y in zip(glmxydata_20_sec['energyJoules'], 
                    np.round((glmxydata_20_sec['lon']-minLon) / degrees_per_pixel_lon).astype(int),
                    np.round((glmxydata_20_sec['lat']-minLat) / degrees_per_pixel_lat).astype(int)):
                if is_within_pixel_box(x,y,box_shape):
                    GLM_data[x,y,1] += value

            # 60-second data
            for value,x,y in zip(glmxydata_60_sec['energyJoules'], 
                    np.round((glmxydata_60_sec['lon']-minLon) / degrees_per_pixel_lon).astype(int),
                    np.round((glmxydata_60_sec['lat']-minLat) / degrees_per_pixel_lat).astype(int)):
                if is_within_pixel_box(x,y,box_shape):
                    GLM_data[x,y,2] += value

            # 1-hour data
            for value,x,y in zip(glmxydata_1_hr['energyJoules'], 
                    np.round((glmxydata_1_hr['lon']-minLon) / degrees_per_pixel_lon).astype(int),
                    np.round((glmxydata_1_hr['lat']-minLat) / degrees_per_pixel_lat).astype(int)):
                if is_within_pixel_box(x,y,box_shape):
                    GLM_data[x,y,3] += value

            multi_band_data[key] = GLM_data

        else:
            raise Exception('Unknown image type')


    return multi_band_data

def is_within_pixel_box(x,y,pixel_box_shape):
    """ This will determine if x,y coordinates are within the bounds of the pixel box of shape pixel_bot_shape

    Dont forget that RGB images are referenced as (row, column)
    x is columns and y is rows, so, (y,x)

    """

    # The less than (and not equal to) is because Python is 0-based, but counting is 1-based.
    x_within = x >= 0 and x < pixel_box_shape[1]
    y_within = y >= 0 and y < pixel_box_shape[0]

    return x_within and y_within

def scatter_plot_mesh(grid_data):
    """ Plots a scatter plot of a 3D array of values

    Assumes the last dimension is size of 4


    # Convert the 3D array into 6 linear arrays
    array_shape = grid_data.shape
    x = []
    y = []
    z0 = []
    z1 = []
    z2 = []
    z3 = []
    for i in array_shape[0]:
        for j in array_shape[1]:
            for k in (0,1,2,3)
                x.append(grid_data[i,j,k]) = 

    plt.imshow(np.flip(skmeasure.block_reduce(GLM_data, 20), axis=0), vmin=0, vmax=1e-16);plt.show()

    """

    return


#************************************************************************************************************

class cutoutOutputClass:
    """ This contains the cutout output information for one detection candidate

    """
    success = False

    # Bolide detection ID
    ID = None

    # If this is true then there are multiple output figures,
    # otherwise, there is just a single vetting figure
    plot_seperate_figures = None
    # This can have one or more filenames
    figure_filenames = None

    # This is the name of the pickle file containing the multi-band image data
    bands = None
    image_pickle_filename = None

    def __init__(self, config, detection_ID):

        # Initialize with no success
        self.success = False
        self.figure_filenames = None
        self.bands = config.bands_to_read
        self.image_pickle_filename = None

        self.plot_seperate_figures = config.plot_seperate_figures
        self.ID = int(detection_ID)

    def copy(self):
        """Returns a deep copy of this object.

        Returns
        -------
        self : 
            A new object which is a copy of the original.
        """
        return deepcopy(self)

    def __repr__(self):
        return ioUtil.print_dictionary(self)

def gen_goes_cutout(bolideDetectionList, outputDir, config):
    """ Top level function to generate the ABI cutout figures with GLM bolide detection overlaid.

    Parameters
    ----------
    bolideDetectionList : BolideDetection list
        List of bolide detection candidates to generate cutout figures for
    outputDir   : str
        Output path to save generated figures
    config: cutoutConfiguration object
        Cutout Configuration object

    Returns
    -------
    cutout_outputs : Dict of cutoutOutputClass objects
        The names of the figures files generated and exit status
        The dictionary key is the detection ID
        Each entry is a cutoutOutputClass that includes the associated cutout filenemas for each detection candidate
    PNGs saved in output directory

    """
    startTime = time.time()

    # If generating seperate figures, we run gen_cutout twice, once to generate the full validation cutout figure, and
    # then a second time for the individual figures.
    # TODO: elliminate this inefficiency!
    # Create a second config object for this second type of processing.
    if config.plot_seperate_figures:
        config_indiv_figs = deepcopy(config)
        # The original config is for generating the full validation cutout figure
        config.plot_seperate_figures = False
    else:
        config_indiv_figs = None

        
    cutout_outputs = {}

    # Set up output dictionary for each detection
    for detection in bolideDetectionList:
        cutout_outputs[detection.ID] = cutoutOutputClass(config, detection.ID)

    if config.n_cores == 1:
        for detection in bolideDetectionList:
            #gen cutout
            detection.ID, success, figure_filenames, image_pickle_filename = gen_cutout(detection, outputDir, config)
            cutout_outputs[detection.ID].success = success
            cutout_outputs[detection.ID].figure_filenames = figure_filenames
            cutout_outputs[detection.ID].image_pickle_filename = image_pickle_filename

            # Individual figures
            if config_indiv_figs is not None:
                cutout_outputs_save = deepcopy(cutout_outputs)
                # Combine success arrays between both runs
                detection.ID, success, figure_filenames = gen_cutout(detection, outputDir, config_indiv_figs)
                cutout_outputs[detection.ID].success = success and cutout_outputs_save[detection.ID].success
                cutout_outputs[detection.ID].figure_filenames = figure_filenames
                cutout_outputs[detection.ID].plot_seperate_figures = True  

    else:
        if config.n_cores == -1:
            print('Warning: Cutout tool: Using maximum number of cores, irrespective of memory constraints.')
            print('We might run out of memory.')
        # This cutout tool can use a lot of memory per thread, like ~30 GB
        # Ensure there is at least config.min_GB per thread of memory available
        mem = psutil.virtual_memory()
        availMemGiB = mem.available / 2**30 # 2**30 = 1 GiB
        maxThreads = int(np.floor(availMemGiB / config.min_GB))
        n_cores = np.min([config.n_cores, maxThreads])
        with mp.Pool(processes=n_cores) as pool:
            results = [pool.apply_async(gen_cutout, 
                args=(detection, outputDir, config)) for detection in bolideDetectionList]
            cutout_outputs_pool = [result.get() for result in results]
            for output in cutout_outputs_pool:
                [detection_ID, success, figure_filenames, image_pickle_filename] = output
                cutout_outputs[detection_ID].success = success
                cutout_outputs[detection_ID].figure_filenames = figure_filenames
                cutout_outputs[detection_ID].image_pickle_filename = image_pickle_filename

      # success_array = [cutout_outputs[key].success for key in cutout_outputs.keys()]


        # Individual figures
        if config_indiv_figs is not None:
            cutout_outputs_save = deepcopy(cutout_outputs)

            with mp.Pool(processes=n_cores) as pool:
                results = [pool.apply_async(gen_cutout, 
                    args=(detection, outputDir, config_indiv_figs)) for detection in bolideDetectionList]
                cutout_outputs_indiv_figs = [result.get() for result in results]
                for output in cutout_outputs_indiv_figs:
                    [detection_ID, success, figure_filenames] = output
                    # Combine success arrays between both runs
                    cutout_outputs[detection_ID].success = success and cutout_outputs_save[detection_ID].success
                    cutout_outputs[detection_ID].figure_filenames = figure_filenames
                    cutout_outputs[detection_ID].plot_seperate_figures = True  

      # if np.logical_not(np.all(success_array)):
      #     raise Exception('Cutout tool: Error processing one or more detections')
            

    endTime = time.time()
    totalTime = endTime - startTime
    print('Cutout tool processing time: {:.2f} seconds, {:.2f} minutes'.format(totalTime, totalTime/60))

    return cutout_outputs 
        
def parse_arguments(arg_list):
    """
    Parse a command line argument list.
   
    Parameters
    ----------
    arg_list   : A list of strings, each containing a command line argument.
                 NOTE that the first element of this list should NOT be the
                 program file name. Instead of passing sys.argv, pass
                 arg_list = sys.argv[1:]
   
    Returns
    -------
    args : parse_arg object
        namespace populated by input arguments
    """

    parser = argparse.ArgumentParser(description='Generate ABI cutouts with bolide detectison overlaid.')
    parser.add_argument('detectionFiles', type=str, nargs='+',
                        help='One or more input detection CSV file names')
    parser.add_argument('--outdir', '-o', dest='outputDir', type=str, default=None,
                        help='Output directory name (default is save to save in same path as each detection file)')
    parser.add_argument('--noannotation', '-noann', dest='annotations', 
            action='store_false', help='Do not annotate the cutout figures with lat/lon lines')
    parser.add_argument('--num_cores', '-n', dest='n_cores', type=int, default=1,
                        help='Number of cores to use in parallelization')

    args = parser.parse_args(arg_list)

    return args


if __name__ == '__main__':
    """ Command line call to generate cutout figures

    """

    args = parse_arguments(sys.argv[1:])

    # Use default configuration
    config = cutoutConfiguration()

    config.annotations = args.annotations
    config.n_cores = args.n_cores

    gen_goes_cutout(args.detectionFiles, args.outputDir, config)
