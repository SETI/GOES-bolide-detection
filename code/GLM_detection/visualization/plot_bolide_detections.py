#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as md
import sys
import os
import numpy as np
import bolide_detections as bd
import bolide_dispositions as bDisp
from datetime import datetime, timedelta
import geometry_utilities as geoUtil
import argparse
import multiprocessing as mp
from tqdm import tqdm
import scatter_lasso_tool as lassoTool

import bolide_database as bolideDB
import bolide_features as bFeatures
import io_utilities as ioUtil
import plotting_utilities as plotUtil
from lat_lon_to_GLM_pixel import LatLon2Pix
import time_utilities as timeUtils

figSize = [10,8]

#******************************************************************************
def plot_detections_parallel(bolideDetectionList, 
        provenance, 
        outDir, 
        latLon2Pix_table_path=None, 
        latLon2Pix_table_path_inverted=None,
        otherSatelliteFlag=False, 
        n_cores=1):
    """ Plots bolid edetection in parallel

    calls plot_detection using the multiprocessing module

    Parameters
    ----------
    bolideDetectionList : list of BolideDetection
        The detections to plot
    provenance : run_glm_pipeline_batch.Provenance
        Configuration of pipeline that generated dataset
        The passed provenenace can either be the original object os a dict derived from the object
    outDir  : str 
        path to save output files to
    latLon2Pix_table_path : str 
        Path to the lookup table file. 
        If passed then pixel boundaries are plotted (slow!)
        There are seperate files for each satellite so pass the correct one!
        Using this assumes all detections are from the same satellite
    latLon2Pix_table_path_inverted  : str 
        Path to the lookup table file for the inverted yaw orientation
        Only relevent for G17.
    otherSatelliteFlag  : bool 
        If True then Plot detection.bolideDetectionOtherSatellite data, if available.
    n_cores : int 
        Number of cores to use in parallel processing
        -1 means us all

    Returns
    -------
    Nothing, just figures saved to outDir
        

    """

    if n_cores == -1:
        # Use os.sched_getaffinity because it gives you the number of CPU cores available to this process, 
        # not the number of physical cores.
        n_cores = len(os.sched_getaffinity(0))

    # Using LatLon2Pix assumes all detections are for the same satellite
    if latLon2Pix_table_path is not None: 
        latLon2PixObj = LatLon2Pix(latLon2Pix_table_path, latLon2Pix_table_path_inverted)
    else:
        latLon2PixObj = None

    if n_cores < 1:
        raise Exception('n_cores should be an integer >= 1')
    elif n_cores == 1:
        fig = plt.figure(figsize=figSize)
        for detection in bolideDetectionList:

            [_, _] = plot_detection(detection, 
                    provenance=provenance, 
                    showDataFileName=True,
                    interactiveEnabled=False,
                    pointSelectionEnabled=False, 
                    figure=fig, 
                    glmGroupsWithNeighbors=None,
                    otherSatelliteFlag=otherSatelliteFlag, 
                    latLon2PixObj=latLon2PixObj,
                    outDir=outDir)
        plt.close(fig)
    else:
        with mp.Pool(processes=n_cores) as pool:
            # TODO: Figure out why named keywords is not working here
          # results = [pool.apply_async(plot_detection, 
          #     args=(detection),
          #     kwds={
          #         'provenance':provenance, 
          #         'showDataFileName':True,
          #         'interactiveEnabled':False,
          #         'pointSelectionEnabled':False, 
          #         'figure':None, 
          #         'glmGroupsWithNeighbors':None,
          #         'otherSatelliteFlag':otherSatelliteFlag, 
          #         'latLon2PixObj':latLon2PixObj,
          #         'outDir':outDir}) 
          #     for detection in bolideDetectionList]
            results = [pool.apply_async(plot_detection, 
                args=(detection,
                    provenance, 
                    True,
                    True, 
                    False,
                    False, 
                    None, 
                    None,
                    otherSatelliteFlag, 
                    latLon2PixObj,
                    True,
                    outDir)) 
                for detection in bolideDetectionList]
            outputs = [result.get() for result in results]
            # There are no results to examine, figures saved to file

            pass
            
    return

#******************************************************************************
def plot_detection(detection, 
        provenance=None, 
        showDataFileName=True, 
        showLineFit=True, 
        interactiveEnabled=False,
        pointSelectionEnabled=False, 
        figure=None,
        glmGroupsWithNeighbors=None, 
        otherSatelliteFlag=False, 
        latLon2PixObj=None,
        plotPixelAsColor=True,
        outDir=None):
    """
    Generate a diagnostic plot from a bolide detection object.
    
    Note: If you wish for the lasso selector to remain active, the selector object must be return and referenced.
    Also, call selector.disconnect to reset the figure when the user is finished.
   
    Parameters
    ----------
    detection           : A bolide_detections.bolideDetection object.
    provenance          : [run_glm_pipeline_batch.Provenance] Configuration of pipeline that generated dataset
                            The passed provenenace can either be the original object or a dict derived from the object
    showDataFileName    : If true, show the netCDF4 file name on the plot
    showLineFit         : If true, display the fitted line feature on the lat/lon plot
    interactiveEnabled  : [bool] If True, show the figure (and pause till figure is destroyed)
    pointSelectionEnabled : Use scatter_lasso_tool.py to allow interactive point selection. (default=False)
    figure              : [matplotlib.figure.Figure] If passed then use this figure handle
                            If you wish to create a new figure then pass None (default=None)
    glmGroupsWithNeighbors : [list of bolide_detection.GlmGroup] 
                A list of objects containing ALL the groups from the netCDF data files associated with this detection
                plus both neighboring files, sorted by time
                If None, then this list is generated from the netCDF files listed in each detection
    otherSatelliteFlag  : [bool] If True then add comment on figure that this data was detected in the other satellite
                                and the data plotted is just what was pulled out to be coincident with the detection
    latLon2PixObj       : [lat_lon_to_GLM_pixel.LatLon2Pix]
                If passed then the pixel boundaries are plotted on the lat/lon plots
    plotPixelAsColor    : [bool] If True, plot color represents which pixel group is on
                            Otherwise, color represents time of group
    outDir  : str 
        path to save output files to
        None means, do not save
        If saving figure then figure is closed after saved
   
    Returns
    -------
    fig                 : A pyplot figure.
    selector            : [SelectFromCollection] The selector object that uses LassoSelector
 
    """

    # If plotting the other satellite figures, then do so only if stereo_figure_plotted 
    if otherSatelliteFlag:
        if not detection.stereoFeatures.stereo_figure_plotted:
            return [None, None]
        assert detection.bolideDetectionOtherSatellite is not None, 'detection.bolideDetectionOtherSatellite is empty'
        assert detection.figureFilenameOtherSatellite is not None, 'detection.figureFilenameOtherSatellite is not available'
        # Save the other satellite figure file path now because we are overwriting the detection and will not be able to
        # construct the filename correctly below.
        figureFilePathOtherSatellite =  os.path.join(outDir, detection.figureFilenameOtherSatellite);
        detection = detection.bolideDetectionOtherSatellite

    # Convert the provennce object to a dict
    if not isinstance(provenance, dict) and provenance is not None:
        provenance = provenance.copy_to_dict()

    # Sort groups by time
    detection.sort_groups()
 
    timeArray       = [g.time for g in detection.groupList]
    timestamps      = [t.timestamp() for t in timeArray]
    lonArray        = np.array([g.longitudeDegreesEast for g in detection.groupList])
    latArray        = np.array([g.latitudeDegreesNorth for g in detection.groupList])
    deltaTSec       = (timeArray[-1] - timeArray[0]).total_seconds()
    energyArray     = np.array([e.energyJoules for e in detection.groupList])
    avgLat, avgLon  = detection.average_group_lat_lon

    eventLatArray    = np.array([e.latitudeDegreesNorth for e in detection.eventList])
    eventLonArray    = np.array([e.longitudeDegreesEast for e in detection.eventList])
    eventEnergyArray    = np.array([e.energyJoules for e in detection.eventList])
    eventTimeArray       = [e.time for e in detection.eventList]

   #[point0_ind, point1_ind, horizontal_flag] = geoUtil.FindFurthestLatLonPointsAlongAxis(latArray, lonArray)
   #dist = geoUtil.DistanceFromLatLonPoints(latArray[point0_ind], lonArray[point0_ind], latArray[point1_ind],
   #                             lonArray[point1_ind])
    dist = detection.features.ground_distance
    if dist == np.nan:
        # Features not computed, generate the ground_distance now
        dist = bFeatures.ground_distance_feature([detection])
        

    midDateAndTime      = detection.bolideTime


    #********************
    # Bolide neighborhood
    # Look at all groups within a window about the detection
    # Anything within the 20 second file that is within a 4 degree box around the detection
    ## load data
    
    neighborhoodFeature, isInNeighborhood, glmGroupsWithNeighbors = bFeatures.neighborhood_feature(detection, glmGroupsWithNeighbors=glmGroupsWithNeighbors)
    # we only want those in the neighborhood or closer

    if (isInNeighborhood is None):
        neighborhoodAvailable = False
    else:
        neighborhoodAvailable = True
        # These are for the groups within the neighborhood
        timeArrayNeighbors   = np.array([group.time for idx, group in enumerate(glmGroupsWithNeighbors) if isInNeighborhood[idx] == 0])
        energyArrayNeighbors = np.array([group.energyJoules for idx, group in enumerate(glmGroupsWithNeighbors) if isInNeighborhood[idx] == 0])
        latArrayNeighbors    = np.array([group.latitudeDegreesNorth for idx, group in enumerate(glmGroupsWithNeighbors) if isInNeighborhood[idx] == 0])
        lonArrayNeighbors    = np.array([group.longitudeDegreesEast for idx, group in enumerate(glmGroupsWithNeighbors) if isInNeighborhood[idx] == 0])
        
        # These are for the groups within the inner radius of the neighborhood
        timeArrayInnerNeighbors   = np.array([group.time for idx, group in enumerate(glmGroupsWithNeighbors) if isInNeighborhood[idx] == -1])
        energyArrayInnerNeighbors = np.array([group.energyJoules for idx, group in enumerate(glmGroupsWithNeighbors) if isInNeighborhood[idx] == -1])
        latArrayInnerNeighbors    = np.array([group.latitudeDegreesNorth for idx, group in enumerate(glmGroupsWithNeighbors) if isInNeighborhood[idx] == -1])
        lonArrayInnerNeighbors    = np.array([group.longitudeDegreesEast for idx, group in enumerate(glmGroupsWithNeighbors) if isInNeighborhood[idx] == -1])
    #********************

    # Only create a new figure if a figure handle is not passed.
    # Otherwise use the passed figure handle
    if figure is None or figure==[]:
        fig = plt.figure(figsize=figSize)
    else:
        fig = figure
        # Clear any previous plotting in the figure
        fig.clf()

    #***
    # Figure plot layout
    figureLayout = (10,2)

    #***
    #********************
    # lat/Long Plot
    latLonAxis = plt.subplot2grid(figureLayout, (1, 0), rowspan=2, colspan=1)

    latLonAxis.get_yaxis().get_major_formatter().set_useOffset(False)
    latLonAxis.get_yaxis().get_major_formatter().set_scientific(False)

    # If showing first data file then show that in title
    # Otherwise, display the GOES Satellite
    if showDataFileName :

        firstFile = os.path.basename(detection.filePathList[0])
        nFiles = len(detection.filePathList)
        if nFiles > 1 :
            label = 'GLM Data File (1st of {}): {}; ID: {}'.format(nFiles, firstFile, detection.ID)
        else :
            label = 'GLM Data File: {}; ID: {}'.format(firstFile, detection.ID)

        plt.title(label, fontsize='x-small', loc='left')
    else:
        plt.title('GOES Satellite: {}; ID: {}'.format(detection.goesSatellite, detection.ID))


    # plot line through points
    latLonAxis.plot(lonArray, latArray, linewidth=0.3)
    # Plot scatter of points
    if plotPixelAsColor:
        # Plot color represents which pixel group is on
        x = [g.x for g in detection.groupList]
        y = [g.y for g in detection.groupList]
        pixelColor = [xi + yi for xi,yi in zip(x,y)]
        latLonPointsCollection = latLonAxis.scatter(lonArray, latArray, marker='o', s=(energyArray / energyArray.max()) * 20 + 15,
            c=pixelColor, linewidth=0.2, edgecolor='none')
    else:
        latLonPointsCollection = latLonAxis.scatter(lonArray, latArray, marker='o', s=(energyArray / energyArray.max()) * 20 + 15,
            c=timestamps, linewidth=0.2, edgecolor='none')

    # Plot pixel boundaries
    if latLon2PixObj is not None:
        lat_boundary, lon_boundary = latLon2PixObj.find_pixel_boundaries(latArray, lonArray,
                yaw_flip_flag=detection.yaw_flip_flag)
        latLonAxis.scatter(lon_boundary, lat_boundary, marker='.', s=10,
            c='k', edgecolors='k', edgecolor='none')

   #latLonAxis.annotate('(a)', xy=(0.95, 0.85), xycoords='axes fraction')

    if showLineFit:
        linearityFeature = bFeatures.linearity_feature(detection, plot_figure=True, ax=latLonAxis)

    plt.ylabel('Latitude [$\degree$]')
    plt.xlabel('Longitude [$\degree$]')
    plt.grid()

    latLonAxis.get_yaxis().get_major_formatter().set_useOffset(False)
    latLonAxis.get_yaxis().get_major_formatter().set_scientific(False)
    latLonAxis.get_xaxis().get_major_formatter().set_useOffset(False)
    latLonAxis.get_xaxis().get_major_formatter().set_scientific(False)
    # We want the plot aspect ratio to default to square. 
    latLonAxis.axis('equal')
    # ...but we also want to be able to zoom in manually, so we need to disable "equal" aspect ratio and manually set to
    # the limits just set above so that zooming works.
    latLonAxis.figure.canvas.draw()
    axisDims = latLonAxis.axis()
    latLonAxis.axis('tight')
    latLonAxis.axis(axisDims)
    plotUtil.set_ticks(latLonAxis, 'x', 5, format_str='%.3f')
    plotUtil.set_ticks(latLonAxis, 'y', 4, format_str='%.3f')

    #***
    #********************
    # Luminous Energy Plot
    energyAxis = plt.subplot2grid(figureLayout, (3, 0), rowspan=2, colspan=2)
    energyAxis.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    energyAxis.xaxis.set_major_formatter(md.DateFormatter('%S.%f'))
    energyAxis.plot(timeArray, energyArray, linewidth=0.5)
    if plotPixelAsColor:
        energyCollection = energyAxis.scatter(timeArray, energyArray, marker='o', s=25, c=pixelColor, edgecolor='none')
    else:
        energyCollection = energyAxis.scatter(timeArray, energyArray, marker='o', s=25, c=timestamps, edgecolor='none')
    energyAxis.scatter(eventTimeArray, eventEnergyArray, marker='o', s=5, c='k', edgecolor='none', alpha=0.3)
    energyAxis.minorticks_on()
    plt.ylabel('Luminuous Energy [J]')
    plt.xlabel('Seconds')
    plt.grid()
    plotUtil.set_ticks(energyAxis, 'x', 6, format_str='%S.%f', date_flag=True)
    plotUtil.set_ticks(energyAxis, 'y', 4, format_str='%.3e')

    #***
    #********************
    # Longitude Plot
    lonAxis = plt.subplot2grid(figureLayout, (5, 0), rowspan=2, colspan=1)
    lonAxis.get_yaxis().get_major_formatter().set_useOffset(False)
    lonAxis.get_yaxis().get_major_formatter().set_scientific(False)
    lonAxis.xaxis.set_major_formatter(md.DateFormatter('%S.%f'))
    lonAxis.plot(timeArray, lonArray, linewidth=0.5)
    lonAxis.scatter(eventTimeArray, eventLonArray, marker='o', s=5, c='k', alpha=0.2)
    if plotPixelAsColor:
        lonCollection = lonAxis.scatter(timeArray, lonArray, marker='+', s=15, c=pixelColor)
    else:
        lonCollection = lonAxis.scatter(timeArray, lonArray, marker='+', s=15, c=timestamps)
    lonAxis.minorticks_on()
    plt.ylabel('Longitude [$\degree$]')
    plt.xlabel('Seconds')
    plt.grid()
    plotUtil.set_ticks(lonAxis, 'x', 4, format_str='%S.%f', date_flag=True)
    plotUtil.set_ticks(lonAxis, 'y', 3, format_str='%.3f')

    #***
    #********************
    # Latitude Plot
    latAxis = plt.subplot2grid(figureLayout, (5, 1), rowspan=2, colspan=1)
    latAxis.get_yaxis().get_major_formatter().set_useOffset(False)
    latAxis.get_yaxis().get_major_formatter().set_scientific(False)
    latAxis.xaxis.set_major_formatter(md.DateFormatter('%S.%f'))
    latAxis.plot(timeArray, latArray, linewidth=0.5)
    latAxis.scatter(eventTimeArray, eventLatArray, marker='o', s=5, c='k', alpha=0.2)
    if plotPixelAsColor:
        latCollection = latAxis.scatter(timeArray, latArray, marker='+', s=15, c=pixelColor)
    else:
        latCollection = latAxis.scatter(timeArray, latArray, marker='+', s=15, c=timestamps)
    latAxis.minorticks_on()
    plt.ylabel('Latitude [$\degree$]')
    plt.xlabel('Seconds')
    plt.grid()
    plotUtil.set_ticks(latAxis, 'x', 4, format_str='%S.%f', date_flag=True)
    plotUtil.set_ticks(latAxis, 'y', 3, format_str='%.3f')

    #********************
    # Bolide neighborhood

    if neighborhoodAvailable:
        #***
        # Plot on globe
        latLonNeighborAxis = plt.subplot2grid(figureLayout, (1, 1), rowspan=2, colspan=1)
        latLonNeighborAxis.get_yaxis().get_major_formatter().set_useOffset(False)
        latLonNeighborAxis.get_yaxis().get_major_formatter().set_scientific(False)
        latLonNeighborAxis.get_xaxis().get_major_formatter().set_useOffset(False)
        latLonNeighborAxis.get_xaxis().get_major_formatter().set_scientific(False)
        # Plot the neighbors
        latLonNeighborAxis .plot(lonArrayNeighbors, latArrayNeighbors, '.r', label='Neighbors')
        # Plot the inner neighbors
        latLonNeighborAxis .plot(lonArrayInnerNeighbors, latArrayInnerNeighbors, '.b', label='Inner Neighbors')
        # Plot the detection
        latLonNeighborAxis .plot(lonArray, latArray, '*g', label='Detection')
        plt.grid()
        latLonNeighborAxis.axis('equal')
        latLonNeighborAxis.figure.canvas.draw()
        axisDims = latLonNeighborAxis.axis()
        latLonNeighborAxis.axis('tight')
        latLonNeighborAxis.axis(axisDims)
        plt.ylabel('Latitude [$\degree$]')
        plt.xlabel('Longitude [$\degree$]')
        latLonNeighborAxis.annotate('Neighborhood Feature = {:.2f}'.format(neighborhoodFeature[0]), xy=(0.1, 0.85), xycoords='axes fraction')
        plotUtil.set_ticks(latLonNeighborAxis, 'x', 5, format_str='%.3f')
        plotUtil.set_ticks(latLonNeighborAxis, 'y', 4, format_str='%.3f')
        
        #***
        # Plot neighborhood groups energy vs time
        energyNeighborAxis = plt.subplot2grid(figureLayout, (7, 0), rowspan=2, colspan=2)
        energyNeighborAxis.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        energyNeighborAxis.xaxis.set_major_formatter(md.DateFormatter('%S.%f'))
        # Plot the neighbors
        energyNeighborAxis.plot(timeArrayNeighbors, energyArrayNeighbors, '.r', label='Neighbors')
        # Plot the inner neighbors
        energyNeighborAxis.plot(timeArrayInnerNeighbors, energyArrayInnerNeighbors, '.b', label='Inner Neighbors')
        # Plot the detection
        energyNeighborAxis.plot(timeArray, energyArray, '*g', label='Detection')
        # plot data file boundaries
        energyNeighborAxis.legend()
        energyNeighborAxis.minorticks_on()
        plt.grid()
        plt.xlabel('Time [Seconds]')
        plt.ylabel('Luminuous Energy [J]')
        plotUtil.set_ticks(energyNeighborAxis, 'x', 6, format_str='%S.%f', date_flag=True)
        plotUtil.set_ticks(energyNeighborAxis, 'y', 4, format_str='%.3e')
    else:
        latLonNeighborAxis = plt.subplot2grid(figureLayout, (1, 1), rowspan=2, colspan=1)
        latLonNeighborAxis.annotate('Neighborhood Not Available', xy=(0.25,   0.5), xycoords='axes fraction', color='red')
        energyNeighborAxis = plt.subplot2grid(figureLayout, (7, 0), rowspan=2, colspan=2)
        energyNeighborAxis.annotate('Neighborhood Not Available', xy=(0.37,   0.5), xycoords='axes fraction', color='red')

    #********************

    # Make the plot pretty
    plt.tight_layout(pad=0.0)
   #plt.tight_layout(pad=0.7)

    #********************
    # Top text
    x_pos = -0.33

    #***
    # Left text column
    # Detection Satellite, Duration and Date
    latLonAxis.annotate('{} ID: {}'.format(detection.goesSatellite, detection.ID), xy=(x_pos,  1.6), xycoords='axes fraction')
    latLonAxis.annotate('UTC Start Time={}'.format(timeArray[0].strftime("%y-%m-%d %H:%M:%S")), xy=(x_pos,  1.4), xycoords='axes fraction')
    latLonAxis.annotate('Duration={:.3f} s'.format(deltaTSec), xy=(x_pos, 1.2), xycoords='axes fraction')

    #***
    # Middle text column
    middle_offset = 0.87
    # Annotate after applyig tight_layout, otherwise the extra-subplot text will distort the tight_layout look 
    # (and shrink the subplot sizes.
    latLonAxis.annotate('Number of Groups: {}'.format(len(timeArray)), xy=(x_pos+middle_offset,   1.6), xycoords='axes fraction')
    # lat/lon distance travelled and speed
    latLonAxis.annotate('Rough Approx Dist: {:.2f} km'.format(dist), xy=(x_pos+middle_offset,   1.4), xycoords='axes fraction')
    if deltaTSec != 0.0:
        latLonAxis.annotate('Approx Ground Speed: {:.2f} km/s'.format(detection.avg_ground_speed),
                xy=(x_pos+middle_offset,  1.2), xycoords='axes fraction')
    else:
        latLonAxis.annotate('Approx Ground Speed: NaN km/s', xy=(x_pos+middle_offset,  1.2), xycoords='axes fraction')

    #***
    # Right text column
    right_offset = 1.7
    # Convert from UTC to solar time
    solar_time = timeUtils.convert_time_to_local(np.array([timeArray[0]]), np.array([avgLon]), local_type='meanSolar')[0]
    latLonAxis.annotate('Mean Solar Time={}'.format(solar_time.strftime("%y-%m-%d %H:%M:%S")), xy=(x_pos+right_offset,  1.6), xycoords='axes fraction')
    # Detection confidence
    if otherSatelliteFlag:
        pass
       #latLonAxis.annotate('Triage Score: {:.4f} (Other Satellite)'.format(detection.assessment.triage.score),
       #        xy=(x_pos+right_offset,   1.4), 
       #    xycoords='axes fraction', color='Blue')
    else:
        latLonAxis.annotate('Triage Score: {:.4f}'.format(detection.assessment.triage.score), xy=(x_pos+right_offset,   1.4), xycoords='axes fraction')
    # Mean lat/lon
    latLonAxis.annotate('Median Lat/Lon: {:.3f},{:.3f}'.format(avgLat, avgLon), xy=(x_pos+right_offset, 1.2), xycoords='axes fraction')

    #********************
    # Bottom text
    # Display creation date and time
    now = datetime.now()
    energyNeighborAxis.annotate('Generated by ATAP GOES GLM Pipeline at {}'.format(now.strftime("%y-%m-%d %H:%M:%S")), xy=(-0.1, -0.5), xycoords='axes fraction')
    # Git branch to display pipeline version
    if provenance is not None:
        energyNeighborAxis.annotate('Branch = {}'.format(provenance['gitBranch']), xy=(0.5, -0.5), xycoords='axes fraction')

    # If this bolide was stereo and detected in the other satellite then print a comment regarding this
    if otherSatelliteFlag:
        energyAxis.annotate('Stereo detection from other satellite', xy=(0.05, 0.85), xycoords='axes fraction', color='Blue') 
    elif detection.isInStereo and detection.stereoFeatures.stereo_figure_plotted:
        energyAxis.annotate('Stereo detection', xy=(0.05, 0.85), xycoords='axes fraction', color='Black') 
    elif detection.stereoFeatures.outsideAltLimits:
        energyAxis.annotate('Stereo detection but below alt. threshold', xy=(0.05, 0.85), xycoords='axes fraction', color='Red') 


    #********************
    if(interactiveEnabled):
        plt.ion()
        plt.show()

    # Scatter point lasso tool
    if pointSelectionEnabled:
        axes = [latLonAxis, energyAxis, lonAxis, latAxis]
        collection = [latLonPointsCollection, energyCollection, lonCollection, latCollection]
        selector = lassoTool.SelectFromCollection(axes, collection, alpha_other=0.0)
    else:
        selector = None

    if (outDir is not None):
        # Save the figure
        if otherSatelliteFlag:
            figureFilePath = figureFilePathOtherSatellite
        else:
            figureFilePath =  os.path.join(outDir, detection.figureFilename);
        fig.savefig(figureFilePath, dpi=150);
        plt.close(fig)

    return [fig, selector]


#******************************************************************************
# def plot_detections_from_file
# 
# Will generate bolide detection figures from a list of bolide detection files and saves figures to files. 
# One figure file per detection.
#
# This function is compatible with both the old pickle-based database and the new ZODB database. The function will
# autodetect which datastore is used based on the filename extension (.p for pickle, .fs for ZODB)
#
# Inputs:
#   inFiles -- [list] A list of bolide detection filenames (.p for pickle, .fs for ZODB)
#   outdir  -- [str] path to save output files to [Default = None]
#               If None then do not save output files
#   interactiveEnabled  -- [bool] If True, show the figure (and pause)
#                           (default=False)
#   statusBarEnabled    -- [bool] If True then the status bar is displayed. 
#                           Is NOT displayed if in interactive mode. Instead a single counter is shown.
#                           {Default=True}
#   IDsToPlot           -- [array of Int64] If not None then plot just the given IDs
#   numRandom           -- [int] Number of random detections to plot per file {Default = None, meaning, plot all in order) 
#   startDate           -- [str] The starting date for bolides to plot
#                           Is ISO format: 'YYYY-MM-DD[*HH[:MM[:SS[.fff[fff]]]]]'
#                           None or '' means no start date
#   endDate             -- [str] The ending date for bolides to plot
#                           Is ISO format: 'YYYY-MM-DD[*HH[:MM[:SS[.fff[fff]]]]]'
#                           None or '' means no end date
#   triageScoreThreshold -- [float] Only show detections >= to this threshold
#   latLon2Pix_table_path           -- [str] Path to the lookup table file. 
#                                       If passed then pixel boundaries are plotted (slow!)
#                                       There are seperate files for G61 and G17 so pass the correct one!
#   latLon2Pix_table_path_inverted  -- [str] Path to the lookup table file for the inverted yaw orientation
#                                           Only relevent for G17.
#   n_cores             -- [int] Number of cores to use in parallel processing
#                           can be >1 only if interactiveEnabled = False
#   
#
# Output:
#   One file per bolide detection
#******************************************************************************
def plot_detections_from_file (inFiles, outdir=None, interactiveEnabled=False, statusBarEnabled=True, 
        IDsToPlot=None, numRandom=None, startDate=None, endDate=None, triageScoreThreshold=0.0, 
            latLon2Pix_table_path=None, latLon2Pix_table_path_inverted=None, n_cores=1):

    if np.logical_or(IDsToPlot is not None, numRandom is not None):
        assert np.logical_xor(IDsToPlot is not None, numRandom is not None), "Cannot select both IDsToPlot and numRandom"

    assert np.logical_not(np.logical_and(interactiveEnabled, n_cores>1)), "n_cores can be >1 only if interactiveEnabled = False"

    if (interactiveEnabled or n_cores > 1):
        statusBarEnabled = False

    # Check if only a single file was passed as a string
    if isinstance(inFiles, str):
        inFiles = [inFiles]

    fig = plt.figure(figsize=figSize)

    # Using LatLon2Pix assumes all detections are for the same satellite
    if latLon2Pix_table_path is not None: 
        latLon2PixObj = LatLon2Pix(latLon2Pix_table_path, latLon2Pix_table_path_inverted)
    else:
        latLon2PixObj = None

    for filename in inFiles:
        if (os.path.splitext(filename)[1] == '.p'): 
            bolideDetectionList = bd.unpickle_bolide_detections(filename)
            provenance=None
        elif (os.path.splitext(filename)[1] == '.fs'): 
            bolideDatabase = bolideDB.BolideDatabase(filename)
            bolideDetectionList = bolideDatabase.extract_bolideDetectionList (dataType='detections')
            provenance=bolideDatabase.provenance
            bolideDatabase.close()
        else:
            raise Exception('Unknown filename extension for {}'.format(filename))

        # Select confidence threshold
        if triageScoreThreshold > 0.0:
            bolideDetectionList = [d for d in bolideDetectionList if d.assessment.triage.score >= triageScoreThreshold]

        # Select datetimes to keep
        if startDate is not None or endDate is not None:
            print('********')
            print('REMOVE ME!!!!!')
            raise Exception('Why is this still in here?')
            for detection in bolideDetectionList:
                detection.features.bolideTime = detection.bolideTime
            print('********')
            bolideDetectionList = bDisp.select_bolideDispositionProfileList_from_dates (bolideDetectionList, 
                    startDate=startDate, endDate=endDate)


        # Random detections to plot
        if (numRandom is not None):
            indicesToPlot = np.random.permutation(len(bolideDetectionList))
            indicesToPlot = indicesToPlot[:min([numRandom, len(bolideDetectionList)])]
        else:
            indicesToPlot = range(len(bolideDetectionList))

        # Specific detections to plot
        if IDsToPlot is not None:
            IDs = [detection.ID for detection in bolideDetectionList]
            indicesToPlot = np.nonzero(np.isin(IDs, IDsToPlot))[0]

        if (statusBarEnabled):
            pbar = tqdm(total=len(indicesToPlot), desc='Plotting bolides', disable=(not statusBarEnabled))


        if n_cores > 1:

            raise Exception('This appears to be test code and incomplete. Decide what to do with this')
            success = _plot_detections_from_file_parallel(bolideDetectionList[indicesToPlot[0]], provenance,
                    latLon2PixObj, filename, outdir)
            with mp.Pool(n_cores) as pool:
                jobs = [pool.apply_async(_plot_detections_from_file_parallel, args=(bolideDetectionList[iBolide], 
                    provenance, latLon2PixObj, filename, outdir)) for iBolide in indicesToPlot]
                outputs = []
                for job in tqdm(jobs):
                    outputs.append(job.get())

                if not np.all(outputs):
                    print('')
    
        else:
            for idx, iBolide in enumerate(indicesToPlot):
            
                # If the detection spans more than one file, use the first file as
                # the base name for the figure file.
                # But make sure each detection only uses its neighbors (not all loaded in)')
                # The filenames in the bolideDetectionList are relative to the directory where the bolideDatabase is located.
                # So, change the paths in bolideDetectionList[iBolide].filePathList to be relative to this path
                for fileIdx in np.arange(len(bolideDetectionList[iBolide].filePathList)):
                    bolideDetectionList[iBolide].filePathList[fileIdx] = \
                            os.path.join(os.path.split(filename)[0], bolideDetectionList[iBolide].filePathList[fileIdx])
                [fig, selector] = plot_detection(bolideDetectionList[iBolide], provenance=provenance, showDataFileName=True,
                                                  interactiveEnabled=interactiveEnabled,
                                                  pointSelectionEnabled=interactiveEnabled, 
                                                  figure=fig, glmGroupsWithNeighbors=None,
                                                  latLon2PixObj=latLon2PixObj)
            
                if (statusBarEnabled):
                    pbar.update()
                elif(interactiveEnabled):
                    print('Displaying {} of {}'.format(idx+1, len(indicesToPlot)))
            
                if(interactiveEnabled):
                    input('Hit the Any key to continue')
            
                    # Reset the figure and not select any points with the lasso
                    selector.disconnect()
            
                if (outdir is not None):
                    # Save the figure
                    figureFilePath =  os.path.join(outdir, bolideDetectionList[iBolide].figureFilename);
                    fig.savefig(figureFilePath, dpi=150);
            
            
            if (statusBarEnabled):
                pbar.close()

    plt.close(fig)

#*************************************************************************************************************
def _plot_detections_from_file_parallel(detection, provenance, latLon2PixObj, filename, outdir):
    """ Generates a plot figure for a single detection when calling plot_detections_from_file in parallel mode


    Returns
    -------
    success : bool
        If True then the processing was successful

    """

   #try:
    if True:
        # If the detection spans more than one file, use the first file as
        # the base name for the figure file.
        # But make sure each detection only uses its neighbors (not all loaded in)')
        # The filenames in the detection are relative to the directory where the bolideDatabase is located.
        # So, change the paths in detection.filePathList to be relative to this path
        for fileIdx in np.arange(len(detection.filePathList)):
            detection.filePathList[fileIdx] = \
                    os.path.join(os.path.split(filename)[0], detection.filePathList[fileIdx])
        [fig, selector] = plot_detection(detection, provenance=provenance, showDataFileName=True,
                                          interactiveEnabled=False,
                                          pointSelectionEnabled=False, 
                                          glmGroupsWithNeighbors=None,
                                          latLon2PixObj=latLon2PixObj)
        
        if (outdir is not None):
            # Save the figure
            figureFilePath =  os.path.join(outdir, detection.figureFilename);
            fig.savefig(figureFilePath, dpi=150);
        
        success = True
   #except:
   #    success = False

        plt.close(fig)

    return success


# *****************************************************************************
# Parse an argument list.
#
# INPUTS
#     arg_list   : A list of strings, each containing a command line argument.
#                  NOTE that the first element of this list should NOT be the
#                  program file name. Instead of passing sys.argv, pass
#                  arg_list = sys.argv[1:]
#
# OUTPUTS
#     args :
# *****************************************************************************
def parse_arguments(arg_list):
    parser = argparse.ArgumentParser(description='Plot bolide detections.')
    parser.add_argument('inFiles', metavar='inFiles', type=str, nargs='+',
                        help='One or more input file names')
    parser.add_argument('--outdir', '-o', dest='outdir', type=str, default=None,
                        help='Output directory name (default is None (do not save))')
    parser.add_argument('--interactive', '-i', dest='interactiveEnabled', 
            action='store_true', help='Display and Pause for each figure (default: False)')
    parser.add_argument('--IDs-to-plot', '-IDs', dest='IDsToPlot', type=int, default=None,
                        help='Specific IDs to plot')
    parser.add_argument('--number-random', '-n', dest='numRandom', type=int, default=None,
                        help='Number of random detections to plot')
    parser.add_argument('--startDate', '-s', dest='startDate', type=str, default=None,
                        help='Start date to plot (ISO format)')
    parser.add_argument('--endDate', '-e', dest='endDate', type=str, default=None,
                        help='End date to plot (ISO format)')
    parser.add_argument('--triage-score-threshold', '-c', dest='triageScoreThreshold', type=float, default=0.0,
                        help='Mininum triage score threshold')

    args = parser.parse_args(arg_list)

    return args

# *****************************************************************************
# Plot detections from one or more files and write the results to the
# designated output directory.
# *****************************************************************************
if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    outdir = args.outdir
    inFiles = args.inFiles
    interactive = args.interactiveEnabled
    IDsToPlot = args.IDsToPlot
    numRandom = args.numRandom
    startDate = args.startDate
    endDate = args.endDate
    triageScoreThreshold = args.triageScoreThreshold

    # Check to see whether the output directory exists and create it if necessary.
    if (outdir is not None):
        ioUtil.create_path(outdir, verbosity=True)

    # Generate and save the plots.
    plot_detections_from_file (inFiles, outdir, interactiveEnabled=interactive, IDsToPlot=IDsToPlot, numRandom=numRandom,
            startDate=startDate, endDate=endDate, triageScoreThreshold=triageScoreThreshold)

# ************************************ EOF ************************************
        
