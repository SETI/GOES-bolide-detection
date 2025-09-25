# This module contains code to take a detection in two GLM instruments in the stereo region
# and "re-navigate" the height of the groups so that they overlap.

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as md
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize_scalar

import bolide_clustering as bCluster
import bolide_detections as bd
import GLM_renavigation as renav
import geometry_utilities as geoUtil
import plotting_utilities as plotUtil

#*************************************************************************************************************
# This class stores the data needed to do the renavigation from each satellite detection
class DetectionData():
    def __init__(self, detection):
        """ Extracts the data from a detection necessary to plot

        When a renavigation fit is complete, the results are stored in this class.
    
        Parameters
        ----------
        detection : A bolide_detections.bolideDetection object.
    
        Attributes
        -------
        """
    
        # Sort groups by time
        detection.sort_groups()
     
        self.figureFilenameStereo   = detection.figureFilenameStereo
        self.ID             = detection.ID
        self.satellite      = detection.goesSatellite   
        self.filePathList   = detection.filePathList
        self.subPointLonDegreesEast = detection.subPointLonDegreesEast 

        [self.avgLat, self.avgLon] = detection.average_group_lat_lon
        self.timeArray      = np.array([g.time for g in detection.groupList])
        # datetime.timestamp() is UNIX time in seconds
        self.timestamps     = np.array([t.timestamp() for t in self.timeArray])
        self.latArray       = np.array([g.latitudeDegreesNorth for g in detection.groupList])
        self.lonArray       = np.array([g.longitudeDegreesEast for g in detection.groupList])
        self.energyArray    = np.array([e.energyJoules for e in detection.groupList])
    
        self.midDateAndTime = detection.bolideTime

        # These are computed in renavigate_stereo_detection
        # alignIndices gives the indices that aligns the two detections (pair of detections 1 and 2 with identical ID. 
        self.alignIndices = []

        # A stereo figure is plotted only if detection is in stereo region and data is available for both satellites
        self.stereo_figure_plotted = False

        # If this is True then there is stereo data but outside the altitude threshold limits set by
        # minAltToGenerateFigureKm and minAltToGenerateFigureKm to generate a figure.
        # None means not yet evaluated
        self.outsideAltLimits = None

        #***
        # Fitted results, initialize to NaN
        self.fitResults = FitResults(len(self.latArray))

        return

#*************************************************************************************************************
class FitResults():
    def __init__(self, nDatums):
        """ Re-navigation fit results

        Initialize all results to NaN.

        Parameters
        ---------
        nDatums : int
            The array length for this data
            Should be the same length as the detection data number of groups
        """

        self.fitSuccess = np.full(nDatums, False)

        # This gives the fitted altitude of the groups
        self.alt = np.full(nDatums, np.nan)

        # This gives the residual distance to the other satellite's renavigated groups in km
        self.residual_dist = np.full(nDatums, np.nan)

        # The fitted latitude and longitude at the re-navigated height
        self.lat = np.full(nDatums, np.nan)
        self.lon = np.full(nDatums, np.nan)

        # The ECEF 3-vector coordinate in km
        self.x = np.full(nDatums, np.nan)
        self.y = np.full(nDatums, np.nan)
        self.z = np.full(nDatums, np.nan)

        self.avgSpeed = np.nan

#*************************************************************************************************************
class RenavigateClass():

    # Some configuration parameters
    closenessSeconds = 1e-3
    height_bounds = [0.0, 500] # Height above lightning ellipse range
    max_iter = 100 # Maximum iterations for brents method
    #e1   -- Lightning ellipsoid altitude, equatorial, to which lat1 an lon1
    #        were originally navigated. Specified as km altitude above Earth
    #        ellipsoid equatorial radius. Original GLM value was 16 km.
    e1 = 16.0
    #p1   -- Lightning ellipsoid altitude, polar, to which lat1 and lon1 were
    #        originally navigated. Specified as km altitude above Earth
    #        ellipsoid polar radius. Original GLM value was 6 km.
    p1 = 6.0

    
    def __init__(self, input_config, provenance, bolideDetectionList):
        """ Sets up the data needed to perform the renavigation analysis. Extract data from the two detections.

        For each detection in bolideDetectionList, this will check if isInStereo == True, if so, it will attempt to
        perform the renavigation.

        If input_config.stereoDetectionOnlyMode is True then the two parameters minAltToGenerateFigureKm and
        maxAltToGenerateFigureKm to check what detections are in the altitude region and are stereo. Only those are kept
        in bolideDetectionList and so figures and the database entries are generated for only these events. 

        If input_config.stereoDetectionOnlyMode is False then only minAltToGenerateFigureKm is used to determein if
        stereo figures are generated. There are not changes to bolideDetectionList. (This is the standard pipeline
        configuration.)

        Parameters
        ----------
        input_config    : bolide_io.input_config object
        provenance      : [bolide_database.Provenance] Contains the provenance for this run
        bolideDetectionList -- [list of BolideDetection objects] The detected bolides

        Attributes
        ----------
        detection1List  : [DetectionData list] The exracted data needed to re-navigate and plot
        detection2List  : [DetectionData list] The exracted data needed to re-navigate and plot

        """

        # Check that IDs are unique for both detection lists (Do not need to be the same between the two. That is checked
        # later
        IDs = np.array([d.ID for d in bolideDetectionList])
        assert len(np.unique(IDs)) == len(IDs), 'bolideDetectionList IDs must be unique'

        self.input_config = input_config
        self.provenance = provenance

        # Search through all detections and find stereo detections
        self.detection1List = []
        self.detection2List = []
        IDsToRemove = []
        for idx, detection1 in enumerate(bolideDetectionList):
            if not detection1.isInStereo or detection1.bolideDetectionOtherSatellite is None:
                # Not in stereo region or no data available from other satellite, skip
                if self.input_config.stereoDetectionOnlyMode:
                    # We only want to keep stereo detections in bolideDetectionList
                   #bolideDetectionList.pop(idx)
                    IDsToRemove.append(detection1.ID)
                    continue
                else:
                    # Just skip, do not remove from bolideDetectionList
                    continue
        
            # Pull out data from both detections
            self.detection1List.append(DetectionData(detection1))
            self.detection2List.append(DetectionData(detection1.bolideDetectionOtherSatellite))

        # Pop cannot be vectorized : (
        for ID in IDsToRemove:
            bolideDetectionListIDs = [b.ID for b in bolideDetectionList]
            idx = bolideDetectionListIDs.index(ID)
            bolideDetectionList.pop(idx)

        assert len(self.detection1List) == len(self.detection2List), 'Bookkeeping error for RenavigateClass'

        # These contain the data for all groups within each set of netCDF files
        self.detection1AllGroupsList = []
        self.detection2AllGroupsList = []

    #*************************************************************************************************************
    def renavigate_all_detections(self):
        """ Loops through all detections and renavigates all detection pairs

        Will also renavigate all groups within the three surrounding 20-second netCDF files.

        Also computes a velocity vector from the renavigation.

        """

        # Reset these to empty sets because they are generated below
        self.detection1AllGroupsList = []
        self.detection2AllGroupsList = []

        detection2IDs = [d.ID for d in self.detection2List]
        for detection1 in self.detection1List:
            detection2 = self.detection2List[detection2IDs.index(detection1.ID)]

            self.renavigate_stereo_detection(detection1, detection2)
            # Compute velocity vector
            detection1.fitResults.avgSpeed = self.compute_velocity_tensor(detection1);
            detection2.fitResults.avgSpeed = self.compute_velocity_tensor(detection2);


            #******************
            # Renavigate all groups in the three 20-second files

            # Detection #1 data
            # First create a new BolideDetection object containing all groups in each file
            # We are just clustering all groups into a single cluster, so there is no need to use the configuration
            # parameters when initializing the BolideClustering object
            BCObj = bCluster.BolideClustering(detection1.filePathList, extractNeighboringFiles=False)
            # create single cluster of all groups
            clusterFound = BCObj.create_cluster_within_box(None, None, None, None, None, None)
            if clusterFound:
                # Generate BolideDetection objects for the single cluster
                # There is only one cluster, it's index is 0
                detection1AllGroups = bd.BolideDetection(BCObj, 0,
                            howFound='all_groups', retrieveEvents=False, 
                            features=None)
                # Set ID to that of the original detection
                detection1AllGroups.ID = detection1.ID
            # Convert to a stereo_renavigation data type
            self.detection1AllGroupsList.append(DetectionData(detection1AllGroups))
                
            # Detection #2 data
            filePathList = detection2.filePathList
            # We are just clustering all groups into a single cluster, so there is no need to use the configuration
            # parameters when initializing the BolideClustering object
            BCObj = bCluster.BolideClustering(filePathList, extractNeighboringFiles=False)
            # create single cluster of all groups
            clusterFound = BCObj.create_cluster_within_box(None, None, None, None, None, None)
            if clusterFound:
                # Generate BolideDetection objects for the single cluster
                # There is only one cluster, it's index is 0
                detection2AllGroups = bd.BolideDetection(BCObj, 0,
                            howFound='all_groups', retrieveEvents=False, 
                            features=None)
                # Set ID to that of the original detection
                detection2AllGroups.ID = detection2.ID
            self.detection2AllGroupsList.append(DetectionData(detection2AllGroups))
            
            # Perform the renavigation of all groups
            # The DetectionData objects were just generated and appened so they are the last elements in the lists.
            self.renavigate_stereo_detection(self.detection1AllGroupsList[-1], self.detection2AllGroupsList[-1])

    #*************************************************************************************************************
    def renavigate_stereo_detection(self, detection1, detection2):
        """ Takes a stereo region detection and attempts to combine the detection data from both satellites and 
        "re-navigate" to find the altitude of the event.
 
        Parameters
        ----------
        detection1   : [DetectionData] event in self.detection1List
        detection2   : [DetectionData] event in self.detection2List
 
        Returns
        success : [bool] True if the fit was successful for all cadences
        detection1.fitted_alt
        detection2.fitted_alt
        """

        # Check that both detections have the same ID
        assert detection1.ID == detection2.ID, 'Detections must have the same ID'

        #***
        # Align the groups in time
        # Find matching groups to within RenavigationClass.closenessSeconds
        # We only want to count each group once so, as pairs are found, check they are not already used in a pairing
        time2Array = np.array([t for t in detection2.timestamps])
        for idx1, time1 in enumerate(detection1.timestamps):
            timeDiff = np.abs(time1 - time2Array)
            sortedIdx = np.argsort(timeDiff)
            # Find the closest pairing that has not already been used yet
            for idx in sortedIdx:
                if (timeDiff[idx] <= self.closenessSeconds):
                    # Check if this index was already used
                    if not np.any(detection2.alignIndices == idx):
                        # Within time window and not used yet, keep this group
                        detection1.alignIndices.append(idx1)
                        detection2.alignIndices.append(idx)
                        # Go to the next detection 1 index
                        break

        #***
        # Take each pair of groups and "re-navigate" to the height where they overlap in Lat/Lon

        for idx1, idx2 in zip(detection1.alignIndices, detection2.alignIndices):
            # Set up the optimization parameters
            self.optimization_params = {'lat1': detection1.latArray[idx1],
                                        'lon1': detection1.lonArray[idx1],
                                        'lat2': detection2.latArray[idx2],
                                        'lon2': detection2.lonArray[idx2],
                                        'sat_lon_1': detection1.subPointLonDegreesEast,
                                        'sat_lon_2': detection2.subPointLonDegreesEast,
                                        }
            
            
            # Use scipy.optimize.minimize_scalar
            minimize_result = minimize_scalar(self._renav_obj_fun, method='Bounded',
                    bounds=self.height_bounds,
                    options={'maxiter':self.max_iter, 'disp': False})
            
            if minimize_result.success:
                detection1.fitResults.fitSuccess[idx1] = True
                detection2.fitResults.fitSuccess[idx2] = True

                # Compute and return all results
                all_results = self._renav_obj_fun(minimize_result.x, return_all_results=True)

                [lat1, lon1, alt1] = renav.event_altitude(all_results['x1'], all_results['y1'], all_results['z1'])
                [lat2, lon2, alt2] = renav.event_altitude(all_results['x2'], all_results['y2'], all_results['z2'])

                assert np.abs(all_results['dist'] - minimize_result.fun) < 1e-6, 'Error recomputing all results'

                # The measured altitude
                detection1.fitResults.alt[idx1] = alt1
                detection2.fitResults.alt[idx2] = alt2
                # The residual distance
                detection1.fitResults.residual_dist[idx1] = all_results['dist']
                detection2.fitResults.residual_dist[idx2] = all_results['dist']

                detection1.fitResults.lat[idx1] = all_results['lat1']
                detection1.fitResults.lon[idx1] = all_results['lon1']
                detection2.fitResults.lat[idx2] = all_results['lat2']
                detection2.fitResults.lon[idx2] = all_results['lon2']

                detection1.fitResults.x[idx1] = all_results['x1']
                detection1.fitResults.y[idx1] = all_results['y1']
                detection1.fitResults.z[idx1] = all_results['z1']
                detection2.fitResults.x[idx2] = all_results['x2']
                detection2.fitResults.y[idx2] = all_results['y2']
                detection2.fitResults.z[idx2] = all_results['z2']

            else:
                raise Exception('Could not converge on altitude solution')

        return True


    #*************************************************************************************************************
    def _renav_obj_fun(self, height, return_all_results=False):
        """ The objective function to minimize with
        scipy.optimize.minimize_scalar

        Adjusts the height of the ellipsoid until the two satellite detections overlap in lat/lon.

        Uses the paramaters in self.optimization_params.

        If return_all_results == True then return all computed values, not just the dist do use by minize_scalar.

        Parameters
        ----------
        height : [float] added height above default lightning ellipsoid in km
        return_all_results : [bool] If True then return all computed values

        Returns
        -------
        dist : float
            Penalty term for minimizer
            Distance between the two renavigated events in ECEF coordinates in km
        all_results : Dict [OPTIONAL]
            All computed values as a dictionary
            'dist'
            'lat1', 'lon1'
            'x1', 'y1', 'z1'
            'lat2', 'lon2'
            'x2', 'y2', 'x2'
        """

        e2 = self.e1 + height
        p2 = self.p1 + height

        [lat1, lon1, x1, y1, z1] = renav.renavigate(self.optimization_params['lat1'], 
                                                                   self.optimization_params['lon1'], 
                                                                   self.optimization_params['sat_lon_1'], 
                                                                   self.e1, 
                                                                   self.p1, 
                                                                   e2, 
                                                                   p2)

        [lat2, lon2, x2, y2, z2] = renav.renavigate(self.optimization_params['lat2'], 
                                                                   self.optimization_params['lon2'], 
                                                                   self.optimization_params['sat_lon_2'], 
                                                                   self.e1, 
                                                                   self.p1, 
                                                                   e2, 
                                                                   p2)

        # Compute distance between the two measured locations 
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

        if return_all_results:
            all_results = {
                            'dist': dist,
                            'lat1': lat1,
                            'lon1': lon1,
                            'x1': x1,
                            'y1': y1,
                            'z1': z1,
                            'lat2': lat2,
                            'lon2': lon2,
                            'x2': x2,
                            'y2': y2,
                            'z2': z2
                            }
            return all_results
        else:
            return dist

    #*************************************************************************************************************
    def find_bolide_neighborhood(self, detection, detectionAllGroups):
        """ Finds all groups within the neighborhood of the detection

        For detection, searches for all groups in detectionAllGroups that are in the neighborhood of detection. 

        Parameters
        ----------
        detection : bolide_detections.bolideDetection object.
        detectionAllGroups : bolide_detections.bolideDetection object conataining all groups in netCDF files around detection

        Returns
        -------
        withinNeighborhood : [logical np.array]
            Logical array of all groups in detection1AllGroups in neighborhood of detection

        """

        # This defines the neighborhodd distance
        radiusKm=500.0 
        radiusSec=2.0 

        distanceKm = geoUtil.DistanceFromLatLonPoints(detection.avgLat, detection.avgLon, detectionAllGroups.latArray, detectionAllGroups.lonArray)
        medianDetectionTime = np.median(detection.timestamps)
        timeDiffSec = np.abs(medianDetectionTime - detectionAllGroups.timestamps)

        withinNeighborhood = np.logical_and(distanceKm <= radiusKm, timeDiffSec <= radiusSec)

        return withinNeighborhood
        
    #*************************************************************************************************************
    @staticmethod
    def compute_velocity_tensor(detection):
        """ After stereo renavigation is performed, computes a veclotiy tensor.

        Returns
        -------
        


        """

        # Align the indices for the two arrays
        idx = detection.alignIndices

        x = detection.fitResults.x[idx]
        y = detection.fitResults.y[idx]
        z = detection.fitResults.z[idx]

        timestamps = detection.timestamps[idx]

        # Make sure all groups are sorted by time
        sortArray = np.argsort(timestamps)
        timestamps = timestamps[sortArray]
        x = x[sortArray]
        y = y[sortArray]
        z = z[sortArray]

        #***
        # Average speed
        speed = []
        # Compute the speed from every two points and average
        # Be smart about computing speed between any two points only once
        for idx in np.arange(len(x)):
            # Distance from this reference group to all groups after it in time
            dist = np.sqrt((x[idx]-x[idx+1:])**2 + (y[idx]-y[idx+1:])**2 + (z[idx]-z[idx+1:])**2)
            # Time from this reference group to all groups later in time
            deltaTSec = [(t - timestamps[idx]) for t in timestamps[idx+1:]]

            speed.extend(np.abs(dist / deltaTSec))
        avgSpeed = np.nanmedian(speed)


        #***
        # Instantaneous speed for every point
        speed = np.full(len(x), np.nan)
        for idx in np.arange(len(x)-1):
            # Distance from this group to the next group
            dist = np.sqrt((x[idx]-x[idx+1])**2 + (y[idx]-y[idx+1])**2 + (z[idx]-z[idx+1])**2)
            deltaTSec = timestamps[idx+1] - timestamps[idx]

            speed[idx] = np.abs(dist / deltaTSec)

        return avgSpeed
    

    #*************************************************************************************************************
    def plot_all_stereo_detections(self, output_dir=None, bolideDetectionList=None):
        """ Loops through all detections and generates renavigation figures

        It uses 
        self.input_config.minAltToGenerateFigureKm : [float] Minimum altitude to generate a figure
        self.input_config.maxAltToGenerateFigureKm : [float] Maximum altitude to generate a figure

        If input_config.stereoDetectionOnlyMode is True then the two parameters minAltToGenerateFigureKm and
        maxAltToGenerateFigureKm to check what detections are in the altitude region and are stereo. Only those are kept
        in bolideDetectionList and so figures and the database entries are generated for only these events. 

        If input_config.stereoDetectionOnlyMode is False then only minAltToGenerateFigureKm is used to determine if
        stereo figures are generated. There are not changes to bolideDetectionList. (This is the standard pipeline
        configuration.)

        Parameters
        ----------
        output_dir  : [str] Output path to save figures (None means do not save)

        """


        detection2IDs = [d.ID for d in self.detection2List]
        detection1AllGroupsIDs = [d.ID for d in self.detection1AllGroupsList]
        detection2AllGroupsIDs = [d.ID for d in self.detection2AllGroupsList]
        for detection1 in self.detection1List:

            detection2 = self.detection2List[detection2IDs.index(detection1.ID)]


            medianAlt1 = np.nanmedian(detection1.fitResults.alt)
            medianAlt2 = np.nanmedian(detection2.fitResults.alt)
            # If stereoDetectionOnlyMode then check if within both altitude limits
            if self.input_config.stereoDetectionOnlyMode:
                if medianAlt1 > self.input_config.maxAltToGenerateFigureKm and \
                   medianAlt2 > self.input_config.maxAltToGenerateFigureKm or \
                   medianAlt1 < self.input_config.minAltToGenerateFigureKm and \
                   medianAlt2 < self.input_config.minAltToGenerateFigureKm:
                    detection1.outsideAltLimits = True
                    detection2.outsideAltLimits = True
                    # Not in desired range, remove from bolideDetectionList
                    # Find index of detection in bolideDetectionList
                    bolideDetectionListIDs = np.array([b.ID for b in bolideDetectionList])
                    idx = np.nonzero(bolideDetectionListIDs == detection1.ID)
                    bolideDetectionList.pop(int(idx[0]))
                    continue

            else:
                # Just check if we are above the minimum altitude to generate a figure
                if medianAlt1 < self.input_config.minAltToGenerateFigureKm and \
                   medianAlt2 < self.input_config.minAltToGenerateFigureKm:
                    detection1.outsideAltLimits = True
                    detection2.outsideAltLimits = True
                    continue
                
            detection1AllGroups = self.detection1AllGroupsList[detection1AllGroupsIDs.index(detection1.ID)]
            detection2AllGroups = self.detection2AllGroupsList[detection2AllGroupsIDs.index(detection1.ID)]

            fig = self.plot_stereo_detection(detection1, detection2, detection1AllGroups, detection2AllGroups)

            # Save the figure
            if output_dir is not None:
                figureFilePath =  os.path.join(output_dir, detection1.figureFilenameStereo)
                fig.savefig(figureFilePath, dpi=150)
                plt.close(fig)

            detection1.stereo_figure_plotted = True
            detection2.stereo_figure_plotted = True
            detection1.outsideAltLimits = False
            detection2.outsideAltLimits = False

        return bolideDetectionList


    #*************************************************************************************************************
    def plot_stereo_detection(self, detection1, detection2, detection1AllGroups, detection2AllGroups):
        """ Plots detections from both satellites on same figure
 
        Parameters
        ----------
        detection1 : bolide_detections.bolideDetection object.
        detection2 : bolide_detections.bolideDetection object.
        detection1AllGroups : bolide_detections.bolideDetection object conataining all groups in netCDF files around detection
        detection2AllGroups : bolide_detections.bolideDetection object conataining all groups in netCDF files around detection

        Returns
        -------
        fig                 : A pyplot figure.

 
        """
        interactiveEnabled=False
        pointSelectionEnabled=False


        # Check that both detections have the same ID
        assert detection1.ID == detection2.ID, 'Detections must have the same ID'

        # Check that the residual distance is the same for both detections 
        assert np.all(detection1.fitResults.residual_dist[detection1.alignIndices] == detection2.fitResults.residual_dist[detection2.alignIndices]), \
                        'Detections must have the same fitted altitude'

        # Set the colors for each satellite
        # No need to have different colors for G16 and G19 or G17 and G18
        # Volor code:
        # G16, G18 => Blue
        # G17, G18 => Red
        det1Color = None
        det2Color = None
        if detection1.satellite in ['G16', 'G19']:
            det1Color = 'Blue'
        elif detection2.satellite in ['G16', 'G19']:
            det2Color = 'Blue'
        else:
            raise Exception('GOES 16 or 19 must be one of the satellites')

        if detection1.satellite in ['G17', 'G18']:
            det1Color = 'Red'
        elif detection2.satellite in ['G17', 'G18']:
            det2Color = 'Red'
        else:
            raise Exception('GOES 17 or 18 must be one of the satellites')
 
        assert det1Color is not None and det2Color is not None, 'Unknown satellite'
 
       #figSize = [8,12]
        figSize = [20,8]
       #figureLayout = (12,2)
        figureLayout = (8,4)
        fig = plt.figure(figsize=figSize)
 
        #********************
        # Original lat/Long Plot
        latLonAxisOrig = plt.subplot2grid(figureLayout, (1, 0), rowspan=2, colspan=1)
        
        latLonAxisOrig.plot(detection1.lonArray, detection1.latArray, linewidth=0.3)
        latLonAxisOrig.scatter(detection1.lonArray, detection1.latArray, marker='o',
                c=det1Color, linewidth=0.2, edgecolor='none', label=detection1.satellite)
        latLonAxisOrig.plot(detection2.lonArray, detection2.latArray, linewidth=0.3)
        latLonAxisOrig.scatter(detection2.lonArray, detection2.latArray, marker='o',
                c=det2Color, linewidth=0.2, edgecolor='none', label=detection2.satellite)
        
        plotUtil.set_ticks(latLonAxisOrig, 'x', 5, format_str='%.3f')
        plotUtil.set_ticks(latLonAxisOrig, 'y', 4, format_str='%.3f')
        
        plt.title('Original Lat/Lon')
        plt.ylabel('Lat [$\degree$]')
        plt.xlabel('Longitude [$\degree$]')
        plt.grid()

        #********************
        # Re-Navigated lat/Long Plot
        latLonAxisFit = plt.subplot2grid(figureLayout, (1, 1), rowspan=2, colspan=1)
        
        latLonAxisFit.plot(detection1.fitResults.lon, detection1.fitResults.lat, linewidth=0.3)
        latLonAxisFit.scatter(detection1.fitResults.lon, detection1.fitResults.lat, marker='o',
                c=det1Color, edgecolor='none', s=25, label=detection1.satellite)
        latLonAxisFit.plot(detection2.fitResults.lon, detection2.fitResults.lat, linewidth=0.3)
        latLonAxisFit.scatter(detection2.fitResults.lon, detection2.fitResults.lat, marker='o',
                c='none', edgecolor=det2Color, s=30, label=detection2.satellite)

        plotUtil.set_ticks(latLonAxisFit, 'x', 5, format_str='%.3f')
        plotUtil.set_ticks(latLonAxisFit, 'y', 4, format_str='%.3f')
        
        plt.title('Re-Navigated Lat/Lon')
        plt.ylabel('Lat [$\degree$]')
        plt.xlabel('Longitude [$\degree$]')
        plt.grid()

        #********************
        # All groups renavigated 
        renavAllAxis = plt.subplot2grid(figureLayout, (3, 0), rowspan=2, colspan=2)

        renavAllAxis.scatter(detection1AllGroups.timeArray, detection1AllGroups.fitResults.alt, marker='o', s=25, c='Cyan', edgecolor='none')
        renavAllAxis.scatter(detection1.timeArray, detection1.fitResults.alt, marker='o', s=25, c=det1Color, edgecolor='none')
        renavAllAxis.scatter(detection2AllGroups.timeArray, detection2AllGroups.fitResults.alt, marker='o', s=25, c='Cyan', edgecolor='none')
        renavAllAxis.scatter(detection2.timeArray, detection2.fitResults.alt, marker='o', s=30, c='none', edgecolor=det2Color)
       #renavAllAxis.set_ylim((0, np.max([100, np.nanmax(detection1AllGroups.fitted_alt)])))
        allTimes = np.concatenate((detection1AllGroups.timeArray, detection2AllGroups.timeArray), 0)
        minVal = np.min(allTimes) - timedelta( milliseconds=int(5) )
        maxVal = np.max(allTimes) + timedelta( milliseconds=int(5) )
        renavAllAxis.set_xlim((minVal, maxVal))
        plotUtil.set_ticks(renavAllAxis, 'x', 5, format_str='%S', date_flag=True)
        plotUtil.set_ticks(renavAllAxis, 'y', 4, format_str='%d')
        plt.title('All Groups Re-Navigated', y=1.0)
        plt.ylabel('Alt [km]')
        plt.xlabel('Seconds')
        plt.grid()

        #********************
        # Neighborhood renavigated 
        # Plot all groups in the neihborhood around the detection
        renavNeighAxis = plt.subplot2grid(figureLayout, (5, 0), rowspan=3, colspan=2)

        detection1WithinNeighborhood =self.find_bolide_neighborhood(detection1, detection1AllGroups)
        renavNeighAxis.scatter(detection1AllGroups.timeArray[detection1WithinNeighborhood],
                detection1AllGroups.fitResults.alt[detection1WithinNeighborhood], marker='o', s=25, c='Cyan', edgecolor='none')

        detection2WithinNeighborhood = self.find_bolide_neighborhood(detection2, detection2AllGroups)
        renavNeighAxis.scatter(detection2AllGroups.timeArray[detection2WithinNeighborhood],
                detection2AllGroups.fitResults.alt[detection2WithinNeighborhood], marker='o', s=25, c='Cyan', edgecolor='none')

        renavNeighAxis.scatter(detection1.timeArray, detection1.fitResults.alt, marker='o', s=25, c=det1Color, edgecolor='none')
        renavNeighAxis.scatter(detection2.timeArray, detection2.fitResults.alt, marker='o', s=30, c='none', edgecolor=det2Color)

       #renavNeighAxis.set_ylim((0, np.max([100, np.nanmax(detection1.fitted_alt)])))
        allTimes = np.concatenate((detection1AllGroups.timeArray[detection1WithinNeighborhood], 
                                   detection2AllGroups.timeArray[detection2WithinNeighborhood]), 0)
        minVal = np.min(allTimes) - timedelta( milliseconds=int(5) )
        maxVal = np.max(allTimes) + timedelta( milliseconds=int(5) )
        renavNeighAxis.set_xlim((minVal, maxVal))
        plotUtil.set_ticks(renavNeighAxis, 'x', 5, format_str='%S', date_flag=True)
        plotUtil.set_ticks(renavNeighAxis, 'y', 4, format_str='%d')
        plt.title('Neighborhood Re-Navigated', y=1.0)
        plt.ylabel('Alt [km]')
        plt.xlabel('Seconds')
        plt.grid()

        #********************
        # Renavigated Residual Plot
        renavResid = plt.subplot2grid(figureLayout, (1, 2), rowspan=2, colspan=2)
        renavResid.plot(detection1.timeArray, detection1.fitResults.residual_dist, linewidth=0.5, color='Green')
        renavResid.scatter(detection1.timeArray, detection1.fitResults.residual_dist, marker='o', s=25, c='Green', edgecolor='none')
        renavResid.minorticks_on()
       #renavResid.set_ylim((0, np.max([100, np.nanmax(detection1.residual_dist)])))
        allTimes = np.concatenate((detection1.timeArray, detection2.timeArray), 0)
        minVal = np.min(allTimes) - timedelta( milliseconds=int(5) )
        maxVal = np.max(allTimes) + timedelta( milliseconds=int(5) )
        renavResid.set_xlim((minVal, maxVal))
        plotUtil.set_ticks(renavResid, 'x', 6, format_str='%S.%f', date_flag=True)
        plotUtil.set_ticks(renavResid, 'y', 4)
       #plt.title('Median Residual Distance = {0:.2f} km'.format(np.nanmedian(detection1.residual_dist)), y=0.8, x=0.2)
        plt.title('Median Residual Distance = {0:.2f} km'.format(np.nanmedian(detection1.fitResults.residual_dist)))
        plt.ylabel('Res. Dist. [km]')
        plt.grid()


        #********************
        # Renavigated Altitude Plot
        renavAxis = plt.subplot2grid(figureLayout, (3, 2), rowspan=2, colspan=2)
        renavAxis.plot(detection1.timeArray, detection1.fitResults.alt, linewidth=0.5, color=det1Color)
        renavAxis.scatter(detection1.timeArray, detection1.fitResults.alt, marker='o', s=25, c=det1Color, edgecolor='none')
        renavAxis.plot(detection2.timeArray, detection2.fitResults.alt, linewidth=0.5, color=det2Color)
        renavAxis.scatter(detection2.timeArray, detection2.fitResults.alt, marker='o', s=30, c='none', edgecolor=det2Color)
        renavAxis.minorticks_on()
        renavAxis.minorticks_on()
       #renavAxis.set_ylim((0, np.max([100, np.nanmax(detection1.fitted_alt)])))
        renavAxis.set_xlim((minVal, maxVal))
        plotUtil.set_ticks(renavAxis, 'x', 6, format_str='%S.%f', date_flag=True)
        plotUtil.set_ticks(renavAxis, 'y', 4)
       #plt.title('Median Re-Navigated Altitude = {0:.2f} km'.format(np.nanmedian(detection1.fitResults.alt)), y=0.8, x=0.2)
        plt.title('Median Re-Navigated Altitude = {0:.2f} km'.format(np.nanmedian(detection1.fitResults.alt)))
        plt.ylabel('Alt [km]')
        plt.grid()
        renavAxis.annotate('{0:s} Speed = {1:.2f} km/s'.format(detection1.satellite, detection1.fitResults.avgSpeed),
                xy=(0.2, 1.17), xycoords='axes fraction')
        renavAxis.annotate('{0:s} Speed = {1:.2f} km/s'.format(detection2.satellite, detection2.fitResults.avgSpeed),
                xy=(0.5, 1.17), xycoords='axes fraction')

        #********************
        # Luminous Energy Plot
        energyAxis = plt.subplot2grid(figureLayout, (5, 2), rowspan=3, colspan=2)
        energyAxis.plot(detection1.timeArray, detection1.energyArray, linewidth=0.5, color=det1Color)
        energyAxis.scatter(detection1.timeArray, detection1.energyArray, marker='o', s=25, c=det1Color, edgecolor='none', label=detection1.satellite)
        energyAxis.plot(detection2.timeArray, detection2.energyArray, linewidth=0.5, color=det2Color)
        energyAxis.scatter(detection2.timeArray, detection2.energyArray, marker='o', s=25, c=det2Color, edgecolor='none', label=detection2.satellite)
        energyAxis.minorticks_on()
        energyAxis.set_xlim((minVal, maxVal))
        energyAxis.legend()
        plotUtil.set_ticks(energyAxis, 'x', 6, format_str='%S.%f', date_flag=True)
        plotUtil.set_ticks(energyAxis, 'y', 5, format_str='%.1e')
        plt.title('GLM Reported Energy')
       #plt.ylabel('Luminuous Energy [J]')
        plt.xlabel('Seconds')
        plt.grid()

        #********************
        # Extra text fields
       
        # Make the plot pretty
        plt.tight_layout(pad=0.0)
       #plt.tight_layout(h_pad=-0.8)
       
        # Annotate after applying tight_layout, otherwise the extra-subplot text will distort the tight_layout look 
        # (and shrink the subplot sizes.
       
        # Duration and Date
        latLonAxisOrig.annotate('Start Time = {}'.format(detection1.timeArray[0].strftime("%y-%m-%d %H:%M:%S")),
                xy=(0.0, 1.55), xycoords='axes fraction')
        latLonAxisOrig.annotate('ID = {}'.format(detection1.ID), xy=(0.0, 1.3), xycoords='axes fraction')
       
        # Mean lat/lon
        latLonAxisOrig.annotate('{} Median Lat/Lon: {:.3f},{:.3f}'.format(detection1.satellite, 
                                        np.median(detection1.latArray), np.median(detection1.lonArray)), 
                                        xy=(1.0, 1.55), xycoords='axes fraction')
        latLonAxisOrig.annotate('{} Median Lat/Lon: {:.3f},{:.3f}'.format(detection2.satellite, 
                                        np.median(detection2.latArray), np.median(detection2.lonArray)), 
                                        xy=(1.0, 1.3), xycoords='axes fraction')

        # Show first data file for detection1
        firstFile = os.path.basename(detection1.filePathList[0])
        nFiles = len(detection1.filePathList)
        if nFiles > 1 :
            label = 'GLM Data File (1st of {}): {}'.format(nFiles, firstFile)
        else :
            label = 'GLM Data File: {}'.format(firstFile)
        latLonAxisOrig.annotate(label, fontsize='medium', xy=(2.2, 1.55), xycoords='axes fraction')
        

       
        # Git branch to display pipeline version
        latLonAxisOrig.annotate('Generated by ATAP GOES GLM Pipeline', fontsize='medium', xy=(2.2, 1.3), xycoords='axes fraction')
        if self.provenance is not None:
            latLonAxisOrig.annotate('Branch = {}'.format(self.provenance.gitBranch), fontsize='medium', xy=(3.2, 1.3), xycoords='axes fraction')
       
        if(interactiveEnabled):
            plt.ion()
            plt.show()
       
        # Scatter point lasso tool
        if pointSelectionEnabled:
            axes = [renavAllAxis, energyAxis, lonAxis, latAxis]
            collection = [latLonPointsCollection, energyCollection, lonCollection, latCollection]
            selector = lassoTool.SelectFromCollection(axes, collection, alpha_other=0.0)
        else:
            selector = None

        return fig

    #*************************************************************************************************************
    def populate_stereo_features(self, bolideDetectionList):
        """ populates the stereo features object in the bolide detection list

        If the stereo figure was plotted then detection.stereoFigureGenerated is set to True

        This modifies the bolideDetectionList, which is returned back by reference (not copied)

        """

        detection1IDs = [d.ID for d in self.detection1List]
        detection2IDs = [d.ID for d in self.detection2List]
        for detection in bolideDetectionList:
            try: 
                detection1 = self.detection1List[detection1IDs.index(detection.ID)]
                detection2 = self.detection2List[detection2IDs.index(detection.ID)]
            except:
                # Detection data not available for both satellites, no stereo features to populate
                continue

            detectionG16 = None
            detectionG17 = None
            detectionG18 = None
            detectionG19 = None
            if detection1.satellite == 'G16':
                detectionG16 = detection1
            elif detection2.satellite == 'G16':
                detectionG16 = detection2

            if detection1.satellite == 'G17':
                detectionG17 = detection1
            elif detection2.satellite == 'G17':
                detectionG17 = detection2

            if detection1.satellite == 'G18':
                detectionG18 = detection1
            elif detection2.satellite == 'G18':
                detectionG18 = detection2

            if detection1.satellite == 'G19':
                detectionG19 = detection1
            elif detection2.satellite == 'G19':
                detectionG19 = detection2

            # Make sure we do not have too many or too few satellite's data
            if not np.logical_xor(detectionG16 is None, detectionG19 is None) or \
                not np.logical_xor(detectionG17 is None, detectionG18 is None):
                    raise Exception('Data for both satellites cannot be found.')

            if detectionG16 is not None:
                detection.stereoFeatures.sat_east = detection.stereoFeatures.G16
                detection.stereoFeatures.G16.medianAlt = np.nanmedian(detectionG16.fitResults.alt[detectionG16.alignIndices])
                detection.stereoFeatures.G16.medianSpeed = detectionG16.fitResults.avgSpeed
                detection.stereoFeatures.G16.medianResDist = np.nanmedian(detectionG16.fitResults.residual_dist[detectionG16.alignIndices])
                detection.stereoFeatures.G16.timestamps = detectionG16.timestamps[detectionG16.alignIndices]
                detection.stereoFeatures.G16.energyJoules = detectionG16.energyArray[detectionG16.alignIndices]
                detection.stereoFeatures.G16.lat  = detectionG16.fitResults.lat[detectionG16.alignIndices]
                detection.stereoFeatures.G16.lon  = detectionG16.fitResults.lon[detectionG16.alignIndices]
                detection.stereoFeatures.G16.alt  = detectionG16.fitResults.alt[detectionG16.alignIndices]
                detection.stereoFeatures.G16.x  = detectionG16.fitResults.x[detectionG16.alignIndices]
                detection.stereoFeatures.G16.y  = detectionG16.fitResults.y[detectionG16.alignIndices]
                detection.stereoFeatures.G16.z  = detectionG16.fitResults.z[detectionG16.alignIndices]
                detection.stereoFeatures.G16.residual_dist  = detectionG16.fitResults.residual_dist[detectionG16.alignIndices]
            elif detectionG19 is not None:
                detection.stereoFeatures.sat_east = detection.stereoFeatures.G19
                detection.stereoFeatures.G19.medianAlt = np.nanmedian(detectionG19.fitResults.alt[detectionG19.alignIndices])
                detection.stereoFeatures.G19.medianSpeed = detectionG19.fitResults.avgSpeed
                detection.stereoFeatures.G19.medianResDist = np.nanmedian(detectionG19.fitResults.residual_dist[detectionG19.alignIndices])
                detection.stereoFeatures.G19.timestamps = detectionG19.timestamps[detectionG19.alignIndices]
                detection.stereoFeatures.G19.energyJoules = detectionG19.energyArray[detectionG19.alignIndices]
                detection.stereoFeatures.G19.lat  = detectionG19.fitResults.lat[detectionG19.alignIndices]
                detection.stereoFeatures.G19.lon  = detectionG19.fitResults.lon[detectionG19.alignIndices]
                detection.stereoFeatures.G19.alt  = detectionG19.fitResults.alt[detectionG19.alignIndices]
                detection.stereoFeatures.G19.x  = detectionG19.fitResults.x[detectionG19.alignIndices]
                detection.stereoFeatures.G19.y  = detectionG19.fitResults.y[detectionG19.alignIndices]
                detection.stereoFeatures.G19.z  = detectionG19.fitResults.z[detectionG19.alignIndices]
                detection.stereoFeatures.G19.residual_dist  = detectionG19.fitResults.residual_dist[detectionG19.alignIndices]
            else:
                raise Exception('Either G16 or G19 must be present')

            if detectionG17 is not None:
                detection.stereoFeatures.sat_west = detection.stereoFeatures.G17
                detection.stereoFeatures.G17.medianAlt = np.nanmedian(detectionG17.fitResults.alt[detectionG17.alignIndices])
                detection.stereoFeatures.G17.medianSpeed = detectionG17.fitResults.avgSpeed 
                detection.stereoFeatures.G17.medianResDist = np.nanmedian(detectionG17.fitResults.residual_dist[detectionG17.alignIndices])
                detection.stereoFeatures.G17.timestamps = detectionG17.timestamps[detectionG17.alignIndices]
                detection.stereoFeatures.G17.energyJoules = detectionG17.energyArray[detectionG17.alignIndices]
                detection.stereoFeatures.G17.lat  = detectionG17.fitResults.lat[detectionG17.alignIndices]
                detection.stereoFeatures.G17.lon  = detectionG17.fitResults.lon[detectionG17.alignIndices]
                detection.stereoFeatures.G17.alt  = detectionG17.fitResults.alt[detectionG17.alignIndices]
                detection.stereoFeatures.G17.x  = detectionG17.fitResults.x[detectionG17.alignIndices]
                detection.stereoFeatures.G17.y  = detectionG17.fitResults.y[detectionG17.alignIndices]
                detection.stereoFeatures.G17.z  = detectionG17.fitResults.z[detectionG17.alignIndices]
                detection.stereoFeatures.G17.residual_dist  = detectionG17.fitResults.residual_dist[detectionG17.alignIndices]
            elif detectionG18 is not None:
                detection.stereoFeatures.sat_west = detection.stereoFeatures.G18
                detection.stereoFeatures.G18.medianAlt = np.nanmedian(detectionG18.fitResults.alt[detectionG18.alignIndices])
                detection.stereoFeatures.G18.medianSpeed = detectionG18.fitResults.avgSpeed 
                detection.stereoFeatures.G18.medianResDist = np.nanmedian(detectionG18.fitResults.residual_dist[detectionG18.alignIndices])
                detection.stereoFeatures.G18.timestamps = detectionG18.timestamps[detectionG18.alignIndices]
                detection.stereoFeatures.G18.energyJoules = detectionG18.energyArray[detectionG18.alignIndices]
                detection.stereoFeatures.G18.lat  = detectionG18.fitResults.lat[detectionG18.alignIndices]
                detection.stereoFeatures.G18.lon  = detectionG18.fitResults.lon[detectionG18.alignIndices]
                detection.stereoFeatures.G18.alt  = detectionG18.fitResults.alt[detectionG18.alignIndices]
                detection.stereoFeatures.G18.x  = detectionG18.fitResults.x[detectionG18.alignIndices]
                detection.stereoFeatures.G18.y  = detectionG18.fitResults.y[detectionG18.alignIndices]
                detection.stereoFeatures.G18.z  = detectionG18.fitResults.z[detectionG18.alignIndices]
                detection.stereoFeatures.G18.residual_dist  = detectionG18.fitResults.residual_dist[detectionG18.alignIndices]
            else:
                raise Exception('Either G17 or G18 must be present')

            detection.stereoFeatures.stereo_figure_plotted = detection1.stereo_figure_plotted and detection2.stereo_figure_plotted
            detection.stereoFeatures.outsideAltLimits = detection1.outsideAltLimits or detection2.outsideAltLimits

        return bolideDetectionList

#*************************************************************************************************************
