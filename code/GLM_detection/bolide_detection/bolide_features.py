# This module contains features to be used for bolide classificiation

import numpy as np
from scipy import stats
from scipy import linalg
import matplotlib.pyplot as plt
from astropy.time import Time
from tqdm import tqdm
import os
import multiprocessing as mp
import datetime
import copy

from sklearn.compose import ColumnTransformer

import bolide_detections as bd
import geometry_utilities as geoUtil
import glint_point as gp
import bolide_filter_functions as bff
from geometry_utilities import equatorialEarthRadiusKm 
import io_utilities as ioUtil

#******************************************************************************
# FeaturesClass
#
# Here we store the computed features for each disposition. 
# All initialized to nan.
#
#******************************************************************************
class FeaturesClass:
    # These are all the feature labels available when generating a training data set.
    # This gives for each feature how to scale using scikit-learn ColumnTransformer
    # Some require robust standardization because there are a lot of outliers
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    featureScaleMethod = {
        'goesSatellite':        None,         
        'bolideTime':           MinMaxScaler(),         
        'bolideMidRange':       None,
        'avgLat':               MinMaxScaler(),         
        'avgLon':               MinMaxScaler(),         
        'startTime':            MinMaxScaler(),         
        'endTime':              MinMaxScaler(),         
        'maxEnergy':            RobustScaler(),         
        'totEnergy':            RobustScaler(),         
        'nGroups':              RobustScaler(),         
        'timeDuration':         RobustScaler(),         
        'latLonLinelets':       RobustScaler(),         
        'energyRatio':          StandardScaler(),         
        'splinelets':           RobustScaler(),         
        'groundTrackDeviation': RobustScaler(),         
        'chop':                 RobustScaler(),         
        'glint':                RobustScaler(),         
        'neighborhood':         RobustScaler(),         
        'latLonLinearity':      'passthrough',
        'hotSpot':              RobustScaler(),         
        'ground_distance':      RobustScaler(), # Total ground track distance travel of bolide candidate in kilometers
        'machine_opinion':      'passthrough', # This is used for post-classifier anlaysis
                }

    def __init__ (self):
        #TODO: If we add features here then also add to featureScaleMethod above and combine_features below.
        # Some simple features
        self.goesSatellite  = None
        self.bolideTime     = np.nan
        self.bolideMidRange = [np.nan, np.nan]
        self.maxEnergy      = np.nan
        self.totEnergy      = np.nan
        self.nGroups        = np.nan
        self.avgLat         = np.nan
        self.avgLon         = np.nan
        self.startTime      = bd.BolideDetection.epoch
        self.endTime        = bd.BolideDetection.epoch
        self.timeDuration   = np.nan # in Seconds
        self.ground_distance       = np.nan # in kilometers

        # These are the 6 filters from the prototype detector (2 are already above)
        self.latLonLinelets         = np.nan
        self.energyRatio            = np.nan
        self.splinelets             = np.nan
        self.groundTrackDeviation   = np.nan

        # Other features
        self.chop               = np.nan
        self.glint              = np.nan
        self.neighborhood       = np.nan
        self.latLonLinearity    = np.nan
        self.hotSpot            = np.nan

    #******************************************************************************
    # Performs a copy of object
    def copy(self):
        """Returns a copy of this object.

        Returns
        -------
        self : 
            A new object which is a copy of the original.
        """
        return copy.copy(self)

    #******************************************************************************
    # combines new feature set to current set
    def combine_features(self, newFeatures):
        """ Combines a new set of features with those in the self object.

        This is useful for example when combining two bolide detections that correspond to the same bolide event
        (as in bolide_detection_performance.combine_bolideDispositionProfiles_from_same_bolidesFromWebsite, and
        BolideDisposition.add_profile)

        Since the actual raw group data is not available, we can only do so much combining the features.
        So, this is a best-effort attempt. 

        DO NOT USE THIS IF YOU ARE ACTUALLY USING THE FEATURES TO CLASSIFY A DETECTION.

        Parameters
        ----------
        newFeatures : [FeatureClass] Features to combine with

        Returns
        -------
        self :
            With features combined

        """

        # Some simple features
        # Make sure satellite is the same
        assert self.goesSatellite  == newFeatures.goesSatellite, 'Can only combien features for saem satellite'

        # New time is average of two times
        self.bolideTime     = self.bolideTime + ((newFeatures.bolideTime - self.bolideTime)/2.0)
        # Midrange is extremes of the old midranges
        self.bolideMidRange = [np.min([self.bolideMidRange[0], newFeatures.bolideMidRange[0]]), 
                               np.max([self.bolideMidRange[1], newFeatures.bolideMidRange[1]])]
        # Energy, nGroups is the sum of the two component energies
        self.totEnergy      += newFeatures.totEnergy
        self.nGroups        += newFeatures.nGroups
        # Lat/lon is average of conponents
        self.avgLat         = np.mean([self.avgLat, newFeatures.avgLat])
        self.avgLon         = np.mean([self.avgLon, newFeatures.avgLon])

        # Time is min/max of components
        self.startTime      = np.min([self.startTime, newFeatures.startTime])
        self.endTime        = np.max([self.endTime, newFeatures.endTime])
        self.timeDuration   = self.endTime - self.startTime

        # Use the combined distance of both detections.
        # This is not entirely accurate because the two bolides could overlap.
        if hasattr(self, 'ground_distance'):
            self.ground_distance = self.ground_distance + newFeatures.ground_distance

        if hasattr(self, 'hotSpot'):
            self.hotSpot        = [np.min([h, n]) for h,n in zip(self.hotSpot, newFeatures.hotSpot)]

        # These cannot be recomputed without the groups, set to NaN
        self.latLonLinelets         = np.nan
        self.energyRatio            = np.nan
        self.splinelets             = np.nan
        self.groundTrackDeviation   = np.nan
        self.chop                   = np.nan
        self.glint                  = np.nan
        self.neighborhood           = np.nan
        self.latLonLinearity        = np.nan

    #******************************************************************************
    # check_for_valid_features
    #
    # Checks if the passed feature list <featuresToUse> are in this class
    #
    # Inputs:
    #   featuresToUse   -- [str list] List of features to use must be in FeaturesClass().__dict__.keys()
    #   allow_special_features -- [bool] If Ture then allow for some extra special features not in FeaturesClass but
    #   still properties of the bolide detection.
    #
    # Outputs:
    #   nProblems  -- [int] number of problem features
    #                   0 => no problems
    #
    #******************************************************************************
    @staticmethod
    def check_for_valid_features(featuresToUse, allow_special_features=False):

        # This is the list of extra special features to also allow in the list
        special_features = ['machine_opinion']

        validFeatureNames = FeaturesClass().__dict__.keys()
        validFeatureNames = [x for x in validFeatureNames]

        if allow_special_features:
            validFeatureNames.extend(special_features)

        nProblems = np.count_nonzero(np.logical_not(np.in1d(featuresToUse, validFeatureNames)))

        return nProblems
 

    # Returns a summary of the attributes in the class via a dict
    def __repr__(self): 
        return ioUtil.print_dictionary(self)

#******************************************************************************
def scale_features(feature_matrix, columnTransformer=None,  featuresToUse=None):
    """ Uses the passed columnTransformer to scale the features in a big matrix with the features in the order given by
    featuresToUse.

    If columnTransformer is not passed then a new columnTransformer is generated and fit using the feature matrix.

    Parameters
    ----------
    feature_matrix : np.array(nDatums, nFeatures)
        The matrix of features for scale
    columnTransformer : sklearn.compose.ColumnTransformer
        The already fit transformer to normalize the features
        if None then create a new transformer
    featuresToUse : [str list] 
        List of features to use in the order given in the feature_matrix
        must be in FeaturesClass().__dict__.keys()
        if columnTransformer is None then this is not used

    Returns
    -------
    feature_matrix_scaled : np.array(nDatums, nFeatures)
        The matrix of features scaled
    columnTransformer : sklearn.compose.ColumnTransformer
        The transformer used to scale th efeatures. 
        If columnTransformer is passed then this is just the same object returned.

    """

    if len(feature_matrix) == 0:
        feature_matrix_scaled = None

    elif (columnTransformer is None):
        assert featuresToUse is not None, 'If columnTransformer is None then featuresToUse should be passed'
        scalerList = []
        for idx, feature in enumerate(featuresToUse):
            scalerList.append(('scaler_{}'.format(idx), FeaturesClass.featureScaleMethod[feature], [idx]))
        columnTransformer = ColumnTransformer(scalerList)
        feature_matrix_scaled = columnTransformer.fit_transform(feature_matrix)

    else:
        feature_matrix_scaled = columnTransformer.transform(feature_matrix)

    return feature_matrix_scaled, columnTransformer

#******************************************************************************
# StereoFeaturesClass
#
# Here we store the computed stereo features for each disposition in the stereo region. 
# All initialized to nan.
#
#******************************************************************************
class StereoFeaturesClass:

    def __init__ (self):
        self.G16 = StereoSatelliteResults()
        self.G17 = StereoSatelliteResults()
        self.G18 = StereoSatelliteResults()
        self.G19 = StereoSatelliteResults()

        # This is a pointer to dictate which are the operational satellites in the East and West positions
        # Point this to one of the satellite objects above.
        self.sat_east = None
        self.sat_west = None

        # See RenavigateClass for details for these
        self.stereo_figure_plotted = False
        self.outsideAltLimits = None

    #******************************************************************************
    def above_min_alt_to_report_stereo(self, minAltToGenerateFigureKm):
        """ Returns True or False if the detection is high enough altitude to report stereo detection information and
        figures. 

        The logic here is that under a certain altitude a lot of false positives for stereo detection can be made. This
        is due to the large amount of lightning below about 20 km. 

        Parameters
        ----------
        minAltToGenerateFigureKm : float
            This is the minimum altitude in kilometers to measure and report a stereo detection
        """

        assert (self.sat_east is not None and self.sat_west is not None), \
            'Both GOES-East and GOES-West data must be available'

        assert self.sat_east.medianAlt is not None and self.sat_west.medianAlt is not None, \
                'Measure stereo altitude first before calling this method'

        if self.sat_east.medianAlt >= minAltToGenerateFigureKm or self.sat_west.medianAlt >= minAltToGenerateFigureKm:
            return True
        else:
            return False

        

    #******************************************************************************
    # Performs a deep copy of object
    def copy(self):
        """Returns a copy of this object.

        Returns
        -------
        self : 
            A new object which is a copy of the original.
        """
        return copy.deepcopy(self)


class StereoSatelliteResults:
    def __init__(self):
        #***
        # Some median values
        # Median measured re-navigated altitude
        self.medianAlt = np.nan
        # Median measured speed of the bolide after re-navigation
        self.medianSpeed = np.nan
        # median residual distance error after re-navigation
        self.medianResDist = np.nan

        #***
        # Arrays

        # timestamps associated with time series arrays
        self.timestamps = None

        self.energyJoules = None

        # Lat/lon/alt
        self.lat = None
        self.lon = None
        self.alt = None

        # Measured re-navigated ECEF coordinates of the groups
        self.x = None
        self.y = None
        self.z = None
        self.residual_dist = None # Residual fit error in km

    @property
    def max_dist(self):
        """ Computes the maximum traversal distance of the event in ECEF coordinates

        """

        if len(self.x) > 0:
            xMax = np.nanmax(self.x)
            xMin = np.nanmin(self.x)
            yMax = np.nanmax(self.y)
            yMin = np.nanmin(self.y)
            zMax = np.nanmax(self.z)
            zMin = np.nanmin(self.z)

            return np.sqrt((xMax-xMin)**2 + (yMax-yMin)**2 + (zMax-zMin)**2)

        else:
            return np.nan


    @property
    def max_ground_track(self):
        """ Computes the maximum ground track traversal distance of the event from lat and lon

        """

        if len(self.lat) > 0:
            latMax = np.nanmax(self.lat)
            latMin = np.nanmin(self.lat)
            lonMax = np.nanmax(self.lon)
            lonMin = np.nanmin(self.lon)
                
            return geoUtil.DistanceFromLatLonPoints(latMax, lonMax, latMin, lonMin)

        else:
            return np.nan

#******************************************************************************
class HotSpots:
    """ This stores any hot spots found in the bolde detections heat map.

    The hot spots are very simply kept as an array of lats and lons. Each pair corresponding to a hot spot.`

    Used in bolide_dispositions.find_bolide_hot_spots and hot_spot_feature
    """
    def __init__ (self, lon_peaks, lat_peaks, radius=None):

        assert len(lon_peaks) == len(lat_peaks), 'lat_peaks and lon_peaks must be the same length'
        if radius is not None:
            assert len(lon_peaks) == len(radius), 'radius must be the same length as the lats and lons'

        self.lon_peaks = lon_peaks
        self.lat_peaks = lat_peaks
        self.radius = radius

        return

    # Returns a summary of the attributes in the class via a dict
    def __repr__(self): 
        return ioUtil.print_dictionary(self)

'''
class HotSpotDistance:
    """ Stores the distance of each detection to each hot spot in km

    """
    def __init__ (self, ID, distArray):
        """ We construct this object with the bolide detection ID and the distance to each hot spot as an array in same
        order as in the HotSpots class.
        """

        self.ID = ID
        self.dist = distArray

        return
'''


#*************************************************************************************************************
# Chop Feature
#
# Measures how often the energy profile decreases back to near the minimum energy. 
#
# Many of the false positives observed are due to a lot of short spikes within the energy profile moving back down to near
# the minimum energy. This feature will measure the number of regions along the energy profile which dips below a
# nominal energy value.
#
# Inputs:
#   bolideDetectionList -- [list of bolide_detections.BolideDetection]
#   sigmaThreshold      -- [float] how many sigma must the energy profile drop below to count as dropping back 
#                           down as a chop point.
#   verbosity           -- [bool] If true, display progress bar
#   debug               -- [bool] If true, then generate diagnostic figures
#
# Outputs:
#   chopFeature -- [np.array float] feature values
#
#*************************************************************************************************************
def chop_feature(bolideDetectionList, sigmaThreshold=1.0, verbosity=False, debug=False):

    if (verbosity):
        pbar = tqdm(total=len(bolideDetectionList), desc='Computing Chop Feature')
    
    chopFeature = []

    for loopCount, detection in enumerate(bolideDetectionList):

        # These should already be sorted by time.
        energyArray     = np.array(detection.energy)
        nCadences = len(energyArray)

        # Compute a sigma from the median absolute deviation of the energy profile
        # sigma = 1.4826 * mad(x)
        # scipy,stats.median_abs_deviation does this scaling automatically if using the scale='normal' setting
        sigma = stats.median_abs_deviation(energyArray, scale='normal')
        threshold = sigma*sigmaThreshold

        #TODO: smooth the time series?

        # Count number of times energy profile dips back below sigmaThreshold
        chopCount = 0
        fallBelowSaveIdx = []
        idx = 0
        while (idx < nCadences-1):
            # Find next cadence well above the threshold
            wellAboveIdx = np.nonzero(energyArray[idx:-1] > 1.1*threshold)[0] + idx
            if (len(wellAboveIdx) < 1):
                # Nothing well above
                break
            else:
                idx = wellAboveIdx[0]

            # Find next cadence that falls below threshold
            fallBelowIdx = np.nonzero(energyArray[idx:-1] < threshold)[0] + idx
            if (len(fallBelowIdx) < 1):
                # Nothing below, reached end of time series
                break
            else:
                # Check next cadence to see if this is just an outlier
                if (np.all(energyArray[fallBelowIdx[0]+1:fallBelowIdx[0]+5] > threshold)):
                    # This is an outlier, reset idx to right after outlier and continue
                    idx = fallBelowIdx[0]+1
                    continue
                fallBelowSaveIdx.append(fallBelowIdx[0])
                chopCount += 1
                idx = fallBelowIdx[0]+1

        # Compute the density of chops per unit length of time
        chopFeature.append(chopCount / nCadences)

        if (verbosity and loopCount > 0 and np.mod(loopCount, 50) == 0):
            pbar.update(50)

        # Debug
        if (debug):
            fontsize=12
            plt.cla()
            plt.plot(energyArray, '*b', label='Group Energy')
            plt.plot([0, nCadences-1], [threshold, threshold], '-k', label='Threshold')
            for i in fallBelowSaveIdx:
                if (i == fallBelowSaveIdx[-1]):
                    plt.plot(i, energyArray[i], 'or', label='Transition')
                else:
                    plt.plot(i, energyArray[i], 'or')
            plt.title('Bolide Candidate Chop Feature = {:.2f}'.format(chopFeature[-1]), fontsize=fontsize)
           #plt.title('ID = {}; Chop Feature = {:.2f}'.format(disposition.BolideDetection.ID, chopFeature[-1]), fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.legend(fontsize=fontsize)
            plt.show()

        pass


    if (verbosity):
        pbar.close()

    return chopFeature

#*************************************************************************************************************
def glint_feature(spice_kernel_path, bolideDetectionList, verbosity=False, multiProcessEnabled=False, nJobs=None):
    """ 
    Computes the distance between the potential bolide object and the glint point in km.

    spiceypy is slow and cannot be parallelized. To speed up this function we have to generate new glint_ploint object at this level.
    The way we parallelize is to chunk the bolideDetectionList and generate a seperate GlintPoint class for each.

    Parameters
    ----------
    spice_kernel_path : str
                Path to kernels to load for glintPoint object
                If unavailabe then returns np.nan
    bolideDetectionList : [list of BolideDetection]
                List of bolide_detection.BolideDetection objects
    verbosity   : Bool
                If true, display progress bar
    multiProcessEnabled : bool
        If true parallelize the glint feature computation
    nJobs       : int
        Number of parallel jobs to use if multiProcessEnabled==True
                    If None then use os.cpu_count()

    Returns
    -------
    glintFeature : np.array float 
                feature values, one for each bolide object


    """

    if nJobs is None:
        nJobs = os.cpu_count()

    if (spice_kernel_path is None or not os.path.isdir(spice_kernel_path)):
        if (verbosity):
            print('No available Spice Kernels. Glint Feature disabled.')
        glintFeature = np.full(len(bolideDetectionList), np.nan)
        return glintFeature
        

    glintFeature = []

    # The spacecraft can move from its nominal point
    # Use the current spacecraft locationas given in the data file
    subPointLatDegreesNorthArray = [detection.subPointLatDegreesNorth for detection in bolideDetectionList]
    subPointLonDegreesEastArray = [detection.subPointLonDegreesEast for detection in bolideDetectionList]
    glmLatLonArray = np.array(list(zip(subPointLatDegreesNorthArray, subPointLonDegreesEastArray)))
    glmRadiusArray = np.array([detection.glmHeight for detection in bolideDetectionList] + \
                        np.full(len(subPointLatDegreesNorthArray), equatorialEarthRadiusKm))


    # Vectorized version
    # Determine location of glint point
    meanTimeArray = np.array([detection.bolideTime for detection in bolideDetectionList])
    satelliteArray = np.array([detection.goesSatellite for detection in bolideDetectionList])

    if (multiProcessEnabled):
        # Set the chunkSize so that the bolide list is divided between the desired number of jobs
        idxList = np.arange(len(bolideDetectionList))
        chunkSize = int(np.ceil(len(idxList) / nJobs))
        idxChunked = [idxList[i:i + chunkSize] for i in range(0, len(idxList), chunkSize)]  

        with mp.Pool(nJobs) as pool:
            results = [pool.apply_async(_glint_feature_elemental, args=(
                                spice_kernel_path,
                                satelliteArray[chunk],
                                meanTimeArray[chunk],
                                glmLatLonArray[chunk],
                                glmRadiusArray[chunk],
                                False)
                                ) for chunk in idxChunked]
            glintLatArray = []
            glintLonArray = []
            [[glintLatArray.extend(result.get()[0]), glintLonArray.extend(result.get()[1])] for result in results]
            glintLatArray = np.array(glintLatArray)
            glintLonArray = np.array(glintLonArray)
        

    else:
        glintLatArray, glintLonArray = _glint_feature_elemental(spice_kernel_path, satelliteArray, meanTimeArray, glmLatLonArray,
            glmRadiusArray, verbosity)

    # When the glint point is occulted by the Earth, the lat/lon returned in np.inf.   
    nanHere = np.logical_or(np.isnan(glintLatArray), np.isnan(glintLonArray))
    notNanHere = np.logical_not(nanHere)
    

    # Calculate distance between bolide center and glint center
    # TODO: Do this in one line (zip?)
    bolideLatArray = np.array([detection.average_group_lat_lon[0] for detection in bolideDetectionList])
    bolideLonArray = np.array([detection.average_group_lat_lon[1] for detection in bolideDetectionList])


    # Distance between glint point and bolide cluster in km
    # If glint is occulted then set distance to 0.5 times Earth's circumference
    circumEarth = geoUtil.faiEarthRadiusKm * 2.0 * np.pi
    glintFeature = np.full(len(meanTimeArray), circumEarth*0.5)
    glintFeature[notNanHere] = geoUtil.DistanceFromLatLonPoints(glintLatArray[notNanHere], glintLonArray[notNanHere], 
                                                                bolideLatArray[notNanHere], bolideLonArray[notNanHere])

    # Check the ranges of the glintFeature
    # It should be between 0 and half the circumference of the Earth
    assert np.all(np.greater_equal(glintFeature[notNanHere], 0.0)) and np.all(np.less(glintFeature[notNanHere], circumEarth*0.5)), \
                        'Out of Range Error for Glint Feature'

    return glintFeature

#*************************************************************************************************************
def _glint_feature_elemental (spice_kernel_path, satelliteArray, meanTimeArray, glmLatLonArray,
        glmRadiusArray, verbosity):
    """ Seperating the computing of the glint point into a helper function so that the glint feature computation can be parallelized.
    """

    glintPoint = gp.GlintPoint(spice_kernel_path)
    
    glintLatArray, glintLonArray = glintPoint.glint_spot(satelliteArray, meanTimeArray, glm_latlon=glmLatLonArray,
            glm_radius=glmRadiusArray, verbosity=verbosity)

    return glintLatArray, glintLonArray

#*************************************************************************************************************
def neighborhood_feature(bolideDetectionList, 
        glmGroupsWithNeighbors=None, 
        innerRadiusLimitKm=30.0, 
        innerRadiusLimitSec=2.0, 
        outerRadiusLimitKm=500.0,
        verbosity=False, debug=False):
    """
    Measures the degree of crowding of GLM groups around the bolide detection.

    There are two circle radii. Anything between the inner and outer circles are counted toward the neighborhood feature.

    This requires reading in the data for all groups in the file. So, to speed this up, one can optionally pass all the
    glmGroupsWithNeighbors as stored by a BolideClustering object.

    If the neighborhood groups cannot be found (perhaps the data files do not exist) then the returned feature value is np.nan.

    Parameters
    ----------
    bolideDetectionList : [list of BolideDetection]
                List of bolide_detection.BolideDetection objects
    glmGroupsWithNeighbors           : [list of bolide_detection.GlmGroup] 
                A list of objects containing ALL the groups from the netCDF data file and neighboring files, sorted by time
                If None then find all groups based on bolideDetectionList.filePathList and both neighboring files
    innerRadiusLimitKm  : [float] Radius of inner circle (not counted toward neighborhood feature)
    innerRadiusLimitSec : [float] How close in time inner circle goups need to be to be not counted toward neighborhood feature
    outerRadiusLimitKm  : [float] Radius of outer circle (IS counted toward neighborhood feature)
    verbosity           : [bool] If true, display progress bar
    debug               : [bool] If true, then generate diagnostic figures

    Returns
    -------
    neighborhoodFeature : np.array float
                feature values, one for each bolide object
    isInNeighborhood : [int nNeighborhoodGroups] 
                logical array giving if each groups is in the neighborhood of each detection
                NOTE: Only returned if len(bolideDetectionList) = 1
                TODO: Do something better here!
                -1 => within inner radius
                 0 => in neighborhhood
                 1 => outside outer radius
                 99 => in the detection itself!
    glmGroupsWithNeighbors           : [list of bolide_detection.GlmGroup] 
                The glmGroupsWithNeighbors list used in the processing


    """

    if (not isinstance(bolideDetectionList, list)):
        bolideDetectionList = [bolideDetectionList]

    assert isinstance(bolideDetectionList[0], bd.BolideDetection), 'Must pass list of bolide_detections.BolideDetection'

    # If we dfoudn the groups with neighbors then we only need to create these lists once.
    # Otherwise, we need to generate for each detection.
    if glmGroupsWithNeighbors is None:
        glmGroupsWithNeighborsLoaded = False
    else:
        glmGroupsWithNeighborsLoaded = True
        # Find the density of groups within an inner and outer radius about each detection
        latArrayNeighbors = np.array([group.latitudeDegreesNorth for group in glmGroupsWithNeighbors])
        lonArrayNeighbors = np.array([group.longitudeDegreesEast for group in glmGroupsWithNeighbors])
        # Set up np.array of time as a float wrt a reference epoch
        # This is for speed since datetime and list completion is slow
        timeArrayNeighborsSecs  = np.array([(group.time - bd.BolideDetection.epoch).total_seconds() for group in glmGroupsWithNeighbors])
        idNeighbors = np.array([group.id for group in glmGroupsWithNeighbors])


    neighborhoodFeature = []
    for detection in bolideDetectionList:

        # Check if we already have the GLM groups or need to extract them
        # If the GLM Groups with neighbors were loaded then do not re-find them
        # This is slow code if we do need to find the neighbors
        if not glmGroupsWithNeighborsLoaded:
            filenames = []
            # Get list of all data files from the detection
            for filePath in detection.filePathList:
                # Check if filename is in the list, if not, add it
                if (os.path.exists(filePath) and filenames.count(filePath) <= 0):
                    filenames.append(filePath)

            if len(filenames) == 0:
                # No data files available, cannot compute feature
                neighborhoodFeature.append(np.nan)
                isInNeighborhood = None
                glmGroupsWithNeighbors = None
                continue
        
            # The detections in bolideDetectionList should already contain a list of all data files, including
            # neighbors, so set extractNeighboringFiles=False
            [glmGroupsWithNeighbors, _] = bd.extract_groups_from_all_files(filenames, eventIdListEnabled=False, ignoreZeroEnergyGroups=True, 
                    ignorePoorQualityGroups=True, extractNeighboringFiles=False)

            # Find the density of groups within an inner and outer radius about each detection
            latArrayNeighbors = np.array([group.latitudeDegreesNorth for group in glmGroupsWithNeighbors])
            lonArrayNeighbors = np.array([group.longitudeDegreesEast for group in glmGroupsWithNeighbors])
            # Set up np.array of time as a float wrt a reference epoch
            # This is for speed since datetime and list completion is slow
            timeArrayNeighborsSecs  = np.array([(group.time - bd.BolideDetection.epoch).total_seconds() for group in glmGroupsWithNeighbors])
            idNeighbors = np.array([group.id for group in glmGroupsWithNeighbors])

        isInNeighborhood = np.full(len(glmGroupsWithNeighbors), 9999)

        [avgLat, avgLon] = detection.average_group_lat_lon
        detectionGroupIds = np.array([group.id for group in detection.groupList])
        distanceKm = geoUtil.DistanceFromLatLonPoints(avgLat, avgLon, latArrayNeighbors, lonArrayNeighbors)
        timeDiffSec = np.abs((detection.bolideTime - bd.BolideDetection.epoch).total_seconds() - timeArrayNeighborsSecs)
        # Find all groups within the distance and time regions, but are not the groups in the detection
        notInDetection = np.logical_not(np.isin(idNeighbors, detectionGroupIds))
        withinOuterRadius = distanceKm <= outerRadiusLimitKm
        withinInnerRadius = np.logical_and(distanceKm <= innerRadiusLimitKm, timeDiffSec <= innerRadiusLimitSec)
        keepList = np.nonzero(np.logical_and(
            np.logical_and(withinOuterRadius, np.logical_not(withinInnerRadius)), notInDetection))[0]

        # Neighborhood feature is ratio of number of neighbors to number of groups in detection
        neighborhoodFeature.append(len(keepList) / len(detectionGroupIds))
        isInNeighborhood[keepList] = int(0)
        isInNeighborhood[np.nonzero(np.logical_and(notInDetection, withinInnerRadius))[0]] = int(-1) 
        isInNeighborhood[np.nonzero(np.logical_and(notInDetection, np.logical_not(withinOuterRadius)))[0]] = int(1)
        isInNeighborhood[np.logical_not(notInDetection)] = int(99)

        # If any are still 9999 then we did not account for all groups
        assert np.logical_not(np.any(isInNeighborhood[:] == 9999)), 'Bookkeeping error'
    

    # Only return a meaningful isInNeighborhood and glmGroupsWithNeighbors if only one detection was passed 
    if (len(bolideDetectionList) > 1):
        isInNeighborhood = None
        glmGroupsWithNeighbors = None


    # Just return the number of groups within the neighborhood borders
    neighborhoodFeature = np.array(neighborhoodFeature)

    return neighborhoodFeature, isInNeighborhood, glmGroupsWithNeighbors

#*************************************************************************************************************
def linearity_feature(bolideDetectionList, plot_figure=False, ax=None):
    """ 
    Compute a linearity feature from the weighted covariance matrix of points
    in x, y. The feature is the proportion of variation explained by the
    principal axis of the weighted point cloud.

    Each point in the model is weighted by the group flux. This is so that the brightest groups points are used to
    assess linearity and to ignore very sim groups, which may be outliers
   
    Parameters
    ----------
    bolideDetectionList : [list of BolideDetection]
                List of bolide_detection.BolideDetection objects
    plot_figure : bool
        If True the plot a figure
        Can only plot if there is only a single detection in bolideDetectionList
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis to plot the data on
        If none then create a new figure

        x : An N-length vector of x coordinates.
        y : An N-length vector of y coordinates.
        w : An N-length vector of weights.
   
    Returns
    -------
    linearityFeature : np.array of floats of range [0.0,1.0]
                        0 => bad, not very linear, 1 => good, very linear
   
    """

    if not isinstance(bolideDetectionList, list):
        bolideDetectionList = [bolideDetectionList]

    if plot_figure:
        assert len(bolideDetectionList) == 1, 'Plotting can only happen if there is only a single detection in bolideDetectionList'
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot()

    linearityFeature = []
    for detection in bolideDetectionList:
        # Extract lat, lon and flux
        latArray    = np.array([g.latitudeDegreesNorth for g in detection.groupList])
        lonArray    = np.array([g.longitudeDegreesEast for g in detection.groupList])
        energyArray = np.array([g.energyJoules         for g in detection.groupList])
        
        
        xy = np.array([lonArray, latArray]).T
        
        # Set point weighting by energy, mean normalized
        w = energyArray / np.mean(energyArray)
        weightSum = np.nansum(w)

        # Center the data.
        mu = np.nansum(xy * np.tile(w, (2, 1)).T, axis=0) / weightSum
        A = xy - np.tile(mu, (np.shape(xy)[0], 1))

        # Compute the weighted covariance matrix.
        covMat = (A.T @ np.diag(w) @ A ) / weightSum
        
        # Compute the eigenvalues.
        eigValues, v = linalg.eig(covMat)
        
        # Calculate the proportion of variation explained by the principal axis. 
        s = np.sqrt(eigValues)
        if np.nanmax(s) > 0.0:
            linearity = np.real(1 - np.nanmin(s) / np.nanmax(s))
        else:
            linearity = 0.0

        linearityFeature.append(linearity)

        if plot_figure:
            # Diagnostic plots
            sp = ax.scatter(lonArray, latArray, c=w)
            #***
            # Plot the primary eignevector
            princVec = v[:,np.argmax(eigValues)]
            # Plot the principle eigenvector across the data
            # Uncenter the principle eigenvector
            princVec = princVec + mu
            xmin = np.min(lonArray)
            xmax = np.max(lonArray)
            ymin = line_fcn(xmin, mu, princVec)
            ymax = line_fcn(xmax, mu, princVec)
            ax.plot([xmin, xmax], [ymin, ymax], '-k', label='Principal Eigenvector')
           #plt.title('Linearity = {}'.format(linearity))
            ax.annotate('Linearity Feature = {:.2f}'.format(linearity), xy=(0.1, 0.85), xycoords='axes fraction')
            plt.xlabel('Longitude [Deg.]')
            plt.ylabel('Latitude [Deg.]')
            plt.colorbar(sp, ax=ax, label='Flux Weighting')
           #plt.grid()
           #plt.show(block=False)
            pass


    linearityFeature = np.array(linearityFeature)

    return linearityFeature

def line_fcn(x, x0, x1):
    """ For a line defined by the two points x0 and x1, 
    Return a y-coord for the given x-coord on this line

    """

    return (x - x0[0]) * (x1[1] - x0[1])/(x1[0] - x0[0]) + x0[1]


#*************************************************************************************************************
def hot_spot_feature(bolideDetectionList, hotSpots=None, plot_figure=False, ax=None):
    """ 
    Computes the distance to the hot spots in the detection heat map.

    It will compute the Haversine distance of each detection to each listed hot spot. It will then report the minimum
    distance of the hot spot list.

    Parameters
    ----------
    bolideDetectionList : [list of BolideDetection]
                List of bolide_detection.BolideDetection objects
    hotSpots : HotSpots class

    Returns
    -------
    hotSpotFeature : np.array
        Distance between detection and each hot spot

    """

    hotSpotFeature = np.full(len(bolideDetectionList), None)
    if hotSpots is None:
        return hotSpotFeature
    
    dist = np.full(len(hotSpots.lon_peaks), np.nan)
    for detIdx,detection in enumerate(bolideDetectionList):
        [avgLat, avgLon] = detection.average_group_lat_lon

        # Compute distance to each hot spot
        dist[:] = np.nan
        for spotIdx,(lon,lat) in enumerate(zip(hotSpots.lon_peaks, hotSpots.lat_peaks)):
            dist[spotIdx] = geoUtil.DistanceFromLatLonPoints(lat, lon, avgLat, avgLon)
            
        hotSpotFeature[detIdx] = copy.copy(dist)


    return hotSpotFeature
    
#*************************************************************************************************************
def ground_distance_feature(bolideDetectionList):
    """ Computes the ground track total distance based on the lat/lon coordinates.

    Parameters
    ----------
    bolideDetectionList : [list of BolideDetection]
                List of bolide_detection.BolideDetection objects

    Returns
    -------
    ground_distance_feature : np.array
        Total ground distance for each bolide candidate

    """

    ground_distance_feature = np.full_like(bolideDetectionList, np.nan)
    for idx, detection in enumerate(bolideDetectionList):
        lat_array, lon_array = detection.group_lat_lon
        ground_distance_feature[idx] = geoUtil.findLatLonBoxDistance(np.array(lat_array), np.array(lon_array))

    return ground_distance_feature
            

#*************************************************************************************************************
def compute_bolide_features(bolideDetectionList, bolideDatabase=None, glmGroupsWithNeighbors=None,
        multiProcessEnabled=False, spice_kernel_path=None, hotSpots=None,
        neighborhood_feature_enabled=True, legacy_filters_enabled=True,
        verbosity=False, chunkSize=100000, debugMode=False):
    """ For each bolide disposition in bolideDatabase generates a set of features.

    If bolideDatabase is passed then the bolide detections are taken from there. Otherwise, all bolide
    information is taken from bolideDetectionList.

    If debugMode then chunkSize=1000 and only 2000 total detections are processed (i.e. fast)

    Parameters
    ----------
    bolideDetectionList : [list of bolide_detections.BolideDetection objects]
    bolideDatabase               : [bolide_database.BolideDatabase] the database of detected and rejected bolides
    glmGroupsWithNeighbors  : [list of bolide_detection.GlmGroup] 
                    A list of objects containing ALL the groups from the netCDF data file associated with the
                    dispositions and in neighboring files, sorted by time. Used by neighborhood feature.
                    If None, then this list is generated from the netCDF files listed in each detection
    multiProcessEnabled : [bool] If True then use multiprocessing
    spice_kernel_path   : [str] Path to kernels to load for glintPoint object
                            If None then glint feature disabled
    hotSpots            : [HotSpots class] List of hot spots to avoid
    neighborhood_filters_enabled : [bool] If True then compute nieghborhood featuere (slow)
    legacy_filters_enabled      : [bool] If True then compute legacy features (slow)
    verbosity           : [bool] If True then print processing status statements
                            Computing features can be slow
    chunkSize           : [int] Number of events to process per chunk
    debugMode           : [bool] If True then only 2000 total detections are processed (i.e. fast)
    
    Returns
    -------
    bolideDetectionList.features -- [list] of BolideDetection objects with features computed
    """

    # Do not be verbose for sub-loops, just be verbose for the chunk loop
    elementalVerbosity = False

    if debugMode:
        chunkSize=1000

    #***
    # If using bolideDatabase then process the data in chunks. 
    # If using bolideDetectionList  then process the data all at once
    if bolideDatabase is not None:
        # The easiest way to implement this is to compute the features for chunks of events.
        # Chunk the detections and rejections but in a single large list of lists
        IDs = bolideDatabase.all_keys.copy()
        IDsChunked = [IDs[i:i + chunkSize] for i in range(0, len(IDs), chunkSize)]  
    else:
        # Just make a big list of all entries
        IDsChunked = [[detection.ID for detection in bolideDetectionList]]

    if (verbosity):
        pbar = tqdm(total=len(IDs), desc='Computing Features')

    if debugMode:
        IDsChunked = IDsChunked[0:2]

    for chunkIdx, IDs in enumerate(IDsChunked):

        # For simplicity call both detections and rejections "detections"
        if bolideDatabase is not None:
            chunkedDetectionList = bolideDatabase.extract_bolideDetectionList(dataType='detections', IDs=IDs)
            chunkedDetectionList.extend(bolideDatabase.extract_bolideDetectionList(dataType='rejections', IDs=IDs))
        else:
            chunkedDetectionList = bolideDetectionList

        if (len(IDs) != len(chunkedDetectionList)):
            raise Exception('Bookkeeping error when computing features')

        #***
        # Some simple features
        bolideTimeDatetime = [detection.bolideTime for detection in chunkedDetectionList]
        maxEnergy           = [detection.get_max_energy for detection in chunkedDetectionList]
        totEnergy           = [detection.get_total_energy for detection in chunkedDetectionList]
        nGroups             = [len(detection.groupList) for detection in chunkedDetectionList]
        
        # These cannot be vectorized (AFAIK, but I'm sure there is a Python trick)
        n_chunk_entries         = len(chunkedDetectionList)
        avgLat              = np.full(n_chunk_entries, np.nan)
        avgLon              = np.full(n_chunk_entries, np.nan)
        startTimeDatetime   = np.full(n_chunk_entries, bd.BolideDetection.epoch)
        endTimeDatetime     = np.full(n_chunk_entries, bd.BolideDetection.epoch)
        for i, detection in enumerate(chunkedDetectionList):
            [startTimeDatetime[i], endTimeDatetime[i]] = detection.get_time_interval
        
        # The start and end times need to be converted from datetime objects to floats
        # Reference time to J2000: bd.BolideDetection.epoch
        startTime   = np.full(n_chunk_entries, np.nan) 
        endTime     = np.full(n_chunk_entries, np.nan) 
        for i in np.arange(len(startTimeDatetime)):
            startTime[i]    = (startTimeDatetime[i] - bd.BolideDetection.epoch).total_seconds()
            endTime[i]      = (endTimeDatetime[i]   - bd.BolideDetection.epoch).total_seconds()

        # Ground track distance
        ground_distance = ground_distance_feature(chunkedDetectionList)
        
        
        #***
        # Compute all the legacy features
        if legacy_filters_enabled:   
            if (elementalVerbosity):
                print('Computing legacy features...')
            legacyFilterResponses = bff.compute_all_filter_responses_on_all_triggers(chunkedDetectionList, elementalVerbosity,
                    multiProcessEnabled=multiProcessEnabled)
        else:
            legacyFilterResponses = bff.FilterResponses(n_chunk_entries)
        
        #***
        # Chop Feature
        if (elementalVerbosity):
            print('Computing chop feature...')
        chopFeature = chop_feature(chunkedDetectionList, verbosity=elementalVerbosity)
        
        #***
        # Glint feature
        if (elementalVerbosity):
            print('Computing glint feature...')
        glintFeature = glint_feature(spice_kernel_path, chunkedDetectionList, verbosity=elementalVerbosity,
                multiProcessEnabled=multiProcessEnabled)

        #***
        # Neighborhood feature
        if neighborhood_feature_enabled:
            if (elementalVerbosity):
                print('Computing neighborhood feature...')
            [neighborFeature, _, _] = neighborhood_feature(chunkedDetectionList, 
                    glmGroupsWithNeighbors=glmGroupsWithNeighbors, verbosity=elementalVerbosity)
        else:
            neighborFeature = np.full(n_chunk_entries, np.nan)
        
        #***
        # Linearity feature
        # Computes how linear is the lat/lon scatter
        if (elementalVerbosity):
            print('Computing linearity feature...')
        linearityFeature = linearity_feature(chunkedDetectionList)

        #***
        # Compute the minimum distance of a detection to a hot spot
        if (elementalVerbosity):
            print('Computing hot spot feature...')
        hotSpotFeature = hot_spot_feature(chunkedDetectionList, hotSpots=hotSpots)
        

        #***
        # Distribute the features into each disposition
        for detectionIdx, detection in enumerate(chunkedDetectionList):

            profileIdx = np.nonzero(np.in1d(IDs, detection.ID))[0]
            # Check for duplicate or missing entry
            if(len(profileIdx) < 1):
                raise Exception('Bookkeeping Error: ID not found')
            elif (len(profileIdx) > 1):
                raise Exception('Bookkeeping Error: Duplicate IDs found')

            profileIdx = profileIdx[0]

            if bolideDetectionList[profileIdx] is None:
                bolideDetectionList[profileIdx] = detection

            # Add in the extra fields so they do not need to be pulled out again from the database
            bolideDetectionList[profileIdx].features.goesSatellite  = detection.goesSatellite
            bolideDetectionList[profileIdx].features.bolideTime     = detection.bolideTime

            # bolideMidRange is the middle 50% of all group times
            bolideDetectionList[profileIdx].features.bolideMidRange        = detection.bolideMidRange
            
            bolideDetectionList[profileIdx].features.maxEnergy    = maxEnergy[detectionIdx] 
            bolideDetectionList[profileIdx].features.totEnergy    = totEnergy[detectionIdx] 
            bolideDetectionList[profileIdx].features.nGroups      = nGroups[detectionIdx]   
            [avgLat, avgLon] = detection.average_group_lat_lon
            bolideDetectionList[profileIdx].features.avgLat       = avgLat    
            bolideDetectionList[profileIdx].features.avgLon       = avgLon
            bolideDetectionList[profileIdx].features.startTime    = startTime[detectionIdx] 
            bolideDetectionList[profileIdx].features.endTime      = endTime[detectionIdx]   
            bolideDetectionList[profileIdx].features.timeDuration = (endTimeDatetime[detectionIdx] -
                            startTimeDatetime[detectionIdx]).total_seconds()
            bolideDetectionList[profileIdx].features.ground_distance    = ground_distance[detectionIdx] 
        
        
            bolideDetectionList[profileIdx].features.latLonLinelets       = legacyFilterResponses.latLonLinelets[detectionIdx]      
            bolideDetectionList[profileIdx].features.energyRatio          = legacyFilterResponses.energyRatio[detectionIdx]         
            bolideDetectionList[profileIdx].features.splinelets           = legacyFilterResponses.splinelets[detectionIdx]          
            bolideDetectionList[profileIdx].features.groundTrackDeviation = legacyFilterResponses.groundTrackDeviation[detectionIdx]
        
            bolideDetectionList[profileIdx].features.chop            = chopFeature[detectionIdx]
            bolideDetectionList[profileIdx].features.glint           = glintFeature[detectionIdx]
            bolideDetectionList[profileIdx].features.neighborhood    = neighborFeature[detectionIdx]
            bolideDetectionList[profileIdx].features.latLonLinearity = linearityFeature[detectionIdx]
            bolideDetectionList[profileIdx].features.hotSpot         = hotSpotFeature[detectionIdx]

        # Clear the ZODB database cache to save memory
        if bolideDatabase is not None:
            bolideDatabase.cacheMinimize()

        if (verbosity):
            pbar.update(chunkSize)

    if (verbosity):
        pbar.close()

    return bolideDetectionList
