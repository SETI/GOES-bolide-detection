import netCDF4
import os
import copy
import glob
import sys
import csv
import persistent
import numpy as np
from datetime import datetime, time, timedelta
import pickle

from itertools import compress
import io_utilities as ioUtil
import time_utilities as timeUtil
import bolide_features as bFeatures
import geometry_utilities as geoUtil
from lat_lon_to_GLM_pixel import LatLon2Pix
from gen_goes_cutout import cutoutOutputClass
from validator import BolideAssessment
#import warnings
#warnings.simplefilter('error', UserWarning)

validSatellites = ('G16', 'G17', 'G18', 'G19')
goesEastSatellites = ('G16', 'G19')
goesWestSatellites = ('G17', 'G18')

def get_satellite_options(satellite):
    """
    Returns a list of satellite options based on the request

    Parameters
    ----------
    satellite : [str] 
        One of bolide_detections.validSatellites
        or 'all' to assess over all satellites
        or 'east' or 'west' to assess GOES-East and GOES-West 
            (i.e. [G16, G19] and [G17, G18] as the same GOES-East or GOES-West data)

    Returns
    -------
    satelliteOptions : list
        A list of satellites selected

    """

    assert satellite in ('all', 'east', 'west') or validSatellites.count(satellite) == 1, 'Unknown GOES satellite'

    if satellite == 'east':
        satelliteOptions = goesEastSatellites
    elif satellite == 'west':
        satelliteOptions = goesWestSatellites
    elif satellite == 'all':
        satelliteOptions = validSatellites
    else:
        # Only a specific satellite
        satelliteOptions = [satellite]

    return satelliteOptions


#******************************************************************************
# GlmEvent
# --------
#
# Stores the event data from a netCDF file.
#
# This is a persistent class so that it can be properly used within a persistent OO database (like ZODB).
#
# INSTANCE ATTRIBUTES
#   datasetFile             : [str] netCDF file name this event came from (IDs are not unique accross files!)    
#                               called 'dataset_name' in the netCDF data set
#   id                      : The unique integer event identification number.
#   time                    : [datetime] The time of the event
#   latitudeDegreesNorth    : The event latitude in degrees North.
#   longitudeDegreesEast    : The event longitude in degrees East.
#   energyJoules            : The measured radiant energy of the event in Joules.
#   parentGroupId           : The integer identification number of the group
#                               (if any) to which the event belongs. May be empty.
#******************************************************************************
#class GlmEvent(persistent.Persistent):
class GlmEvent():
    def __init__(self, datasetFile, id, time, latitudeDegreesNorth, longitudeDegreesEast,
                  energyJoules, dataQuality, parentGroupId=-1):

        self.datasetFile            = datasetFile
        self.id                     = int(id)
        self.time                   = time
        self.latitudeDegreesNorth   = float(latitudeDegreesNorth)
        self.longitudeDegreesEast   = float(longitudeDegreesEast)
        self.energyJoules           = float(energyJoules)
        self.dataQuality            = np.uint16(dataQuality)
        self.parentGroupId          = int(parentGroupId)

    #******************************************************************************
    # Performs a deep copy of object
    def copy(self):
        """Returns a copy of this `GlmEvent` object.

        This method uses Python's `copy.deepcopy` function to ensure that all
        objects stored within the instance are fully copied.

        Returns
        -------
        self : 
            A new object which is a copy of the original.
        """
        return copy.deepcopy(self)

    # Returns a summary of the attributes in the GlmEvent class
    def __repr__(self): 
        return ioUtil.print_dictionary(self)


#******************************************************************************
# Construct a list of GlmEvent objects from a file.
#
# INPUTS
#     nc4Data     : A netCDF4 Dataset object.
#     eventIdList : A list or ndarray of integer event ID numbers.
#                   'all' means extract all events from netCDF4 file
#
# OUTPUTS
#     eventList   : A list of GlmEvent objects.
#
#******************************************************************************
def get_events_by_id(nc4Data, eventIdList='all'):

    assert isinstance(eventIdList, list) or (isinstance(eventIdList, str) and eventIdList == 'all'), "eventIdList must be either a list or 'all'"

    allEventIds = nc4Data.variables['event_id'][:]

    if eventIdList == []:
        return []

    if eventIdList == 'all':
        eventIdList = allEventIds

    indicatorList = np.nonzero(np.in1d(allEventIds, eventIdList))[0]

    eventIds = nc4Data.variables['event_id'][indicatorList]

    # -------------------------------------------------------------------------
    # Compute event time offset in miliseconds
    # -------------------------------------------------------------------------
    timeOffsetMsec = nc4Data.variables['event_time_offset'][indicatorList].data
    # The nc file format changed after 10/15/2018 @ 16:00 afterwhich event_time_offset is in fractions of second
    productTime = BolideDetection.epoch + timedelta( seconds=int(nc4Data.variables['product_time'][0].data) )
    if productTime > datetime(2018, 10, 15, 16, 0): 
        # nc file format changed and event_time_offset is in fractions of second
        timeOffsetMsec = [o * 1000.0 for o in timeOffsetMsec]
    else: 
        # nc file old format and event_time_offset is already in milliseconds
        pass

    eventTime = [productTime + timedelta(milliseconds=float(offset)) for offset in timeOffsetMsec]

    # -------------------------------------------------------------------------
    # Get event coordinates.
    # -------------------------------------------------------------------------
    latitudeDegreesNorth   = nc4Data.variables['event_lat'][indicatorList]
    # Convert masked_array to numpy.ndarray
    latitudeDegreesNorth = np.array(latitudeDegreesNorth)
    longitudeDegreesEast   = nc4Data.variables['event_lon'][indicatorList]
    longitudeDegreesEast = np.array(longitudeDegreesEast)
    # Wrap longitude to -180...180 degrees
    longitudeDegreesEast = geoUtil.wrap_longitude(longitudeDegreesEast)
    # But unwrap the western side of the GOES-West so that it is contiguous on a map
    longitudeDegreesEast = geoUtil.unwrap_longitude(longitudeDegreesEast)

    # -------------------------------------------------------------------------
    # Retrieve event energy in Joules.
    # -------------------------------------------------------------------------
    energyJoules = nc4Data.variables['event_energy'][indicatorList]
    energyJoules = np.array(energyJoules)

    parentGroupId = nc4Data.variables['event_parent_group_id'][indicatorList]
    parentGroupId = np.array(parentGroupId)

    eventList = []
    for i in np.arange(len(indicatorList)):
        eventList.append(GlmEvent(nc4Data.dataset_name, eventIds[i], eventTime[i], latitudeDegreesNorth[i], longitudeDegreesEast[i],
                                 energyJoules[i], parentGroupId[i]))

    return eventList

#******************************************************************************
def extract_events_from_all_files(glm_data_path, central_time, time_window_sec, exclude_list=None):
    """ Extracts GLM event data from all GLM netCDF files within the specified time window.

    Parameters
    ----------
    glm_data_path : str
        Top level path to the GLM satellite data with subdirectoires of the form:
        glm_data_path/YYYY/DOY/HOD
        YYYY : 4-digit year
        DOY : 3-digit day of year (1 to 366)
        HOD : 2-digit Hour of day (0 to 23) 
    central_time : datetime
        The central time to extract all events around
    time_window_sec : float
        Time window in seconds
    exclude_list : GlmEvent list
        List of GlmEvent objects that we want to exclude from this list
        For example, if we want to exclude all events from a detection, we pass detection.eventList

    Returns
    -------
    event_list : GlmEvent list
        A list of all GLM event objects within the specified window
    
    """

    # By limiting this function to a time window of just 1 hour we only have to deal with no more than two data
    # directories. Otherwise, we need code to sequentially go through each directory.
    assert time_window_sec <= 60*60, 'This function only works with time_window_sec <= 3600 (1 hour)'

    # This is a buffer to add to the time_window so that when finding all data files, 
    # we properly include the first and last.
    window_buffer_sec = 25

    # Get GLM format timestamp from central_time
    glm_central_time = timeUtil.generate_glm_timestamp(central_time)

    # Get the starting and ending times for window
    start_time = central_time - timedelta(seconds=time_window_sec/2.0)
    end_time = central_time + timedelta(seconds=time_window_sec/2.0)
    start_time_sec = (start_time - BolideDetection.epoch).total_seconds()
    end_time_sec = (end_time - BolideDetection.epoch).total_seconds()

   #bolide_time_sec = (central_time - BolideDetection.epoch).total_seconds()
   #bolide_time_sec_test = timeUtil.extract_total_seconds_from_glm_timestamp(glm_central_time)
   #assert (start_time_sec == bolide_time_sec_test), 'Error in converting times'
   #start_time_sec = round(bolide_time_sec - time_window_sec/2.0)
   #end_time_sec = round(bolide_time_sec + time_window_sec/2.0)

    #***
    # Get all files within window
    # determine the directories to extract GLM data from
    # By limiting time window to 1 hour, we have no more than 2 directories to examine.
    first_dir = os.path.join(glm_data_path, '{:04}'.format(start_time.year), '{:03}'.format(start_time.timetuple().tm_yday), '{:02}'.format(start_time.hour))
    last_dir = os.path.join(glm_data_path, '{:04}'.format(end_time.year), '{:03}'.format(end_time.timetuple().tm_yday), '{:02}'.format(end_time.hour))

    # Pull all events from first to last directory
    more_data = True
    pull_dir = first_dir
    event_list = []
    while more_data:
        files_in_dir = glob.glob(os.path.join(pull_dir, '*.nc'), recursive=True)
        # Only keep files within start and stop times
        # Add in a buffer so that the shoulder 20-second files are included to get the full range.
        file_start_times_sec = []
        file_end_times_sec = []
        for filename in files_in_dir:
            idx = filename.find('_s')
            file_start_times_sec.append(timeUtil.extract_total_seconds_from_glm_timestamp(filename[idx+2:idx+16]))
            idx = filename.find('_e')
            file_end_times_sec.append(timeUtil.extract_total_seconds_from_glm_timestamp(filename[idx+2:idx+16]))
        # Find files to read
        files_to_read = [f for idx,f in enumerate(files_in_dir) if 
                (file_start_times_sec[idx] >= (start_time_sec - window_buffer_sec) and 
                 file_end_times_sec[idx] <= (end_time_sec + window_buffer_sec))  ]

        # Extract events for all files to read
        for filename in files_to_read:
            nc4Data = netCDF4.Dataset(filename)
            eventListThisFile = get_events_by_id(nc4Data, eventIdList='all')
            # Only keep events within time window
            eventListThisFileTruncated = [event for event in eventListThisFile if 
                    (event.time >= start_time and event.time <= end_time)]
            event_list.extend(eventListThisFileTruncated)

        # Check if there is a second data directory. If so, then read that one too.
        if last_dir != first_dir and pull_dir == first_dir:
            pull_dir = last_dir
        else: 
            more_data = False

    # Exclude event list
    # We want to be efficient, sequentially checking each event against each exclude event could be slow
    # So, we compare the integer ids, which should be very fast
    # For any ids that match, we then confirm the datasetFile is the same (since the event id is not unqie accross data files)
    if exclude_list is not None:
        eventIds = [e.id for e in event_list]
        eventFilenames = [e.datasetFile for e in event_list]
        eventIdsExclude = [e.id for e in exclude_list]
        eventExcludeFilenames = [e.datasetFile for e in exclude_list]

        # We cannot assume unique, because the same id can be in two different data files!
        # intersect1d with return_indices=True returns only the first instance, and we can have duplicates
        match_ids = np.intersect1d(eventIds, eventIdsExclude, assume_unique=False, return_indices=False)
        match_indices = np.nonzero(np.isin(eventIds, match_ids))[0]
       #match_exclude_indices = np.isin(eventIdsExclude, match_ids)

        event_indices_to_remove = []
        # For any matches, check if the datasetFile also matches. If so, remove from list
        for match_index in match_indices:
            exclude_indices = np.nonzero(np.isin(eventIdsExclude, eventIds[match_index]))[0]
            for exclude_index in exclude_indices:
                if eventExcludeFilenames[exclude_index] == eventFilenames[match_index]:
                    # It's a match, remove from event_list
                    if match_index in event_indices_to_remove:
                        raise Exception('match_index already in event_indices_to_remove, does exclude_list have duplicates?')
                    event_indices_to_remove.append(match_index)

        # Remove the events in the exclude list
        # Sort indices in descending order so that we can sequentially delete each element using the same indices
        event_indices_to_remove.sort(reverse=True)
        # del is the most efficient way to delete an element from a list
        for idx in event_indices_to_remove:
            try:
                del event_list[idx]
            except:
                raise Exception('Error in deleting exclude list for directory {}'.format(pull_dir))


    return event_list

#******************************************************************************
# GlmGroup
# --------
#
# Stores the group data from a netCDF file.
#
# This is a persistent class so that it can be properly used within a persistent OO database (like ZODB).
#
# INSTANCE ATTRIBUTES
#   datasetFile          : [str] netCDF file name this group came from (IDs are not unique accross files!)    
#                           called 'dataset_name' in the netCDF data set
#   id                  : [int] The unique integer group identification number. (Only unique within each netCDF file!)
#   time                : [datetime] The time of the group
#   latitudeDegreesNorth: The group latitude in degrees North.
#   longitudeDegreesEast: The group longitude in degrees East.
#   areaSquareKm        : The estimated area covered by events in the group.
#   energyJoules        : The measured radiant energy of the group in Joules.
#   dataQuality         : An integer in the set {0,1,3,5} (see NOTES for details).
#   eventIdList         : A list of event IDs belonging to the group.
#
# DERIVED ATTRIBUTES
#   x                   : [float] Pixel x coordinate
#   y                   : [float] Pixel y coordinate
#
# NOTES
#   Group Data Quality Flags (group_quality_flag)
#   Flag Value  Flag Meaning
#   ----------  ------------
#   0		good_quality_qf
#   1		degraded_due_to_group_constituent_events_out_of_time_order_or_parent_flash_abnormal_qf
#   3		degraded_due_to_group_constituent_event_count_exceeds_threshold_qf
#   5		degraded_due_to_group_duration_exceeds_threshold_qf
#******************************************************************************
#class GlmGroup(persistent.Persistent):
class GlmGroup():
    def __init__(self, datasetFile, id, time, latitudeDegreesNorth, longitudeDegreesEast,
                  areaSquareKm, energyJoules, dataQuality, eventIdList):

        self.datasetFile            = datasetFile
        self.id                     = int(id)
        self.time                   = time
        self.latitudeDegreesNorth   = float(latitudeDegreesNorth)
        self.longitudeDegreesEast   = float(longitudeDegreesEast)
        self.areaSquareKm           = float(areaSquareKm)
        self.energyJoules           = float(energyJoules)
        self.dataQuality            = np.uint16(dataQuality)
        self.eventIdList            = eventIdList


        # These quantities are derived and not extracted from the netCDF data files
        self.x = -1
        self.y = -1

    #******************************************************************************
    # Performs a deep copy of object
    def copy(self):
        """Returns a copy of this `GlmGroup` object.

        This method uses Python's `copy.deepcopy` function to ensure that all
        objects stored within the instance are fully copied.

        Returns
        -------
        self : 
            A new object which is a copy of the original.
        """
        return copy.deepcopy(self)


    # Returns a summary of the attributes in the glmGroup class
    def __repr__(self): 
        return ioUtil.print_dictionary(self)


#******************************************************************************
# Extracts the requested list of GLM groups from a netCDF file dataset
#
# Use with extract_groups_from_all_files
#
# Constructs a list of GlmGroup objects from a file.
#
# INPUTS
#   nc4Data     : [netCDF4._netCDF4.Dataset] A netCDF4 Dataset.
#   groupIdList : A list or ndarray of integer group ID numbers. ([] => load all groups)
#                   Group IDs not in the nd4Data file will be ignored.
#   eventIdListEnabled      : [bool] If True then find the event IDs corresponding to each group
#                               This is very slow to run so it can be disabled as an option
#   ignoreZeroEnergyGroups  : [bool] If True then do not include groups with zero energyJoules
#                               Cannot be set to True if groupIdList != []
#   ignorePoorQualityGroups : [bool] If True then do not include groups with quality flag > 0
#                               Cannot be set to True if groupIdList != []
#   ignorePoorDateGroups    : [bool] If True then do not include groups with a poor date
#                               (Meaning a date not consistent with the file dates
#
# OUTPUTS
#   glmGroups   : A list of GlmGroup objects.
#
#******************************************************************************
def get_groups_by_id(nc4Data, groupIdList=[], eventIdListEnabled=True, 
        ignoreZeroEnergyGroups=False, ignorePoorQualityGroups=False, ignorePoorDateGroups=False):

    if (groupIdList != [] and ignoreZeroEnergyGroups):
        raise Exception('ignoreZeroEnergyGroups cannot be True if groupIdList ~= []')
    if (groupIdList != [] and ignorePoorQualityGroups):
        raise Exception('ignorePoorQualityGroups cannot be True if groupIdList ~= []')

    # Start and end dates of data in file
    time_coverage_start = datetime.fromisoformat(timeUtil.convert_GLM_date_str_to_ISO(nc4Data.time_coverage_start))
    time_coverage_end   = datetime.fromisoformat(timeUtil.convert_GLM_date_str_to_ISO(nc4Data.time_coverage_end))
    # However, the group times will commonly extend past the ends of this covereg limit (WTF!?!?)
    # So, pad the times by a couple seconds (10 seconds?)
    time_coverage_start = time_coverage_start - timedelta(seconds=10.0)
    time_coverage_end   = time_coverage_end + timedelta(seconds=10.0)

    allGroupIds = nc4Data.variables['group_id'][:]

    if groupIdList == [] :
        groupIdList = allGroupIds

    indicatorList = np.nonzero(np.in1d(allGroupIds, groupIdList))[0]

    groupIds = nc4Data.variables['group_id'][indicatorList]

    # -------------------------------------------------------------------------
    # Compute group time offset in miliseconds and then the group time
    # -------------------------------------------------------------------------
    timeOffsetMsec = nc4Data.variables['group_time_offset'][indicatorList].data
    # The nc file format changed after 10/15/2018 @ 16:00 afterwhich group_time_offset is in fractions of second
    productTime = BolideDetection.epoch + timedelta( seconds=int(nc4Data.variables['product_time'][0].data) )
    if productTime > datetime(2018, 10, 15, 16, 0): 
        # nc file format changed and group_time_offset is in fractions of second
        timeOffsetMsec = [o * 1000.0 for o in timeOffsetMsec]
    else: 
        # nc file old format and group_time_offset is already in milliseconds
        pass

    groupTime = [productTime + timedelta(milliseconds=float(offset)) for offset in timeOffsetMsec]


    # -------------------------------------------------------------------------
    # Get group coordinates and area.
    # -------------------------------------------------------------------------
    latitudeDegreesNorth = nc4Data.variables['group_lat'][indicatorList]
    # Convert masked_array to numpy.ndarray
    latitudeDegreesNorth = np.array(latitudeDegreesNorth)

    # Compute group longitude in degrees.
    longitudeDegreesEast = nc4Data.variables['group_lon'][indicatorList]
    longitudeDegreesEast = np.array(longitudeDegreesEast)
    # Wrap longitude to -180...180 degrees
    longitudeDegreesEast = geoUtil.wrap_longitude(longitudeDegreesEast)
    # But unwrap the western side of the GOES-West so that it is contiguous on a map
    longitudeDegreesEast = geoUtil.unwrap_longitude(longitudeDegreesEast)

    areaSquareKm = nc4Data.variables['group_area'][indicatorList]
    areaSquareKm = np.array(areaSquareKm)

    # -------------------------------------------------------------------------
    # Compute group energy in Joules. Scaling and offset are automatically
    # applied.
    # -------------------------------------------------------------------------
    energyJoules = nc4Data.variables['group_energy'][indicatorList]
    energyJoules = np.array(energyJoules)
    dataQuality  = nc4Data.variables['group_quality_flag'][indicatorList]
    dataQuality  = np.array(dataQuality)

    # -------------------------------------------------------------------------
    # Construct a list of component events for each group.
    # -------------------------------------------------------------------------
    glmGroups = []
    for i in np.arange(len(indicatorList)):
        if (ignoreZeroEnergyGroups and energyJoules[i] == 0.0):
            # Zero energy, skip this group
            continue
        if (ignorePoorQualityGroups and dataQuality[i] > 0):
            # Poor quality data, skip this group
            continue
        if (ignorePoorDateGroups and
            (groupTime[i] < time_coverage_start or
             groupTime[i] > time_coverage_end)    ):
            # group time is outside valid range
            continue
        # build a list of events (IDs) comprising the group
        if eventIdListEnabled:
            eventIdList = find_events_by_group_id( nc4Data, groupIds[i] )
        else:
            eventIdList = []
        glmGroups.append( GlmGroup(nc4Data.dataset_name, groupIds[i], groupTime[i], 
                                latitudeDegreesNorth[i], longitudeDegreesEast[i],
                                 areaSquareKm[i], energyJoules[i], dataQuality[i], eventIdList) )

    return glmGroups

#******************************************************************************
def extract_groups_from_all_files(filePathList, extractNeighboringFiles=False, **kwargs):
    """ Extracts the GLM groups from all netCDF files passed

    Sorts groups by time
  
    Parameters
    ----------
    filePathList : [list of str] List if pathnames for the netCDF files
    extractNeighboringFiles : [bool] If True then also attempt to extract glmGroups from the netCDF files on either side
                in time of the ones requested.
    groupIdList : A list or ndarray of integer group ID numbers. ([] => load all groups)
    eventIdListEnabled      : [bool] If True then find the event IDs corresponding to each group
                                This is very slow to run so it can be disabled as an option
    ignoreZeroEnergyGroups  : [bool] If True then do not include groups with zero energyJoules
                                Cannot be set to True if groupIdList != []
    ignorePoorQualityGroups : [bool] If True then do not include groups with quality flag > 0
                                Cannot be set to True if groupIdList != []
    ignorePoorDateGroups    : [bool] If True then do not include groups with a poor date
                                (Meaning a date not consistent with the file dates
 
    Returns
    -------
    glmGroups   : [list of bolide_detection.GlmGroup] 
                A list of objects containing ALL the groups from the netCDF data files, sorted by time
    filePathList : [list of str] List if pathnames for the netCDF files, including neighboring files
                    The first index in the list is the primary netCDF file

    """
    # If passed single file then place in a list
    if filePathList.count('.nc'):
        filePathList = [filePathList]

    glmGroups = []
    filePathListSave = filePathList.copy()
    for filename in filePathListSave:
        if (not os.path.exists(filename)):
            continue
        nc4Data = netCDF4.Dataset(filename)
        glmGroups.extend(get_groups_by_id(nc4Data, **kwargs))

        # If no groups in main file then do not search for neighbors and just return
        if len(glmGroups) == 0:
            continue

        if extractNeighboringFiles:
            goesSatellite = nc4Data.platform_ID
            # Get the time stamp of the file under question.
            # just get the filename without the path so that the '_s', '_e' and '_c' searches do not find a directory name
            filenameNoPath = os.path.basename(filename)
            startTime   = int(filenameNoPath[filenameNoPath.index('_s')+2: filenameNoPath.index('_e')])
            endTime     = int(filenameNoPath[filenameNoPath.index('_e')+2: filenameNoPath.index('_c')])

            # Find file with next earlier time
            # Compare end time of previous file with start time of FUS (File Under Study)
            dirName = os.path.dirname(filename)
            # Check if this is geonex data, if so, search over all hourOfDay subdirectories
            # We find this by checking if the highest level directory name is two characters long (i.e. 01, 02, ...,23 etc...
            if len(os.path.basename(dirName)) == 2:
                dirName = os.path.join(os.path.dirname(dirName), '*')
            fileList = glob.glob(os.path.join(dirName, '*'+goesSatellite+'*.nc'))
            fileList = [os.path.basename(f) for f in fileList]
            # Get StartTimes and EndTimes for all files
            startTimes  = np.sort(np.array([fn[fn.index('_s')+2: fn.index('_e')] for fn in fileList], dtype=int))
            endTimes    = np.sort(np.array([fn[fn.index('_e')+2: fn.index('_c')] for fn in fileList], dtype=int))
            fusStartIdxAll = np.nonzero(startTimes == startTime)[0]
            fusEndIdxAll   = np.nonzero(endTimes == endTime)[0]

            extraFileList = []
            # Rarely, we have multiple files covering the same time range (with different creation times)
            # For these rare cases, save all of the files
            # Find the first previous startTime, this is the previous file
            for fusStartIdx in fusStartIdxAll:
                if (fusStartIdx > 0):
                    # if fusStartIdx == 0 then this is the first file in list so no previous file
                    # Check if the previous file is indeed the previous 20-seconds' file
                    if (timeUtil.extract_total_seconds_from_glm_timestamp(startTime) - 
                            timeUtil.extract_total_seconds_from_glm_timestamp(int(startTimes[fusStartIdx-1])) < 30):
                        extraFileList.extend(glob.glob(os.path.join(dirName, '*'+goesSatellite+'*'+'_e'+str(endTimes[fusStartIdx-1]) +'*.nc')))
            # Find the next endTime, this is the next file
            for fusEndIdx in fusEndIdxAll:
                if (fusEndIdx < len(fileList)-1):
                    # if fusEndIdx == len(fileList) then this is the last file in list so no next file
                    # Check if the next file is indeed the next 20-seconds' file
                    if (timeUtil.extract_total_seconds_from_glm_timestamp(str(endTimes[fusEndIdx+1])) - 
                            timeUtil.extract_total_seconds_from_glm_timestamp(endTime) < 30):
                        extraFileList.extend(glob.glob(os.path.join(dirName, '*'+goesSatellite+'*'+'_s'+str(startTimes[fusEndIdx+1]) +'*.nc')))
            # extraFileList now contains a list of the previous and next files to read glmGroups from

            # Read in the extra groups on either side
            for extraFilename in extraFileList:
                # Check that extraFilename is not already in filePathList
                if (filePathList.count(extraFilename) == 0):
                    nc4Data = netCDF4.Dataset(extraFilename)
                    glmGroups.extend(get_groups_by_id(nc4Data, **kwargs))
                    filePathList.append(extraFilename)


    #***
    # Sort groups by time
    glmGroups.sort(key=lambda x: x.time)

    return glmGroups, filePathList

# *****************************************************************************
#
# *****************************************************************************
def get_netcdf_vars_as_masked_arrays( nc4Data, varList=[]) :

    varDict = {}

    if varList == [] :
        # The '*' operator unpacks any iterable object. The enclosing brackets
        # "catch" the results in a list.
        varList = [*nc4Data.variables.keys()]

    for key in varList:
        varDict[key] = nc4Data.variables[key][:]

    return varDict


# *****************************************************************************
# Find the GLM events comprising a group and return a list of their event ID
# numbers.
#
# INPUTS
#     nc4Data : A netCDF4 Dataset.
#     groupId : An integer group ID number.
#
# OUTPUTS
#     eventIdsThisGroup : A list of event IDs for events comprising the group
#                         specified by groupId.
# *****************************************************************************
def find_events_by_group_id( nc4Data, groupId ) :
    eventIds  = nc4Data.variables['event_id'][:]
    parentIds = nc4Data.variables['event_parent_group_id'][:]
    eventIdsThisGroup = eventIds[np.in1d(parentIds, groupId)]
    return list(eventIdsThisGroup)

# *****************************************************************************
# Convert Bolide objects (defined in bolide_support_functions.py) to
# bolideDetection objects (defined in bolide_detections.py).
#
# INPUTS
#   bolideList          : A list of Bolide objects. See
#                           bolide_support_functions.py for details.
#   confidence          : Confidence score for this detection
#                           low=> 0.0, high=> 1.0, unknown=> -1.0
#   confidenceSource    : str, Where confidence came from (I.e. neo-bolides website)
#   howFound            : str, How the detection was found
#
# OUTPUTS
#     bolideDetectionList   : A list of bolideDetection objects. See
#                             bolide_detections.py for details.
#
# NOTES
#   - For a given Bolide object, the conversion is done by first constructing
#     a glmDataSubsetRecord object, which specifies the original data file
#     along with the event or group IDs comprising the detection. The detection
#     data is then extracted from the original netCDF4 data file to construct
#     the bolideDetection object.
# *****************************************************************************
def convert_detection_representation(bolideList, confidence=-1.0, 
        confidenceSource='unknown', howFound='unknown'):

    raise Exception('This code is not being maintained')

    bolideDetectionList= []

    for iBolide in range( len(bolideList) ) :
        print('Converting {} of {} detections ...'.format(iBolide+1, len(bolideList)))

        detection = bolideList[iBolide]

        # Get the group ID list for this detection.
        idList = []
        for grp in detection.group:
            idList.append(grp.id)

        detectionRecord = bd.glmDataSubsetRecord([detection.filePathStr], 'group_id', idList)
        bolideDetectionList.append(bd.bolideDetection.fromGlmDataFiles(detectionRecord, 
            confidence=confidence, confidenceSource=confidenceSource, howFound=howFound))

    return bolideDetectionList


#******************************************************************************
# BolideDetection
# ---------------
#
# This is the principle object which contains bolide detection information.
# If a bolide was detected by two satellites then create two BolideDetection objects, one for each detection.
#
# This is a persistent class so that it can be properly used within a persistent OO database (like ZODB).
#
# There are three ways to construct this object:
#   1) __init__: Explicitely give it all the information it needs to populate its attributes
#   2) fromGlmDataFiles: Using glmDataSubsetRecord to tell the constructor what data to retrieve from the netCDF files.
#   3) fromBolideCluster: Using bolideClustering Object
#
# CLASS ATTRIBUTES
#     epochStr      : A string.
#     epoch         : A datetime object set to the J2000 epoch.
#
# INSTANCE ATTRIBUTES
#   filePathList    : A list of files (path strings) containing the detection groups
#                   The first index in filePathList is the primary data file (not the neighborhood)
#   goesSatellite   : [str] Which GOES Satellite {'G16', 'G17', etc...}
#   productTime      : Reference time for event offsets in integer seconds
#                       since J2000 epoch.
#   subPointLatDegreesNorth : Latitude of the sub-satellite point on the Earth's surface.
#   subPointLonDegreesEast  : Longitude of the sub-satellite point on the Earth's surface.
#   lat_field_of_view_bounds: [float array] latitude coordinates for north/south extent of field of view
#   lon_field_of_view_bounds: [float array] longitude coordinates for west/east extent of field of view 
#   glmHeight               : nominal satellite height above GRS 80 ellipsoid (platform altitude)
#   yaw_flip_flag           : bool yaw flip flag (for G17)
#   groupList       : A list of GlmGroup objects comprising the detection.
#   eventList       : A list of GlmEvent objects comprising the detection. This
#                       list must always be populated.
#   bolideTime      : datetime, The time of the detection (the median time of the groups in the detection)
#   ID              : int64, Unique ID for each detection, see bolide_database._generate_ID 
#   isInStereo      : bool, If True then detection is in the stereo region detectable by two satellites
#   features        : [bolide_features.FeaturesClass]  An object of features computed for this detection
#                       (can either be passed when initialized or computed later)
#   bolideDetectionOtherSatellite : [BolideDetection] For the detection data in the other satellite
#                   If isInStereo = True then this should be populated.
#   stereoFeatures  : [] Stereo features
#   cutoutFeatures  : [cutoutOutputClass] Contains the information generated by the cutout tool
#   howFound        : str, How the detection was found
#   assessment      : validator.BolideAssessment, The bolideness assesment of the detection 
#
# KEY PROPERTY METHODS
#   group_times
#   get_time_interval
#   bolideMidRange
#   group_lat_lon
#   average_group_lat_lon
#   energy
#   get_total_energy
#
#******************************************************************************
#class BolideDetection(persistent.Persistent):
class BolideDetection():
    epochStr = 'J2000 (2000-01-01 12:00:00 UTC)'
    epoch    = datetime(2000, 1, 1, 12)

    #******************************************************************************
    #******************************************************************************
    def __init__(self, 
            BCObj, 
            clusterIndex, 
            howFound='clustering', 
            assessment=None,
            retrieveEvents=False, 
            features=None, 
            latLon2Pix_table_path=None, 
            latLon2Pix_table_path_inverted=None):
        """
        This constructor is used to load the group/event data from a bolide_clustering.BolideClustering object
        It reads in the data from the cluster and generates a BolideDetection object
       
        Parameters
        ----------
        BCObj           : bolideClustering object
        clusterIndex    : int 
            index of cluster in BCObj.clusteredGroups
        howFound : str
            How was this bolide detection found?
        assessment : validator.BolideAssessment class
            Contains assessment of the bolideness of this detection
        retrieveEvents  : bool 
            If True then also retrieve the event info (Slow!)
        features        : FeaturesClass 
            The features computed for this detection
        latLon2Pix_table_path           : str 
            Path to the lookup table file.
        latLon2Pix_table_path_inverted  : str 
            Path to the lookup table file for the inverted yaw orientation 
            Only relevent for G17.
     
        """

        self.filePathList = BCObj.netCDFFilenameList.copy()
        if (validSatellites.count(BCObj.goesSatellite) == 1):
            self.goesSatellite = BCObj.goesSatellite
        else:
            raise Exception('Unknown goesSatellite')
        self.productTime = copy.copy(BCObj.productTime)
        self.subPointLatDegreesNorth = copy.copy(BCObj.subPointLatDegreesNorth)
        self.subPointLonDegreesEast = copy.copy(BCObj.subPointLonDegreesEast)
        self.lat_field_of_view_bounds  = copy.copy(BCObj.lat_field_of_view_bounds)
        self.lon_field_of_view_bounds  = copy.copy(BCObj.lon_field_of_view_bounds)
        self.glmHeight = copy.copy(BCObj.glmHeight)
        self.yaw_flip_flag = copy.copy(BCObj.yaw_flip_flag)

        self.groupList = [BCObj.glmGroups[idx].copy() for idx in BCObj.clusteredGroups[clusterIndex]['groupIndices']] 
        self.howFound = howFound

        assert assessment is None or isinstance(assessment, BolideAssessment), 'Passed assessment must be type validator.BolideAssessment'
        self.assessment = assessment
        
        if 'streakIdx' in BCObj.clusteredGroups[clusterIndex]:
            self.streakIdx = BCObj.clusteredGroups[clusterIndex]['streakIdx']

        self.eventList = []
        if (retrieveEvents):
            eventIdList = []
            # Go through each file and find all event IDs associated with each.
            for filename in self.filePathList:
                nc4Data = netCDF4.Dataset(filename)
                filename = os.path.basename(filename)
                for groupId in [group.id for group in self.groupList if group.datasetFile == filename]:
                    eventIdList = find_events_by_group_id(nc4Data, groupId)
                    self.eventList.extend(get_events_by_id(nc4Data, eventIdList))


            allEventIds = [e.id for e in self.eventList]
            if len(allEventIds) != len(np.unique(allEventIds)):
                raise Exception('Non-unique allEventIds')

        
        # Make sure groups and events are sorted by time
        self.groupList.sort(key=lambda x: x.time)
        self.eventList.sort(key=lambda x: x.time)

        # np.datetime64(datetime.utcnow()).astype(datetime)

        # Time of event and ID
        # Define the time of event as the median time of the groups in the detection
        timeArray  = [g.time for g in self.groupList]
        self.bolideTime = timeArray[len(timeArray)//2]
        # Initiliaze this to zero for now. ID generation occurs in bolide_database.
        # We only generate IDs for true detections (not just clusters)
        self.ID         = None

        # We determine if in stereo region in detect_bolide.copy_netCDF_files_from_other_satellite
        # So here we set to None
        self.isInStereo = None
        self.bolideDetectionOtherSatellite = None

        if features is None:
            self.features   = bFeatures.FeaturesClass()
        else:
            self.features   = features

        self.stereoFeatures = bFeatures.StereoFeaturesClass()
        self.cutoutFeatures = None

        # Compute the derived pixel coordinates
        if latLon2Pix_table_path is not None: 
            try:
                latLon2PixObj = LatLon2Pix(latLon2Pix_table_path, latLon2Pix_table_path_inverted)
                self.glm_pixel_coords(latLon2PixObj)
            except:
                print('\n****** LatLon2Pix: Error generating pixel coordinates for bolide candidate at time {}'.format(self.bolideTime))

    #******************************************************************************
    def add_cutout_features(self, cutoutFeatures):
        """ Adds cutout features generated by gen_goes_cutout 

        """
        assert isinstance(cutoutFeatures, cutoutOutputClass), 'cutoutFeatures must be of type cutoutOutputClass'

        assert cutoutFeatures.ID == self.ID, 'This cutout feature data does not have the correct ID'

        self.cutoutFeatures = cutoutFeatures

    #******************************************************************************
    # Adds more groups to a detection in the form of a second bolideDetection object
    # The new assessment score is the mean the two original assessment scores.
    # NOTE: Be very careful with this method. Not all attributes are properly combined. But it will combine the group and
    # event lists
    def add_data(self, otherBolideDetection):

        self.filePathList.extend(otherBolideDetection.filePathList)
        
        # Added data must be from the same satellite
        if (self.goesSatellite != otherBolideDetection.goesSatellite):
            raise Exception ('Data for added bolide must be from the same satellite')

        # Product time also has to be the same, for now...
        # If both data sets are from the same file then their product times should be the same!
     #  if (self.productTime != otherBolideDetection.productTime):
     #      raise Exception ('Data for added bolide must be from the same productTime')

        # These should also be the same
     #  if (self.subPointLatDegreesNorth != otherBolideDetection.subPointLatDegreesNorth or 
     #          self.subPointLonDegreesEast  != otherBolideDetection.subPointLonDegreesEast): 
     #      raise Exception ('Data for added bolide must have the same lat/lon sub-point')

        self.groupList.extend(otherBolideDetection.groupList)
        self.eventList.extend(otherBolideDetection.eventList)

        # detection score is the mean detection score of the two being combined
        self.assessment.triage.score = np.mean([self.assessment.triage.score, otherBolideDetection.assessment.triage.score])
        self.assessment.validation.score = np.mean([self.assessment.validation.score, otherBolideDetection.assessment.validation.score])

        if otherBolideDetection.assessment.triage.method not in self.assessment.triage.method:
            self.assessment.triage.method = 'Combined: ' + self.assessment.triage.method + ' + ' + otherBolideDetection.assessment.triage.method
        if otherBolideDetection.assessment.validation.method not in self.assessment.validation.method:
            self.assessment.validation.method = 'Combined: ' + self.assessment.validation.method + ' + ' + otherBolideDetection.assessment.validation.method
        if otherBolideDetection.howFound not in self.howFound:
            self.howFound =  'Combined: ' + self.howFound + ' + ' + otherBolideDetection.howFound

        # Be sure groups and events are sorted by time
        self.sort_groups()
        if len(self.eventList) > 0:
            self.eventList.sort(key=lambda x: x.time)


    #******************************************************************************
    # Helper method to sort groups by time
    def sort_groups(self):
        self.groupList.sort(key=lambda x: x.time)

    #******************************************************************************
    # Returns the group time for each group
    # Does not sort, however in the constructor the groups should have already been sorted.
    @property
    def group_times(self):
        return np.array([group.time for group in self.groupList])

    #******************************************************************************
    # Returns the start and end time of the bolide detection as datetime objects
    @property
    def get_time_interval(self):
        # Use group time
        return np.min(self.group_times), np.max(self.group_times)

    #******************************************************************************
    # Returns the middle 50% of all groups times
    @property
    def bolideMidRange(self):
        # Make sure groups are sorted by time
        self.sort_groups()
        twentyFiveTime  = self.group_times[len(self.group_times)//4]
        seventyFiveTime = self.group_times[-len(self.group_times)//4]
        return [twentyFiveTime, seventyFiveTime]
        

    #******************************************************************************
    # List of all group latitudes and longitudes
    @property
    def group_lat_lon(self):
        return [group.latitudeDegreesNorth for group in self.groupList], [group.longitudeDegreesEast for group in self.groupList]

    #******************************************************************************
    # Returns the median latitude and longitude for the detection
    @property
    def average_group_lat_lon(self):
        latArray, lonArray = self.group_lat_lon
        avgLat = np.nanmedian(latArray)
        avgLon = np.nanmedian(lonArray)
        return avgLat, avgLon

    #******************************************************************************
    # Returns Group energy in Joules from self.groupList
    @property
    def energy(self):
        return np.array([group.energyJoules for group in self.groupList])

    #******************************************************************************
    @property
    def get_max_energy(self):
        """ returns the robust maximum energy, 
        meaning, the 3rd highest energy group record.

        However, if less than min_groups then just take the maximum energy

        """
        min_groups = 15
        # If less than min_groups then just take the maximum energy
        if len(self.energy) < min_groups:
            return self.energy.max()
        else:
            return np.sort(self.energy)[-3]

    #******************************************************************************
    @property
    def get_total_energy(self):
        # Use group energy
        energy = np.nansum(self.energy)
        return energy

    #******************************************************************************
    @property
    def avg_ground_speed(self):
        """ Returns the average speed scalar.

        It computes the speed by computing the distance and time between every two groups and then computing the median
        speed of the ratio of each of these.

        This does not account for direction

        """

        latArray = np.array([g.latitudeDegreesNorth for g in self.groupList])
        lonArray = np.array([g.longitudeDegreesEast for g in self.groupList])
        timeArray = [g.time for g in self.groupList]

        speed = []
        # Compute the speed from every two points and average
        # Be smart about computing speed between any two points only once
        for idx,_ in enumerate(self.groupList):
            # Distance from this reference group to all groups after it in time
            dist = geoUtil.DistanceFromLatLonPoints(latArray[idx], lonArray[idx], latArray[idx+1:], lonArray[idx+1:])
            # Time from this reference group to all groups later in time
            deltaTSec = np.array([(t - timeArray[idx]).total_seconds() for t in timeArray[idx+1:]])

            deltaTSec[np.nonzero(deltaTSec == 0.0)[0]] = np.nan

            speed.extend(np.abs(dist / deltaTSec))


        return np.nanmedian(speed)
        
        

    #******************************************************************************
    def glm_pixel_coords(self, latLon2Pix):
        """ Returns the pixel coordinates corresponding to the latitude and longitudes.

        Uses the lat_long_to_GLM_pixel.LatLong2Pix object, which must be passed to this method.

        Parameters
        ----------
        latLon2Pix : lat_lon_to_GLM_pixel.LatLon2Pix

        Returns
        -------
        self.groupList[:].x
        self.groupList[:].y
        """

        assert isinstance(latLon2Pix, LatLon2Pix), \
                'latLon2Pix must be passed'

        if not hasattr(self, 'yaw_flip_flag'):
            yaw_flip_flag = 0
        else:
            yaw_flip_flag = self.yaw_flip_flag
        
        lat = [g.latitudeDegreesNorth for g in self.groupList]
        lon = [g.longitudeDegreesEast for g in self.groupList]
        x, y = latLon2Pix.latLon2pix(lat, lon, yaw_flip_flag)

        for group, x1, y1 in zip(self.groupList, x, y):
            group.x = x1
            group.y = y1

    #******************************************************************************
    # This returns the name of the file that would be associated with this detection.
    @property
    def figureFilename(self):
        # The first index in filePathList is the primary data file (not the neighborhood)
        return os.path.basename(os.path.splitext(self.filePathList[0])[0] + '_' + str(self.ID) + '_detection.png')

    #******************************************************************************
    # This returns the name of the file that would be associated with this detection.
    # If we are detecting in the other satellite
    @property
    def figureFilenameOtherSatellite(self):
        # The first index in filePathList is the primary data file (not the neighborhood)
        # If the other satellite data does not exist then there is no othe rsatellite figure. 
        # We do not know the raw netCDF filename to use in the filename
        if self.bolideDetectionOtherSatellite is None:
            return None
        else:
            return os.path.basename(os.path.splitext(self.bolideDetectionOtherSatellite.filePathList[0])[0] + '_' + str(self.ID) + '_otherSatellite_detection.png')

    #******************************************************************************
    # This returns the name of the file that would be associated with this detection.
    # If we are plotting a stereo detection figure (see stereo_renavigation.py)
    @property
    def figureFilenameStereo(self):
        # The first index in filePathList is the primary data file (not the neighborhood)
        return os.path.basename(os.path.splitext(self.filePathList[0])[0] + '_' + str(self.ID) + '_stereo' + '.png')

    #******************************************************************************
    # Write a header for the CSV file 
    def write_csv_header(self, csvFileName, GJ_data=False):
        """ Writes the header for the CSV file.

        Parameters
        ----------
        csvFileName : [str] 
            full path to CSV file
        GJ_data : bool
            If True then append the extra data needed for the Boggs Gigantic Jets (GJ) study

        """

        with open(csvFileName, 'w') as csvfile:

            fieldnames = [
                    '# detection_ID', 
                    'triage_score', 
                    'validation_score', 
                    'figure_filename', 
                    'Data_filenames_list', 
                    'event_or_group_IDs_recorded', 
                    'ID:fileIndexForThisID',
                     'avgLat', 
                     'avgLon']

            if GJ_data:
                # Add in explicit date stamp to front
                tmp = copy.copy(fieldnames)
                fieldnames = ['# time']
                fieldnames.extend(tmp)
                fieldnames.extend([
                        'stereo_east_3D_max_dist_km',
                        'stereo_east_ground_max_dist_km',
                        'stereo_east_lat_deg_north',
                        'stereo_east_lon_deg_east', 
                        'stereo_east_UTC_time_POSIX',
                        'stereo_east_energyJoules',
                        'stereo_east_alt_km',
                        'stereo_east_alt_error_km',
                        'stereo_west_3D_max_dist_km',
                        'stereo_west_ground_max_dist_km',
                        'stereo_west_lat_deg_north',
                        'stereo_west_lon_deg_east', 
                        'stereo_west_UTC_time_POSIX',
                        'stereo_west_energyJoules',
                        'stereo_west_alt_km',
                        'stereo_west_alt_error_km',
                        ]
                        )

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    #******************************************************************************
    # Append a bolide detection to a CSV file.
    def append_csv(self, csvFileName, GJ_data=False):
        """ Writes the header for the CSV file.

        Parameters
        ----------
        csvFileName : [str] 
            full path to CSV file
        GJ_data : bool
            If True then append the extra data needed for the Boggs Gigantic Jets (GJ) study

        """

        # Remove full path from self.filePathList
        filePathList = [os.path.basename(filename) for filename in self.filePathList]
        
        # The event and group ID is unique only WITHIN each 20-second data file. 
        # We need to keep track from which data file.

        avgLat, avgLon = self.average_group_lat_lon

        # Record event_id if available and we are not recording data for GJs
        if (self.eventList != [] and not GJ_data):
            eventIdAndFile = [str(event.id)+':'+str(filePathList.index(event.datasetFile)) for event in self.eventList]
            row = [self.ID] + [self.assessment.triage.score] + [self.assessment.validation.score] + [self.figureFilename] + [filePathList] + ['event_id'] + \
                    [eventIdAndFile] + [avgLat] + [avgLon]

        else:
            groupIdAndFile = [str(group.id)+':'+str(filePathList.index(group.datasetFile)) for group in self.groupList]
            row = [self.ID] + [self.assessment.triage.score] + [self.assessment.validation.score] + [self.figureFilename] + [filePathList] + ['group_id'] + \
                    [groupIdAndFile] + [avgLat] + [avgLon]

            if GJ_data:
                # Append the extra data needed to study Gigantic Jets

                # Add in a ISO time stamp right at the beginning
                tmp = copy.copy(row)
                row = [self.bolideTime.isoformat()]
                row.extend(tmp)

                # We want to measured stereo data, however the pipeline reports stereo data for both
                # satellites. Return data for both satellites.
                row.extend([
                        self.stereoFeatures.sat_east.max_dist,
                        self.stereoFeatures.sat_east.max_ground_track,
                        self.stereoFeatures.sat_east.lat.tolist(),
                        self.stereoFeatures.sat_east.lon.tolist(), 
                        self.stereoFeatures.sat_east.timestamps.tolist(),
                        self.stereoFeatures.sat_east.energyJoules.tolist(),
                        self.stereoFeatures.sat_east.alt.tolist(),
                        self.stereoFeatures.sat_east.residual_dist.tolist(),
                        self.stereoFeatures.sat_west.max_dist,
                        self.stereoFeatures.sat_west.max_ground_track,
                        self.stereoFeatures.sat_west.lat.tolist(),
                        self.stereoFeatures.sat_west.lon.tolist(),
                        self.stereoFeatures.sat_west.timestamps.tolist(),
                        self.stereoFeatures.sat_west.energyJoules.tolist(),
                        self.stereoFeatures.sat_west.alt.tolist(),
                        self.stereoFeatures.sat_west.residual_dist.tolist()
                        ])

        try:
            # If the file exists, append the record.
            with open(csvFileName, 'a') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(row)
        except:
            sys.exit('Could not write to file {}.'.format(csvFileName))

        return 0

    
    #******************************************************************************
    # Performs a deep copy of object
    def copy(self):
        """Returns a copy of this `BolideDetection` object.

        Returns
        -------
        bolideDetection_copy : `BolideDetection`
            A new object which is a copy of the original.
        """

        return copy.deepcopy(self)


    #******************************************************************************
    # Returns a summary of the attributes in the BolideDetection class
    def __repr__(self): 

        bolideDetectionDict = self.__dict__.copy()
        bolideDetectionDict['subPointLonDegreesEast'] = float(self.subPointLonDegreesEast)
        bolideDetectionDict['subPointLatDegreesNorth'] = float(self.subPointLatDegreesNorth)
        bolideDetectionDict['glmHeight'] = float(self.glmHeight)
        bolideDetectionDict['groupList'] = '{} groups total'.format(len(self.groupList))
        bolideDetectionDict['eventList'] = '{} events total'.format(len(self.eventList))
        
        return ioUtil.print_dictionary(bolideDetectionDict)

#******************************************************************************
# Removes duplicates from bolideDetectionList
#
# Groups are clustered including a small fraction of groups from neighboring files if close in time
# to the main file. This means if a cluster is contained in the overlap region, the cluster (i.e. detection) will
# be detected twice. We need to clean these duplicates from the list.
#
# Note that this will only remove a duplicate bolide if ALL gouprs are identical between both. 
#
# Inputs:
#   bolideDetectionList : [list of BolideDetection] The list of bolide detections (or rejections)
#
# Outputs:
#   bolideDetectionListCleaned : [list of BolideDetection] The list of bolide detections (or rejections)
#                                With duplicates removed
#******************************************************************************
def remove_duplicates_from_bolideDetectionList (bolideDetectionList):

    # Do nothing if list is empty
    if len(bolideDetectionList) == 0:
        return bolideDetectionList

    # If there is a duplicate it is probably because the detection was found near the end of the
    # target file and so we found the bolide boht in the shoulder region of the center file and in the middle of the
    # neighboring file. We want to keep the detection whcih is mostly in the target file and not in either side
    # file. So, we need to keep track of whcih file each detection is mostly in.
    isInPrimary = np.full(len(bolideDetectionList), False)
    for groupIdx, detection in enumerate(bolideDetectionList):
        groupFileName = [g.datasetFile for g in detection.groupList]
        # count number of groups in each file
        numInThisFile = []
        uniqueFilenames = np.unique(groupFileName) 
        for filename in uniqueFilenames :
            numInThisFile.append(np.count_nonzero([gFilename == filename for gFilename in groupFileName]))
        primaryFilename = uniqueFilenames[np.argmax(numInThisFile)]
        baseFilePathList = [os.path.basename(f) for f in detection.filePathList]
        isInPrimary[groupIdx] = baseFilePathList.index(primaryFilename) == 0

    # Sort all detections so that the ones in the primary file are first.
    # In the code below, the first of each duplicate is kept, so this sorting will force the algorithm to kjeep the
    # detection from the primary file.
    bolideDetectionList = [bolideDetectionList[idx] for idx in np.argsort(np.logical_not(isInPrimary))]


    groupListKeepers = [] # List of groups for each detection to keep
    bolideIndicesToRemove = []
    for groupIdx, detection in enumerate(bolideDetectionList):
        duplicate = False
        # Check if all groups in this detection are identical to any other detection
        groupListThisBolide = np.sort([g.id for g in detection.groupList])
        for idxKeeper, listItem in enumerate(groupListKeepers):
            if (len(groupListThisBolide) == len(listItem) and
                    np.all(groupListThisBolide == listItem)):
                duplicate = True
                break
        if duplicate:
            # Duplicate! Because we sorted the bolideDetectionList above we know the detection already in the keeper
            # list is from the primary file, if either is in the primary file
            # So, add this one to the list to remove
            bolideIndicesToRemove.append(groupIdx)
        else:
            # Not a duplicate add to list
            groupListKeepers.append(groupListThisBolide)


    # Remove the duplicates
    bolideDetectionListCleaned = [detection for idx, detection in enumerate(bolideDetectionList) if 
                                    bolideIndicesToRemove.count(idx) == 0]

    return bolideDetectionListCleaned 

def keep_groups_within_spatiotemporal_box(spatiotemporal_box, glmGroups):
    """
    Keeps only the groups that are within the specified spatiotemporal_box.

    If streak_width is not None then we are finding a streak and the spatiotemporal box just gives the end-points and
    the streak_width gives the width of the streak in kilometers.

    Parameters
    ----------
    spatiotemporal_box : run_glm_pipeline_quick.SpatiotemporalBox
    glmGroups   : [list of bolide_detection.GlmGroup] 
            A list of GlmGroup objects
 
    Returns
    -------
    glmGroupsKept   : [list of bolide_detection.GlmGroup] 
        A list of GlmGroup objects within the spatiotemporal_box
    idxToKeep : list of int
        The indices of the groups to keep in the glmGroups array 

    """

    # Do nothing if list is empty
    if len(glmGroups) == 0:
        return glmGroups, []

    # Do nothing if spatiotemporal box is None
    if spatiotemporal_box is None:
        return glmGroups, np.arange(len(glmGroups))

    spBox = fill_spatiotemporal_box(spatiotemporal_box)

    idxToKeep = []
    for idx, group in enumerate(glmGroups):

        # Within spatio-temporal box
        lat = group.latitudeDegreesNorth
        lon = geoUtil.unwrap_longitude([group.longitudeDegreesEast])[0]
        if spBox.streak_width is None:
            # The spatiotemporal_box is a true box, find all groups within the box
            if (lat >= spBox.minLat and
                lat <= spBox.maxLat and
                lon >= spBox.minLon and
                lon <= spBox.maxLon and
                group.time >= spBox.minDate and 
                group.time <= spBox.maxDate):
            
                idxToKeep.append(idx)
        else:
            # First check if within time brackets
            if (group.time >= spBox.minDate and group.time <= spBox.maxDate):
                # The spatiotemporal_box is a line with a width, find all groups on the line
                P1 = (spBox.minLon, spBox.minLat)
                P2 = (spBox.maxLon, spBox.maxLat)
                if geoUtil.dist_point_to_line(P1, P2, (lon, lat)) <= spBox.streak_width:
                    idxToKeep.append(idx)


    # Create new list of just the files we want
    glmGroupsKept = [glmGroups[idx] for idx in idxToKeep]

    return glmGroupsKept, idxToKeep

def keep_bolides_within_spatiotemporal_box(spatiotemporal_box, bolideDetectionList):
    """

    Keeps only the BolideDetection objects in the list that are within the specified spatio-temporal box given in
    spatiotemporal_box

    Time is based on BolideDetection.bolideTime
    lat/lon is based on BolideDetection.average_group_lat_lon

    Parameters
    ----------
    spatiotemporal_box : run_glm_pipeline_quick.SpatioTemporalBox
    bolideDetectionList : [list of BolideDetection] 
        The list of bolide detections (or rejections)
 
    Returns
    -------
    bolideDetectionListKept : [list of BolideDetection] 
        The list of bolide detections (or rejections) but only within the spatio-temporal box

    """

    # Do nothing if list is empty
    if len(bolideDetectionList) == 0:
        return bolideDetectionList

    # Do nothing if spatiotemporal box is None
    if spatiotemporal_box is None:
        return bolideDetectionList

    spBox = fill_spatiotemporal_box(spatiotemporal_box)

    idxToKeep = []
    for idx, detection in enumerate(bolideDetectionList):

        # Within spatio-temporal box
        avgLat, avgLon = detection.average_group_lat_lon
        avgLon = geoUtil.unwrap_longitude([avgLon])[0]
        if spBox.streak_width is None:
            if (avgLat >= spBox.minLat and
                avgLat <= spBox.maxLat and
                avgLon >= spBox.minLon and
                avgLon <= spBox.maxLon and
                detection.bolideTime >= spBox.minDate and 
                detection.bolideTime <= spBox.maxDate):
            
                idxToKeep.append(idx)
        else:
            # First check if within time brackets
            if (detection.bolideTime >= spBox.minDate and detection.bolideTime <= spBox.maxDate):
                # The spatiotemporal_box is a line with a width, find all groups on the line
                P1 = (spBox.minLon, spBox.minLat)
                P2 = (spBox.maxLon, spBox.maxLat)
                if geoUtil.dist_point_to_line(P1, P2, (avgLon, avgLat)) <= spBox.streak_width:
                    idxToKeep.append(idx)


    # Create new list of just the files we want
    bolideDetectionListKept = [bolideDetectionList[idx] for idx in idxToKeep]

    return bolideDetectionListKept

def fill_spatiotemporal_box(spatiotemporal_box):
    """
    Fills in missing values in spatiotemporal_box with default extreme values

    Parameters
    ----------
    spatiotemporal_box : run_glm_pipeline_quick.SpatiotemporalBox

    Returns
    -------
    spatiotemporal_box : run_glm_pipeline_quick.SpatiotemporalBox

    """

    # Make a local copy because we modify the values.
    spBox = copy.copy(spatiotemporal_box)

    # Fill non-specified limits with extreme values
    if spBox.minDate is None:
            spBox.minDate = datetime.min
    if spBox.maxDate is None:
            spBox.maxDate = datetime.max
    if spBox.minLat is None:
            spBox.minLat = -91.0
    if spBox.maxLat is None:
            spBox.maxLat = 91.0
    if spBox.minLon is None:
            spBox.minLon = -361.0
    if spBox.maxLon is None:
            spBox.maxLon = 361.0

    # If minDate and maxDate are equal to the same day, then set the maxDate to the next day, 
    # so that we include the entire day in the box.
    # This only works for valid POSIX timestamps (1970 or later)
    if spBox.minDate.year >= 1970:
        dayTimestamp = datetime.combine(spBox.minDate.date(), time()).timestamp()
        if (spBox.minDate.timestamp() - dayTimestamp) == 0.0 and \
                spBox.minDate == spBox.maxDate:
            spBox.maxDate = spBox.maxDate + timedelta(days=1)

    return spBox

# *****************************************************************************
# Write a list of bolide detection objects to a binary file.
#
# If the file exists, append the detection list.
#
# INPUTS
#   filename            -- [str] path and name fo file to write to
#   bolideDetections    -- [bolie_detections.BolideDetection list] A list of bolide detection objects.
#
# OUTPUTS
#     (none)
#
# NOTES
#
# *****************************************************************************
def pickle_bolide_detections(filename, bolideDetectionList) :

    # Write the data.
    try:
        # If the file exists, append the detection list.
        with open(filename, 'ab') as fp :
            pickle.dump( bolideDetectionList, fp)
    except:
        sys.exit('Could not write to file {}.'.format(filename))

    fp.close()


# *****************************************************************************
# unpickle_bolide_detections(filename)
#
# Unpickle bolide detections and return them as a list. The pickle file may
# contain either a list of objects or a series of individual objects that are
# not part of a list.
#
# This function will load in a list of objects from a pickle file. It technically is not specific to a
# bolideDetectionsList  and so can really be used for any list of objects in a pickle file.
#
# INPUTS
#   filename              : A string designating the path and name of
#                             the pickle file. The file may contain serial
#                             objects or one or more lists of objects.
# OUTPUTS
#   bolideDetectionsList    -- [bolie_detections.BolideDetection list] A list of bolide detection objects.
# *****************************************************************************
def unpickle_bolide_detections(filename) :

    bolideDetectionList = []

    with open(filename, 'rb') as fp:
        try:
            obj = pickle.load(fp)
            if isinstance(obj, list) :
                bolideDetectionList = obj
                while True:
                    bolideDetectionList.extend(pickle.load(fp))
            else :
                bolideDetectionList = [obj]
                while True:
                    bolideDetectionList.append(pickle.load(fp))
        except EOFError:
            pass # We've reached the end of the file.

    fp.close()

    return bolideDetectionList

# *****************************************************************************
# This __main__ function is to qwuickly test elements of this module
#
# *****************************************************************************
if __name__ == "__main__":


    pass

# ************************************ EOF ************************************
