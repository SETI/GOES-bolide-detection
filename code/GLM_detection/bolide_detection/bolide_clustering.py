#*************************************************************************************************************
# bolide_clustering.py
# 
# Class definitions to generate clusters of GLM L2 events or groups.
#
# This will cluster all groups within the specified input fileName but will als look for groups in neighboring files on
# either side of the file under study. All clusters found must have the median time within the file under study. This is
# so that when working one's way through all 20-secon files we do not generate up to three copies of each cluster. 
#
# In time, this clustering tool can be expanded to also work on L0 data.
#
#*************************************************************************************************************
import netCDF4
import os
import numpy as np
import datetime
import time
import scipy.cluster.hierarchy as spHierarchy
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import warnings
from numba import jit, prange, set_num_threads

import bolide_detections as bd
#import bolide_support_functions as bsf
import geometry_utilities as geoUtil
import io_utilities as ioUtil

#*************************************************************************************************************

class BolideCluster:
    """ Defines a cluster of bolide groups as a "cluster"

    Attributes:
    -----------
    filePathList    : [list of str] A list of files (path strings) containing the detection groups
    glmGroups       : [bd.GlmGroup] A list of objects containing the GLM groups in this cluster
    ID              : [int64] Bolide detection ID
    figureFilename  : [str] The name of file corresponding to detection, if a figure exists
    confidence      : [float] detection confidence, if one exists

    """
    def __init__(self, filePathList, glmGroups, ID=None):

        self.filePathList = filePathList
        self.glmGroups = glmGroups
        self.ID = ID
    
    def __repr__(self): 

        bolideDataSubsetDict = self.__dict__.copy()
        bolideDataSubsetDict['idList'] = '{} groups total'.format(len(glmGroups))
        
        return ioUtil.print_dictionary(bolideDataSubsetDict)



#*************************************************************************************************************

class BolideClustering:

#*************************************************************************************************************
# BolideClustering class
#
# Loads in GLM .nc file groups and stores relevant information prepping for clustering steps
#
# If extractNeighboringFiles, then self.glmGroupsWithNeighbors also contains all groups from neighboring files on either
# side in time, and self.glmGroups contains those groups in neighboring files within 2*closeness_seconds to the file under study
#
# If there are no groups in the netCDF file then self.glmGroups and self.glmGroupsWithNeighbors are empty sets (=[])
#
# INSTANCE ATTRIBUTES:
#   netCDFFilenameList          -- [str list] Path to the files the groups are to be extracted from
#   cluster_3D_enabled : [bool] If True then use a 3-dimensional hierarchical clustering (See atap github ticket 127)
#   numba_threads       : [int] The number of parallel threads to use for numba
#   min_num_groups_for_outliers -- [int] minimum number of groups in cluster for outlier rejection to be performed
#   outlierSigmaThreshold       -- [float] Outlier sigma threshold factor to clip at
#   outlierMinClipValueDegrees  -- [float] Outlier Minimum value to clip at when looking at lat/lon timeseries. 
#                                   Nothing is clipped below this value, no matter what sigmaThreshold is set at.
#   closeness_seconds           -- [float] The distance in time considered close for clustering (the time scaling factor)
#   closeness_km                -- [float] The distance in kilometers considered close for clustering (the distance scaling factor)
#   goesSatellite   -- [str] Which GOES Satellite 'G**'
#   productTime     -- [datetime.datetime] start time of observations associated with product, in seconds since 2000-01-01 12:00:00
#   subPointLatDegreesNorth : [float] Latitude of the sub-satellite point on the Earth's surface.
#   subPointLonDegreesEast  : [float] Longitude of the sub-satellite point on the Earth's surface.
#   glmHeight               : [float] Height of satellite in km (?)
#   lat_field_of_view_bounds: [float array] latitude coordinates for north/south extent of field of view
#   lon_field_of_view_bounds: [float array] longitude coordinates for west/east extent of field of view 
#   yaw_flip_flag           : [int] Yaw orientation of the satellite
#   glmGroups       -- [bd.GlmGroup] A list of objects containing ALL the groups from the netCDF data file, sorted by time
#                       Also contains those groups in neighboring files within 2*closeness_seconds to the file under study
#   glmGroupsWithNeighbors  -- [bd.GlmGroup] A list of objects containing ALL the groups from the netCDF data file and
#                       in neighboring files on either side, sorted by time
#   minTime         -- [datetime] Time of first group in filename
#   maxTime         -- [datetime] Time of last group in filename
#   spatiotemporal_box : [list of dict] Only group data within these spatiotemporal boxes
#   clusteredGroups     -- [dict list nClusters] All the groups gathered by cluster (initialized to [])
#       'groupIndices'      -- [list] the group indices in the glmGroups list
#       'groupIds'          -- [list] the groupIds in this cluster 
#       'groupFilenamesIndex' -- [list] the index to the source filename in self.netCDFFilenameList for this group
#       'time'              -- [list of datetime] the time for each group in this cluster 
#
#*************************************************************************************************************

    # Some constants (Well, not really, there are no constants in Python!)
    MAX_N_GROUPS = 30000 # Maximum number of groups hierachical clustering can handle without bogging down

    def __init__ (self, 
            fileName, 
            cluster_3D_enabled=True,
            numba_threads=16,
            closeness_seconds=0.2, 
            closeness_km=7.0, 
            extractNeighboringFiles=True, 
            min_num_groups_for_outliers=0,
            outlierSigmaThreshold=10.0, 
            outlierMinClipValueDegrees=0.15,
            spatiotemporal_box=None):
        """ Initializes the object.

        Parameters
        ----------
        fileName    : [str] path to netCDF file to load in GLM data under study
                        Can be a list of files, If it is, then groups are loaded from all files.
        cluster_3D_enabled : [bool] If True then use a 3-dimensional hierarchical clustering (See atap github ticket 127)
        numba_threads       : [int] The number of parallel threads to use for numba
        closeness_seconds           : [float] The distance in time considered close for clustering (the time scaling factor)
        closeness_km                : [float] The distance in kilometers considered close for clustering (the distance scaling factor)
        extractNeighboringFiles : [bool] If True then self.glmGroups also contains all groups from neighboring files on either side in time
                                This functionality is if loading in a single netCDF, if multiple files passed then no
                                neighbor exracting will occur.
        min_num_groups_for_outliers : [int] minimum number of groups in cluster for outlier rejection to be performed
        outlierSigmaThreshold       : [float] Outlier sigma threshold factor to clip at
        outlierMinClipValueDegrees  : [float] Outlier Minimum value to clip at. Nothing is clipped below this value, no matter what
                                        sigmaThreshold is set at.
        spatiotemporal_box : list of dict

        """

        # We want the minimum lat/lon sigma clipping to be larger than a pixel.
        self.cluster_3D_enabled             = cluster_3D_enabled
        self.numba_threads                  = numba_threads
        self.closeness_seconds              = closeness_seconds
        self.closeness_km                   = closeness_km
        self.min_num_groups_for_outliers    = min_num_groups_for_outliers 
        self.outlierSigmaThreshold          = outlierSigmaThreshold 
        self.outlierMinClipValueDegrees     = outlierMinClipValueDegrees 
        self.spatiotemporal_box = spatiotemporal_box

        # Determine if we are reading in from a single netCDF file or multiple files
        if isinstance(fileName, list) and extractNeighboringFiles:
            raise Exception('If extracting neighboring files then only a single netCDF fileName can be passed')
        elif isinstance(fileName, list):
            if len(fileName) == 0:
                raise Exception('Empty file list passed to BolideClustering')
            self.init_with_multiple_files(fileName)
        else:
            self.init_with_single_file(fileName, extractNeighboringFiles)

        # Initialize to an empty set
        self.clusteredGroups = []

        # These contain the group IDs and filenames for all files in self.glmGroups
        # Stored here for speed because this list is needed many,many times in determine_group_proximity
        self.groupIdList = np.array([g.id for g in self.glmGroups])
        filePathList = [os.path.basename(filename) for filename in self.netCDFFilenameList]
        self.filenameListIndex = [filePathList.index(g.datasetFile) for g in self.glmGroups]

        return

    #*************************************************************************************************************
    def init_with_single_file(self, fileName, extractNeighboringFiles):
        """ This will initialize the BolideClustering class with a single netCDF file.

        TODO: init_with_single_file and init_with_multiple_files can probably be merged into a single method.
        """

        ## load data, including neighoring files on either side
        nc4Data = netCDF4.Dataset(fileName)
        # Collect all the groups from the file and neighboring files
        self.goesSatellite = nc4Data.platform_ID
        self.glmGroupsWithNeighbors, self.netCDFFilenameList = bd.extract_groups_from_all_files(fileName, eventIdListEnabled=False, ignoreZeroEnergyGroups=True, 
                ignorePoorQualityGroups=True, ignorePoorDateGroups=True, extractNeighboringFiles=extractNeighboringFiles)

        # If not doing an explicit box, then only keeps groups within each spatiotemporal_box
        # That way the clustering algorithm will only make clusters of the groups within the box
        if self.spatiotemporal_box is not None:
            if not self.spatiotemporal_box[0].explicit_cluster:
                glmGroupsAllClusters = []
                for box in self.spatiotemporal_box:
                    glmGroupsOneCluster, _ = bd.keep_groups_within_spatiotemporal_box(box, self.glmGroupsWithNeighbors)
                    glmGroupsAllClusters.extend(glmGroupsOneCluster)
                self.glmGroupsWithNeighbors = glmGroupsAllClusters

            
        if (len(self.glmGroupsWithNeighbors) == 0):
            self.glmGroups = []
            return

        self.productTime = bd.BolideDetection.epoch + datetime.timedelta( seconds=int(nc4Data.variables['product_time'][0].data) )

        # Record the start and end times of all groups in file under study (not neighboring files)
        # HOWEVER, also keep those groups in neighboring files within 2*closeness_seconds to the file under study
        # TODO: This uses repeat code from bolide_detections.get_groups_by_id, consolodate code?
        timeOffsetMsec = nc4Data.variables['group_time_offset'][:].data
        # The nc file format changed after 10/15/2018 @ 16:00 afterwhich group_time_offset is in fractions of second
        if self.productTime > datetime.datetime(2018, 10, 15, 16, 0): 
            # nc file format changed and group_time_offset is in fractions of second
            timeOffsetMsec = [o * 1000.0 for o in timeOffsetMsec]
        groupTime = [self.productTime + datetime.timedelta(milliseconds=float(offset)) for offset in timeOffsetMsec]
        self.minTime = np.min(groupTime)
        self.maxTime = np.max(groupTime)

        minTimeWithFudge = self.minTime - datetime.timedelta(seconds=10.0*self.closeness_seconds)
        maxTimeWithFudge = self.maxTime + datetime.timedelta(seconds=10.0*self.closeness_seconds)

        # Only keep groups for glmGroups with a time within minTime and maxTime plus and minus a fudge factor to
        # consider cluster leakage into neighboring files.
        self.glmGroups = [g for g in self.glmGroupsWithNeighbors if (g.time >= minTimeWithFudge and g.time <= maxTimeWithFudge)]

        #***
        # Check that all group IDs are unique (they better be!)
        # Note: now that groups span multiple files we cannot guarantee all group IDs are unique
       #groupIds = np.array([g.id for g in self.glmGroupsWithNeighbors ])
       #assert len(np.unique(groupIds)) == len(groupIds), 'Groups IDs are NOT unique!'


        # TODO: Properly handle masked arrays
        self.subPointLatDegreesNorth    = nc4Data.variables['nominal_satellite_subpoint_lat'][0].data.tolist()
        self.subPointLonDegreesEast     = nc4Data.variables['nominal_satellite_subpoint_lon'][0].data.tolist()
        self.glmHeight                  = nc4Data.variables['nominal_satellite_height'][0].data.tolist()

        self.lat_field_of_view_bounds = [nc4Data.variables['lat_field_of_view_bounds'][0].data.tolist(), 
                                         nc4Data.variables['lat_field_of_view_bounds'][1].data.tolist()]
        self.lon_field_of_view_bounds = [nc4Data.variables['lon_field_of_view_bounds'][0].data.tolist(), 
                                         nc4Data.variables['lon_field_of_view_bounds'][1].data.tolist()]
        
        self.yaw_flip_flag = nc4Data.variables['yaw_flip_flag'][0].data.tolist()

        return
        
    #*************************************************************************************************************
    def init_with_multiple_files(self, filenames):
        """ This will initialize the BolideClustering class with multiple netCDF files.

        """
    
        # Work through all files and load the group data
        self.glmGroups = []
        self.glmGroupsWithNeighbors = None
        self.netCDFFilenameList  = []
        groupTime = []
        for idx, filename in enumerate(filenames):
            nc4Data = netCDF4.Dataset(filename)
            glmGroupsWithNeighborsThisFile, netCDFFilenameListThisFile = bd.extract_groups_from_all_files(filename, eventIdListEnabled=False, 
                    ignoreZeroEnergyGroups=True, ignorePoorQualityGroups=True, ignorePoorDateGroups=True, extractNeighboringFiles=False)
            # If not doing an explicit box, then Only keeps groups within each spatiotemporal_box
            # That way the clustering algorithm will only make clusters of the gourps within the box
            if self.spatiotemporal_box is not None:
                if not self.spatiotemporal_box[0].explicit_cluster:
                    glmGroupsAllClusters = []
                    for box in self.spatiotemporal_box:
                        glmGroupsOneCluster, _  = bd.keep_groups_within_spatiotemporal_box(box, glmGroupsWithNeighborsThisFile)
                        glmGroupsAllClusters.extend(glmGroupsOneCluster)
                    glmGroupsWithNeighborsThisFile = glmGroupsAllClusters
            self.glmGroups.extend(glmGroupsWithNeighborsThisFile)
            self.netCDFFilenameList.extend(netCDFFilenameListThisFile)

            # productTime and some other fields will be for the first file
            if idx == 0:
                self.goesSatellite = nc4Data.platform_ID
                self.productTime = bd.BolideDetection.epoch + datetime.timedelta( seconds=int(nc4Data.variables['product_time'][0].data) )
                self.subPointLatDegreesNorth    = nc4Data.variables['nominal_satellite_subpoint_lat'][0].data.tolist()
                self.subPointLonDegreesEast     = nc4Data.variables['nominal_satellite_subpoint_lon'][0].data.tolist()
                self.glmHeight                  = nc4Data.variables['nominal_satellite_height'][0].data.tolist()
                
                self.lat_field_of_view_bounds = [nc4Data.variables['lat_field_of_view_bounds'][0].data.tolist(), 
                                                 nc4Data.variables['lat_field_of_view_bounds'][1].data.tolist()]
                self.lon_field_of_view_bounds = [nc4Data.variables['lon_field_of_view_bounds'][0].data.tolist(), 
                                                 nc4Data.variables['lon_field_of_view_bounds'][1].data.tolist()]

                self.yaw_flip_flag = nc4Data.variables['yaw_flip_flag'][0].data.tolist()


            # Record the start and end times of all groups in all files under study
            # TODO: This uses repeat code from bolide_detections.get_groups_by_id, consolodate code?
            productTime = bd.BolideDetection.epoch + datetime.timedelta( seconds=int(nc4Data.variables['product_time'][0].data) )
            timeOffsetMsec = nc4Data.variables['group_time_offset'][:].data
            # The nc file format changed after 10/15/2018 @ 16:00 afterwhich group_time_offset is in fractions of second
            if productTime > datetime.datetime(2018, 10, 15, 16, 0): 
                # nc file format changed and group_time_offset is in fractions of second
                timeOffsetMsec = [o * 1000.0 for o in timeOffsetMsec]
            groupTimeThisFile = [productTime + datetime.timedelta(milliseconds=float(offset)) for offset in timeOffsetMsec]
            groupTime.extend(groupTimeThisFile)

        if len(groupTime) > 0:
            self.minTime = np.min(groupTime)
            self.maxTime = np.max(groupTime)
        else:
            self.minTime = datetime.datetime(year=datetime.MINYEAR, month=1, day=1)
            self.maxTime = datetime.datetime(year=datetime.MAXYEAR, month=12, day=31)

        return

    #*************************************************************************************************************
    def __repr__(self):
        repr_string =   'netCDF Filenames: {} '.format(self.netCDFFilenameList) + '\n' \
                        'goesSatellite: {} '.format(self.goesSatellite) + '\n' \
                        'productTime: {} '.format(self.productTime) + '\n' \
                        'Num GLM Groups: {} '.format(len(self.glmGroups)) + '\n' \
                        'Num Clustered Groups: {} '.format(len(self.clusteredGroups)) + '\n'

        return repr_string

    #*************************************************************************************************************
    def get_groups_this_cluster(self, clusterIndex):
        """ Returns the bd.GlmGroup objects for this cluster

        Parameters
        ----------
        clusterIndex : [int] cluster index in self.clusteredGroups

        Returns
        -------
        glmGroups   : [bd.GlmGroup] A list of objects containing the groups from this cluster

        """

        return [self.glmGroups[i] for i in self.clusteredGroups[clusterIndex]['groupIndices']]

    #*************************************************************************************************************
    def remove_groups_from_cluster(self, clusterIndex, groupIndicesToRemove):
        """ Removes the selected groups from the selected cluster

        Parameters
        ----------
        clusterIndex    : [int] The cluster index in the self.clueteredGroups array
        groupIndicesToRemove    : [int array] The group indices inside the selected cluster

        Returns
        -------
        self.clusteredGroups[clusterIndex] with requested groups removed
            'groupIndices'      -- [list of ] the group indices in the glmGroups list
            'groupIds'          -- [dict] the groupIds in this cluster 
            'time'              -- [list of datetime] the time for each group in this cluster 

        """

        assert self.clusteredGroups != [], 'Clustering has not yet been performed'

        if (len(groupIndicesToRemove) == 0):
            return

        # Only keep datums NOT in groupIndicesToRemove list
        for key in self.clusteredGroups[clusterIndex].keys():
            self.clusteredGroups[clusterIndex][key] = np.array([self.clusteredGroups[clusterIndex][key][idx] for idx in
                np.arange(len(self.clusteredGroups[clusterIndex][key])) if
                    np.logical_not(np.count_nonzero(groupIndicesToRemove == idx) > 0)])


    #*************************************************************************************************************
    # cluster_glm_groups
    #
    # Uses hierachical clustering and Clemens' sequential clustering to generate groups to consider as bolide detections.
    #
    # Right now, this just works on one .nc file at a time.
    #
    # Inputs:
    #   sequential_clustering_enabled   -- [bool] If True then perform sequential clustering
    #   outlier_rejection_enabled       -- [bool] If True then remove outliers in energy, lat and lon for each cluster 
    #
    # Outputs:
    #   self.clusteredGroups    -- [list nClusters] All the groups gathered by cluster
    #       'groupIndices'          -- [list of ] the group indices in the glmGroups list
    #       'groupIds'              -- [list] the groupIds in this cluster 
    #       'groupFilenamesIndex'   -- [list] the index to the source filename in self.netCDFFilenameList for this group
    #       'time'                  -- [list of datetime] the time for each group in this cluster 
    #   success                 -- [boolean] True only if the clustering was succesfful
    #
    #
    #*************************************************************************************************************
    def cluster_glm_groups(self, sequential_clustering_enabled=False, outlier_rejection_enabled=True):

        self.clusteredGroups = []

        if self.spatiotemporal_box is not None and self.spatiotemporal_box[0].explicit_cluster:
            # Explicit clustering

            # Work through each spatiotemporal box available and create a cluster for each
            self.cluster_spatiotemporal_boxes()

            if (self.clusteredGroups == []):
                # No clusters created
                success = False
            else:
                success = True

        else:
            # Normal clustering
            #***
            # Perform hierachical clustering in order to do a first fast clustering pass
            success = self.hierarchical_clustering()
            if (not success):
                # ATAPG-37: The only times success=False so far have been due to there being no groups to cluster, which should not throw an error.
                # JS cannot think of other reasons to throw this error, other than an algorithm faliure, which should raise an exception. So, commenting out this warning message.
               #print ('hierarchical clustering failed!')
                return success
            
            #***
            # Now perform the original sequential clustering to subdivide the above clusters
            if sequential_clustering_enabled:
                self.sequential_clustering()

        #***
        # Ensure all groups are sorted by time
        for iCluster, cluster in enumerate(self.clusteredGroups):
            timeArray = np.array(cluster['time'])
            groupsIdArray = np.array(cluster['groupIds'])
            groupsFilenameIndexArray = np.array(cluster['groupFilenamesIndex'])
            groupsIndicesArray = np.array(cluster['groupIndices'])
            timeGroupList = [[t, x, f, i] for t,x,f,i in sorted(zip(timeArray,groupsIdArray,groupsFilenameIndexArray,groupsIndicesArray))]
            [timeSorted, groupIdsSorted, groupFilenamesIndexSorted, groupIndicesSorted] = zip(*timeGroupList)
            # TODO: get rid of this optional keyname: 'streakIdx'
            if 'streakIdx' in cluster:
                self.clusteredGroups[iCluster] = {'groupIds': groupIdsSorted, 
                                                  'groupFilenamesIndex': groupFilenamesIndexSorted,
                                                  'time': timeSorted,
                                                  'groupIndices': groupIndicesSorted,
                                                  'streakIdx': cluster['streakIdx']} 
            else:
                self.clusteredGroups[iCluster] = {'groupIds': groupIdsSorted, 
                                                  'groupFilenamesIndex': groupFilenamesIndexSorted,
                                                  'time': timeSorted,
                                                  'groupIndices': groupIndicesSorted} 
            
        #***
        # Outlier rejection
        # This requires the groups to be sorted in time
        if self.spatiotemporal_box is not None and not self.spatiotemporal_box[0].explicit_cluster and outlier_rejection_enabled:
            self.outlier_rejection()

        #***
        # Only keep clusters with nonzero number of groups
        # Also only keep clusters with a median time within minTime and maxTime
        # Also remove any clusters where all groups have the same time
        clustersToRemoveIdx = []
        for iCluster, cluster in enumerate(self.clusteredGroups):
            timeArray = cluster['time']
            if len(timeArray) == 0:
                # Zero length cluster, remove
                clustersToRemoveIdx.append(iCluster)
                continue

            clusterTime = timeArray[len(timeArray)//2]
            if (clusterTime < self.minTime or clusterTime > self.maxTime):
                # Outside center file time range, remove from cluster list
                clustersToRemoveIdx.append(iCluster)

            elif len(timeArray) > 1 and np.all(np.array(timeArray) == timeArray[0]):
                # All groups have the same time
                clustersToRemoveIdx.append(iCluster)

        # Now do the removing
        self.clusteredGroups = [cluster for idx, cluster in enumerate(self.clusteredGroups) if clustersToRemoveIdx.count(idx) == 0]

       ## Check if any clusters have groups in more than two files
       #for cluster in self.clusteredGroups:
       #    if len(np.unique(cluster['groupFilenamesIndex'])) > 2:
       #        raise Exception('A cluster covers more than two data files, this should not be!')


        return success

    #*************************************************************************************************************
    # Perform hierarchical clustering
    #
    # Using SciPy's package
    #
    # Outputs:
    #   self.clusteredGroups    -- [A list of dicts of length nClusters] All the groups grouped into clusters
    #       'groupIds'          -- [dict] the groupIds in this cluster 
    #       'time'              -- [dict] the time for each group in this cluster 
    #   success             -- [boolean] True only if the clustering was succesfful
    #
    #*************************************************************************************************************
    def hierarchical_clustering(self):

        filePathList = [os.path.basename(filename) for filename in self.netCDFFilenameList]

        # Check if we have already generated clusters.
        # If so, issue a warning
        if (self.clusteredGroups != []):
            warnings.warn('This GLM data appears to have already been clustered. Overwriting old clusters...')

        # Not successful until otherwise noted
        success = False
 
        bolides = [ ]
 
        # Form the data matrix X with n samples and m features with X.shape := (n,m)
        # The m features are latitude, longitude and time
        # All three normalized
        # We need a time reference. Choose the first group time
        timeZero = self.glmGroups[0].time
        nGroups = len(self.glmGroups)

        timeArray   = np.array([(group.time - timeZero).total_seconds() for group in self.glmGroups])
        latArray    = np.array([group.latitudeDegreesNorth for group in self.glmGroups])
        lonArray    = np.array([group.longitudeDegreesEast for group in self.glmGroups])

        if self.cluster_3D_enabled:
            X = _3D_distance_matrix(latArray, lonArray, timeArray, self.closeness_seconds, self.closeness_km, numba_threads=self.numba_threads)
        else:
            X = self._2D_distance_array(latArray, lonArray, timeArray)
 
        # The hierarchical clustering "linkage" function bogs down if the number of groups become too large. 
        # If number of groups is larger than MAX_N_GROUPS then subdivide the groups into smaller chunks
        self.clusteredGroups = []
        nChunks = int(np.ceil(nGroups / self.MAX_N_GROUPS))
        countIndex = 0
        max_d = np.sqrt(2.0) # Set because we normalized our dimensions to 1.0
        for iChunk in np.arange(nChunks):
            chunkRange = np.arange(countIndex,np.min([countIndex+self.MAX_N_GROUPS, nGroups]))
            if self.cluster_3D_enabled:
                # linkage() can take a condensed distance matrix, Convert to condenced with squareform function
                Xchunk = squareform(X[countIndex:np.min([countIndex+self.MAX_N_GROUPS, nGroups]), countIndex:np.min([countIndex+self.MAX_N_GROUPS, nGroups])])
            else:
                Xchunk = X[chunkRange,:]
            glmGroupsChunk = []
            for idx in chunkRange:
                glmGroupsChunk.append(self.glmGroups[idx])
        
            # generate the Linkage Matrix
            # The old sequential clustering method adds on groups to the "bolide" in chronological order comparing each new group
            # to the distance to the previously added group. The best way to replicate this with hierarchical clustering is to
            # use the "single" distance metric
            if np.shape(Xchunk)[0] >= 2:
                Z = spHierarchy.linkage(Xchunk,'single')
                
                #********
                # Create the clusters
                clusters = spHierarchy.fcluster(Z, max_d, criterion='distance')
                
                #********
                # Create the group clusters to return
                groupIdList = np.array([g.id for g in glmGroupsChunk])
                groupFilenameList = np.array([g.datasetFile for g in glmGroupsChunk])
                groupTimeList = np.array([g.time for g in glmGroupsChunk])
                for iCluster in np.arange(1,np.max(clusters)+1):
                    groupsThisBolideDetection = np.nonzero(clusters==iCluster)[0].astype(int)
                    # List the groups in this cluster and get the times for each
                    self.clusteredGroups.append({   'groupIndices':   chunkRange[groupsThisBolideDetection],
                                                    'groupIds':       groupIdList[groupsThisBolideDetection],
                                'groupFilenamesIndex': [filePathList.index(filename) for filename in groupFilenameList[groupsThisBolideDetection]],
                                                    'time': groupTimeList[groupsThisBolideDetection]})

            countIndex = countIndex+self.MAX_N_GROUPS

 
        #********
        #********
        #********
        # plotting
        #
        # This is for diagnostic purposes to examine the hierachical clustering
        if (False):
            fontSize = 30
            # Plot the dendrogram
            plt.figure(figsize=(20,8))
            plt.title('Hierarchical Clustering Dendrogram (truncated)', fontsize=fontSize)
            plt.xlabel('sample index or (cluster size)', fontsize=fontSize)
            plt.ylabel('distance', fontsize=fontSize)
            spHierarchy.dendrogram(
                Z,
                truncate_mode='lastp',  # show only the last p merged clusters
                p=100,  # show only the last p merged clusters
                leaf_rotation=90.,
                leaf_font_size=12.,
                show_contracted=True,  # to get a distribution impression in truncated branches
                color_threshold=max_d
            )
            plt.show()
            
            #**********
            # Visualizing the clusters
            plt.figure(figsize=(20, 8))
            plt.title('Hierarchical Clustering Time/Distance', fontsize=fontSize)
            plt.xlabel('Time [seconds]', fontsize=fontSize)
            plt.ylabel('Distance [km]', fontsize=fontSize)
            plt.scatter(timeArray, distanceKm, c=clusters, cmap='prism')  # plot points with cluster dependent colors
            plt.show()
 
           #input("Press the <ENTER> key to continue...")
        #********
        #********
        #********
 
        if (self.clusteredGroups == []):
            # No clusters created
            success = False
        else:
            success = True
 
        return success

    #*************************************************************************************************************
   #@jit(parallel=False)
    def _2D_distance_array(self, latArray, lonArray, timeArray):
        """ Compute a 2-D distance array as a combination of time, latitude and longitude

        The latitude and longitude are combined into a single distance value. 

        Parameters
        ----------
        latArray : np.array
            Array of latitude in degrees
        lonArray : np.array
            Array of longitude in degrees
        timeArray : np.array
            Array of time in seconds from an arbitrary reference time
            (Such as the first time point in the array)

        Returns
        -------
        distanceArray : 2D np.array of time and distance of shape (n_elements, 2)
            The normalized time and distance for each point in the arrays

        """

        # Time closeness is set at 0.2 seconds, So scale time dimension such that 0.2 => 1.0
        timeArrayNorm = timeArray / self.closeness_seconds

        # Convert to absolute distance
        # closeness_km sets the closeness scale
        distanceKm = geoUtil.DistanceFromLatLonPoints(latArray[0], lonArray[0], latArray, lonArray)
        distanceNorm = distanceKm / self.closeness_km
        distanceArray = np.array([timeArrayNorm.T, distanceNorm.T]).T

        return distanceArray
 
        
    #*************************************************************************************************************
    # sequential_clustering()
    #
    # Performs the sequential clustering operation on the clusters.
    #
    # This method requires the self.clusteredGroups to have already been generated.
    # TODO: allow this to work on non-clustered data.
    #
    # Inputs:
    #
    # Outputs:
    #   self.clusteredGroups    -- [list nClusters] All the groups gathered by cluster
    #       'groupIndices'      -- [list] the group indices in the glmGroups list
    #       'groupIds'          -- [dict] the groupIds in this cluster 
    #       'groupFilenamesIndex' -- [list] the index to the source filename in self.netCDFFilenameList for this group
    #       'time'              -- [dict] the time for each group in this cluster 
    #
    #*************************************************************************************************************
    def sequential_clustering(self):

        # Check if another clustering method has already been performed
        if (self.clusteredGroups == []):
            raise Exception("Looks like the groups have not been clustered yet.")
 
        # The code below changes the length of self.clusteredGroups so we need to save the original while we work on it
        groupsThisClusterSave = self.clusteredGroups
 
        for cluster in groupsThisClusterSave:
 
            # Make sure groups are time ordered
            timeArray = np.array(cluster['time'])
            groupsIdArray = np.array(cluster['groupIds'])
            groupsFilenameIndexArray = np.array(cluster['groupFilenamesIndex'])
            groupsIndicesArray = np.array(cluster['groupIndices'])
            timeGroupList = [[t, x, f, i] for t,x,f,i in sorted(zip(timeArray,groupsIdArray,groupsFilenameIndexArray,groupsIndicesArray))]
            [timeSorted, groupIdsSorted, groupFilenamesIndexSorted, groupIndicesSorted] = zip(*timeGroupList)
 
            clustersTemp = [ ]
            #*******************************
            # Vectorized Sequential clustering (FAST)
            # Take each group and first try to attach it to the last cluster, if the new group is too far away
            # then start a new group. This method requires all groups to be chronologically ordered.
            
            while (len(groupIdsSorted) > 0):
                # Create a new potential bolide with the next non-grouped group
                clustersTemp.append({'groupIds': [groupIdsSorted[0]], 
                                     'groupFilenamesIndex': [groupFilenamesIndexSorted[0]],
                                     'time': [timeSorted[0]],
                                     'groupIndices': [groupIndicesSorted[0]]}) 
                groupIdsSorted              = np.delete(groupIdsSorted, 0)
                groupFilenamesIndexSorted   = np.delete(groupFilenamesIndexSorted, 0)
                groupIndicesSorted          = np.delete(groupIndicesSorted, 0)
                timeSorted                  = np.delete(timeSorted, 0)
                # Find the groups that belong to this potential bolide cluster
                # Iterate the grouping to better replicate the sequential clustering.
                stillAdding = True
                while stillAdding:
                    inCluster = self.determine_group_proximity(
                            clustersTemp[-1]['groupIds'][-1], clustersTemp[-1]['groupFilenamesIndex'][-1], 
                            groupIdsSorted, groupFilenamesIndexSorted)
                    if (not np.any(inCluster)):
                        stillAdding = False
                    else:
                        # Add the groups to this bolide
                        clustersTemp[-1]['groupIds'] = np.append(clustersTemp[-1]['groupIds'], groupIdsSorted[inCluster])
                        clustersTemp[-1]['groupFilenamesIndex'] = np.append(clustersTemp[-1]['groupFilenamesIndex'], groupFilenamesIndexSorted[inCluster])
                        clustersTemp[-1]['groupIndices'] = np.append(clustersTemp[-1]['groupIndices'], groupIndicesSorted[inCluster])
                        clustersTemp[-1]['time'] = np.append(clustersTemp[-1]['time'], timeSorted[inCluster])
                        # Remove groups from list of all non-grouped groups
                        groupIdsSorted              = np.delete(groupIdsSorted, np.nonzero(inCluster))
                        groupFilenamesIndexSorted   = np.delete(groupFilenamesIndexSorted, np.nonzero(inCluster))
                        groupIndicesSorted          = np.delete(groupIndicesSorted, np.nonzero(inCluster))
                        timeSorted   = np.delete(timeSorted, np.nonzero(inCluster))
                        
 
            #*******************************
            # Replace old big bolide with new subdivided bolides in big bolides array
            self.clusteredGroups = self.clusteredGroups[1:] # Remove the old big cluster
            self.clusteredGroups.extend(clustersTemp)
 


    #*************************************************************************************************************
    #
    # Determines if an array of GLM groups are proximal to a reference group
    #
    # Proximal means the lat/lon distance and time distance are witin self.closeness_km and self.closeness_seconds
    #
    # the list of glmGroups can cover multiple netCDF files so we have to track the filenames too.
    #
    # Inputs:
    #   refGroupId          -- [int] The groupID for the reference group
    #   refSourceFileIndex  -- [str] The index for the filename for the reference group in self.netCDFFilenameList
    #   groupIds            -- [int list] The group IDs for all the groups to compute proximity to reference group
    #   groupSourceFilesIndex -- [int list] The group filenames index in self.netCDFFilenameList for all the groups to compute proximity to reference group
    #
    # Outputs:
    #   isProximal          -- [lgical list(len(groups))] Logical array of which groups are proximal to the reference group 
    #
    #*************************************************************************************************************
    def determine_group_proximity(self, refGroupId, refSourceFileIndex, groupIds, groupSourceFilesIndex):

        # Locate the groups in the glmGroups
        refGroupIndex = list(np.nonzero(np.logical_and(np.in1d(self.groupIdList, refGroupId), np.in1d(self.filenameListIndex, refSourceFileIndex)))[0])
        assert len(refGroupIndex) == 1, 'Bookkeeping Error: group ID and filename is not unique.'
        refGroup = self.glmGroups[refGroupIndex[0]]
 
        useTheseGroups = list(np.nonzero(np.logical_and(np.in1d(self.groupIdList, groupIds), np.in1d(self.filenameListIndex, groupSourceFilesIndex)))[0])
        assert len(useTheseGroups) == len(groupIds), 'Bookkeeping Error: Could not find all groupIds in self.glmGroups.'
        groups = [self.glmGroups[o] for o in useTheseGroups]
 
        timeDiffs   = np.array([(o.time - refGroup.time).total_seconds() for o in groups])
        latList     = np.array([o.latitudeDegreesNorth for o in groups])
        lonList     = np.array([o.longitudeDegreesEast for o in groups])
 
        dist_km = geoUtil.DistanceFromLatLonPoints(refGroup.latitudeDegreesNorth, refGroup.longitudeDegreesEast, latList, lonList)
 
        return np.logical_and(dist_km < self.closeness_km, timeDiffs < self.closeness_seconds)

    #*************************************************************************************************************
    def outlier_rejection (self):
        """ Examines energy and lat/lon ground travel time series and remove outliers

        Only remove outliers from clusters with a minimum number of groups
        
        """
        
        for idx, cluster in enumerate(self.clusteredGroups):


            glmGroupsThisCluster = self.get_groups_this_cluster(idx)
        
            energyArray = [g.energyJoules for g in glmGroupsThisCluster]
           #timeArray   = [g.time for g in glmGroupsThisCluster]
            latArray    = [g.latitudeDegreesNorth for g in glmGroupsThisCluster]
            lonArray    = [g.longitudeDegreesEast for g in glmGroupsThisCluster]
        
            # Do not perform outlier rejection to those clusters with fewer than minumum number of groups
            if len(energyArray) < self.min_num_groups_for_outliers:
                continue
        
            datumsToRemove = _sigma_clipping(energyArray, sigmaThreshold=self.outlierSigmaThreshold)
            datumsToRemove = np.union1d(datumsToRemove, _sigma_clipping(latArray, sigmaThreshold=self.outlierSigmaThreshold,
                minClipValue=self.outlierMinClipValueDegrees))
            datumsToRemove = np.union1d(datumsToRemove, _sigma_clipping(lonArray, sigmaThreshold=self.outlierSigmaThreshold,
                minClipValue=self.outlierMinClipValueDegrees))

            # Remove outliers
            self.remove_groups_from_cluster(idx, datumsToRemove)

            pass

    #*************************************************************************************************************
    def create_cluster_within_box(self, minLat, maxLat, minLon, maxLon, minTime, maxTime):
        """ Creates a cluster of all groups within a given box

            TODO: change code to use spatiotemporal_box instead of this function.

            Appends cluster to end of self.clusteredGroups

            If you want no limit then set the range value to None.
            So, if you want to extract all groups and place into a single cluster then set all ranges to None.

            Parameters
            ----------
            minLat  : [float] in degrees
            maxLat  : [float] in degrees
            minLon  : [float] in degrees
            maxLon  : [float] in degrees
            minTime : [datetime] 
            maxTime : [datetime]

            Returns
            -------
            clusterFound : [bool] True if a cluster was found. False means no groups found within the box
    
        """
        clusterFound = False

        # Set to max range if set to None on input
        if minLat is None:
            minLat = -90.0
        if maxLat is None:
            maxLat = 90.0
        if minLon is None:
            minLon = -180.0
        if maxLon is None:
            maxLon = 180.0
        if minTime is None:
            minTime = datetime.datetime(year=datetime.MINYEAR, month=1, day=1)
        if maxTime is None:
            maxTime = datetime.datetime(year=datetime.MAXYEAR, month=12, day=31)

        # Strip path from filenames
        netCDFFilenameList = [os.path.basename(f) for f in self.netCDFFilenameList]

        groupIndices = []
        groupIds = []
        groupFilenamesIndex = []
        timeList = []
        # step through all groups and add those within the box
        for idx, group in enumerate(self.glmGroups):
            if ( group.latitudeDegreesNorth <= maxLat and
                 group.latitudeDegreesNorth >= minLat and
                 group.longitudeDegreesEast <= maxLon and
                 group.longitudeDegreesEast >= minLon and
                 group.time <= maxTime and 
                 group.time >= minTime):
                groupIndices.append(idx)
                groupIds.append(group.id)
                groupFilenamesIndex.append(netCDFFilenameList.index(group.datasetFile))
                timeList.append(group.time)

        # only generate a cluster if one or more groups were found within box
        if len(groupIndices) > 0:
            self.clusteredGroups.append({'groupIndices': groupIndices,
                                         'groupIds': groupIds,
                                         'groupFilenamesIndex': groupFilenamesIndex,
                                         'time': timeList})
            clusterFound = True

        return clusterFound
    
    #*************************************************************************************************************
    def cluster_spatiotemporal_boxes(self):
        """ Works through spatiotemporal_box list and creates a cluster for each

        Returns:
          self.clusteredGroups    -- [list nClusters] All the groups gathered by cluster
              'groupIndices'        : [list of ] the group indices in the glmGroups list
              'groupIds'            : [list] the groupIds in this cluster 
              'groupFilenamesIndex' : [list] the index to the source filename in self.netCDFFilenameList for this group
              'time'                : [list of datetime] the time for each group in this cluster 
              'streakIdx'           : [list of ints] The index in the spatiotemporal_box lists for this streak cluster
        """

        for idx, box in enumerate(self.spatiotemporal_box):

            glmGroupsKept, idxToKeep = bd.keep_groups_within_spatiotemporal_box(box, self.glmGroups)
            
            # Create a single cluster for all these groups
            if len(idxToKeep) > 0:
                filePathList = [os.path.basename(filename) for filename in self.netCDFFilenameList]
                self.clusteredGroups.append({   'groupIndices':   idxToKeep,
                                                'groupIds':       np.array([g.id for g in glmGroupsKept]),
                            'groupFilenamesIndex': np.array([filePathList.index(g.datasetFile) for g in glmGroupsKept]),
                                                'time': np.array([g.time for g in glmGroupsKept]),
                                                'streakIdx' : idx})

        return

# END class BolideClustering

#*************************************************************************************************************
def _sigma_clipping(timeSeries, sigmaThreshold=10.0, minClipValue=0.0):
    """ Performs sigma clipping

    First Savitisky-Golay filters then computes a median absolute deviation derived sigma then clips!

    Note: This method requires the groups to be sorted in time.

    Parameters
    ----------
    timeSeries      : [float np.array] The time series to sigma clip
    sigmaThreshold  : [float] sigma threshold factor to clip at
    minClipValue    : [float] Minimum value to clip at. Nothing is clipped below this value, no matter what
                            sigmaThreshold is set at.

    Returns
    -------
    datumsToRemove : [int no.array] Array of indices to remove.

    """

    # Median filter then find outliers
    # Set the kernel to 1/5th time series length
    kernelLength = len(timeSeries) // 5
    # Must be odd
    if (np.mod(kernelLength ,2) == 0):
        kernelLength += int(1)
   #timeSeriesFit = timeSeries - medfilt(timeSeries, kernel_size=kernelLength)
    timeSeriesFit = timeSeries - savgol_filter(timeSeries, window_length=kernelLength, 
                                polyorder=3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
    
    # Use a median absolute deviation derived standard deviation to find the threshold for outliers
    # sigma = 1.4826 * mad(x)
    sigma = 1.4826*np.median(np.abs(timeSeriesFit - np.median(timeSeriesFit)))
    
    sigmaClip = np.max([sigma*sigmaThreshold, minClipValue])

    return np.nonzero(np.abs(timeSeriesFit) > sigmaClip)[0]


@jit(nopython=True, parallel=True)
def _3D_distance_matrix(latArray, lonArray, timeArray, closeness_seconds, closeness_km, numba_threads=16):
    """ Compute a 3-D distance matrix as a combination of time, latitude and longitude

    distance = sqrt(haversineDist ** 2 + timeDist ** 2)

    This method is slow so pulling out of object so that we can use parallelizm Numba. Numba is very efficient at
    parallizing for-loops if you use the numba prange instead of Python's range. So, instead of vectoring the numpy
    operations, we here use for-loops using prange. This is faster than vectorising using numpy.

    Parameters
    ----------
    latArray : np.array
        Array of latitude in degrees
    lonArray : np.array
        Array of longitude in degrees
    timeArray : np.array
        Array of time in seconds from an arbitrary reference time
        (Such as the first time point in the array)
    closeness_seconds : float
        Normalization factor for time in seconds
    closeness_km : float
        Normalization factor for distance in km
    numba_threads : int
        Sets the number of parallel threads to use in numba jit parallelization
        0 => use all currently available threads as returned by get_num_threads

    Returns
    -------
    distanceMatric : square np.array matrix
        The normalized distance between any two points in the arrays

    """

    if numba_threads > 0:
        set_num_threads(numba_threads)

    distanceMatrix = np.zeros((len(timeArray), len(timeArray)))

    # Convert to radians and normalize
    lat = np.zeros(len(latArray))
    lon = np.zeros(len(lonArray))
    timeArrayNorm = np.zeros(len(timeArray))
    for i in prange(len(latArray)):
        lat[i] = latArray[i] * geoUtil.deg2rad
        lon[i] = lonArray[i] * geoUtil.deg2rad
        timeArrayNorm[i] = timeArray[i] / closeness_seconds
    
    # This double forloop is parallelized in numba
    for i in prange(len(timeArray)):
        for j in prange(i + 1, len(timeArray)):  # Notice the range starts from i+1 to avoid repetition

            timeDiff = timeArrayNorm[i] - timeArrayNorm[j]

            # Compute haversine distance
            a = (np.sin((lat[j] - lat[i]) * 0.5)**2.0) + np.cos(lat[i]) * np.cos(lat[j]) * (np.sin((lon[j] - lon[i]) * 0.5)**2.0)
            c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
            distanceGeoNorm = geoUtil.faiEarthRadiusKm * c / closeness_km

            distance = np.sqrt(np.square(distanceGeoNorm) + np.square(timeDiff))

            distanceMatrix[i, j] = distance
            distanceMatrix[j, i] = distance  # Ensure symmetry

    return distanceMatrix


#*************************************************************************************************************
# This __main__ function will perform a unit test of the class definitions

if __name__ == "__main__":

    raise Exception('This function is not maintained')

    startTime = time.time()
    
    print('bolide_clustering.py: tools to cluster GLM data into bolide detections.')

    fileName = '/Users/bohr/data/ATAP/GLM_bolide_detector/sample_data/cuba_event/Data/Day032/OR_GLM-L2-LCFA_G16_s20190321816000_e20190321816200_c20190321816227.nc'

    BCObj = BolideClustering(fileName)

    success = BCObj.cluster_glm_groups()

    if not success:
        raise Exception("GLM clusterung was not successful")
        

    endTime = time.time()
    totalTime = endTime - startTime
    print("Total processing time {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))

    print("********")
    print('Try old clustering method...')

    startTime = time.time()
    [productTime, gid, glat, glon, genergy, group_time_offset, goesSatellite] = bsf.LoadBolideData(fileName)
    [groups_per_bolide, bolides] = bsf.ExtractGroupsFromEvents(productTime, gid, glat, glon, genergy, group_time_offset)
    endTime = time.time()
    totalTime = endTime - startTime
    print("Total processing time {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))

    print("********")
    # Compare old to new clusterBolideDetections
    clusterNumDiff = abs(len(BCObj.clusteredGroups) - len(bolides))
    if (clusterNumDiff != 0):
        #raise Exception("Old bolides clusters and new clusters arrays are not the same length")
        print("Old bolides clusters and new clusters arrays are not the same length")
        print('Old Number of clusters = {}; New number of clusters = {}'.format(len(bolides), len(BCObj.clusteredGroups)))
    # There might be a vectorized way to do this but I'll start with sequential for-loops
    for iCluster, cluster in enumerate(BCObj.clusteredGroups):
        groupIds = cluster['groupIds']
        # Find a group in bolides that shares these group Ids
        foundGroup = False
        for iGroup, group in enumerate(bolides):
            groupArray = np.array([g.id for g in group.group])
            foundHere= np.in1d(groupArray, groupIds, assume_unique=False)
            if (np.any(foundHere)):
                if (not np.all(foundHere)):
                    # One but not all groups were found in this cluster/bolide. This is bad
                    #raise Exception("A group was found to not be consistent between the two methods")
                    print("A group was found to not be consistent between the two methods")
                else:
                    # Found the group.
                    # Remove this grouping from bolides
                    del bolides[iGroup]
                    # No need to continue searching
                    foundGroup = True
                    break

        if (not foundGroup):
           #raise Exception("This cluster was not found in any bolide groupings")
            print("This cluster was not found in any bolide groupings")
                    

    pass



