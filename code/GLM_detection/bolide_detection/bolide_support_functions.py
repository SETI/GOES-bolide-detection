#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 16:51:32 2018

@author: crumpf
DrCRumpf@gmail.com
"""

# BOLIDE SUPPORT FUNCTIONS
import datetime
import matplotlib.pyplot as plt
import numpy as np
import bolide_filter_functions as bff
import matplotlib.ticker as mtick
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import netCDF4
import geometry_utilities as geoUtil


closeness_seconds = 0.200
closeness_km = 7.0
closeness_lat_deg = 0.05
closeness_lon_deg = 0.05
recording_period = 0.002


class Group:
    id = 0
    lastTime = datetime.datetime(2000,1,1)
    lastLat  = -100.0
    lasLon   = -500.0
    energy = -1000.0
  
    def __init__( self, id, lat, lon, dtime, energy ):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.dtime = dtime
        self.energy = energy



class Bolide:
    group = []
    counter = 1

    def __init__( self, group):
        self.group = [ group ]

        # Initialize other instance attributes to default values.
        self.filePathStr = ''
        self.score = ''
        self.paramDict = {}

    def close( self, group ):
        last = self.group[-1]
        diff = group.dtime - last.dtime
        dist_km = geoUtil.DistanceFromLatLonPoints(last.lat, last.lon, group.lat, group.lon)
        return ( dist_km < closeness_km and
                 diff.total_seconds() < closeness_seconds )
        
    # Vectorized version of the close determination method above
    def vectorized_close( self, groups ):
        last = self.group[-1]
        # Need a vector of time diffs and lat/long (which looks to not be vectorizable, but I didn't try that hard!)
        nGroups = len(groups)
        timeDiffs = np.zeros(nGroups)
        latList = np.zeros(nGroups)
        lonList = np.zeros(nGroups)
        for i in range(nGroups):
            timeDiffs[i] = (groups[i].dtime - last.dtime).total_seconds()
            latList[i]   = groups[i].lat
            lonList[i]   = groups[i].lon
        dist_km = geoUtil.DistanceFromLatLonPoints(last.lat, last.lon, latList, lonList)
        return np.logical_and(dist_km < closeness_km, timeDiffs < closeness_seconds)
        
    def old( self, group ):
        diff = group.dtime - self.group[-1].dtime
        return diff.total_seconds() > 1.0
    
    def add( self, group ):
        self.group.append(group)
    
    def vector_add( self, groups ):
        for g in groups:
            self.group.append(g)
    
    def condenseEnergiesInTime(self):
        group_list = self.group
        last_group = group_list[0]
        condensed_group_list = []
        condensed = False
        for group in group_list[1:]:
            diff_dtime = group.dtime - last_group.dtime
            diff_secs = diff_dtime.total_seconds()
            if abs(diff_secs) < recording_period:
                # maybe average lat/lons???
                condensed = True
                condensed_group = Group(group.id, group.lat, group.lon, last_group.dtime, group.energy + last_group.energy)
                last_group = condensed_group
            else:
                if condensed:
                    condensed_group_list.append(condensed_group)
                    last_group = condensed_group
                    condensed = False
                else:
                    condensed_group_list.append(group)
                    last_group = group

        return condensed_group_list






def LoadBolideData(file_path):
    ## load data
    raise Exception('This function is not used or maintained')
    f = netCDF4.Dataset(file_path)
    epoch = datetime.datetime(2000,1,1,12)
    productTime = epoch + datetime.timedelta( seconds=int(f.variables['product_time'][0].data) )
    
    
    gid     = f.variables['group_id'][:].data
    glat    = f.variables['group_lat'][:].data
    glon    = f.variables['group_lon'][:].data
    genergy = f.variables['group_energy'][:].data
    goesSatellite = f.platform_ID
    group_time_offset = f.variables['group_time_offset'][:].data
    
    f.close()
    
    return [productTime, gid, glat, glon, genergy, group_time_offset, goesSatellite]

#************************************************************************************
# ExtractGroupsFromEvents(productTime, gid, glat, glon, genergy, group_time_offset):
#
# Generates Bolide detection clusters of GLM "groups."
#
# A two step cliustering procedure is utilized:
#   1) A standard SciPy Hierachical Clustering method is used with a 'single' linkage method is first used to generate
#       groups.
#   2) Each groups is passed through a vectorized version of the sequential clustering method from Clemens original
#       prototype to make sugroups
#
# This two step process well replicates the original method ut at greatly increased speed.
#
#************************************************************************************

def ExtractGroupsFromEvents(productTime, gid, glat, glon, genergy, group_time_offset):
    groups = []

    if productTime > datetime.datetime(2018, 10, 15, 16, 0): # nc file format changed and group_time_offset is in fractions of second
        for i in range( len(genergy) ):
            dtime = productTime + datetime.timedelta( milliseconds=int(group_time_offset[i]*1000) )
            groups.append( Group(gid[i],glat[i],glon[i],dtime,genergy[i] ) )
    else: # nc file old format and group_time_offset is in milliseconds
        for i in range( len(genergy) ):
            dtime = productTime + datetime.timedelta( milliseconds=int(group_time_offset[i]) )
            groups.append( Group(gid[i],glat[i],glon[i],dtime,genergy[i] ) )

    # The hierarchical clustering "Linkage" function bogs down if the number of groups become too large. So, skip all
    # files when the number of groups is larger than MAX_N_GROUPS
    MAX_N_GROUPS = 30000
    if len(groups) > MAX_N_GROUPS:
        groups_per_bolide = len(groups) * np.ones(1)
        bolides = []
        return [groups_per_bolide, bolides]

    # Make sure groups are time ordered
    groups.sort(key=lambda x: x.dtime)
    
   ##*******************************
   ## Sequential clustering (SLOW)
   ## This is the original, unaltered method from the prototype, preserved here for reference.
   ### check groups for consistency with bolides and place close groups in groups
   #last_g = groups[0]
   #bolides = [ ]
   #found = False
   #
   #counter = 0
   #for g in groups[1:]: # go through all groups in file
   #   #print(len(groups)-counter)
   #    counter += 1
   #    if not found: # this group has not been assigned to a bolide yet
   #        bolides.append(Bolide(last_g)) # start a new bolide instance
   #    for bolide in bolides: # go through all existing bolide instances and compare the current group if it belongs (found) to one of those bolides
   #        #print(bolide.counter)
   #        ### groups are spatially and temporally close
   #        if bolide.close(g) and not found: # is the group close to the current bolide and has not been assigned to any other bolide?
   #            bolide.add(g)
   #            found = True
   #            bolide.counter += 1
   #            break
   #        else: # this group does not belong to any existing bolide and can potentially become a new bolide sequence.
   #            last_g = g
   #            found = False

    #*******************************
    #*******************************
    #*******************************
    # Hierarchical clustering
    bolides = hierarchical_clustering(groups)

    #*******************************
    # Run new group clusters through original clustering algorithm
    bolidesSave = bolides

    for i in range(len(bolidesSave)):
        bolide = bolidesSave[i]
        bolidesTemp = [ ]
        # Make sure groups are time ordered
        bolide.group.sort(key=lambda x: x.dtime)

        #*******************************
        # Vectorized Sequential clustering (FAST)
        # Take each group and first try to attach it to attach it to the last cluster, if the new group is too far away
        # then start a new group. This method requires all groups to be chronologically ordered.
        groupsArray = np.array(bolide.group)
        
        while (len(groupsArray) > 0):
            # Create a new potential bolide with the next non-grouped group
            bolidesTemp.append(Bolide(groupsArray[0])) # start a new bolide instance
            groupsArray = np.delete(groupsArray, 0)
            # Find the groups that belong to this potential bolide
            # Iterate the grouping to better replicate the sequential clustering.
            stillAdding = True
            while stillAdding:
                inBolide = bolidesTemp[-1].vectorized_close(groupsArray)
                if (not np.any(inBolide)):
                    stillAdding = False
                else:
                    # Add the groups to this bolide
                    bolidesTemp[-1].vector_add(groupsArray[inBolide].tolist())
                    bolidesTemp[-1].counter = len(bolidesTemp[-1].group)
                    # Remove groups from list of all non-grouped groups
                    groupsArray = np.delete(groupsArray, np.nonzero(inBolide))
                    

        #*******************************
        # Replace old big bolide with new subdivided bolides in big bolides array
        bolides = bolides[1:]
        for bolide in bolidesTemp:
            bolides.append(bolide)
    

    #*******************************
    #*******************************
    #*******************************


    #*******************************
    groups_per_bolide = np.zeros(len(bolides))
    for i, b in enumerate(bolides):
        groups_per_bolide[i] = b.counter
    
    return [groups_per_bolide, bolides]


#*************************************************************************************************************
# Perform hierarchical clustering
#
# Using SciPy's package
#
# Inputs:
#   groups          -- [list] All groups sorted by time
#
# Outputs:
#   bolides         -- [list] All the groups grouped into bolide detections
#


def hierarchical_clustering(groups):

    bolides = [ ]
    groupsArray = np.array(groups)

    # Form the data matrix X with n samples and m features with X.shape := (n,m)
    # The m features are latitude, longitude and time
    # All three normalized
    # We need a time reference. Choose the first group time
    timeZero = groupsArray[0].dtime
    nGroups = len(groupsArray)
    timeArray    = np.zeros(nGroups)
    latArray     = np.zeros(nGroups)
    lonArray     = np.zeros(nGroups)
    for i in range(nGroups):
        timeArray[i] = (groupsArray[i].dtime - timeZero).total_seconds()
        latArray[i]  = groupsArray[i].lat
        lonArray[i]  = groupsArray[i].lon

    # Time closeness is set at 0.2 seconds, So scale time dimension such that 0.2 => 1.0
    timeArrayNorm = timeArray / closeness_seconds

    # Convert to absolute distance
    # closeness_km set the closeness scale
    distanceKm = geoUtil.DistanceFromLatLonPoints(latArray[0], lonArray[0], latArray, lonArray)
    distanceNorm = distanceKm / closeness_km
    X = np.array([timeArrayNorm.T, distanceNorm.T]).T


    # generate the Linkage Matrix
    # The old sequential clustering method adds on groups to the "bolide" in chronological order comparing each new group
    # to the distance to the previosuly added group. The best way to replicate this with hierarchical clustering is to
    # use the "single" distance metric
    Z = linkage(X,'single')

    #********
    # Create the clusters
    max_d = np.sqrt(2.0) # Set because we normalized our dimensions to 1.0
    clusters = fcluster(Z, max_d, criterion='distance')

    # Define the bolides
    bolides = [ ]
    for i in np.arange(1,np.max(clusters)):
        # First create each bolide using the first group in each cluster
        groupsThisBolide = np.nonzero(clusters==i)[0]
        firstGroup = groupsThisBolide[0]
        bolides.append(Bolide(groupsArray[firstGroup])) # start a new bolide instance

        # Then add the rest of the groups to this bolide
        bolides[-1].vector_add(groupsArray[groupsThisBolide[1:]].tolist())
        bolides[-1].counter = len(bolides[-1].group)

    #********
    #********
    #********
    # plotting
    #
    # This is for diagnostic purposes to examine the hierachical clustering
    if (False):
        # Plot the dendrogram
        plt.figure(figsize=(20,8))
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        dendrogram(
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
        plt.title('Hierarchical Clustering Time/Distance')
        plt.xlabel('Time [seconds]')
        plt.ylabel('Distance [km]')
        plt.scatter(timeArray, distanceKm, c=clusters, cmap='prism')  # plot points with cluster dependent colors
        plt.show()

       #input("Press the <ENTER> key to continue...")
    #********
    #********
    #********

    return bolides

#*************************************************************************************************************
def ExtractGroupsFromEventsOld(productTime, glat, glon, genergy, group_time_offset):
    groups = []

    for i in range( len(genergy) ):
        dtime = productTime + datetime.timedelta( milliseconds=int(group_time_offset[i]) )
        groups.append( Group(gid[i],glat[i],glon[i],dtime,genergy[i] ) )
    
    # Make sure groups are time ordered
    groups.sort(key=lambda x: x.dtime)
    
    
    ## check groups for consistency with bolides and place close groups in groups
    last_g = groups[0]
    bolides = [ ]
    
    found = False
    
    counter = 0
    for g in groups[1:]:
        #print(len(groups)-counter)
        counter += 1
        if not found:
            bolides.append(Bolide(last_g))
        for b in bolides:
            #print(b.counter)
            ### groups are spatially and temporally close
            if b.close(g) and not found:
                b.add(g)
                found = True
                b.counter += 1
                break
            else:
                last_g = g
                found = False
    
    
    groups_per_bolide = np.zeros(len(bolides))
    for i, b in enumerate(bolides):
        groups_per_bolide[i] = b.counter
    
    return [groups_per_bolide, bolides]



#    def persistence( self ):
#        longestRun = 0
#        run = 0
#        e0 = self.groups[0]
#        for e in self.groups[1:]:
#            diff = e.dtime - e0.dtime
#            if abs(diff.total_seconds()-recording_period) < 0.0001:
#                run += 1
#                longestRun = max(longestRun,run)
#            else:
#                run = 0
#            e0 = e
#        return longestRun



def PlotLatLonOfBolide(index, poly_coefficients, horizontal_flag, bolides):
    plt.figure()
    plt.axis('equal')
    plt.plot([g.lon for g in bolides[index].group], [g.lat for g in bolides[index].group], marker='.',linewidth=0.3, markersize=5)
    plt.title('Ground Path of Bolide'); plt.xlabel('Longitude [deg]'); plt.ylabel('Latitude [deg]')
    if horizontal_flag:
        plt.plot([g.lon for g in bolides[index].group], [np.polyval(poly_coefficients, g.lon) for g in bolides[index].group])
    else:
        plt.plot([np.polyval(poly_coefficients, g.lat) for g in bolides[index].group], [g.lat for g in bolides[index].group], )
    plt.legend(['bolide', 'line fit'])
    #fig.savefig('./Figures/lat_lon_bolide.png', dpi=300)
    
def PlotEnergyOfBolide(index, bolides):
    plt.figure()
    date_array = [e.dtime for e in bolides[index].group]
    plt.plot(date_array, [e.energy for e in bolides[index].group])
    #locs, labels = plt.xticks()
    #print(locs)
    #plt.xticks(locs, [labels])
    date_middle_index = int(len(date_array)/2)
    plt.xticks([date_array[date_middle_index]], [date_array[date_middle_index].strftime("%y-%m-%d %H:%M:%S")])
    plt.title('Energy Time History of Bolide'); plt.xlabel('Time [hh:mm:ss]'); plt.ylabel('Energy')
    #fig.savefig('./Figures/energy_time.png', dpi=300)
    
def PlotEnergyOfEventList(group_list):
    plt.figure()
    plt.plot([e.dtime for e in group_list], [e.energy for e in group_list])
    #fig.savefig('./Figures/condensed_energy_time.png', dpi=300)
    
def PlotLatLonOfEventList(group_list):
    plt.figure()
    plt.axis('equal')
    plt.plot([g.lon for g in group_list], [g.lat for g in group_list])
    
##%%
#poly_coefficients = line_fit_results[0]
#plt.figure()
#plt.axis('equal')
#plt.plot([g.lon for g in bolides[1].group], [g.lat for g in bolides[1].group])
#plt.title('Ground Path of Bolide'); plt.xlabel('Longitude [deg]'); plt.ylabel('Latitude [deg]')
#plt.plot([g.lon for g in bolides[1].group], [np.polyval(poly_coefficients, g.lon) for g in bolides[1].group])
#plt.legend(['bolide', 'line fit'])
#print(poly_coefficients)
#print(line_fit_results[1,0])
#
#plt.figure()
#plt.axis('equal')
#plt.title('Long range fit'); plt.xlabel('Longitude [deg]'); plt.ylabel('Latitude [deg]')
#plt.plot(np.arange(-100, 0, 0.1), [np.polyval(poly_coefficients, g) for g in np.arange(-100, 0, 0.1)])
    
    
    
def FindFurthestLatLonPoints(lat_array, lon_array):
    raise Exception("This function is no longer being maintained")
    most_west_lon = lon_array.min()
    most_east_lon = lon_array.max()
    most_south_lat = lat_array.min()
    most_north_lat = lat_array.max()
    west_ind = np.where(lon_array==most_west_lon)
    east_ind = np.where(lon_array==most_east_lon)
    north_ind = np.where(lat_array==most_north_lat)
    south_ind = np.where(lat_array==most_south_lat)
    
    north_south_dist = geoUtil.DistanceFromLatLonPoints(lat_array[north_ind[0][0]], lon_array[north_ind[0][0]], lat_array[south_ind[0][0]], lon_array[south_ind[0][0]])
    east_west_dist = geoUtil.DistanceFromLatLonPoints(lat_array[east_ind[0][0]], lon_array[east_ind[0][0]], lat_array[west_ind[0][0]], lon_array[west_ind[0][0]])
    
    if north_south_dist > east_west_dist:
        return [north_ind[0][0], south_ind[0][0], False]
    else:
        return [east_ind[0][0], west_ind[0][0], True]
    
    
    
    
    
def PlotBolide(bolides, bolide_index, save_flag = False, save_name = '', show_energy_ratio = False, show_line_fit = False):
    raise Exception("This function is no longer being maintained")
    date_array = [e.dtime for e in bolides[bolide_index].group]
    timestamps = [t.timestamp() for t in date_array]
    lon_array = np.array([g.lon for g in bolides[bolide_index].group])
    lat_array = np.array([g.lat for g in bolides[bolide_index].group])
    delta_t = date_array[-1] - date_array[0]
    energy_array = np.array([e.energy for e in bolides[bolide_index].group])
    [point0_ind, point1_ind, horizontal_flag] = FindFurthestLatLonPoints(lat_array, lon_array)
    dist = geoUtil.DistanceFromLatLonPoints(lat_array[point0_ind], lon_array[point0_ind], lat_array[point1_ind], lon_array[point1_ind])
    
    fig = plt.figure()
    
    ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
    
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_scientific(False)
    
    plt.plot(lon_array, lat_array,linewidth=0.3)
    plt.scatter(lon_array, lat_array, marker='o', s=(energy_array/energy_array.max())*20+5, c=timestamps, edgecolors='k', linewidth=0.2)
    #plt.annotate('Ground Track length: {:.2f} km'.format(dist), xy=(0.05, 0.9), xycoords='axes fraction')
    plt.annotate('{:.2f} km'.format(dist), xy=(0.05, 0.9), xycoords='axes fraction')
    plt.annotate('{:.2f} km/s'.format(dist/delta_t.total_seconds()), xy=(0.05, 0.8), xycoords='axes fraction')
    plt.annotate('(a)', xy=(0.95, 0.9), xycoords='axes fraction')
    
    if show_line_fit:
        lon_range = np.linspace(lon_array.min(), lon_array.max(), 100)
        lat_range = np.linspace(lat_array.min(), lat_array.max(), 100)
        if horizontal_flag:
            line_fit_results = np.polyfit([g.lon for g in bolides[bolide_index].group], [g.lat for g in bolides[bolide_index].group], 1,full=True)
            poly_coefficients = line_fit_results[0]
            plt.plot(lon_range, np.polyval(poly_coefficients, lon_range),'k--', linewidth=0.5)
        else:
            line_fit_results = np.polyfit([g.lat for g in bolides[bolide_index].group], [g.lon for g in bolides[bolide_index].group], 1,full=True)
            poly_coefficients = line_fit_results[0]
            plt.plot(np.polyval(poly_coefficients, lat_range),lat_range, 'k--', linewidth=0.5)
    
    plt.ylabel('Latitude [$\degree$]')
    plt.xlabel('Longitude [$\degree$]')
    plt.axis('equal')
    
    ax = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.plot(date_array, energy_array, linewidth=0.5)
    plt.scatter(date_array, energy_array, marker='o', s=25, c=timestamps, edgecolor='none')
    date_middle_index = int(len(date_array)/2)
    plt.xticks([date_array[date_middle_index]], [date_array[date_middle_index].strftime("%y-%m-%d %H:%M:%S")])
    #plt.annotate('Light curve duration T={:.3f}s'.format(delta_t.total_seconds()), xy=(0.1, 0.9), xycoords='axes fraction')
    plt.annotate('T={:.3f} s'.format(delta_t.total_seconds()), xy=(0.1, 0.9), xycoords='axes fraction')
    plt.annotate('(b)', xy=(0.95, 0.9), xycoords='axes fraction')
    plt.ylabel('Luminuous Energy [J]')
    
    ratio_string = ''
    if show_energy_ratio:
        ratio = bff.CalculateProfileBalanceRatio(bolides[bolide_index].group)
        ratio_index = int(len(bolides[bolide_index].group)*ratio)
        plt.axvline(bolides[bolide_index].group[ratio_index].dtime, color='k', ls='--', linewidth=0.7)
        #plt.text(bolides[bolide_index].group[ratio_index+2].dtime, energy_array.max()*0.6, '50/50 at {:.2f} T'.format(ratio))
        ratio_string = str(ratio)
    
    plt.tight_layout()
    
    #plt.show()
    #plt.pause(0.0001)
    if save_flag:
        if len(save_name)>0:
            fig.savefig('./signature-'+date_array[date_middle_index].strftime("%y-%m-%d--%H-%M-%S_")+ save_name + ratio_string + '.png', dpi=300)
        else:
            fig.savefig('./signature-'+date_array[date_middle_index].strftime("%y-%m-%d--%H-%M-%S")+ ratio_string +'.png', dpi=300)
            
    plt.close(fig)
            
    
def PlotBolideWLatLon(bolides, bolide_index, save_flag = False, save_name = '', show_energy_ratio = False, show_line_fit = False):
    raise Exception("This function is no longer being maintained")
    date_array = [e.dtime for e in bolides[bolide_index].group]
    timestamps = [t.timestamp() for t in date_array]
    lon_array = np.array([g.lon for g in bolides[bolide_index].group])
    lat_array = np.array([g.lat for g in bolides[bolide_index].group])
    delta_t = date_array[-1] - date_array[0]
    energy_array = np.array([e.energy for e in bolides[bolide_index].group])
    [point0_ind, point1_ind, horizontal_flag] = FindFurthestLatLonPoints(lat_array, lon_array)
    dist = geoUtil.DistanceFromLatLonPoints(lat_array[point0_ind], lon_array[point0_ind], lat_array[point1_ind], lon_array[point1_ind])
    
    fig = plt.figure()
    
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=2)
    
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_scientific(False)

    plt.plot(lon_array, lat_array,linewidth=0.3)
    plt.scatter(lon_array, lat_array, marker='o', s=(energy_array/energy_array.max())*20+5, c=timestamps, edgecolors='k', linewidth=0.2)
    #plt.annotate('Ground Track length: {:.2f} km'.format(dist), xy=(0.05, 0.9), xycoords='axes fraction')
    plt.annotate('{:.2f} km'.format(dist), xy=(0.05, 0.85), xycoords='axes fraction')
    plt.annotate('{:.2f} km/s'.format(dist/delta_t.total_seconds()), xy=(0.05, 0.7), xycoords='axes fraction')
    plt.annotate('(a)', xy=(0.95, 0.85), xycoords='axes fraction')
    
    if show_line_fit:
        lon_range = np.linspace(lon_array.min(), lon_array.max(), 100)
        lat_range = np.linspace(lat_array.min(), lat_array.max(), 100)
        if horizontal_flag:
            line_fit_results = np.polyfit([g.lon for g in bolides[bolide_index].group], [g.lat for g in bolides[bolide_index].group], 1,full=True)
            poly_coefficients = line_fit_results[0]
            plt.plot(lon_range, np.polyval(poly_coefficients, lon_range),'k--', linewidth=0.5)
        else:
            line_fit_results = np.polyfit([g.lat for g in bolides[bolide_index].group], [g.lon for g in bolides[bolide_index].group], 1,full=True)
            poly_coefficients = line_fit_results[0]
            plt.plot(np.polyval(poly_coefficients, lat_range),lat_range, 'k--', linewidth=0.5)
    
    plt.ylabel('Latitude [$\degree$]')
    plt.xlabel('Longitude [$\degree$]')
    plt.axis('equal')
    
    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=2)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.plot(date_array, energy_array, linewidth=0.5)
    plt.scatter(date_array, energy_array, marker='o', s=25, c=timestamps, edgecolor='none')
    date_middle_index = int(len(date_array)/2)
    plt.xticks([date_array[date_middle_index]], [date_array[date_middle_index].strftime("%y-%m-%d %H:%M:%S")])
    #plt.annotate('Light curve duration T={:.3f}s'.format(delta_t.total_seconds()), xy=(0.1, 0.9), xycoords='axes fraction')
    plt.annotate('T={:.3f} s'.format(delta_t.total_seconds()), xy=(0.1, 0.85), xycoords='axes fraction')
    plt.annotate('(b)', xy=(0.95, 0.85), xycoords='axes fraction')
    plt.ylabel('Luminuous Energy [J]')
    
    ratio_string = ''
    if show_energy_ratio:
        ratio = bff.CalculateProfileBalanceRatio(bolides[bolide_index].group)
        ratio_index = int(len(bolides[bolide_index].group)*ratio)
        plt.axvline(bolides[bolide_index].group[ratio_index].dtime, color='k', ls='--', linewidth=0.7)
        #plt.text(bolides[bolide_index].group[ratio_index+2].dtime, energy_array.max()*0.6, '50/50 at {:.2f} T'.format(ratio))
        ratio_string = str(ratio)
        
        
        
        
    ax3 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=1)
    ax3.get_yaxis().get_major_formatter().set_useOffset(False)
    ax3.get_yaxis().get_major_formatter().set_scientific(False)
    plt.plot(date_array, lon_array, linewidth=0.5)
    plt.scatter(date_array, lon_array, marker='+', s=5, c=timestamps, edgecolor='none')
    date_middle_index = int(len(date_array)/2)
    plt.xticks([date_array[date_middle_index]], [date_array[date_middle_index].strftime("%y-%m-%d %H:%M:%S")])
    plt.annotate('(c)', xy=(0.85, 0.85), xycoords='axes fraction')
    plt.ylabel('Longitude [$\degree$]')
      
    ax4 = plt.subplot2grid((3, 2), (2, 1), rowspan=1, colspan=1)
    ax4.get_yaxis().get_major_formatter().set_useOffset(False)
    ax4.get_yaxis().get_major_formatter().set_scientific(False)
    plt.plot(date_array, lat_array, linewidth=0.5)
    plt.scatter(date_array, lat_array, marker='+', s=5, c=timestamps, edgecolor='none')
    date_middle_index = int(len(date_array)/2)
    plt.xticks([date_array[date_middle_index]], [date_array[date_middle_index].strftime("%y-%m-%d %H:%M:%S")])
    plt.annotate('(d)', xy=(0.85, 0.85), xycoords='axes fraction')
    plt.ylabel('Latitude [$\degree$]')
    

    plt.tight_layout()
    
    #plt.show()
    #plt.pause(0.0001)
    if save_flag:
        if len(save_name)>0:
            filename = './signature-'+date_array[date_middle_index].strftime("%y-%m-%d--%H-%M-%S_")+ save_name + ratio_string + '.png'
        else:
            filename = './signature-'+date_array[date_middle_index].strftime("%y-%m-%d--%H-%M-%S")+ ratio_string +'.png'
            
        fig.savefig(filename, dpi=300)
    else:
        filename = ''
        
    plt.close(fig)

    return filename
