#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 07:06:38 2018

@author: crumpf
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scipyInterp
import datetime
import time
#from astropy.time import Time
from tqdm import tqdm
import multiprocessing as mp
import os

import bolide_dispositions as bDisp
import bolide_support_functions as bsf
import geometry_utilities as geoUtil

plt.close('all')

plotting_flag = False

verbosity = False

#*************************************************************************************************************
# Elemental filter functions
#

def MaxDistanceKMFilter(max_dist_km, dist_km=1.0):
    """
    Parameters
    ----------
    max_dist_km : float
        Maximum distance between fitted line and group.
    
    Returns
    -------
    bolide_probability : float
        probability that we have a bolide based on the spline fit residual
        
    Description
    -----------
    
    """

    bolide_probability = 1. - (1. / (1. + np.exp(-(max_dist_km-dist_km)*8)))
    if bolide_probability > 1.0:
        bolide_probability = 1.0
    elif bolide_probability < 0.0:
        bolide_probability = 0.0
     
    return bolide_probability


if plotting_flag:
    dist_array = np.arange(0, 4+0.01, 0.01)
    plt.figure()
    plt.plot(dist_array, [MaxDistanceKMFilter(dist) for dist in dist_array])
    plt.title("Maximum km distance"); plt.xlabel("distance [km]"); plt.ylabel("P")
    plt.grid()





def MedianLineletLatLonFilter(root_sqared_sum_median_linelet_residual, linelet_05=-5.0):
    """
    Parameters
    ----------
    root_sqared_sum_median_linelet_residual : float
        The root of the sum of the squared median residuals in lat and lon.
    
    Returns
    -------
    bolide_probability : float
        probability that we have a bolide based on the spline fit residual
        
    Description
    -----------
    
    """

    neg_res_log = np.log10(root_sqared_sum_median_linelet_residual)
    bolide_probability = 1. - (1. / (1. + np.exp(-(neg_res_log-linelet_05)*3)))
    if bolide_probability > 1.0:
        bolide_probability = 1.0
    elif bolide_probability < 0.0:
        bolide_probability = 0.0
     
    return bolide_probability


if plotting_flag:
    residual_log_array = np.arange(-8, 0+0.01, 0.01)
    residual_array = 10.0**residual_log_array
    plt.figure()
    plt.semilogx(residual_array, [MedianLineletLatLonFilter(res) for res in residual_array])
    plt.title("Median Linelet Filter"); plt.xlabel("Log Residual"); plt.ylabel("P")
    plt.grid()



def TimeDurationFilter(seconds, time_05=6.0):
    """
    Parameters
    ----------
    seconds : float
        duration of the energy signature in seconds [s]
    time_05 : float
        maximum pulse duration in seconds [s]
    
    Returns
    -------
    bolide_probability : float
        probability that we have a bolide based on energy signature duration
        
    Description
    -----------
    
    """

    bolide_probability = 1. - (1. / (1. + np.exp(-(seconds-time_05)*2)))
    if bolide_probability > 1.0:
        bolide_probability = 1.0
    elif bolide_probability < 0.0:
        bolide_probability = 0.0
     
    return bolide_probability



if plotting_flag:
    time_array = np.arange(0,15,0.01)
    plt.figure()
    plt.plot(time_array, [TimeDurationFilter(t) for t in time_array])
    plt.title("Time Duration Filter"); plt.xlabel("time [s]"); plt.ylabel("P")
    plt.grid()





def LineFitResidualFilter(residual, line_05=-5.0):
    """
    Parameters
    ----------
    residual : float
        line fit residual (normalized by number of considered data points)
    
    Returns
    -------
    bolide_probability : float
        probability that we have a bolide based on line fit
        
    Description
    -----------
    
    """
    neg_res_log = np.log10(residual)
    #    if press <= 150000.0:
    #        vuln_press = 0.0
    #    elif press >= 900000.0:
    #        vuln_press = 1.0
    #    else:
    bolide_probability = 1. - (1. / (1. + np.exp(-(neg_res_log-(line_05))*3)))
    if bolide_probability > 1.0:
        bolide_probability = 1.0
    elif bolide_probability < 0.0:
        bolide_probability = 0.0
     
    return bolide_probability



if plotting_flag:
    residual_log_array = np.arange(-8, 0+0.01, 0.01)
    residual_array = 10.0**residual_log_array
    plt.figure()
    plt.semilogx(residual_array, [LineFitResidualFilter(res) for res in residual_array])
    plt.title("Line Fit Residual Filter"); plt.xlabel("Residual"); plt.ylabel("P")
    plt.grid()







def GroupCountFilter(group_count, group_05=25.0):
    """
    Parameters
    ----------
    group_count : float
        group count that constitutes the bolide detection
    
    Returns
    -------
    bolide_probability : float
        probability that we have a bolide based on group count
        
    Description
    -----------
    
    """

    #50, #150, 0.07 # 25, 0.03; 30, 0.07 picks up small events but also lots of false positives
    bolide_probability = 1. / (1. + np.exp(-(group_count-group_05)*0.07))
    if bolide_probability > 1.0:
        bolide_probability = 1.0
    elif bolide_probability < 0.0:
        bolide_probability = 0.0
     
    return bolide_probability



if plotting_flag:
    group_count_array = np.arange(0,400,0.1)
    plt.figure()
    plt.plot(group_count_array, [GroupCountFilter(res) for res in group_count_array])
    plt.title("Group Count Filter"); plt.xlabel("Group Count"); plt.ylabel("P")
    plt.ylim(-0.1,1.1)
    plt.grid()






def EnergyBalanceFilter(profile_balance_ratio, energy_05=0.3):
    """
    Parameters
    ----------
    profile_balance_ratio : float
        The index ratio for which the energy profile has equal deposited
        energy before and after.
    
    Returns
    -------
    bolide_probability : float
        probability that we have a bolide based on the energy deposition profile
        
    Description
    -----------
    
    """

    bolide_probability = 1. / (1. + np.exp(-(profile_balance_ratio-energy_05)*25)) #0.45, 12
    if bolide_probability > 1.0:
        bolide_probability = 1.0
    elif bolide_probability < 0.0:
        bolide_probability = 0.0
     
    return bolide_probability


if plotting_flag:
    energy_balance_array = np.arange(0,1,0.01)
    plt.figure()
    plt.plot(energy_balance_array, [EnergyBalanceFilter(res) for res in energy_balance_array])
    plt.title("Energy Balance Filter"); plt.xlabel("Energy Balance Ratio"); plt.ylabel("P")
    plt.grid()



def MaxLineDistanceFilter(max_line_dist):
    """
    Parameters
    ----------
    max_line_dist : float
        Maximum event distance from line fit to group
    
    Returns
    -------
    bolide_probability : float
        probability that we have a bolide based on the maximum line speration distance
        
    Description
    -----------
    
    """

    bolide_probability = 1.0 - (1. / (1. + np.exp(-(max_line_dist-0.005)*3000)))
    if bolide_probability > 1.0:
        bolide_probability = 1.0
    elif bolide_probability < 0.0:
        bolide_probability = 0.0
     
    return bolide_probability


if plotting_flag:
    event_dist_array = np.arange(0,0.01,0.0001)
    plt.figure()
    plt.plot(event_dist_array, [MaxLineDistanceFilter(res) for res in event_dist_array])
    plt.title("Line Distance Filter"); plt.xlabel("Distance [deg]"); plt.ylabel("P")
    plt.grid()
    
    
    
    
def MaxLineDistanceRatioFilter(max_line_dist_ratio, distance_ratio_05=0.4):
    """
    Parameters
    ----------
    max_line_dist_ratio : float
        Maximum event distance from line fit to group normalized by the lat/lon range in the group.
    
    Returns
    -------
    bolide_probability : float
        probability that we have a bolide based on the maximum line speration distance
        
    Description
    -----------
    
    """

    bolide_probability = 1.0 - (1. / (1. + np.exp(-(max_line_dist_ratio-distance_ratio_05)*80))) #0.12
    if bolide_probability > 1.0:
        bolide_probability = 1.0
    elif bolide_probability < 0.0:
        bolide_probability = 0.0
     
    return bolide_probability


if plotting_flag:
    event_dist_array = np.arange(0,1,0.001)
    plt.figure()
    plt.plot(event_dist_array, [MaxLineDistanceRatioFilter(res) for res in event_dist_array])
    plt.title("Line Distance Ratio Filter"); plt.xlabel("Distance Ratio [-]"); plt.ylabel("P")
    plt.grid()





def MeanSplineResidualFilter(mean_spline_residual_squared, spline_05=-2.0):
    """
    Parameters
    ----------
    mean_spline_residual_squared : float
        The average residual in the spline fit filter.
    
    Returns
    -------
    bolide_probability : float
        probability that we have a bolide based on the spline fit residual
        
    Description
    -----------
    
    """

    if (mean_spline_residual_squared <= 0.0):
        neg_res_log = -np.inf
    else:
        neg_res_log = np.log10(mean_spline_residual_squared)
    bolide_probability = 1. - (1. / (1. + np.exp(-(neg_res_log-spline_05)*3)))
    if bolide_probability > 1.0:
        bolide_probability = 1.0
    elif bolide_probability < 0.0:
        bolide_probability = 0.0
     
    return bolide_probability


if plotting_flag:
    residual_log_array = np.arange(-8, 0+0.01, 0.01)
    residual_array = 10.0**residual_log_array
    plt.figure()
    plt.semilogx(residual_array, [MeanSplineResidualFilter(res) for res in residual_array])
    plt.title("Mean Spline Residual Ratio Filter"); plt.xlabel("Residual Ratio Squared"); plt.ylabel("P")
    plt.grid()
    
#*************************************************************************************************************
# The main filter calls 
# To return the filter responses for all bolides, set bolide_probability_cutoff = -1
    
# This filter MUST be called first!
def FilterBolidesByGroupCount(bolides, bolide_probability_cutoff, group_05=25):
    bolide_index_list = []
    bolide_probability_list = []

    for index, b in enumerate(bolides):
        [counter_passed_flag, bolide_probability] = FilterDetectionsByEventCount(b, bolide_probability_cutoff, group_05)

        if bolide_probability > bolide_probability_cutoff:
            bolide_index_list.append(index)
            bolide_probability_list.append(bolide_probability)
            
    if (verbosity): print("{:d} detections passed group count filter".format(len(bolide_index_list)))

    return [bolide_index_list, bolide_probability_list]

    


def FilterBolidesByTime(bolides, bolide_index_list, cum_bolide_probability_list, bolide_probability_cutoff, time_05=6.0):
    temp_bolide_index_list = []
    temp_cum_bolide_probability_list = []
    
    for i, bolide_index in enumerate(bolide_index_list):
        date_array = [e.dtime for e in bolides[bolide_index].group]
        delta_t = date_array[-1] - date_array[0]
        time_probability = TimeDurationFilter(delta_t.total_seconds(), time_05)
        cum_bolide_probability = cum_bolide_probability_list[i] * time_probability

        if cum_bolide_probability > bolide_probability_cutoff:
            temp_bolide_index_list.append(bolide_index_list[i])
            temp_cum_bolide_probability_list.append(cum_bolide_probability)
            
    bolide_index_list = [x for x in temp_bolide_index_list]
    cum_bolide_probability_list = [x for x in temp_cum_bolide_probability_list]
                  
    
    if (verbosity): print("{:d} detections passed time filter".format(len(bolide_index_list)))
    
    return [bolide_index_list, cum_bolide_probability_list]



# median residual of a 5 data-point-wide window over which straight “linelets” are fit to lat and Lon time histories. 
def FilterBolidesByLatLonLinelets(bolides, bolide_index_list, cum_bolide_probability_list, bolide_probability_cutoff, linelet_05 = -5.0):
    temp_bolide_index_list = []
    temp_cum_bolide_probability_list = []
    
    for ind, bolide_index in enumerate(bolide_index_list):
        date_array = [e.dtime for e in bolides[bolide_index].group]
        timestamps = [t.timestamp() for t in date_array]
        lon_array = np.array([g.lon for g in bolides[bolide_index].group])
        lat_array = np.array([g.lat for g in bolides[bolide_index].group])
        
        window_width = 5
        n_windows = int(len(timestamps)/window_width)-1
        residuals_lon = np.zeros(n_windows)
        residuals_lat = np.zeros(n_windows)
        
        for i in range(n_windows):
            line_fit_results = np.polyfit(timestamps[i*window_width:(i+1)*window_width], lon_array[i*window_width:(i+1)*window_width], 1,full=True)
            residuals_lon[i] = line_fit_results[1][0]
            line_fit_results = np.polyfit(timestamps[i*window_width:(i+1)*window_width], lat_array[i*window_width:(i+1)*window_width], 1,full=True)
            residuals_lat[i] = line_fit_results[1][0]
            
        median_root_sum_squared_residuals = np.sqrt(np.median(residuals_lon)**2.0 + np.median(residuals_lat)**2.0)
        
        linelet_probability = MedianLineletLatLonFilter(median_root_sum_squared_residuals, linelet_05)
        temp_cum_probability = linelet_probability * cum_bolide_probability_list[ind]
        
        if temp_cum_probability > bolide_probability_cutoff:
            temp_cum_bolide_probability_list.append(temp_cum_probability)
            temp_bolide_index_list.append(bolide_index_list[ind])
            
    if len(temp_cum_bolide_probability_list) > 0:
        bolide_index_list = [x for x in temp_bolide_index_list]
        cum_bolide_probability_list = [x for x in temp_cum_bolide_probability_list]
        
    if (verbosity): print("{:d} detections passed linelet filter".format(len(bolide_index_list)))

    return [bolide_index_list, cum_bolide_probability_list]




# Describes the energy balance of the light curve.
def FilterBolidesByEnergyRatio(bolides, bolide_index_list, cum_bolide_probability_list, bolide_probability_cutoff, energy_05=0.3):
    temp_bolide_index_list = []
    temp_cum_bolide_probability_list = []
    
    for i, cand_index in enumerate(bolide_index_list):
        profile_balance_ratio = CalculateProfileBalanceRatio(bolides[cand_index].group)
        energy_bolide_probability = EnergyBalanceFilter(profile_balance_ratio, energy_05)
        cum_bolide_probability = cum_bolide_probability_list[i] * energy_bolide_probability
        if cum_bolide_probability > bolide_probability_cutoff:
            temp_cum_bolide_probability_list.append(cum_bolide_probability)
            temp_bolide_index_list.append(bolide_index_list[i])
            
    bolide_index_list = [x for x in temp_bolide_index_list]
    cum_bolide_probability_list = [x for x in temp_cum_bolide_probability_list]          
            
    if (verbosity): print("{:d} detections passed energy profile filter".format(len(bolide_index_list)))
    
    return [bolide_index_list, cum_bolide_probability_list]



#%% Filter by fitting splinelets to a moving window over the energy deposition profile
def FilterBolidesBySplinelets(bolides, bolide_index_list, cum_bolide_probability_list, bolide_probability_cutoff, spline_05):
    
    temp_bolide_index_list = []
    temp_cum_bolide_probability_list = []
    
    for i, bolide_index in enumerate(bolide_index_list):
        [spline_probability, mean_spline_fit, spline_residual_array] = FilterBolidesByEnergySplineResidual(bolides[bolide_index], spline_05)
        cum_bolide_probability = cum_bolide_probability_list[i] * spline_probability
        if cum_bolide_probability > bolide_probability_cutoff:
            temp_bolide_index_list.append(bolide_index_list[i])
            temp_cum_bolide_probability_list.append(cum_bolide_probability)
            
    bolide_index_list = [x for x in temp_bolide_index_list]
    cum_bolide_probability_list = [x for x in temp_cum_bolide_probability_list]

    
    if (verbosity): print("{:d} detections passed splinelets filter".format(len(bolide_index_list)))

    return [bolide_index_list, cum_bolide_probability_list]

# Filter by deviations from a ground track line fit
def FilterBolidesByGroundTrackDeviation(bolides, bolide_index_list, cum_bolide_probability_list, bolide_probability_cutoff, dist_km_05):

    temp_bolide_index_list = []
    temp_cum_bolide_probability_list = []

    horizontal_flag = False
    
    for i, (index, b) in enumerate(zip(bolide_index_list, bolides)):
        # check if line is more horizontal or vertical
        
        if abs(bolides[index].group[0].lon - bolides[index].group[-1].lon) > abs(bolides[index].group[0].lat - bolides[index].group[-1].lat):
            line_fit_results = np.polyfit([g.lon for g in bolides[index].group], [g.lat for g in bolides[index].group], 1,full=True)
            horizontal_flag = True
        else:
            line_fit_results = np.polyfit([g.lat for g in bolides[index].group], [g.lon for g in bolides[index].group], 1,full=True)
            horizontal_flag = False
            
        if len(line_fit_results[0]>0):
            poly_coefficient = line_fit_results[0]
            
            lat_array = np.array([event.lat for event in bolides[index].group])
            lon_array = np.array([event.lon for event in bolides[index].group])
            
            if horizontal_flag:
                predicted_lats_deg = np.polyval(poly_coefficient, lon_array)
                dists_km = geoUtil.DistanceFromLatLonPoints(predicted_lats_deg, lon_array, lat_array, lon_array)
                
            else:
                predicted_lons_deg = np.polyval(poly_coefficient, lat_array)
                dists_km = geoUtil.DistanceFromLatLonPoints(lat_array, predicted_lons_deg, lat_array, lon_array)
            
            max_dist_km = dists_km.max()
            # Calculate maximum distance ignoring outliers
            # Use a median absolute deviation derived standard deviation to find the threshold for outliers
            # sigma = 1.4826 * mad(x)
          # sigma = 1.4826*np.median(np.abs(dists_km - np.median(dists_km)))
          # sigmaThreshold = 5.0;
          # max_dist_km = dists_km[dists_km <= (sigma*sigmaThreshold)].max()
            
          # if (verbosity): print("Max dist in km is {0:.3f}".format(max_dist_km))
            
            distance_bolide_probability = MaxDistanceKMFilter(max_dist_km, dist_km_05)
                    
            cum_bolide_probability = cum_bolide_probability_list[i] * distance_bolide_probability
            if cum_bolide_probability > bolide_probability_cutoff:
                temp_bolide_index_list.append(bolide_index_list[i])
                temp_cum_bolide_probability_list.append(cum_bolide_probability)
                
    bolide_index_list = [x for x in temp_bolide_index_list]
    cum_bolide_probability_list = [x for x in temp_cum_bolide_probability_list]
    
    if (verbosity): print("{:d} detections passed event distance filter".format(len(bolide_index_list)))
    
    return [bolide_index_list, cum_bolide_probability_list]

#*************************************************************************************************************
# Other helper functions

def FilterBolidesByEnergySplineResidual(bolide, spline_05=-2.0):
    energies = np.array([e.energy for e in bolide.group])
    window_width = np.min([5, len(energies)])
    arrayLength = np.max([len(energies)-window_width,1])
    
    if (arrayLength <= 3):
        # This metric cannot be computed
        return [0.0, np.inf, np.zeros(arrayLength)]

    spline_residual_array = np.zeros(arrayLength)
    spline_counter = 0
    energy_range_inverse = 1.0 / energies.max() - energies.min()
    energy_range_inverse_sqred = energy_range_inverse * energy_range_inverse
    spline_residual_relative_array = np.zeros(arrayLength)
    for window_start in range(arrayLength):
        window_end = window_start + window_width
        energy_window = energies[window_start:window_end]
        index_window = np.arange(window_start,window_end)
        spline = scipyInterp.UnivariateSpline(index_window, energy_window, k=3)
        spline_residual_array[spline_counter] = spline.get_residual()
        spline_residual_relative_array[spline_counter] = spline.get_residual() * energy_range_inverse_sqred
        spline_counter += 1
    
    mean_spline_fit = spline_residual_relative_array.mean()
    spline_probability = MeanSplineResidualFilter(mean_spline_fit, spline_05)
    return [spline_probability, mean_spline_fit, spline_residual_relative_array]




def FilterBolidesByLine(bolides, cum_bolide_index_list, cum_bolide_probability_list, bolide_probability_cutoff, line_05=-5.0):
    bolide_index_list = []
    poly_coefficients_list = []
    horizontal_flag_output_list = []
    bolide_probability_list = []
    
    best_line_fit = 100 # arbitrary high number
    horizontal_flag = False
    temp_line_fit_probability = 0.0
    bolide_probability = 0.0
    for i, (index, b) in enumerate(zip(cum_bolide_index_list, bolides)):
        # check if line is more horizontal or vertical
        if abs(bolides[index].group[0].lon - bolides[index].group[-1].lon) > abs(bolides[index].group[0].lat - bolides[index].group[-1].lat):
            line_fit_results = np.polyfit([g.lon for g in bolides[index].group], [g.lat for g in bolides[index].group], 1,full=True)
            horizontal_flag = True
        else:
            line_fit_results = np.polyfit([g.lat for g in bolides[index].group], [g.lon for g in bolides[index].group], 1,full=True)
            horizontal_flag = False
        if line_fit_results[1].size>0:
            residual = line_fit_results[1][0]
                
            normalized_residual = residual / float(len(bolides[index].group))
            #if (verbosity): print(normalized_residual)
            temp_line_fit_probability = LineFitResidualFilter(normalized_residual, line_05)
            bolide_probability = temp_line_fit_probability * cum_bolide_probability_list[i]
            #print(temp_bolide_probability)
            if bolide_probability > bolide_probability_cutoff:
                bolide_index_list.append(index)
                bolide_probability_list.append(bolide_probability)
                poly_coefficients_list.append(line_fit_results[0])
                horizontal_flag_output_list.append(horizontal_flag)

                    
    if len(bolide_index_list) > 0:
#       if (verbosity): print("PASSED group count filter with count of {:d} and P = {:.2f}".format(bolides[bolide_index].counter, group_count_probability))
#       if (verbosity): print("PASSED line fitting with residual of {:.1E} and P = {:.2f}".format(best_line_fit, line_fit_bolide_probability))
        if (verbosity): print("{:d} bolide detections passed line fitting.".format(len(bolide_index_list)))
        return [bolide_index_list, poly_coefficients_list, horizontal_flag_output_list, bolide_probability_list]
    else:
        ##print(bolide_probability)
        if (verbosity): print("FAILED line fitting with residual of {:.1E} and P = {:.2f}".format(best_line_fit, bolide_probability))
        return []



def CalculateProfileBalanceRatio(groups):
    """
    Parameters
    ----------
    groups : float
        The groups that could be a bolide which is made up of individual groups with energy
    
    Returns
    -------
    profile_balance_ratio : float
        The index ratio along the profile where the energy deposition before and
        after this index is equal. In other words, the index at which the 
        energy deposition profile reaches 50% of the total energy deposited.
        
    Description
    -----------
    
    """
    # If there is only one group then fail this test
    if (len(groups) < 2):
        return 0.0

    # determine at which point we reach half energy level in the energy deposition time profile
    energies = np.array([event.energy for event in groups])
    total_energy = energies.sum()
    half_energy = 0.5 * total_energy
    #cum_energies = np.zeros(len(energies))
    cum_energy = 0.0
    profile_balance_ratio = 0.0
    for i, energy in enumerate(energies):
        cum_energy += energy
        if cum_energy > half_energy:
            #profile_balance_ratio = float(i) / (len(energies)-1)
            break
    
    total_time_range = groups[-1].dtime - groups[0].dtime
    half_energy_time_range = groups[i].dtime - groups[0].dtime
    if (total_time_range.total_seconds() == 0):
        return 0.0
    else:
        profile_balance_ratio = half_energy_time_range.total_seconds() / total_time_range.total_seconds()
    
    
    
    return profile_balance_ratio










def FilterBolidesByCurve(bolides, bolide_probability_cutoff):
    best_line_fit = 100 # arbitrary high number
    best_probability = -0.1
    horizontal_flag = False
    horizontal_flag_output = False
    group_count_probability = 0.0
    temp_line_fit_probability = 0.0
    line_fit_bolide_probability = 0.0
    for index, b in enumerate(bolides):
        [counter_passed_flag, group_count_probability_detection] = FilterDetectionsByEventCount(b, bolide_probability_cutoff)
        if counter_passed_flag: #consider only bolides with more than X groups
            # check if line is more horizontal or vertical
            if abs(bolides[index].group[0].lon - bolides[index].group[-1].lon) > abs(bolides[index].group[0].lat - bolides[index].group[-1].lat):
                line_fit_results = np.polyfit([g.lon for g in bolides[index].group], [g.lat for g in bolides[index].group], 1,full=True)
                horizontal_flag = True
            else:
                line_fit_results = np.polyfit([g.lat for g in bolides[index].group], [g.lon for g in bolides[index].group], 1,full=True)
                horizontal_flag = False
            if line_fit_results[1].size>0:
                residual = line_fit_results[1]
                temp_line_fit_probability = LineFitResidualFilter(residual[0])
                temp_bolide_probability = temp_line_fit_probability * group_count_probability_detection
                #if (verbosity): print(temp_bolide_probability)
                if temp_bolide_probability > best_probability:
                    bolide_index = index
                    best_line_fit = residual[0]
                    poly_coefficients = line_fit_results[0]
                    horizontal_flag_output = horizontal_flag
                    group_count_probability = group_count_probability_detection
                    line_fit_bolide_probability = temp_line_fit_probability
                    best_probability = temp_bolide_probability
                    
    #if (verbosity): print(best_line_fit)
    bolide_probability = line_fit_bolide_probability * group_count_probability
    if bolide_probability > bolide_probability_cutoff:
        if (verbosity): print("PASSED group count filter with count of {:d} and P = {:.2f}".format(bolides[bolide_index].counter, group_count_probability))
        if (verbosity): print("PASSED line fitting with residual of {:.1E} and P = {:.2f}".format(best_line_fit, line_fit_bolide_probability))
        return [bolide_index, poly_coefficients, horizontal_flag_output, bolide_probability]
    else:
        ##print(bolide_probability)
        if (verbosity): print("FAILED line fitting with residual of {:.1E} and P = {:.2f}".format(best_line_fit, bolide_probability))
        return [] 
    
    
def FilterDetectionsByEventCount(bolide, bolide_probability_cutoff, group_05):
    group_count_probability = GroupCountFilter(bolide.counter, group_05)
    if group_count_probability > bolide_probability_cutoff: 
        #if (verbosity): print("PASSED group count filter with count of {0} and P = {:.2f}".format(b.counter, group_count_probability))
        return [True, group_count_probability]
    else:
        #if (verbosity): print("FAILED group count filter with count of {0} and P = {:.2f}".format(b.counter, group_count_probability))
        return [False, group_count_probability]
    
    
#*************************************************************************************************************
# apply_glint_filter
#
# Applies the glint filter from glint_filter module. It will remove all groups that reside within the glint region.
#
# This function requires a GlintFilter object to be already instantiated.
#
# Inputs:
#   glintFilter     -- [GlintFilter class] Instantiated glint filter object 
#   goesSatellite   -- [str] which GOES satellite to lookup {'G16', 'G17', etc...}
#   productTime     -- [datetime] start time for data
#   glmGroups       -- [bd.GlmGroup] A list of objects containing the groups from the netCDF data file to filter
#
# Outputs:
#   glmGroups       -- [bd.GlmGroup] A FILTERED list of objects containing the groups
#
#*************************************************************************************************************
def apply_glint_filter(glintFilter, goesSatellite, productTime, glmGroups):
    
    # Compute Julian Date for each group
    julianDay = []

   #timeThisGroup = []
   #timeOffsetMsec = np.array([o.timeOffsetMsec for o in glmGroups])
   #for i in range( len(timeOffsetMsec) ):
   #    # Add in the offset to the product time to get the time for each event in datetime format
   #    timeThisGroup.append(productTime + datetime.timedelta( milliseconds=int(timeOffsetMsec[i])))

    timeThisGroup = [group.time for group in glmGroups]

    # Convert to astropy time object in julian date format
    raise Exception('Figure out why astropy.time module not working on NAS')
   #julianDay = Time(timeThisGroup).jd

    julianDay           = np.array(julianDay)
    latitudeDegrees     = np.array([o.latitudeDegreesNorth for o in glmGroups])
    longitudeDegrees    = np.array([o.longitudeDegreesEast for o in glmGroups])
    
    # Run on the test data
    withinGlintRegion = glintFilter.glint_filter(goesSatellite, julianDay, latitudeDegrees, longitudeDegrees)

    # Remove all groups within glint region
    notWithinGlintRegion = np.logical_not(withinGlintRegion)
    glmGroups = [glmGroups[i] for i in np.arange(len(glmGroups)) if notWithinGlintRegion[i]]

    if (verbosity): print("{:d} groups passed glint filter".format(np.sum(np.logical_not(withinGlintRegion))))

    return glmGroups

#*************************************************************************************************************
# The classes used below to return the filter repsonses for all bolide detections
class FilterResponses:
    def __init__ (self, nDetections):
        self.latLonLinelets         = np.full(nDetections, np.nan)
        self.energyRatio            = np.full(nDetections, np.nan)
        self.splinelets             = np.full(nDetections, np.nan)
        self.groundTrackDeviation   = np.full(nDetections, np.nan)

class FilterTuningParams:
    def __init__ (self):
        # Set to the default parameters used by the pipeline in 2019 and 2020
        self.linelet   = -5.0
        self.energy    = 0.3
        self.spline    = -2.0
        self.dist_km   = 1.0

#*************************************************************************************************************
# compute_all_filter_responses_on_all_triggers ()
#
# The prototype detector uses sequential filters: the latter filters are only computed if the earlier ones pass.
#
# For a ML classifier, we need all filter repsonses computed for all triggers. This function does just that.
#
# Also, the prototype automatically applies the logit function, here we want the full range feature values.
#
# Inputs:
#   bolideDetectionList -- [list of bolideDetection]
#   verbosity   -- [bool] If True then print processing status statements
#                   Computing features can be slow
#   multiProcessEnabled -- [bool] If true parallelize the filter computation
#   nJobs               -- [int] Number of parallel jobs to use if multiProcessEnabled==True
#                           If None then use os.cpu_count()
#
#
# Outputs:
#   filterResponses     -- [FilterResponses class]
#       .latLonLinelets      
#       .energyRatio         
#       .splinelets          
#       .groundTrackDeviation
#
#*************************************************************************************************************

def compute_all_filter_responses_on_all_triggers (bolideDetectionList, verbosity=False, multiProcessEnabled=False,
        nJobs=None):
    
    if nJobs is None:
        nJobs = os.cpu_count()

    #***
    # The prototype filters uses the "bolide_support_functions.bolide data class. We need to convert our
    # bolideDetectionList into the bolide data class

    # Form an old-style "bolide" class from the bolide detections
    # Define the bolides
    startTime = time.time()
    bolides = [ ]
    for detection in bolideDetectionList:
        groupsThisBolide = detection.groupList
        # Sort the groups by time (required by some of the filters)
        groupsThisBolide.sort(key=lambda x: x.time)

        # First create each bolide using the first group in each detection
       #dtime = detection.productTime + \
       #    datetime.timedelta(milliseconds=int(groupsThisBolide[0].timeOffsetMsec))
        dtime = groupsThisBolide[0].time


        firstGroupOldStyle = bsf.Group(groupsThisBolide[0].id, groupsThisBolide[0].latitudeDegreesNorth, 
                groupsThisBolide[0].longitudeDegreesEast, dtime, groupsThisBolide[0].energyJoules)
        bolides.append(bsf.Bolide(firstGroupOldStyle)) # start a new bolide instance
    
        # Then add the rest of the groups to this bolide (which is the last in the bolides list)
        groupsArray = []
        for group in groupsThisBolide[1:]:
           #dtime = detection.productTime + \
           #    datetime.timedelta(milliseconds=int(group.timeOffsetMsec))
            dtime = group.time
            groupsArray.append(bsf.Group(group.id, group.latitudeDegreesNorth, 
                    group.longitudeDegreesEast, dtime, group.energyJoules))
        bolides[-1].vector_add(groupsArray)
        bolides[-1].counter = len(bolides[-1].group)

    bolides = np.array(bolides)

    totalTime = time.time() - startTime
    if (verbosity): print("Forming Old-style Bolide class total processing time {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))

    nBolides = len(bolides)

    #***
    # The prototype detector uses sequential filters. We are hijacking this code in order to generate all filter
    # responses on all triggers

    # initialize the output
    filterResponses = FilterResponses(nBolides)

    #***
    #%% Group count filter
    # If not performing the sigmoid, the group count filter is simply the group count, nothing to do here

    #***
    #%% Time filter
    # Time filter is just the total duration of the cluster

    #***
    #%% Apply linelet filter
    startTime = time.time()

    if (multiProcessEnabled):
        idxList = np.arange(nBolides)
        # Set the chunkSize so that the bolide list is divided between the desired number of jobs
        chunkSize = int(np.ceil(len(idxList) / nJobs))
        idxChunked = [idxList[i:i + chunkSize] for i in range(0, len(idxList), chunkSize)]  

        with mp.Pool(nJobs) as pool:
            results = [pool.apply_async(_linelet_filter_elemental, args=([bolides[chunk]])) for chunk in idxChunked]
                    
            resultArray = []
            [resultArray.extend(result.get()) for result in results]
            filterResponses.latLonLinelets =  resultArray
    else:
        # Do all at once
        filterResponses.latLonLinelets = _linelet_filter_elemental(bolides)


    totalTime = time.time() - startTime
    if (verbosity): print("latLonLinelets total processing time {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))

    #***
    #%% Energy Deposition Profile Balance filter

    startTime = time.time()

    for bolideIndex in np.arange(nBolides):
        filterResponses.energyRatio[bolideIndex] = CalculateProfileBalanceRatio(bolides[bolideIndex].group)


    totalTime = time.time() - startTime
    if (verbosity): print("Energy Ratio total processing time {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))


    #***
    #%% Filter by fitting splinelets to a moving window over the energy deposition profile

    startTime = time.time()
    if (verbosity):
        pbar = tqdm(total=nBolides, desc='Computing Splinelets Feature')

    if (multiProcessEnabled):

        idxList = np.arange(nBolides)
        # Set the chunkSize so that the bolide list is divided between the desired number of jobs
        chunkSize = int(np.ceil(len(idxList) / nJobs))
        idxChunked = [idxList[i:i + chunkSize] for i in range(0, len(idxList), chunkSize)]  

        with mp.Pool(nJobs) as pool:
            results = [pool.apply_async(_splinelet_elemental, args=([bolides[chunk]])) for chunk in idxChunked]
                    
            resultArray = []
            [resultArray.extend(result.get()) for result in results]
            filterResponses.splinelets =  resultArray
    else:
        filterResponses.splinelets = _splinelet_elemental(bolides)

    if (verbosity):
        pbar.close()

    totalTime = time.time() - startTime
    if (verbosity): print("Splinelets total processing time {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))

    #***
    #%% Remove events that have large deviations from a ground track line fit

    startTime = time.time()

    horizontal_flag = False
    for bolideIndex in np.arange(nBolides):
        # check if line is more horizontal or vertical
        
        if abs(bolides[bolideIndex].group[0].lon - bolides[bolideIndex].group[-1].lon) > abs(bolides[bolideIndex].group[0].lat - bolides[bolideIndex].group[-1].lat):
            line_fit_results = np.polyfit([g.lon for g in bolides[bolideIndex].group], [g.lat for g in bolides[bolideIndex].group], 1,full=True)
            horizontal_flag = True
        else:
            line_fit_results = np.polyfit([g.lat for g in bolides[bolideIndex].group], [g.lon for g in bolides[bolideIndex].group], 1,full=True)
            horizontal_flag = False
            
        if len(line_fit_results[0]>0):
            poly_coefficient = line_fit_results[0]
            
            lat_array = np.array([event.lat for event in bolides[bolideIndex].group])
            lon_array = np.array([event.lon for event in bolides[bolideIndex].group])
            
            if horizontal_flag:
                predicted_lats_deg = np.polyval(poly_coefficient, lon_array)
                dists_km = geoUtil.DistanceFromLatLonPoints(predicted_lats_deg, lon_array, lat_array, lon_array)
                
            else:
                predicted_lons_deg = np.polyval(poly_coefficient, lat_array)
                dists_km = geoUtil.DistanceFromLatLonPoints(lat_array, predicted_lons_deg, lat_array, lon_array)
            
            # Calculate maximum distance ignoring outliers
            # Use a median absolute deviation derived standard deviation to find the threshold for outliers
            # sigma = 1.4826 * mad(x)
            if (np.median(dists_km) == 0.0):
                norm_dists_km = 0.0
            else:
                norm_dists_km = np.abs(dists_km - np.median(dists_km)) / np.median(dists_km)
            sigma = 1.4826*np.median(np.abs(norm_dists_km - np.median(norm_dists_km)))
            sigmaThreshold = 5.0
            belowThreshold = np.abs(norm_dists_km)  <= (sigma*sigmaThreshold)
            if (np.all(np.logical_not(belowThreshold))):
                # Outlier rejector failed
                max_dist_km = dists_km.max()
            else:
                max_dist_km = dists_km[belowThreshold].max()

        else:
            raise Exception('Figure out what to do in this condition!')

        filterResponses.groundTrackDeviation[bolideIndex] = max_dist_km
            



    totalTime = time.time() - startTime
    if (verbosity): print("groundTrackDeviation total processing time {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))

    pass

    return filterResponses

#*************************************************************************************************************
# def _linelet_filter_elemental()
#
# Subfunction used to parallelize the lat/lon linelet filter computation
#
# Inputs:
#   bolides -- [class list] old style bolide class. The list of bolides to compute filter for
#
# Returns:
#   latLonLinelets -- [float list] the filter response for each bolide in <bolides>
#
#*************************************************************************************************************
def _linelet_filter_elemental(bolides):

    window_width = 5

    latLonLinelets = []
    for bolide in bolides:
        date_array = [e.dtime for e in bolide.group]
        timestamps = [t.timestamp() for t in date_array]
        lon_array = np.array([g.lon for g in bolide.group])
        lat_array = np.array([g.lat for g in bolide.group])
        
        n_windows = np.max([int(len(timestamps)/window_width)-int(1), int(1)])
        residuals_lon = np.zeros(n_windows)
        residuals_lat = np.zeros(n_windows)
        
        for i in range(n_windows):
            line_fit_results = np.polyfit(timestamps[i*window_width:(i+1)*window_width], lon_array[i*window_width:(i+1)*window_width], 1,full=True)
            if (len(line_fit_results[1]) < 1):
                # polyfit fails. Set to None
                residuals_lon[i] = np.nan
            else:
                residuals_lon[i] = line_fit_results[1][0]
            line_fit_results = np.polyfit(timestamps[i*window_width:(i+1)*window_width], lat_array[i*window_width:(i+1)*window_width], 1,full=True)
            if (len(line_fit_results[1]) < 1):
                # polyfit fails. Set to None
                residuals_lat[i] = np.nan
            else:
                residuals_lat[i] = line_fit_results[1][0]
            
        if np.all(np.isnan(residuals_lon)) or np.all(np.isnan(residuals_lat)):
            score = np.nan
        else:
            score = np.sqrt(np.nanmedian(residuals_lon)**2.0 + np.nanmedian(residuals_lat)**2.0)
        if np.isnan(score) or np.isinf(score):
            # If the score is still NaN or inf then give up and set to a very large number
            score = 10000.0
        latLonLinelets.append(score)

    return latLonLinelets
        
#*************************************************************************************************************
# def _splinelet_elemental()
#
# Subfunction used to parallelize the splinlet filter computation
#
# Inputs:
#   bolides -- [class list] old style bolide class. The list of bolides to compute filter for
#
# Returns:
#   splineletResponse -- [float list] the filter response for each bolide in <bolides>
#
#*************************************************************************************************************
def _splinelet_elemental(bolides):

    spline_05 = -2.0 # This can be ANYTHING since we are not computing the sigmoid
    splineletResponse = np.full_like(bolides, 0.0)

    for idx, bolide in enumerate(bolides):

        [_, splineletResponse[idx], _] = FilterBolidesByEnergySplineResidual(bolide, spline_05)

    return splineletResponse
    

