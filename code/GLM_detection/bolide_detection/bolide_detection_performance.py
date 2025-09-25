#*************************************************************************************************************
# bolide_detection_performance module
#
# Tools to assess performance of bolide detector compared to the NEO-bolides website
#
#

import os
import datetime
import warnings
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import bolide_detections as bd
import bolide_dispositions as bDisp
import geometry_utilities as geoUtil

from bolides.plotting import plot_density, plot_scatter, generate_plot

#*************************************************************************************************************
def find_bolide_match_from_website(bolideDispositionProfileList, bolidesFromWebsite, satellite,
        confidenceThreshold=0.0, beliefSource=None):
    """
    Compares a bolide candidate with the list from the neo-bolides website and finds any matches.
   
    Parameters
    ----------
    bolideDispositionProfileList : [list] of BolideDispositionProfile objects
    bolidesFromWebsite  : [WebsiteBolideEvent list] 
        created by: bolide_dispositions.pull_dispositions_from_website 
    satellite       : [str] 
        Which GLM to search for 'G**' {'G16', 'G17', etc...}
        If 'all' then match across all available satellites.
        If 'east' or 'west' to assess GOES-East and GOES-West 
            (i.e. [G16, G19] and [G17, G18] as the same GOES-East or GOES-West data)
    confidenceThreshold : [float] 
        Classifier confidence score to use as detection threshold
    beliefSource        : [str] 
        Which source to use for belief confidence score {'human', 'triage', 'validation'}
        if confidenceThreshold==0.0 then ignore beliefSource
   
    Returns
    -------
    matchIndex : [int np.array] Same size and len(bolideDispositionProfileList). 
        Gives the bolide index in bolidesFromWebsite which matches this detection. 
        -1 means no match but same satellite, 
        -2 means below confidenceThreshold
        -3 means no match and not even same satellite (not used if satellite == 'all')
    """
   

    assert confidenceThreshold >= 0.0 and confidenceThreshold <= 1.0, 'confidenceThreshold must be between 0 and 1, inclusive'

    satelliteOptions = bd.get_satellite_options(satellite)
    
    matchIndex = np.full([len(bolideDispositionProfileList)], int(-1))

    # How close does the beginning or end the clusters need to be to the truth bolides to be considered a hit?
    distDeltaRange = 50.0 # kilometers (was: 50.0)

    #***
    # Get the data from the truth bolides
    # Extend ranges by expansion amount to account for bracketing errors
   #timeRangeExt    = 0.0 / 60 / 60 / 24 # convert second => day (was: 0.0)
    timeRangeExt    = 0.5 # seconds
   #latLonRangeExt  = 0.0 # degrees
    # There can be data from multiple satellites per bolide in bolidesFromWebsite
    # In Unix Time
    bolideStartTimes    = []
    bolideEndTimes      = []
    bolideAvgLat        = []
    bolideAvgLon        = []
    nBolides = len(bolidesFromWebsite)
    for bIdx, bolide in enumerate(bolidesFromWebsite):
        if satellite == 'all':
            idx = np.arange(len(bolide.satellite))
        else:
            idx = np.nonzero(np.in1d(bolide.satellite, satelliteOptions))[0]
        if (len(idx) == 1):
            idx = idx[0]
            # Found a bolide for the GOES Satelite data in file_path
            bolideStartTimes.append(bolide.timeRange[idx][0].timestamp() - timeRangeExt)
            bolideEndTimes.append(bolide.timeRange[idx][1].timestamp()   + timeRangeExt)
            bolideAvgLat.append(np.mean(bolide.latRange[idx]))
            bolideAvgLon.append(np.mean(bolide.lonRange[idx]))
        elif (len(idx) > 1):
            # Sometimes the bolide is divided into two (or more?) parts. Combine these.
            bolideStartTimes.append(np.min([bolide.timeRange[i][0].timestamp() for i in idx]) - timeRangeExt)
            bolideEndTimes.append(np.max([bolide.timeRange[i][1].timestamp() for i in idx])  + timeRangeExt)
            bolideAvgLat.append(np.mean(np.array([bolide.latRange[i] for i in idx]).flatten()))
            bolideAvgLon.append(np.mean(np.array([bolide.lonRange[i] for i in idx]).flatten()))
        else:
            bolideStartTimes.append(np.nan)
            bolideEndTimes.append(np.nan)
            bolideAvgLat.append(np.nan)
            bolideAvgLon.append(np.nan)

    bolideStartTimes    = np.array(bolideStartTimes)
    bolideEndTimes      = np.array(bolideEndTimes)
    bolideAvgLat        = np.array(bolideAvgLat)
    bolideAvgLon        = np.array(bolideAvgLon)

    # Pick which belief to return
    # But only if confidenceThreshold > 0.0
    if confidenceThreshold > 0.0:
        selection = bDisp.bolideBeliefSwitcher.get(beliefSource, bDisp.bolideBeliefSwitcher['unknown'])
        if (selection == bDisp.bolideBeliefSwitcher['unknown']):
            raise Exception("Unknown belief source")
    
    #***
    # For each bolide disposition, search for a match in bolidesFromWebsite
    for idx, disposition in enumerate(bolideDispositionProfileList):

        # Check if not even the correct satellite, set index to -2
        if disposition.features.goesSatellite not in satelliteOptions:
            matchIndex[idx] = int(-3)
            continue
        
        # Check if confidence score is too low
        # But only if confidenceThreshold > 0.0
        if confidenceThreshold > 0.0:
            if selection == bDisp.bolideBeliefSwitcher['human']:
                # TODO: Get this working for multiple opinions
                if (disposition.humanOpinions is not None and disposition.humanOpinions[0].belief < confidenceThreshold):
                    matchIndex[idx] = int(-2)
                    continue
            elif (selection == bDisp.bolideBeliefSwitcher['triage'] or selection == bDisp.bolideBeliefSwitcher['validation']):
                opinionIdx = [i for i, machineOpinion in enumerate(disposition.machineOpinions) if machineOpinion.source == beliefSource]
                # If there are multiple machine opinions from this source, then average the scores
                if len(opinionIdx) >= 1:
                    bolideBelief = np.mean([disposition.machineOpinions[i].bolideBelief for i in opinionIdx])
                else:
                    raise Exception('Error finding machine opinion')
                if (disposition.machineOpinions is not None and bolideBelief < confidenceThreshold):
                    matchIndex[idx] = int(-2)
                    continue
        
        # Does this bolide candidate agree with any ground truth bolides?
        # First find any true bolides that line up in time with this one.
        # "Lining up" means 50% of the groups lie within the extended range of the true bolide
        # Ignore warning about Nans
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            potentialTimeMatchHere = np.logical_and(disposition.features.bolideMidRange[0].timestamp() > bolideStartTimes,
                    disposition.features.bolideMidRange[1].timestamp() < bolideEndTimes)
        
        if (len(np.nonzero(potentialTimeMatchHere)[0]) > 0):
            # Some line up in time. Now check lat/lon
            avgLat = disposition.features.avgLat
            avgLon = disposition.features.avgLon
            latLonDist = geoUtil.DistanceFromLatLonPoints(avgLat, avgLon,
                    np.transpose(bolideAvgLat), np.transpose(bolideAvgLon))

            # Check for any lat/lon matches in kilometers
            # Ignore warning about Nans
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                potentialDistMatchHere = np.abs(latLonDist) <= distDeltaRange

            thisMatchIndex = np.nonzero(np.logical_and(potentialTimeMatchHere, potentialDistMatchHere))[0]

            if (len(thisMatchIndex) == 1):
                # We found a match! 
                matchIndex[idx] = int(thisMatchIndex)
            if (len(thisMatchIndex) > 1):
                # Found more than one match!
               #raise Exception('Found more than one website bolide match.')
                print('Found more than one website bolide match. Choosing the first one')
                matchIndex[idx] = int(thisMatchIndex[0])

    return matchIndex


#*************************************************************************************************************
# combine_bolideDispositionProfiles_from_same_bolidesFromWebsite
#
# Combines bolideDispositionProfileList elements that belong to the same website bolide event.
# This function is called on the output from find_bolide_match_from_website and returns a new 
# bolideDispositionProfileList and matchIndex with profiles combined.
#
# Inputs:
#   bolideDispositionProfileList -- [list] of BolideDispositionProfile objects
#   matchIndex -- [int np.array] Same size and len(bolideDispositionProfileList). 
#       The output from find_bolide_match_from_website
#       Gives the bolide index in bolidesFromWebsite which matches this detection. 
#       [-1 means no match but same satellite, 
#        -2 means below confidenceThreshold
#        -3 means no match and not even same satellite]
#
# Outputs:
#   bolideDispositionProfileListCombined -- [list] of BolideDispositionProfile objects after combining
#   matchIndexCombined -- [int np.array] Same size and len(bolideDispositionProfileList). 
#       Gives the bolide index in bolidesFromWebsite which matches this detection. 
#
#*************************************************************************************************************
def combine_bolideDispositionProfiles_from_same_bolidesFromWebsite(bolideDispositionProfileList, matchIndex):

    bolideDispositionProfileListCombined = []
    matchIndexCombined = []

    # Step through each bolide profile and see if there are any other profiles that correspond to the same website
    # bolide. If so, then combine them, all other profiles leave the same.
    for idx in range(len(bolideDispositionProfileList)):
        thisWebsiteIndex = matchIndex[idx]
        if (thisWebsiteIndex >= 0):
            # There is a website match
            allMatchIndices = np.nonzero(thisWebsiteIndex == matchIndex)[0]
            assert idx == allMatchIndices[0], 'Bookkeeping error'
            if (len(allMatchIndices) > 1):
                # More than one match, so combine them
                bolideDispositionProfileListCombined.append(bolideDispositionProfileList[allMatchIndices[0]])
                for matchProfileIdx in allMatchIndices[1:].tolist():
                    profile = bolideDispositionProfileList[matchProfileIdx]
                    bolideDispositionProfileListCombined[-1].add_profile(profile)
                matchIndexCombined.append(thisWebsiteIndex)
                # Change the matchIndex values for the combined profiles so that we no longer consider these profiles in
                # the outer loop
                matchIndex[allMatchIndices] = -99
                continue

        if (thisWebsiteIndex > -99):
            # A value of -99 means this profile has already been combined.
            # No existing match, just propagate this profile as a new instance    
            bolideDispositionProfileListCombined.append(bolideDispositionProfileList[idx])
            matchIndexCombined.append(thisWebsiteIndex)

    return bolideDispositionProfileListCombined, np.array(matchIndexCombined)

#*************************************************************************************************************
def compare_bolideDispositionProfileList_to_bolidesFromWebsite(bolideDispositionProfileList, bolidesFromWebsite,
        satellite='all', startDate=None, endDate=None, confidenceThreshold=0.0,
        useOnlyIntersectDays=True, deepCopyInputs=True, beliefSource=None):
    """
    Compares a bolideDispositionProfileList, which is generated from the bolide_detections.p and bolide_rejections.p
    output of the pipeline, to a bolidesFromWebsite, which is generated from
    bolide_dispositions.pull_dispositions_from_website and records all events from the website.
   
    Parameters
    ----------
    bolideDispositionProfileList : [list] of BolideDispositionProfile objects 
        created by: bolide_dispositions.generate_bolideDispositionProfileList_from_bolide_detections
    bolidesFromWebsite  : [WebsiteBolideEvent list] 
        created by: bolide_dispositions.pull_dispositions_from_website 
    satellite : [str] 
        One of bolide_detections.validSatellites
        or 'all' to assess over all satellites
        or 'east' or 'west' to assess GOES-East and GOES-West 
            (i.e. [G16, G19] and [G17, G18] as the same GOES-East or GOES-West data)
    startDate : [str] 
        The starting date for analysis
        Is ISO format: 'YYYY-MM-DD'
        None or '' means no start date
    endDate : [str] 
        The ending date for analysis
        Is ISO format: 'YYYY-MM-DD'
        None or '' means no end date
    confidenceThreshold : [float] 
        Confidence threshold to use to declare a detection
    useOnlyIntersectDays : [bool] 
        If True then only use days that have both website bolides and pipeline bolides.
        Otherwise, use all data in statistics
    deepCopyInputs : [logical] 
        If True then make a deep copy of the input lists so that they are preserved.
        Otherwise, the are manipulated. THIS IS SLOW!  
    beliefSource : [str] 
        Which source to use for belief confidence score {'human', 'triage', 'validation'}
   
   
    Returns
    -------
    histFig : [matplotlib Figure class] 
        Histogram figure object
    globeFigScatter : [matplotlib Figure class] 
        Globe figure object of scatter plot
    globeFigDensityOn : [matplotlib Figure class] 
        Globe figure object of desnity plot on website
    globeFigDensityNotOn : [matplotlib Figure class] 
        Globe figure object of scatter plot NOT on website

    """

    # Make local deep copies to not disturb the originals
    if deepCopyInputs:
        print(f'Making deep copies of bolides lists, this is slow...')
        bolideDispositionProfileListLocal = copy.deepcopy(bolideDispositionProfileList)
        bolidesFromWebsiteLocal = copy.deepcopy(bolidesFromWebsite)
    else:
        print(f'********')
        print(f'Not deepcopying; bolideDispositionProfileList and bolidesFromWebsite will be manipulated')
        print(f'********')
        bolideDispositionProfileListLocal = bolideDispositionProfileList
        bolidesFromWebsiteLocal = bolidesFromWebsite

    #**************************
    # Only use data from selected days
    bolidesFromWebsiteLocal = bDisp.select_bolidesFromWebsite_from_dates (bolidesFromWebsiteLocal,
            startDate=startDate, endDate=endDate)
    [bolideDispositionProfileListLocal, _] = bDisp.select_bolideDispositionProfileList_from_dates (bolideDispositionProfileListLocal,
            startDate=startDate, endDate=endDate)

    #**************************
    # Only use days that have both website data and pipeline data
    # This is to correct the performance statistics to days that were vetted and data was available.
    if useOnlyIntersectDays:
        [bolideDispositionProfileListLocal , bolidesFromWebsiteLocal] = bDisp.intersect_days_bolideDispositionProfileList_and_bolidesFromWebsite(
            bolideDispositionProfileListLocal, bolidesFromWebsiteLocal, satellite=satellite)

    #**************************
    # Determine how many website matches were found
    matchIndex = find_bolide_match_from_website(bolideDispositionProfileListLocal , bolidesFromWebsiteLocal, satellite,
            confidenceThreshold=confidenceThreshold, beliefSource=beliefSource)
    # Combined detections corresponding to the same website event
    bolideDispositionProfileListLocal , matchIndex = combine_bolideDispositionProfiles_from_same_bolidesFromWebsite(
            bolideDispositionProfileListLocal , matchIndex)
    # bolideDispositionProfileListLocal  now has all profiles corresponding to different website events
    

    pipeMatches = np.nonzero(matchIndex > -1)[0]
    pipeNotMatches = np.nonzero(matchIndex == -1)[0]
    print('Number of {} Website Bolides found = {}'.format(satellite, len(pipeMatches)))
    print('***')

    #**************************
    # Compare total number
    satelliteOptions = bd.get_satellite_options(satellite)
    webBolidesIdx = [idx for [idx, bolide] in enumerate(bolidesFromWebsiteLocal) if np.any(np.in1d(satelliteOptions, bolide.satellite))]
    nWebBolide = len(webBolidesIdx)

    selection = bDisp.bolideBeliefSwitcher.get(beliefSource, bDisp.bolideBeliefSwitcher['unknown'])
    if (selection == bDisp.bolideBeliefSwitcher['unknown']):
        raise Exception("Unknown belief source")
    elif selection == bDisp.bolideBeliefSwitcher['human']:
        # TODO: Get this working for multiple opinions
        pipeBolidesIdx = [idx for [idx, disposition] in enumerate(bolideDispositionProfileListLocal ) if 
                np.any(np.in1d(satelliteOptions, disposition.features.goesSatellite)) and 
                disposition.humanOpinions[0].belief >= confidenceThreshold]
    else:
        pipeBolidesIdx = []
        for bolideIdx, disposition in enumerate(bolideDispositionProfileListLocal):
            idx = [idx for idx, machineOpinion in enumerate(disposition.machineOpinions) if machineOpinion.source == beliefSource]
            bolideBelief = np.mean([disposition.machineOpinions[i].bolideBelief for i in idx])
            if np.any(np.in1d(satelliteOptions, disposition.features.goesSatellite)) and bolideBelief >= confidenceThreshold:
                pipeBolidesIdx.append(bolideIdx)
    nPipeBolidesIdx = len(pipeBolidesIdx)

    print('Number of Website  {} Bolides = {}'.format(satellite, nWebBolide))
    print('Number of Pipeline {} Bolides = {}'.format(satellite, nPipeBolidesIdx))
    print('***')

    
    #**************************
    # Precision and Recall, assuming the website is the truth

    totFound = nPipeBolidesIdx
    P = nWebBolide
    TP = len(pipeMatches)
    FP = totFound - TP
    FN = P - TP
    
    if (totFound == 0):
        precision = -1
    else:
        precision = TP / totFound
    recall = TP / P
    print('{0} Precision = {1:.3f}'.format(satellite, precision))
    print('{0} Recall = {1:.3f}'.format(satellite, recall))
    print('')


    #**************************
    # Detections as a function of time, histogram

    histFig, ax = plt.subplots(1,1)
    histFig.set_figwidth(14.0)
    histFig.set_figheight(9.0)
    locator1 = mdates.AutoDateLocator()
    # Count number of days in data set
    numberOfDays = (bolidesFromWebsiteLocal[0].timeRange[0][0] - bolidesFromWebsiteLocal[-1].timeRange[0][0]).days
    # One day per bin, but at least 100 bins
    timeBins = np.max([100, numberOfDays])

    mdateTimeOriginPipe = \
        mdates.date2num([disposition.features.bolideTime for disposition in 
            [bolideDispositionProfileListLocal[i] for i  in pipeBolidesIdx] ])
    mdateTimeOriginPipeMatch = \
        mdates.date2num([disposition.features.bolideTime for disposition in 
            [bolideDispositionProfileListLocal[i] for i  in pipeMatches] ])

    # Collect the dates for the website bolides
    # This assumes there is only one website bolide for each satellite East or West
    # TODO: this is awkward. Figure out a more elegent and readable way to handle this.
    mdateTimeOriginWeb = []
    websiteBolidesThisSat = [bolidesFromWebsiteLocal[i] for i in webBolidesIdx]
    for webBolide in websiteBolidesThisSat:
        for sat in satelliteOptions:
            if webBolide.satellite.count(sat) > 0:
                idx = webBolide.satellite.index(sat)
                mdateTimeOriginWeb.append(mdates.date2num(webBolide.timeRange[idx][0]))
                continue

    minBin = np.min(mdateTimeOriginWeb)
    maxBin = np.max(mdateTimeOriginWeb)
    ax.hist(mdateTimeOriginPipe, bins=timeBins, range=(minBin, maxBin), log=False, facecolor='b', alpha=0.5, label='{} Pipeline'.format(satellite))
    ax.hist(mdateTimeOriginPipeMatch, bins=timeBins, range=(minBin, maxBin), log=False, facecolor='g', alpha=0.5,
            label='{} Pipeline Matches'.format(satellite))
    ax.hist(mdateTimeOriginWeb,  bins=timeBins, range=(minBin, maxBin), log=False, facecolor='r', alpha=0.5, label='{} Website'.format(satellite))
    ax.grid(axis='y')
    ax.legend()
    ax.xaxis.set_major_locator(locator1)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator1))
    ax.set_xlabel('Date of Event')
    ax.set_title('GOES-{} Distribution of All Bolide Detections in Time'.format(satellite))
    

    #**************************
    # Detections space and time
    # Scatter plot of bolide locations in lat and lon

    #***
    # All pipeline detections
    meanLatPipe = [disposition.features.avgLat for disposition in [bolideDispositionProfileListLocal[i] for i  in pipeBolidesIdx]]
    meanLonPipe = [disposition.features.avgLon for disposition in [bolideDispositionProfileListLocal[i] for i  in pipeBolidesIdx]]
    
    # All pipeline detections that match website
    meanLatPipeMatch = [disposition.features.avgLat for disposition in [bolideDispositionProfileListLocal[i] for i  in pipeMatches]]
    meanLonPipeMatch = [disposition.features.avgLon for disposition in [bolideDispositionProfileListLocal[i] for i  in pipeMatches]]
    
    # All pipeline detections that do NOT match website
    meanLatPipeNotMatch = [disposition.features.avgLat for disposition in [bolideDispositionProfileListLocal[i] for i  in pipeNotMatches]]
    meanLonPipeNotMatch = [disposition.features.avgLon for disposition in [bolideDispositionProfileListLocal[i] for i  in pipeNotMatches]]

    # This assumes there is only one website bolide for each satellite East or West
    # TODO: this is awkward. Figure out a more elegent and readable way to handle this.
    meanLatWeb = []
    for webBolide in websiteBolidesThisSat:
        for sat in satelliteOptions:
            if webBolide.satellite.count(sat) > 0:
                idx = webBolide.satellite.index(sat)
                meanLatWeb.append(np.mean(webBolide.latRange[idx]))
                continue
    meanLonWeb = []
    for webBolide in websiteBolidesThisSat:
        for sat in satelliteOptions:
            if webBolide.satellite.count(sat) > 0:
                idx = webBolide.satellite.index(sat)
                meanLonWeb.append(np.mean(webBolide.lonRange[idx]))
                continue
    
    #***
    # All detections from pipeline and website on globe
    globeFigScatter, ax = generate_plot(figsize=(20,10))
    plot_scatter(meanLatPipe, meanLonPipe, boundary=['goes-e', 'goes-w'],
        marker="o", color="r", alpha=0.9, edgecolor=None, s=40, zorder=900, label='{} Pipeline'.format(satellite), fig=globeFigScatter, ax=ax)
    plot_scatter(meanLatWeb, meanLonWeb, boundary=['goes-e', 'goes-w'],
        marker="+", color="g", alpha=0.9, edgecolor='k', s=30, zorder=900, label='{} Website'.format(satellite), fig=globeFigScatter, ax=ax)
    plt.legend()
    plt.title('Scatter plot of all bolide detection from Pipeline versus Website')
    plt.show()

    #***
    # plot all detections as a density heat map

    if (len(meanLatPipeMatch) > 1):
        globeFigDensityOn, ax = plot_density(meanLatPipeMatch, meanLonPipeMatch, bandwidth=2, 
            boundary=['goes-e', 'goes-w'], lat_resolution=200,  lon_resolution=100, figsize=(20, 10), 
            title='Density Distribution of All {} Pipeline Detections on Website'.format(satellite))
        plt.show()
    else:
        globeFigDensityOn = None

    if (len(meanLatPipeNotMatch) > 1):
        globeFigDensityNotOn, ax = plot_density(meanLatPipeNotMatch, meanLonPipeNotMatch, bandwidth=2, 
            boundary=['goes-e', 'goes-w'], lat_resolution=200,  lon_resolution=100, figsize=(20, 10), 
            title='Density Distribution of All {} Pipeline Detections NOT on Website'.format(satellite))
        plt.show()
    else:
        globeFigDensityNotOn = None

    # Return Figure handles
    return histFig, globeFigScatter, globeFigDensityOn, globeFigDensityNotOn

#*************************************************************************************************************
def plot_precision_recall_vs_confidence (bolideDispositionProfileList, bolidesFromWebsite,
        satellite, beliefSource='human', rebalanceRatio=1.0, useOnlyIntersectDays=True,  
        useMaxPossibleRecall=False, minConfidenceThreshold=0.01, 
        startDate=None, endDate=None, deepCopyInputs=True):
    """
    plot_precision_recall_vs_confidence
  
    Plots the Precision/Recall curve of bolideDispositionProfileList with bolidesFromWebsite as ground truth as the
    confidence threshold is adjusted.

    You can plot versus each individual satellite or the GOES-East or GOES-West positions, The latter meaning any
    saytellites associstaed with that position are treated together.
  
    The rebalanceRatio assumes that the vast majority of clusters are true negatives (A very, very small fraction of
    clusters are legitimate bolides). It will adjust the precision and nDetectionsPerDay to account for the imbalance.
  
    Parameters
    ----------
    bolideDispositionProfileList -- [list] of BolideDispositionProfile objects created by:
        bolide_dispositions.generate_bolideDispositionProfileList_from_bolide_detections
    bolidesFromWebsite  -- [WebsiteBolideEvent list] created by:
        bolide_dispositions.pull_dispositions_from_website 
    satellite           -- [str] one if bolide_detections.validSatellites
                            or 'all' to assess over all satellites
                            or 'east' or 'west' to assess GOES-East and GOES-West 
                                (i.e. [G16, G19] and [G17, G18] as the same GOES-East or GOES-West data)
    beliefSource        -- [str] Which source to use for belief {'human', 'triage', 'validation'}
    rebalanceRatio      -- [float] Ratio of cluster in list to total clusters
                            Use this to re-balance statistics if only a fraction of total clusters per day are available
                            1.0 means no rebalancing. 0.1 mean only 10% of total clusters is in list
    useOnlyIntersectDays -- [bool] If True then only use days that have both website bolides and pipeline bolides.
                                    Otherwise, use all data in statistics
    useMaxPossibleRecall -- [bool] If true then scale the recall by the maximum possible recall id detection threshold = 0.0
                            Note: This is meaningful only if all clusters are stored in bolideDispositionProfileList (I.e. training data)
    minConfidenceThreshold -- [float] Minimum confidence threshold to plot.
    startDate           -- [str] The starting date for analysis
                            Is ISO format: 'YYYY-MM-DD'
                            None or '' means no start date
    endDate             -- [str] The ending date for analysis
                            Is ISO format: 'YYYY-MM-DD'
                            None or '' means no end date
    deepCopyInputs      -- [logical] If True then make a deep copy of the input lists so that they are preserved. THIS IS SLOW!  
                                Otherwise, the bolideDispositionProfileList manipulated. 
  
  
    Returns
    -------
    precision           -- [float array] Precision vs confidence threshold
    recall              -- [float array] Recall vs confidence threshold
    confidenceScanVals  -- [float array] Confidecne threshold corresponding to precison and recall
    nDetectionsPerDay -- [float array] Average number of detectison per day vs confidence threshold
 
    """

    assert satellite in ('all', 'east', 'west') or bd.validSatellites.count(satellite) == 1, 'Unknown satellite'

    # Make local deep copies to not disturb the originals
    if deepCopyInputs:
        print(f'Making deep copies of bolides lists, this is slow...')
        bolideDispositionProfileListLocal = copy.deepcopy(bolideDispositionProfileList)
        bolidesFromWebsiteLocal = copy.deepcopy(bolidesFromWebsite)
    else:
        print(f'********')
        print(f'Input lists will be manipulated')
        print(f'********')
        bolideDispositionProfileListLocal = bolideDispositionProfileList
        bolidesFromWebsiteLocal = bolidesFromWebsite

    #**************************
    # Only use data from selected days
    bolidesFromWebsiteLocal = bDisp.select_bolidesFromWebsite_from_dates (bolidesFromWebsiteLocal,
            startDate=startDate, endDate=endDate)
    [bolideDispositionProfileListLocal, _] = bDisp.select_bolideDispositionProfileList_from_dates (bolideDispositionProfileListLocal,
            startDate=startDate, endDate=endDate)

    #**************************
    # Only use days that have both website data and pipeline data
    # This is to correct the performance statistics to days that were vetted and data was available.
    if useOnlyIntersectDays:
        [bolideDispositionProfileListLocal , bolidesFromWebsiteLocal] = bDisp.intersect_days_bolideDispositionProfileList_and_bolidesFromWebsite(
            bolideDispositionProfileListLocal, bolidesFromWebsiteLocal, satellite=satellite)

    if len(bolideDispositionProfileListLocal) == 0: 
        print('There are no detected bolides to assess (after taking desired cuts)')
        return

    # Find all website bolides on this satellite
    satelliteOptions = bd.get_satellite_options(satellite)
    webBolidesIdx = [idx for [idx, bolide] in enumerate(bolidesFromWebsiteLocal) if np.any(np.in1d(satelliteOptions, bolide.satellite))]
    nWebBolide = len(webBolidesIdx)

    # Determine how many website matches were found, ignoring confidenceThreshold
    matchIndex = find_bolide_match_from_website(bolideDispositionProfileListLocal , bolidesFromWebsiteLocal, satellite,
            confidenceThreshold=0.0, beliefSource=beliefSource)
    bolideDispositionProfileListLocal , matchIndex = combine_bolideDispositionProfiles_from_same_bolidesFromWebsite(
            bolideDispositionProfileListLocal , matchIndex)
    # bolideDispositionProfileListLocal  now has all profiles corresponding to different website events
    
    #***
    # Now extract the bolide beliefs
    # TODO: Get this working for multiple opinions
    selection = bDisp.bolideBeliefSwitcher.get(beliefSource, bDisp.bolideBeliefSwitcher['unknown'])
    if (selection == bDisp.bolideBeliefSwitcher['unknown']):
        raise Exception("Unknown belief source")
    if (selection == bDisp.bolideBeliefSwitcher['human']):
        bolideConfidenceArray = np.array([disposition.humanOpinions[0].belief for disposition in bolideDispositionProfileListLocal])
    else:
        bolideConfidenceArray = []
        for disposition in bolideDispositionProfileListLocal:
            idx = [idx for idx, machineOpinion in enumerate(disposition.machineOpinions) if machineOpinion.source == beliefSource]
            # If there are multiple machine opinions from this source, then average the scores
            if len(idx) >= 1:
                bolideConfidenceArray.append(np.mean([disposition.machineOpinions[i].bolideBelief for i in idx]))
            else:
                raise Exception('Error finding machine opinion')
        bolideConfidenceArray = np.array(bolideConfidenceArray)

    #***
    # Count total number of days of data
    # There is no JulianDate in Python.datatime but...
    # A way to do this is convert the productTime for each detection to a Gregorian ordinal (different integer every
    # day) then count the unique set of days
    ordinalDates = [disposition.features.bolideTime.toordinal() for disposition in bolideDispositionProfileListLocal ]
    totNumDaysWithDetections = len(np.unique(ordinalDates))

    #**************************
    # Compute the theoretical maximum Recall
    # This is because there are bolides on the website that are not found in the GLM data using our clustering method
    # We do this by setting the confidenceThreshold to 0.0 (declare everthing a bolide)
    # Note: This is meaningful only if all clusters are stored in bolideDispositionProfileList (I.e. training data)
    if useMaxPossibleRecall:
        nMaxPossiblePipelineMatches = len(np.nonzero(np.logical_and(matchIndex > -1, bolideConfidenceArray>=0.0))[0])
        maxPossibleRecall = nMaxPossiblePipelineMatches / nWebBolide
    else:
        maxPossibleRecall = 1.0

    #**************************
    # Scan versus confidence threshold
    confidenceScanVals = np.linspace(minConfidenceThreshold, 1.0, num=99)
    precision   = np.full_like(confidenceScanVals, np.nan)
    recall      = np.full_like(confidenceScanVals, np.nan)
    nDetectionsPerDay = np.full_like(confidenceScanVals, np.nan)
    for idx, confidenceThreshold in enumerate(confidenceScanVals):
        # Count total number of detections in same satellite that are at or above threshold
        # This means match index is greater than or equal to -1
        nPipelineBolides = len(np.nonzero(np.logical_and(matchIndex >= -1, bolideConfidenceArray>=confidenceThreshold))[0])

        # Count all matches to website at or above confidence threshold 
        # This means the match index is a natural number index (> -1)
        nPipelineMatches = len(np.nonzero(np.logical_and(matchIndex > -1, bolideConfidenceArray>=confidenceThreshold))[0])

        totFound = nPipelineBolides
        P = nWebBolide
        TP = nPipelineMatches
        FP = totFound - TP
        FN = P - TP
        
        if (totFound == 0):
            precision[idx] = -1
        else:
            # Scale by rebalanceRatio
            # We do so by increasing the number of false positives (FP) by the rebalanceRatio
            precision[idx] = (TP / (TP + FP / rebalanceRatio))
        # Scale by maximum theoretically possible Recall
        # but only do this if 
        recall[idx] = (TP / P) / maxPossibleRecall 

        # Average number of detections per day
        # Scale by rebalanceRatio
        # We do so by increasing the number of false positives (FP) by the rebalanceRatio
        nDetectionsPerDay[idx] = ((TP + FP / rebalanceRatio) / totNumDaysWithDetections)
        
    
    #**************************
    # Plot curves
    fig, ax = plt.subplots(4,1, figsize=(10, 10), sharex=True)

    # Only plot for realistic precision values
    badIdx = np.nonzero(np.logical_or(precision < 0.0, precision > 1.0))[0]
    precision[badIdx] = np.nan

    # precision vs recall
    ax[0].plot(recall, precision, '.-b')
    ax[0].grid()
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
   #ax[0].set_title('Precision Vs. Recall (Scaled by Rebalance Ratio and Maximum Possible Recall)')
    ax[0].set_title('Precision Vs. Recall')
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(0.0, 1.0);

    # Precision vs threshold
    ax[1].plot(confidenceScanVals, precision, '.-r')
    ax[1].grid()
    ax[1].set_xlabel('Confidence Threshold')
    ax[1].set_ylabel('Precision')
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(0.0, 1.0);

    # Recall vs threshold
    ax[2].plot(confidenceScanVals, recall, '.-m')
    ax[2].grid()
    ax[2].set_xlabel('Confidence Threshold')
    ax[2].set_ylabel('Recall')
    ax[2].set_xlim(0.0, 1.0)
    ax[2].set_ylim(0.0, 1.0);

    # Total number of detections vs threshold
    ax[3].plot(confidenceScanVals, nDetectionsPerDay, '.-k')
    ax[3].grid()
    ax[3].set_xlabel('Confidence Threshold')
    ax[3].set_ylabel('Ave. # Detections Per Day')
    ax[3].set_xlim(0.0, 1.0)
    ax[3].set_ylim(0.0, 40.0);

    plt.tight_layout(pad=0.0)

    return [precision, recall, confidenceScanVals, nDetectionsPerDay]


    
#*************************************************************************************************************
# plot_G17_lon
#
# Plots the 'nominal_satellite_subpoint_lon' for G17. This is used to determine when G17 moved to its final location in
# longitude.
#
#*************************************************************************************************************
def plot_G17_lon (bolideDispositionProfileList):

    # Get the lon for G17 data
    pipeBolidesG17Idx = [idx for [idx, disposition] in enumerate(bolideDispositionProfileList) if 
            np.any(np.in1d('G17', disposition.bolideDetection.goesSatellite))]

    lon = [disposition.bolideDetection.subPointLonDegreesEast  for disposition in 
            [bolideDispositionProfileList[i] for i in pipeBolidesG17Idx] ]
    
    mdateTimeOriginPipeG17 = \
        mdates.date2num([disposition.bolideDetection.productTime for disposition in
            [bolideDispositionProfileList[i] for i  in pipeBolidesG17Idx] ])


    # plot
    minBin = np.min(np.min(mdateTimeOriginPipeG17))
    maxBin = np.max(np.max(mdateTimeOriginPipeG17))
    timeBins = 600
    fig, ax = plt.subplots(1,1)
    fig.set_figwidth(14.0)
    fig.set_figheight(9.0)
    locator1 = mdates.AutoDateLocator()

    ax.plot(mdateTimeOriginPipeG17, lon, '.b')
    ax.grid()
    ax.xaxis.set_major_locator(locator1)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator1))
    ax.set_title('GOES 17 nominal_satellite_subpoint_lon')
    ax.set_ylabel('Longitude [Degrees East]')
    ax.set_xlabel('Date of Event')

    
    

#*************************************************************************************************************
# plot_website_bolides_on_globe
# 
# Plots all bolides on website on a basemap globe.
#
# Inputs:
#   bolidesFromWebsite  -- [WebsiteBolideEvent list] created by:
#       bolide_dispositions.pull_dispositions_from_website 
#   PROJ_LIB            -- [str] Basemap requires the environment variable PROJ_LIB to be properly set
#                                   TODO: figure out a more robust way to do this.
#   sigmaThreshold -- [float] The factor above the STD of the density before density color change begins
#
#*************************************************************************************************************
def plot_website_bolides_on_globe (bolidesFromWebsite, PROJ_LIB=None, sigmaThreshold=2.0):

    raise Exception('Set up for G18 and G19')
    raise Exception('Switch to Bolides package for plotting')

    assert PROJ_LIB is not None, 'PROJ_LIB path must be passed'

    webBolidesG16Idx = [idx for [idx, bolide] in enumerate(bolidesFromWebsite) if 
            np.any(np.in1d('G16', bolide.satellite)) and not np.any(np.in1d('G17', bolide.satellite))]
    webBolidesG17Idx = [idx for [idx, bolide] in enumerate(bolidesFromWebsite) if 
            np.any(np.in1d('G17', bolide.satellite)) and not np.any(np.in1d('G16', bolide.satellite))]
    webBolidesStereoIdx = [idx for [idx, bolide] in enumerate(bolidesFromWebsite) if 
            np.any(np.in1d('G16', bolide.satellite)) and np.any(np.in1d('G17', bolide.satellite))]

    #***
    # Get Lat/Lon for each bolide
    meanLatWebG16 = \
        [np.mean(bolide.latRange[bolide.satellite.index('G16')]) for bolide in 
            [bolidesFromWebsite[i] for i in webBolidesG16Idx]]
    meanLonWebG16 = \
        [np.mean(bolide.lonRange[bolide.satellite.index('G16')]) for bolide in 
            [bolidesFromWebsite[i] for i in webBolidesG16Idx]]
    meanLatWebG17 = \
        [np.mean(bolide.latRange[bolide.satellite.index('G17')]) for bolide in 
            [bolidesFromWebsite[i] for i in webBolidesG17Idx]]
    meanLonWebG17 = \
        [np.mean(bolide.lonRange[bolide.satellite.index('G17')]) for bolide in 
            [bolidesFromWebsite[i] for i in webBolidesG17Idx]]
    meanLatWebStereo = \
        [np.mean(bolide.latRange[bolide.satellite.index('G17')]) for bolide in 
            [bolidesFromWebsite[i] for i in webBolidesStereoIdx]]
    meanLonWebStereo = \
        [np.mean(bolide.lonRange[bolide.satellite.index('G17')]) for bolide in 
            [bolidesFromWebsite[i] for i in webBolidesStereoIdx]]

    #***
    # Get energy for each bolide
    totEnergyWebG16 = \
        [bolide.totEnergy[bolide.satellite.index('G16')] for bolide in 
            [bolidesFromWebsite[i] for i in webBolidesG16Idx]]
    totEnergyWebG17 = \
        [bolide.totEnergy[bolide.satellite.index('G17')] for bolide in 
            [bolidesFromWebsite[i] for i in webBolidesG17Idx]]
    # Take mean of G16 and G17 energy for Stereo
    totEnergyWebStereo = \
        [np.mean(bolide.totEnergy) for bolide in 
            [bolidesFromWebsite[i] for i in webBolidesStereoIdx]]
    
    #***
    # Exclude those with negative energy!?!?
    # TODO: figure out why there are any negative totEnergy bolides!
    positiveEnergyG16 = np.nonzero(np.greater(totEnergyWebG16, 0.0))[0]
    positiveEnergyG17 = np.nonzero(np.greater(totEnergyWebG17, 0.0))[0]
    positiveEnergyStereo = np.nonzero(np.greater(totEnergyWebStereo, 0.0))[0]
    meanLatWebG16 = np.array(meanLatWebG16)[positiveEnergyG16]
    meanLonWebG16 = np.array(meanLonWebG16)[positiveEnergyG16]
    meanLatWebG17 = np.array(meanLatWebG17)[positiveEnergyG17]
    meanLonWebG17 = np.array(meanLonWebG17)[positiveEnergyG17]
    meanLatWebStereo = np.array(meanLatWebStereo)[positiveEnergyStereo]
    meanLonWebStereo = np.array(meanLonWebStereo)[positiveEnergyStereo]
    totEnergyWebG16 = np.array(totEnergyWebG16)[positiveEnergyG16]
    totEnergyWebG17 = np.array(totEnergyWebG17)[positiveEnergyG17]
    totEnergyWebStereo = np.array(totEnergyWebStereo)[positiveEnergyStereo]

    totNum = len(positiveEnergyG16) + len(positiveEnergyG17) + len(positiveEnergyStereo)
    
    #***
    # Plot website detections
    globePlotter = pGlobe.GlobePlotter(PROJ_LIB)
    sz = 20 # minimum point size on scatter plot
    fc = 30 # scaling factor for point size based on energy

    # G16
    globePlotter.plot_scatter_on_globe(meanLatWebG16, meanLonWebG16, 
            marker="o", color="r", alpha=0.9, edgecolor=None, 
            s=sz+fc*np.log(totEnergyWebG16/np.median(totEnergyWebG16)), zorder=900, label='G16')

    # G17
    globePlotter.plot_scatter_on_globe(meanLatWebG17, meanLonWebG17, 
            marker="o", color="b", alpha=0.9, edgecolor=None, 
            s=sz+fc*np.log(totEnergyWebG17/np.median(totEnergyWebG17)), zorder=900, label='G17')

    # Stereo
    globePlotter.plot_scatter_on_globe(meanLatWebStereo, meanLonWebStereo, 
            marker="o", color="c", alpha=0.9, edgecolor=None, 
            s=sz+fc*np.log(totEnergyWebStereo/np.median(totEnergyWebStereo)), zorder=900, label='Stereo')
    
    globePlotter.plot_title_and_legend('Distribution of Bolides on https://neo-bolide.ndc.nasa.gov; Number Bolides: {}'.format(totNum))

    #***
    # Plot density distribution
    # Include the stereo detections in each
    meanLatWebAll = np.append(meanLatWebG16, meanLatWebG17) 
    meanLatWebAll = np.append(meanLatWebAll, meanLatWebStereo) 
    meanLonWebAll = np.append(meanLonWebG16, meanLonWebG17) 
    meanLonWebAll = np.append(meanLonWebAll, meanLonWebStereo) 

    meanLatWebG16All = np.append(meanLatWebG16, meanLatWebStereo)
    meanLonWebG16All = np.append(meanLonWebG16, meanLonWebStereo)
    meanLatWebG17All = np.append(meanLatWebG17, meanLatWebStereo)
    meanLonWebG17All = np.append(meanLonWebG17, meanLonWebStereo)

    globePlotter = pGlobe.GlobePlotter(PROJ_LIB)
    globePlotter.plot_density_on_globe(meanLatWebG16All, meanLonWebG16All, sigmaThreshold=sigmaThreshold, label='G16 Pipeline')
    globePlotter.plot_title_and_legend('Density Distribution of G16 Website Bolides')

    globePlotter = pGlobe.GlobePlotter(PROJ_LIB)
    globePlotter.plot_density_on_globe(meanLatWebG17All, meanLonWebG17All, sigmaThreshold=sigmaThreshold, label='G17 Pipeline')
    globePlotter.plot_title_and_legend('Density Distribution of G17 Website Bolides')

    globePlotter = pGlobe.GlobePlotter(PROJ_LIB)
    globePlotter.plot_density_on_globe(meanLatWebAll, meanLonWebAll, sigmaThreshold=sigmaThreshold, label='All Website Bolides')
    globePlotter.plot_title_and_legend('Density Distribution of All Website Bolides')

#*************************************************************************************************************
# histporam_of_website_bolides
# 
# Generates Histogram of all bolides on website.
#
# Inputs:
#   bolidesFromWebsite  -- [WebsiteBolideEvent list] created by:
#       bolide_dispositions.pull_dispositions_from_website 
#   PROJ_LIB            -- [str] Basemap requires the environment variable PROJ_LIB to be properly set
#                                   TODO: figure out a more robust way to do this.
#   startDate           -- [str] The starting date for analysis
#                           Is ISO format: 'YYYY-MM-DD'
#                           None or '' means no start date
#   endDate             -- [str] The ending date for analysis
#                           Is ISO format: 'YYYY-MM-DD'
#                           None or '' means no end date
#
#*************************************************************************************************************
def histogram_of_website_bolides (bolidesFromWebsite, PROJ_LIB=None, startDate=None, endDate=None):

    raise Exception('Set up for G18 and G19')

    assert PROJ_LIB is not None, 'PROJ_LIB path must be passed'

    #**************************
    # Only use data from selected days
    bolidesFromWebsite = bDisp.select_bolidesFromWebsite_from_dates (bolidesFromWebsite,
            startDate=startDate, endDate=endDate)

    webBolidesG16Idx = [idx for [idx, bolide] in enumerate(bolidesFromWebsite) if 
            np.any(np.in1d('G16', bolide.satellite))]
    webBolidesG17Idx = [idx for [idx, bolide] in enumerate(bolidesFromWebsite) if 
            np.any(np.in1d('G17', bolide.satellite))]


    #**************************
    # Bolides as a function of time, histogram

    histFig, ax = plt.subplots(2,1, figsize=(30, 15))
    histFig.set_figwidth(16.0)
    histFig.set_figheight(9.0)
    locator1 = mdates.AutoDateLocator()

    # Have one bin for each day
    # Count number of days in data set
    numberOfDays = (bolidesFromWebsite[0].timeRange[0][0] - bolidesFromWebsite[-1].timeRange[0][0]).days
    # One day per bin, but at least 100 bins
    timeBins = np.max([100, numberOfDays])

    mdateTimeOriginWebG16 = \
        mdates.date2num([bolide.timeRange[bolide.satellite.index('G16')][0] for bolide in 
            [bolidesFromWebsite[i] for i in webBolidesG16Idx]])
    mdateTimeOriginWebG17 = \
        mdates.date2num([bolide.timeRange[bolide.satellite.index('G17')][0] for bolide in
            [bolidesFromWebsite[i] for i in webBolidesG17Idx]])

    fontsize=20
    minBin = np.min([np.min(mdateTimeOriginWebG16), np.min(mdateTimeOriginWebG17)])
    maxBin = np.max([np.max(mdateTimeOriginWebG16), np.max(mdateTimeOriginWebG17)])
    ax[0].hist(mdateTimeOriginWebG16,  bins=timeBins, range=(minBin, maxBin), log=False, facecolor='b', alpha=1.0, label='G16 Website')
    ax[0].grid(axis='y')
   #ax[0].legend()
    ax[0].xaxis.set_major_locator(locator1)
    ax[0].xaxis.set_major_formatter(mdates.AutoDateFormatter(locator1))
    plt.axes(ax[0])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax[0].set_xlabel('Date of Event', fontsize=fontsize)
    ax[0].set_title('Goes-16 Distribution of All Website Bolides in Time', fontsize=fontsize+4)
    
    ax[1].hist(mdateTimeOriginWebG17,  bins=timeBins, range=(minBin, maxBin), log=False, facecolor='r', alpha=1.0, label='G17 Website')
    ax[1].grid(axis='y')
   #ax[1].legend()
    ax[1].xaxis.set_major_locator(locator1)
    ax[1].xaxis.set_major_formatter(mdates.AutoDateFormatter(locator1))
    plt.axes(ax[1])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax[1].set_xlabel('Date of Event', fontsize=fontsize)
    ax[1].set_title('Goes-17 Distribution of All Website Bolides in Time', fontsize=fontsize+4)
    plt.tight_layout(pad=0.7)

