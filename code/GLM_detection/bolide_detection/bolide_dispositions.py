#******************************************************************************
#
# Dispositions Methods
# 
# The following are used to disposition bolide detections (I.e. classify them)

import sys
import os
import warnings
import shutil
import argparse
import datetime
import pickle
import copy
import csv

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from sklearn.metrics import roc_curve
#from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
#from scipy.stats import multivariate_normal
from PIL import Image

import bolide_detections as bd
import io_utilities as ioUtil
import bolide_features as bFeatures
from gen_goes_cutout import cutoutConfiguration, cutoutOutputClass
import bolide_database as bolideDB
from time_utilities import convert_time_to_local
from lat_lon_to_GLM_pixel import LatLon2Pix
import bolide_detection_performance as bDetPerform
import geometry_utilities as geoUtil

from bolides.plotting import plot_density, plot_scatter, generate_plot

#******************************************************************************
class humanDispositionError(Exception):
    pass

class HumanOpinion:
    '''
   
    This holds human dispositions of the bolide detections.
   
    Not all fields must be populated. This class is used for multiple purposed, using different fields.
   
    INSTANCE ATTRIBUTES
      ID        : np.int64
            Detection candidate ID
      belief    : float
            A real-valued measure of belief on the interval [0,1] in the
            hypothesis "This detection is a bolide signature."  
      name      : str
            The name or ID of the person issuing the opinion.
      expertise : flaot
            A subjective assessment of the opinion-giver's expertise on the interval [0,1]
      time      : datetime.dateimte
            An optional time stamp indicating when the opinion was issued.
      comments  : str
            A string containing optional notes regarding the detection.
      disposition : str
            The human opinion ('TP', 'TN', 'FP', 'FN')
 
    '''

    # 'UNC' := uncertain
   #valid_dispositions = ('TP', 'TN', 'FP', 'FN', 'UNC')

    def __init__(self, 
            ID=None, 
            belief=0.0, 
            name=None, 
            expertise=None, 
            time=None, 
            comments='', 
            valid_dispositions=('TP', 'TN', 'FP', 'FN', 'UNC')):

        self.ID         = ID
        self.belief     = belief
        self.name       = name
        self.expertise  = expertise
        self.time       = time
        self.comments   = comments

        self.valid_dispositions = valid_dispositions

        self._disposition = None

    @property
    def disposition(self):
        ''' This is the human opinion disposition ('TP', 'TN', 'FP', 'FN')
        '''
        return self._disposition

    @disposition.setter
    def disposition(self, opinion : str):
        ''' Set the disposition to human opinion

        Parameters
        ----------
        opinion : str
            The user disposition opinion ('TP', 'TN', 'FP', 'FN')

        '''

        opinion = opinion.upper()

        # Check if opinion is valid
        if opinion in self.valid_dispositions:
            self._disposition = opinion
        else:
            raise humanDispositionError('Invalid opinion')


    # Returns a summary of the attributes in the class via a dict
    def __repr__(self): 
        return ioUtil.print_dictionary(self)

# *****************************************************************************
class MachineOpinion:
    """
    This holds computer dispositions opinions of the bolide candidates.
   
    ATTRIBUTES
    ----------
    bolideBelief : float [0.0, 1.0]
        A measure of belief score on the interval [0,1] in the hypothesis "This detection is a bolide signature."
    glintBelief : float [0.0, 1.0] 
        A real-valued measure of belief score on the interval [0,1] in the hypothesis "This detection is a glint signature."
    method : str
        The classification method issuing opinion. 
        {e.g. 'SVM-RBF', 'logisticRegression', 'random-forest', etc...}
    source : str
        What step in the detection pipeline related to this opinion
        {e.g. 'triage', 'validation'}
    ID : np.int64
        Detection candidate ID
    comments : str 
        Optional notes regarding the detection opinion
    """
    def __init__(self, bolideBelief=None, glintBelief=None, method=None, source=None, time=None, ID=None, comments=None) :
        self.bolideBelief   = bolideBelief
        self.glintBelief    = glintBelief
        self.method         = method
        self.source         = source
        self.time           = time
        self.ID             = ID
        self.comments       = comments

    # Returns a summary of the attributes in the class via a dict
    def __repr__(self): 
        return ioUtil.print_dictionary(self)

# *****************************************************************************
# bolideBeliefSwitcher is to have a list of belief options
# It's to be able to have a switch-like mechanism in Python
bolideBeliefSwitcher = {
                        'human': 1,
                        'triage': 2,
                        'validation': 3,
                        'unknown': 1001
                        }

# *****************************************************************************
# WebsiteBolideEvent
#
# Contains information extracted from the website detailing a detected bolide.
#
# INSTANCE ATTRIBUTES
#   detectedBy  -- 'Goes-16/17' means a human has looked at the GLM L2 data and can see the bolide in the data
#   howFound    -- 'algorithm' means the pipeline detected it
#   confidence  -- 
#   eventId -- This is a MongoDB id from the website database, not from the netCDF files
#   satellite   -- [str list]
#   netCDFFilename  -- one file for each satellite  
#   timeRange   -- [datetime list]
#   peakTime    -- [datetime list]
#   peakLat     -- [float list]
#   peakLon     -- [float list]
#   peakEnergy  -- [float list]
#   latRange    -- [float list]
#   lonRange    -- [float list]
#   totEnergy   -- [float]
#
# *****************************************************************************
class WebsiteBolideEvent:

    
    # *****************************************************************************
    # Constructor:
    # 
    # Inputs:
    #   bolideEventFromWebsite  -- [bolides.bolide.Bolide] from Geert and Clements tool
    #
    # *****************************************************************************
    def __init__ (self, bolideEventFromWebsite):

        self.detectedBy = bolideEventFromWebsite.detectedBy
        self.howFound   = bolideEventFromWebsite.howFound
        self.confidence = bolideEventFromWebsite.confidenceRating
        self.eventId    = bolideEventFromWebsite._id
        self.netCDFFilename    = bolideEventFromWebsite.netCDFFilename
        
        # Event can be detected by any satellite, including all
        try: 
            for satellite in bolideEventFromWebsite.satellite:
                bd.validSatellites.index(satellite)
            self.satellite = bolideEventFromWebsite.satellite
        except:
            raise Exception('Unknown goesSatellite')
        self.timeRange = []
        self.latRange  = []
        self.lonRange  = []
        self.totEnergy = []
        self.peakEnergy = []
        self.peakTime = []
        self.peakLat = []
        self.peakLon = []
        for idx in range(bolideEventFromWebsite.nSatellites):

            # Time is in unix time milliseconds, so divide by 1000 to get to unix time in seconds
            unixTime = np.transpose(bolideEventFromWebsite.times[idx]) / 1000.0
            minTime = np.min(unixTime)
            maxTime = np.max(unixTime)
            self.timeRange.append([datetime.datetime.utcfromtimestamp(minTime), 
                                datetime.datetime.utcfromtimestamp(maxTime)])
            # Values at peak energy
            peakIdx = np.argmax(bolideEventFromWebsite.energies[idx])
            self.peakEnergy.append(bolideEventFromWebsite.energies[idx][peakIdx])
            self.peakTime.append(datetime.datetime.utcfromtimestamp(unixTime[peakIdx]))
            self.peakLat.append(bolideEventFromWebsite.latitudes[idx][peakIdx])
            self.peakLon.append(bolideEventFromWebsite.longitudes[idx][peakIdx])
            
            # Lat/Lon
            self.latRange.append([np.min(bolideEventFromWebsite.latitudes[idx]), 
                                    np.max(bolideEventFromWebsite.latitudes[idx])])
            self.lonRange.append([np.min(bolideEventFromWebsite.longitudes[idx]), 
                                    np.max(bolideEventFromWebsite.longitudes[idx])])

            # Energy
            self.totEnergy.append(np.nansum(bolideEventFromWebsite.energies[idx]))


    #******************************************************************************
    # Write a header for the CSV file 
    def write_csv_header(self, csvFileName):

        with open(csvFileName, 'w') as csvfile:
            fieldnames = ['# UTC_POSIX_timestamp_at_Energy_Peak', 'Satellite', 'Peak_Energy_[J]', 'Latitude_[degrees]', 'Longitude_[degrees]']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    #******************************************************************************
    # Append a Website detection to a CSV file.
    def append_csv(self, csvFileName):
    
        # If Stereo then average the numbers
        if len(self.satellite) > 1:
            sat = 'stereo'
            peakTime = np.mean([t.timestamp() for t in self.peakTime])
            peakEnergy = np.mean([e for e in self.peakEnergy])
            peakLat = np.mean([lat for lat in self.peakLat])
            peakLon = np.mean([lon for lon in self.peakLon])

            row = [peakTime] + [sat] + [peakEnergy] + [peakLat] + [peakLon]

        else:
           #for idx,sat in enumerate(self.satellite):
            idx = 0
            row = [self.peakTime[idx].timestamp()] + [self.satellite[idx]] + [self.peakEnergy[idx]] + \
                    [self.peakLat[idx]] + [self.peakLon[idx]]
            
        try:
            # If the file exists, append the record.
            with open(csvFileName, 'a') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(row)
        except:
            sys.exit('Could not write to file {}.'.format(csvFileName))



        return 0

    

    def __repr__(self):
        return ioUtil.print_dictionary(self)

# ************************************************************************************************************
# pull_dispositions_from_website (file_path)
#
# Function to pull dispositions from the website: https://neo-bolide.ndc.nasa.gov
#
# Inputs:
#   file_path -- [str] if not empty then save data to this file
#
# Outputs:
#   bolidesFromWebsite  -- [WebsiteBolideEvent list]
#
# ************************************************************************************************************

def pull_dispositions_from_website (file_path=[]):

    # Use the tool Geert and Clemens wrote to pull the data: https://github.com/barentsen/bolides
    import bolides
    bl=bolides.BolideList()
    #bl.to_pandas()

    # Cycle through each bolide detection and create a WebsiteBolideEvent object
    bolidesFromWebsite = []
    for bolideEvent in tqdm(bl, 'Extracting bolide dispositions from website...'):
        
        bolidesFromWebsite.append(WebsiteBolideEvent(bolideEvent))

    # Pickle the data
    if (file_path != []):
        try:
            with open(file_path, 'wb') as fp:
                pickle.dump(bolidesFromWebsite, fp)
        except:
            raise Exception('Could not write to file {}.'.format(file_path))
        fp.close()


    return bolidesFromWebsite

# *****************************************************************************
# select_bolidesFromWebsite_from_dates ()
#
# Selects a subset of bolides in bolideFromWebsite based on selected dates
#
# Inputs:
#   bolidesFromWebsite  -- [WebsiteBolideEvent list] created by:
#       bolide_dispositions.pull_dispositions_from_website 
#   startDate           -- [str] The starting date for bolides to keep
#                           Is ISO format: 'YYYY-MM-DD'
#                           None or '' means no start date
#   endDate             -- [str] The ending date for bolides to keep
#                           Is ISO format: 'YYYY-MM-DD'
#                           None or '' means no end date
# Outputs:
#   bolidesFromWebsiteReduced -- [list] of BolideDispositionProfile objects only within selected dates
#
# *****************************************************************************
def select_bolidesFromWebsite_from_dates (bolidesFromWebsite, startDate=None, endDate=None):

    # Only use data from selected days
    if startDate == '' or startDate is None:
        startDate = datetime.date.min
    else:
        startDate = datetime.date.fromisoformat(startDate)

    if endDate == '' or endDate is None:
        endDate = datetime.date.max
    else:
        endDate = datetime.date.fromisoformat(endDate)

    # Reduce website data to only days desired
    bolidesToKeep = []
    for idx, bolide in enumerate(bolidesFromWebsite):
        # Each bolidesFromWebsite event can be from any satellite. In any case, they better damn well
        # correspond to the same event in time. So, only look at the first item in each bolide event
        if bolide.timeRange[0][0].date() >= startDate and bolide.timeRange[0][1].date() <= endDate:
            bolidesToKeep.append(idx)
    return [bolidesFromWebsite[i] for i in bolidesToKeep]


# *****************************************************************************
# BolideDispositionProfile
# ----------------------
# 
# Contains a bolide detection object and disposition objects.
#
# One can directly pass in the BolideDetection objects or associate this profile with a bolideDatabase to save on
# memory.
#
# INSTANCE ATTRIBUTES
#   ID              : Bolide Detection ID
#   bolideDetection : A bolide_detections.BolideDetection object. (OPTIONAL)
#   machineOpinions : A list of MachineOpinion objects.
#   humanOpinions   : A list of HumanOpinion objects.
#   features        : [bolide_features.FeaturesClass] An object of features computed for this disposition (initialized to [])
#
# *****************************************************************************
class BolideDispositionProfile:

    #*************************************************************************************************************
    # Constructor
    #
    # If BolideDetection is passed then compute extra stored fields
    # Otherwise, they need to remain None. Compute them in bolide_features.compute_bolide_features
    #
    # Inputs:
    #   ID                  -- Bolide Detection ID
    #   detectionFlag       -- [bool] If this is a detection or rejection
    #   bolideDetection     -- bolide_detections.BolideDetection object containing the bolide detection information
    #                           OPTIONAL: if not passed then instead associate this object with a bolideDatabase
    #                                       We do this to save memory.
    #   
    #   machineOpinions     -- MachineOpinion object list
    #   humanOpinions       -- HumanOpinion object list
    #   features            -- [bolide_features.FeaturesClass] An object of features computed for this disposition
    #                           Initialized to an empty FeatureClass if not passed.
    #   stereoFeatures      -- [bolide_features.StereoFeatureClass] Features computed for stereo detection events
    #   cutoutFeatures      --  [cutoutOutputClass] Contains the information generated by the cutout tool
    #
    #*************************************************************************************************************
    def __init__(self, ID, detectionFlag=None, bolideDetection=None, machineOpinions=None, humanOpinions=None,
            features=None, stereoFeatures=None, cutoutFeatures=None):
        
        if (bolideDetection is not None and ID != bolideDetection.ID):
            raise Exception('ID does not match bolideDetection.ID')

        self.ID = ID
        self.detectionFlag   = detectionFlag
        self.bolideDetection = bolideDetection
        self.machineOpinions = machineOpinions
        self.humanOpinions   = humanOpinions
        if (features is not None):
            self.features        = features
        else:
            # Empty features, we fill it in when we generate a BolideDispositions object
            self.features        = bFeatures.FeaturesClass()

        if (stereoFeatures is not None):
            self.stereoFeatures        = stereoFeatures
        else:
            self.stereoFeatures        = bFeatures.StereoFeaturesClass()

        if (cutoutFeatures is not None):
            self.cutoutFeatures        = cutoutFeatures
        else:
            self.cutoutFeatures        = cutoutOutputClass(cutoutConfiguration(), ID)

    #*************************************************************************************************************
    # Add a BolideDispositionProfile into this profile
    # This can be used, for example, to combine two bolide detections that correspond to the same event.
    # Use the ID from the initial detection
    # Also: Combine the features as best we can, given what information we have
    def add_profile(self, otherBolideDispositionProfile):

        # Just append any human or machine opinions
        if (self.machineOpinions is not None and otherBolideDispositionProfile.machineOpinions is not None):
            self.machineOpinions.extend(otherBolideDispositionProfile.machineOpinions)
        if (self.humanOpinions is not None and otherBolideDispositionProfile.humanOpinions is not None):
            self.humanOpinions.extend(otherBolideDispositionProfile.humanOpinions)

        # Now for the bolideDetection
        # If using a bolide database then bolideDetection is empty and all we can do is try to combine the features as
        # best we can
        if (otherBolideDispositionProfile.bolideDetection is not None):
            self.bolideDetection.add_data(otherBolideDispositionProfile.bolideDetection)
            raise Exception('Need to recompute features with the larger profile')
            self.goesSatellite  = otherBolideDispositionProfile.bolideDetection.features.goesSatellite
            self.bolideTime     = otherBolideDispositionProfile.bolideDetection.features.bolideTime
            self.bolideMidRange = otherBolideDispositionProfile.bolideDetection.features.bolideMidRange
            self.avgLatLon      = [otherBolideDispositionProfile.bolideDetection.features.avgLat, 
                    otherBolideDispositionProfile.bolideDetection.features.avgLon]
            self.features = bFeatures.FeaturesClass()
        else:
            # Combine the features as best we can, given the bolideDetection groups are not present
            self.features.combine_features(otherBolideDispositionProfile.features)


    #*************************************************************************************************************
    # Returns a summary of the attributes in the class via a dict
    # Add newline on either side of the string
    def __repr__(self): 

        classDict = self.__dict__.copy()

        print_str = ' \n {}'.format(classDict)

        # Add Newline between each key/value pair
        loc = 0
        while True:
            loc = print_str.find(", '", loc+1)
            if (loc == -1):
                break
            print_str = print_str[:loc] + '\n' + print_str[loc+1:]

        print_str = print_str + '\n'
            
        return print_str

#******************************************************************************
# generate_bolideDispositionProfileList_from_bolide_detections ()
#
# Use this function to generate a bolideDispositionProfileList of BolideDispositionProfile objects
# For use to construct a BolideDispositions object (see bolide_dispositions).
#
# Inputs:
#   bolideDetectionsFilename -- [str] path to file containing the bolide_detections.fs or bolide_detections.p 
#                                   file generated by the pipeline (detect_bolides.py)
#   humanVetter     -- [str] Name of human who vetted the data (if known)
#   dataType        -- [str] extract 'detections' or 'rejections'
#
# Outputs:
#   bolideDispositionProfileList -- [list] of BolideDispositionProfile objects
#
#******************************************************************************
def generate_bolideDispositionProfileList_from_bolide_detections (bolideDetectionsFilename, 
        humanVetter=None, dataType='detections'):

    raise Exception('This function is old, please use BolideDispositions.from_bolideDatabase or .from_bolideDispositionProfileList')

    # Extract the data
    if (os.path.splitext(bolideDetectionsFilename)[1] == '.p'): 
        bolideDetectionList = bd.unpickle_bolide_detections(bolideDetectionsFilename)
    elif (os.path.splitext(bolideDetectionsFilename)[1] == '.fs'): 
        bolideDatabase = bolideDB.BolideDatabase(bolideDetectionsFilename, purgeDatabase=False)
        bolideDetectionList = bolideDatabase.extract_bolideDetectionList(dataType=dataType)
    else:
        raise Exception('Unknown <inFiles> bolideDetectionsFilename extension')


    bolideDispositionProfileList = []

    # Convert to bolide dispositions
    for detection in bolideDetectionList:
        bolideDispositionProfileList.append(BolideDispositionProfile(detection.ID, detectionFlag=True,
            bolideDetection=detection,
            humanOpinions=[HumanOpinion(belief=detection.confidence, name=humanVetter, time=None,
                comments=detection.howFound)]))
        
    return bolideDispositionProfileList 

#******************************************************************************
# select_bolideDispositionProfileList_from_dates ()
#
# Selects a subset of bolides in bolideDispositionProfileList based on selected dates
#
# Inputs:
#   bolideDispositionProfileList -- [list] of BolideDispositionProfile objects created by:
#       bolide_dispositions.generate_bolideDispositionProfileList_from_bolide_detections
#       Note: can also use list of BolideDetection
#   startDate           -- [str] The starting date and time for bolides to keep
#                           Is ISO format: 'YYYY-MM-DD[*HH[:MM[:SS[.fff[fff]]]]]'
#                           None or '' means no start date
#   endDate             -- [str] The ending date and time for bolides to keep
#                           Is ISO format: 'YYYY-MM-DD[*HH[:MM[:SS[.fff[fff]]]]]'
#                           None or '' means no end date
# Outputs:
#   bolideDispositionProfileListReduced -- [list] of BolideDispositionProfile objects only within selected dates
#   bolidesToKeep       -- [list] indices of bolides to keep
#
#******************************************************************************
def select_bolideDispositionProfileList_from_dates (bolideDispositionProfileList, startDate=None, endDate=None):

    # Only use data from selected days
    if startDate == '' or startDate is None:
        startDate = datetime.datetime.min
    else:
        startDate = datetime.datetime.fromisoformat(startDate)

    if endDate == '' or endDate is None:
        endDate = datetime.datetime.max
    else:
        endDate = datetime.datetime.fromisoformat(endDate)

    bolidesToKeep = []
    for idx, bolide in enumerate(bolideDispositionProfileList):
        if bolide.features.bolideTime >= startDate and bolide.features.bolideTime <= endDate:
            bolidesToKeep.append(idx)
    return [bolideDispositionProfileList[i] for i in bolidesToKeep], bolidesToKeep

# *****************************************************************************
# intersect_days_bolideDispositionProfileList_and_bolidesFromWebsite
#
# Finds the intersection of days with events from both bolideDispositionProfileList and bolidesFromWebsite
#
# This can be used, for example, to compute correct statistics for only when both data is available and the pipeline was
# run.
#
# Inputs:
#   bolideDispositionProfileList -- [list] of BolideDispositionProfile objects created by:
#       bolide_dispositions.generate_bolideDispositionProfileList_from_bolide_detections
#   bolidesFromWebsite  -- [WebsiteBolideEvent list] created by:
#       bolide_dispositions.pull_dispositions_from_website 
#   satellite           -- [str] One of bolide_detections.validSatellites or 'all' or ['east', 'west']
#   
# Outputs:
#   bolideDispositionProfileListReduced -- [list] of BolideDispositionProfile objects only within selected dates
#   bolidesFromWebsiteReduced           -- [list] of BolideDispositionProfile objects only within selected dates
# ************************************************************************************************************
def intersect_days_bolideDispositionProfileList_and_bolidesFromWebsite(bolideDispositionProfileList,
        bolidesFromWebsite, satellite='all'):

    satelliteOptions = bd.get_satellite_options(satellite)

    # Find all dates in website data
    websiteDates = []
    for bolide in bolidesFromWebsite:
        if np.any(np.isin(bolide.satellite, satelliteOptions) > 0):
            # Assume no bolides will span more than one day
            # Search list of days. If bolide date is not in list then append to list
            try:
                websiteDates.index(bolide.timeRange[0][0].date())
            except:
                websiteDates.append(bolide.timeRange[0][0].date())

    # Find all dates in pipeline data
    pipelineDates = []
    for bolide in bolideDispositionProfileList:
        # Assume no bolides will span more than one day
        # Search list of days. If bolide date is not in list then append to list
        try:
            pipelineDates.index(bolide.features.bolideTime.date())
        except:
            pipelineDates.append(bolide.features.bolideTime.date())

    #***
    # Reduce website data to only days desired
    bolidesToKeep = []
    for idx, bolide in enumerate(bolidesFromWebsite):
        # Each bolidesFromWebsite event can be from any satellite. In any case, they better damn well
        # correspond to the same event in time. So, only look at the first item in each bolide event.
        try:
            websiteDates.index(bolide.timeRange[0][0].date())
            pipelineDates.index(bolide.timeRange[0][0].date())
            bolidesToKeep.append(idx)
        except:
            pass

    bolidesFromWebsiteReduced = [bolidesFromWebsite[i] for i in bolidesToKeep]

    #***
    # Reduce pipeline data to only days desired
    bolidesToKeep = []
    for idx, bolide in enumerate(bolideDispositionProfileList):
        try:
            websiteDates.index(bolide.features.bolideTime.date())
            pipelineDates.index(bolide.features.bolideTime.date())
            bolidesToKeep.append(idx)
        except:
            pass

    bolideDispositionProfileListReduced = [bolideDispositionProfileList[i] for i in bolidesToKeep]

    return [bolideDispositionProfileListReduced, bolidesFromWebsiteReduced]
    

#*************************************************************************************************************
#*************************************************************************************************************
#*************************************************************************************************************
# class BolideDispositions

def create_BolideDispositions_from_multiple_database_files(database_files, copyOverBolideDetection=False, 
        beliefSource='triage', verbosity=True, useRamDisk=False):
    """ Constructs a BolideDispositions object from multiple ZODB databse files

    This right now does not compute extra features. Will we ever do that again?

    This also will close the ZODB database after loading the data.

    Parameters
    ----------
    database_files : str or list of str
        Path to the database file, or files, to load data from
    copyOverBolideDetection : bool
        If True then copy over the bolideDisposition.bolideDetection from the database to the
        bolideDispositionProfileList. This will significantly increase the memory usage of the object
    verbosity : bool
    useRamDisk : bool
        If true then temporarily store the bolid database in a RAM disk for speed

    Returns
    -------
    bDispObj : BolideDispositions
        Bolide Dispositions object withh all database files loaded

    """

    # Generate the bolide dispositions object
    # If passing in multiple database files then combine all databases
    if len(database_files) == 1:
        bDispObj = BolideDispositions.from_bolideDatabase(database_files[0], spice_kernel_path='', verbosity=verbosity,
            compute_features=False, useRamDisk=useRamDisk, beliefSource=beliefSource, copyOverBolideDetection=copyOverBolideDetection)
    else:
        bDispObj = BolideDispositions.from_bolideDatabase(database_files[0], spice_kernel_path='', verbosity=verbosity,
            compute_features=False, useRamDisk=useRamDisk, beliefSource=beliefSource, copyOverBolideDetection=copyOverBolideDetection)
        bDispObj.bolideDatabase.close()
        for database_file in database_files[1:]:
            bDispObj = BolideDispositions.from_bolideDatabase(database_file,
                extra_bolideDispositionProfileList=bDispObj.bolideDispositionProfileList, spice_kernel_path='', verbosity=verbosity,
                compute_features=False, useRamDisk=useRamDisk, beliefSource=beliefSource, copyOverBolideDetection=copyOverBolideDetection)
            bDispObj.bolideDatabase.close()

    return bDispObj

#*************************************************************************************************************
# BolideDispositions
#
# A class to handle the dispositions of a set of bolide detections. The main attribute is a bolideDatabase and 
# a list of BolideDispositionProfiles. 
#
# The object initialization input, bolideDispositionProfileList, can be generated with the stand alone function 
# generate_bolideDispositionProfileList_from_bolide_detections 
#
# INSTANCE ATTRIBUTES
#   bolideDatabase          -- [BolideDatabase] the database of detected and rejected bolides
#   database_path           -- [str] path to the database file
#   bolideDispositionProfileList -- [list] of BolideDispositionProfile objects
#   featuresToUse           -- [str list] Features used in classifier
#   columnTransformer       -- [sklearn ColumnTransformer object] The fit transformer used to normalize the features
#   hotSpots                -- [HotSpots] The found hot spots via find_bolide_hot_spots
#
#   HELPER Attributes
#   n_tot_entries           -- [int] Number of detections in bolideDispositionProfileList
#   featuresComputed        -- [bool] True if bolide_features.compute_bolide_features was called
#   spice_kernel_path       -- [str] Path to kernels to load for glintPoint object
#   multiProcessEnabled     -- [bool] If True then all multiprocessing
#   debugMode               -- [bool] If true then sets chunkSize=1000 and inly compute 2000 detectison total (I.e. runs fast)
#   
#
#*************************************************************************************************************
class BolideDispositions:

    #*************************************************************************************************************
    # Constructor
    # 
    # This is the bare constructor method. It is not intended to be called on its own, but only through the two
    # classmethods:
    # 1) from_bolideDatabase -- If using a ZODB database
    #
    # 2) from_bolideDispositionProfileList -- If using a simple bolideDispositionProfile list to contain the bolide
    #                                           detections
    #                                         ...Or for loading in an already cimpute BolideDispositions object
    #
    # Inputs:
    #   spice_kernel_path   -- [str] Path to kernels to load for glintPoint object
    #                               Only needed if computing features.
    #   verbosity           -- [bool] If True then print processing status statements
    #   compute_features    -- [bool] If to recompute the features. 
    #                           This is slow, so when loading in a saved BolideDispositions object use the stored features
    #   glmGroupsWithNeighbors -- [list of bolide_detection.GlmGroup] 
    #                   A list of objects containing ALL the groups from the netCDF data file associated with the
    #                   dispositions along with neighboring files, sorted by time.
    #                   If None, then this list is generated from the netCDF files listed in each detection
    #   multiProcessEnabled -- [bool] If true then use multiprocessing when computing features
    #   debugMode           -- [bool] If True then test with a very short list of dispositions
    #   lookup_table_path           -- [str] Path to the lookup table file.
    #   lookup_table_path_inverted  -- [str] Path to the lookup table file for the inverted yaw orientation
    #                                           Only relevent for G17.
    #   hotSpots            -- [HotSpots class] List of hot spots to avoid
    #   startDate           -- [str] The starting date and time for bolides to keep
    #                           ISO format: 'YYYY-MM-DD[*HH[:MM[:SS[.fff[fff]]]]]'
    #                           None or '' means no start date
    #   endDate             -- [str] The ending date and time for bolides to keep
    #                           ISO format: 'YYYY-MM-DD[*HH[:MM[:SS[.fff[fff]]]]]'
    #                           None or '' means no end date
    #   
    #*************************************************************************************************************
    def __init__(self, spice_kernel_path=None, verbosity=False, compute_features=False, glmGroupsWithNeighbors=None,
            multiProcessEnabled=True, debugMode=False, lookup_table_path=None, lookup_table_path_inverted=None,
            hotSpots=None, startDate=None, endDate=None):

        assert hasattr(self, 'bolideDispositionProfileList'), 'Do not call this constructor on its own, use one of the classmethods'

        self.multiProcessEnabled = multiProcessEnabled
        self.debugMode = debugMode

        self.hotSpots = hotSpots

        if (compute_features and (spice_kernel_path is None or not os.path.isdir(spice_kernel_path))):
            print('spice_kernel_path unavailable, glint feature disabled')
            self.spice_kernel_path = None
        else:
            self.spice_kernel_path = spice_kernel_path

        #************
        # Truncate bolideDispositionProfileList to specified start and end dates
        if startDate is not None or endDate is not None:
            [self.bolideDispositionProfileList, bolidesToKeep] = select_bolideDispositionProfileList_from_dates (self.bolideDispositionProfileList,
                startDate=startDate, endDate=endDate)

        #************
        if (compute_features):
            # Compute the features
            # If using a database, the bolideDetection objects were not copied to self.bolideDispositionProfileList to save memory.
            # So, this should be all None
            bolideDetectionList = [disposition.bolideDetection for disposition in self.bolideDispositionProfileList]
            bolideDetectionList = bFeatures.compute_bolide_features(bolideDetectionList, 
                    bolideDatabase=self.bolideDatabase, glmGroupsWithNeighbors=glmGroupsWithNeighbors, 
                    multiProcessEnabled=self.multiProcessEnabled, spice_kernel_path=self.spice_kernel_path,
                    hotSpots=hotSpots,
                    verbosity=verbosity, debugMode=self.debugMode)
            # Point the disposition-level features to the detection-level just computed features
            for idx, detection in enumerate(bolideDetectionList):
                self.bolideDispositionProfileList[idx].features = detection.features
            
            # Generate the machine opinion in the disposition profile, if not yet existant
            # This would happen if we are using a bolideDatabase
            for disposition in self.bolideDispositionProfileList:
                if (disposition.machineOpinions is None):
                    if disposition.bolideDetection.assessment is None:
                        disposition.machineOpinions = None
                    else:
                        disposition.machineOpinions = \
                            [MachineOpinion(
                                bolideBelief=copy.copy(disposition.bolideDetection.assessment.triage.score),
                                method=copy.copy(disposition.bolideDetection.assessment.triage.method),
                                source='triage',
                                comments=copy.copy(disposition.bolideDetection.howFound)
                                )]


        else:
            # Check if the features are properly in the database
            if np.all([np.isnan(profile.features.nGroups) for profile in self.bolideDispositionProfileList]):
                raise Exception('Bolide features do not appear to already be computed.')
        self.featuresComputed = True

        #***
        #************
        # Cleanup
        # Remove bolideDispositionProfileList entries that are not valid
        # TODO: Identify and elliminate cases where entries are not valid
        stepSize = 500
        if (verbosity):
            pbar = tqdm(total=len(self.bolideDispositionProfileList), desc='Cleaning bolideDispositionProfileList of invalid entries')
        
        datumsToRemove = []
        for idx, record in enumerate(self.bolideDispositionProfileList):
            if (
                record.features.bolideTime is None or 
                record.features.bolideTime is np.nan or 
                (record.machineOpinions is None and record.humanOpinions is None) or
                record.features.nGroups is np.nan
                ):
                    datumsToRemove.append(idx)
        
            if (verbosity and np.mod(idx, stepSize) == 0):
                pbar.update(stepSize)
        
        if (verbosity & (len(datumsToRemove) > 0)):
            pbar.close()
            print('******')
            print('******')
            print('******')
            print('Number of invalid bolideDispositionProfiles to be removed: {}'.format(len(datumsToRemove)))
            print('******')
            print('******')
            print('******')
            pbar = tqdm(total=len(datumsToRemove), desc='removing bolideDispositionProfileList invalid entries')
        
        if (len(datumsToRemove) > 0):
            for datumIdx in sorted(datumsToRemove, reverse=True):
                del self.bolideDispositionProfileList[datumIdx]
                if (verbosity):
                    pbar.update()
        if (verbosity):
            pbar.close()
        #***

        self.featuresToUse = None
        self.columnTransformer = None


    #*************************************************************************************************************
    # Constructor
    # classmethod from_bolideDatabase  
    #
    # This classmethod constructor will take the path to the bolide databases and construct a BolideDispositions object.
    # This database is only used in the generation of the bolideDispositionProfileList then the actual constructor
    # method above is used for the rest.
    #
    # If the optional bolideDispositionProfileList is passed then these will be appended to the 
    # self.bolideDispositionProfileList. This is used, for example, to add a large list of rejectison to the object.
    #
    # We are pulling this data from the database, that means the dispositions are machine opinions. In the
    # bolideDispositionProfileList set the machineOpinions to the disposition values.
    #
    # Computing features can be slow.
    #
    # Inputs:
    #   database_filename   -- [list of str] Path to the bolide databases to include in object
    #   extra_bolideDispositionProfileList -- [list] of BolideDispositionProfile objects
    #                                   Or file path [str] to pickled bolideDispositionProfileList 
    #   spice_kernel_path   -- [str] Path to kernels to load for glintPoint object
    #                               Only needed if computing features.
    #   verbosity           -- [bool] If True then print processing status statements
    #   humanVetter         -- [str] Name of human who vetted the data (if known)
    #   useRamDisk          -- [bool] If true then temporarily store the bolid database in a RAM disk for speed
    #   confidenceThreshold -- [float] Detection score threshold to store in bolideDispositions object
    #   beliefSource       -- [str] The source of the detections score for confidenceThreshold
    #   copyOverBolideDetection -- [bool] If True then copy over the bolideDisposition.bolideDetection from the database to the
    #       bolideDispositionProfileList. This will significantly increase the memory requirements for the object
    #   database_wait_time_sec -- [float] Wait this number of seconds for the database file to become available and no
    #       longer locked.
    #   requireCutoutFeatures -- [bool] If True then require the cutout feature be present.
    #                               If the cutout tool is disabled in the pipeline run that generated this datam then
    #                               the cutoutFeatures is None. Depending on what you are doing you might require these
    #                               features.
    #
    #*************************************************************************************************************
    @classmethod
    def from_bolideDatabase(cls, database_filename, extra_bolideDispositionProfileList=None, spice_kernel_path=None, 
            verbosity=False, humanVetter=None, useRamDisk=True, confidenceThreshold=0.0, beliefSource='triage', copyOverBolideDetection=False, 
            database_wait_time_sec=0.0, requireCutoutFeatures=True, **kwargs):

        assert os.path.splitext(database_filename).index('.fs'), "<database_filename> must be a ZODB ending with '.fs'"

        cls.bolideDatabase = cls._associate_bolideDatabase (database_filename, useRamDisk, database_wait_time_sec=database_wait_time_sec)
        cls.database_path = os.path.dirname(database_filename)

        # How many iterations before the status bar is updated
        # A new step every 1% of the total number of entries.
        loopStep = round(cls.bolideDatabase.n_tot_entries / 100)

        #************
        # Generate a bolideDispositionProfileList to be associated with this bolideDatabase
        # copy over features in database
        if (verbosity):
            pbar = tqdm(total=cls.bolideDatabase.n_tot_entries, desc='Generating bolideDispositionProfileList')
        # First work through the detections then do the rejections
        cls.bolideDispositionProfileList = []
        loopCount = 0
        first_catch = True
        for key, value in cls.bolideDatabase.detections.iteritems():
            ID = key
            detection = value
            # We have to be intelligent about what to copy over to the profile in order to be memory efficient
            # Do not copy over the entire bolideDetection, only keep the features, unless copyOverBolideDetection=True

            # Remove bolide candidates if there is no cutout feature data
            if requireCutoutFeatures and detection.cutoutFeatures is None:
                if first_catch:
                    warnings.warn('CutoutFeatures required but found detections without cutoutFeatures, they are being removed from bolideDispositionProfileList')
                    first_catch = False
                    
                if (verbosity and loopCount > 0 and np.mod(loopCount, loopStep) == 0):
                    pbar.update(loopStep)
                    # Clear the ZODB database cache to save memory
                    cls.bolideDatabase.cacheMinimize()
                loopCount += 1
                continue
            elif hasattr(detection,'cutoutFeatures') and detection.cutoutFeatures is not None:
                cutoutFeatures = detection.cutoutFeatures.copy()
            else:
                cutoutFeatures = None


            # Only store detections with a detection score above confidenceThreshold
            # For the validation step, if validation could not be performed then the score is None, we must skip these
            # detections
            if beliefSource == 'triage':
                score = detection.assessment.triage.score
            elif beliefSource == 'validation':
                score = detection.assessment.validation.score
            elif beliefSource == 'human':
                score = detection.assessment.human.score
            else:
                raise Exception('Unknown beliefSource')
            if score is None or score < confidenceThreshold:
                if (verbosity and loopCount > 0 and np.mod(loopCount, loopStep) == 0):
                    pbar.update(loopStep)
                    # Clear the ZODB database cache to save memory
                    cls.bolideDatabase.cacheMinimize()
                loopCount += 1
                continue

            if copyOverBolideDetection:
                bolideDetection=detection
            else:
                bolideDetection=None

            if beliefSource == 'triage':
                machineOpinions=[MachineOpinion(
                    bolideBelief=copy.copy(detection.assessment.triage.score),
                    method=copy.copy(detection.assessment.triage.method),
                    source='triage',
                    comments='howFound: '.format(copy.copy(detection.howFound))
                    )]
                humanOpinions = None
            elif beliefSource == 'validation':
                machineOpinions=[MachineOpinion(
                    bolideBelief=copy.copy(detection.assessment.validation.score),
                    method=copy.copy(detection.assessment.validation.method),
                    source='validation',
                    comments='howFound: '.format(copy.copy(detection.howFound))
                    )]
                humanOpinions = None
            elif beliefSource == 'human':
                machineOpinions = None
                humanOpinions=[HumanOpinion(
                   belief=copy.copy(detection.assessment.human.score),
                   name=copy.copy(detection.assessment.human.source),
                   comments='howFound: '.format(copy.copy(detection.assessment.human.source))
                   )]
            cls.bolideDispositionProfileList.append(
                    BolideDispositionProfile(ID, 
                        detectionFlag=True, 
                        bolideDetection=bolideDetection,
                        machineOpinions=machineOpinions,
                        humanOpinions=humanOpinions,
                        features=detection.features.copy(),
                        stereoFeatures=detection.stereoFeatures.copy(),
                        cutoutFeatures=cutoutFeatures))

            if (verbosity and loopCount > 0 and np.mod(loopCount, loopStep) == 0):
                pbar.update(loopStep)
                # Clear the ZODB database cache to save memory
                cls.bolideDatabase.cacheMinimize()
            loopCount += 1

        #***
        # Rejections, if there are any
        for key, value in cls.bolideDatabase.rejections.iteritems():
            ID = key
            rejection = value
            if copyOverBolideDetection:
                bolideRejection=rejection
            else:
                bolideRejection=None

            if hasattr(rejection,'cutoutFeatures') and rejection.cutoutFeatures is not None:
                cutoutFeatures = rejection.cutoutFeatures.copy()
            else:
                cutoutFeatures = None

            if beliefSource == 'triage':
                machineOpinions=[MachineOpinion(
                    bolideBelief=copy.copy(rejection.assessment.triage.score),
                    method=copy.copy(rejection.assessment.triage.method),
                    source='triage',
                    comments='howFound: '.format(copy.copy(rejection.howFound))
                    )]
                humanOpinions = None
            elif beliefSource == 'validation':
                machineOpinions=[MachineOpinion(
                    bolideBelief=copy.copy(detection.assessment.validation.score),
                    method=copy.copy(detection.assessment.validation.method),
                    source='validation',
                    comments='howFound: '.format(copy.copy(detection.howFound))
                    )]
                humanOpinions = None
            elif beliefSource == 'website':
                machineOpinions = None
                humanOpinions=[HumanOpinion(
                   belief=copy.copy(rejection.assessment.human.score),
                   name=copy.copy(rejection.assessment.human.source),
                   comments='howFound: '.format(copy.copy(rejection.assessment.human.source))
                   )]
            else:
                raise Exception('Unknown confidendeSource')
            cls.bolideDispositionProfileList.append(
                    BolideDispositionProfile(ID, 
                        detectionFlag=False, 
                        bolideDetection=bolideRejection,
                        machineOpinions=machineOpinions,
                        humanOpinions=humanOpinions,
                        features=rejection.features.copy(),
                        stereoFeatures=rejection.stereoFeatures.copy(),
                        cutoutFeatures=cutoutFeatures))

            if (verbosity and loopCount > 0 and np.mod(loopCount, loopStep) == 0):
                pbar.update(loopStep)
                # Clear the ZODB database cache to save memory
                cls.bolideDatabase.cacheMinimize()
            loopCount += 1

        assert len(cls.bolideDispositionProfileList) > 0, 'No detections were added to bolideDispositionProfileList'

        # Add in the extra_bolideDispositionProfileList
        if extra_bolideDispositionProfileList is not None:
            cls.add_bolideDispositionProfileList(cls, extra_bolideDispositionProfileList)
            print('{} extra bolideDispositionProfiles added'.format(len(extra_bolideDispositionProfileList)))

        # one more clearing of the database cache
        cls.bolideDatabase.cacheMinimize()

        if (verbosity):
            pbar.close()

        # Here we instantiate the actual object and return it 
        return cls(spice_kernel_path=spice_kernel_path, verbosity=verbosity, **kwargs)

    #*************************************************************************************************************
    # classmethod from_bolideDispositionProfileList  
    #
    # Constructs a BolideDispositions object using a bolideDispositionProfileList.
    # The pipeline uses the ZODB bolide database.
    #
    # If a database_filename is passed then also open a ZODB database connection to associate with this object.
    #
    # Inputs:
    #   bolideDispositionProfileList -- [list] of BolideDispositionProfile objects
    #                                   Or file path [str] to pickled bolideDispositionProfileList 
    #   database_filename            --  [list of str] Path to the bolide database to associate in this object
    #                                   if None then assume data is in bolideDispositionProfileList.bolideDetection
    #   spice_kernel_path            -- [str] Path to kernels to load for glintPoint object
    #                                       Only needed if computing features.
    #   verbosity   -- [bool] If True then print processing status statements
    #                   Computing features can be slow
    #   useRamDisk          -- [bool] If true then temporarily store the bolide database in a RAM disk for speed
    #   database_wait_time_sec -- [float] Wait this number of seconds for the database file to become available and no
    #       longer locked.
    #   compute_features  -- [bool] If to recompute the features. 
    #                           This is slow so when loading in a saved BolideDispositions object, use the stored features
    #
    #*************************************************************************************************************
    @classmethod
    def from_bolideDispositionProfileList(cls, bolideDispositionProfileList, database_filename=None, spice_kernel_path=None, 
                useRamDisk=True, database_wait_time_sec=0.0, **kwargs):

        cls.bolideDispositionProfileList = []
        cls.add_bolideDispositionProfileList(cls, bolideDispositionProfileList)

        cls.bolideDatabase = cls._associate_bolideDatabase (database_filename, useRamDisk, database_wait_time_sec=database_wait_time_sec)
        if database_filename is not None:
            cls.database_path = os.path.dirname(database_filename)
        else:
            cls.database_path = None

        # Here we instantiate the actual object and return it 
        return cls(spice_kernel_path=spice_kernel_path, **kwargs)

    #*************************************************************************************************************
    def add_bolideDispositionProfileList(self, bolideDispositionProfileList):
        """ Appends a list of bolideDispositionProfile objects to self.bolideDispositionProfileList.

        This method can be used, for example, to add a list of rejections fomr a training data set run to the detections
        stored in the database used to create this object with the class method from_bolideDatabase.

        Parameters
        ----------
        bolideDispositionProfileList : [list] of BolideDispositionProfile objects
                                        Or file path [str] to pickled bolideDispositionProfileList 

        Returns
        -------
        self.bolideDispositionProfileList with list appended

        """

        if (type(bolideDispositionProfileList) == str):
            self.bolideDispositionProfileList.extend(bd.unpickle_bolide_detections(bolideDispositionProfileList))
        else:
            # Use deepcopy to ensure the objects are fully copied
            self.bolideDispositionProfileList.extend(copy.deepcopy(bolideDispositionProfileList))


    #*************************************************************************************************************
    # def _associate_bolideDatabase 
    # 
    # Associates this object with a ZODB bolide database.
    #
    # Inputs:
    #   database_filename       -- [list of str] Path to the bolide databases to include in object
    #   useRamDisk          -- [bool] If true then temporarily store the bolid database in a RAM disk for speed
    #   database_wait_time_sec -- [float] Wait this number of seconds for the database file to become available and no
    #       longer locked.
    #   
    #
    # Outputs:
    #   bolideDatabase -- [BolideDatabase] ZODB database
    #*************************************************************************************************************
    @staticmethod
    def _associate_bolideDatabase (database_filename, useRamDisk=True, database_wait_time_sec=0.0):

        if (database_filename is None):
            # Nothing to do!
            return None
    
        # The bolide database is a non-resident ZODB database
        # TODO: get this to work with a list of databases
        if (useRamDisk):
            # Tmeporarily copy bolide database to the RAM disk
            # On the NAS the RAM disk is located at /tmp or /tmp/ramdisk
            tmpFilePath = os.path.join('/tmp/ramdisk', 'glm_tmp/database')
            if not os.path.isdir(tmpFilePath):
                try:
                    os.makedirs(tmpFilePath)
                except OSError:
                    raise Exception('Creation of the directory {} failed'.format(tmpFilePath))
            indexFilename   = database_filename + '.index'
            lockFilename    = database_filename + '.lock'
            databaseTmpFilename   = os.path.join(tmpFilePath, os.path.split(database_filename)[1])
            os.system('cp {} {}'.format(database_filename, tmpFilePath))
            os.system('cp {} {}'.format(indexFilename, tmpFilePath))
            os.system('cp {} {}'.format(lockFilename, tmpFilePath))
            bolideDatabase = bolideDB.BolideDatabase(databaseTmpFilename, purgeDatabase=False, wait_time_sec=database_wait_time_sec)
        else:
            bolideDatabase = bolideDB.BolideDatabase(database_filename, purgeDatabase=False, wait_time_sec=database_wait_time_sec)

        return bolideDatabase
        
    
    #*************************************************************************************************************
    @property
    def n_tot_entries(self):
        """ The total number of detections and rejections in the database
        """
        return len(self.bolideDispositionProfileList)

    @property
    def IDs(self):
        """ A np.array of IDs from self.bolideDispositionProfileList
        """
        return np.array([bolide.ID for bolide in self.bolideDispositionProfileList])

    #*************************************************************************************************************
    # Converts bolideTime from UTC to local time
    def convert_time_to_local(self, local_type='meanSolar'):
        """ Converts the time of each detection (bolideTime) from UTC time (as is in the netCDF file) to local time.

        Uses time_utilities.convert_time_to_local

        Parameters
        ----------
        local_type : str
            The type of local time to convert to.

        Returns
        -------
            self.bolideTime             : datetime converted to local time
            self.features.bolideTime    : datetime converted to local time

        """

        bolideTime  = np.array([disposition.features.bolideTime for disposition in self.bolideDispositionProfileList])
        longitude   = np.array([disposition.features.avgLon for disposition in self.bolideDispositionProfileList])

        localTime = convert_time_to_local(bolideTime, longitude, local_type=local_type)

        for idx, disposition in enumerate(self.bolideDispositionProfileList):
            disposition.features.bolideTime = localTime[idx]
            

        return

    #******************************************************************************
    def generate_bolide_belief_array(self, beliefSource, bolidesFromWebsite=None, acrossSatellites=False):
        """
        Returns the belief score [0,1] for each bolide disposition.
      
        If bolidesFromWebsite is passed and beliefSource = 'human' then the human belief is computed based on the website
        database. If acrossSatellites is True then disregard the satllite on the website, otherewise only retrieve the
        truth value only if the satellite is a match for both the detection and the website bolide.
      
        If the machine or human beliefs are not set then this returns all NaNs.
      
        TODO: Set this up so it can handle multiple votes.
      
        Parameters
        ----------
        beliefSource    : [str] 
            Which source to use for belief {'human', 'triage', 'validation'}
            If None then return all zeros
        bolidesFromWebsite  : [WebsiteBolideEvent list] 
            created by: bolide_dispositions.pull_dispositions_from_website 
        acrossSatellites : bool
            If True then search for bolide belief across all satellites and not just if there is a match for the same
            satellite on the website.
      
        Returns:
        --------
        bolideBelief    : [np.array length(nDispositions)] 
            0 => not a bolide, 1 => bolide
     
        """

        # We need to place this import statement here so that we do not have a circular import in bolide_cnn.py
        from validator import bolideWebsiteBeliefSwitcher

        bolideBelief  = []
 
        # Pick which belief to return
        selection = bolideBeliefSwitcher.get(beliefSource, bolideBeliefSwitcher['unknown'])
        if (selection== bolideBeliefSwitcher['unknown']):
            bolideBelief = np.zeros(self.n_tot_entries)

        # If bolidesFromWebsite is passed then compute the belief from website data
        if isinstance(bolidesFromWebsite, list) and isinstance(bolidesFromWebsite[0], WebsiteBolideEvent):
            assert selection == bolideBeliefSwitcher['human'], "If bolidesFromWebsite is passed then beliefSource should be set to 'human'"

            # Bolide belief will be the arithmetic mean for all satellite beliefs
            # If matching across all satellites then we just have the arrithmetic mean of the same number. 
            beliefsAllSatellites = np.full((len(bd.validSatellites), self.n_tot_entries), np.nan)
            for idx, satellite in enumerate(bd.validSatellites):
                if acrossSatellites:
                    matchIndexThisSatellite = bDetPerform.find_bolide_match_from_website(self.bolideDispositionProfileList, bolidesFromWebsite, 'all')
                else:
                    matchIndexThisSatellite = bDetPerform.find_bolide_match_from_website(self.bolideDispositionProfileList, bolidesFromWebsite, satellite)

                # If there is a match with a website bolide then return the confidence for that satellite. 
                # If no match then return a confidence of nan
                beliefsAllSatellites[idx,:] = \
                    [bolideWebsiteBeliefSwitcher.get(bolidesFromWebsite[matchIndex].confidence, bolideWebsiteBeliefSwitcher['unknown']) \
                        if matchIndex >= 0 else np.nan  for matchIndex in matchIndexThisSatellite]

            assert not np.any(beliefsAllSatellites.flatten() < 0.0), 'Error in computing website-based bolide belief'

            # Ignore RuntimeWarning due to taking nanmean of all NaNs, which returns NaN. The next line converts these
            # to 0.0
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                bolideBelief = np.nanmean(beliefsAllSatellites, axis=0)
            # Convert all NaN beliefs to 0.0
            bolideBelief[np.isnan(bolideBelief)] = 0.0

        else:
            for record in self.bolideDispositionProfileList:
            
                if (selection == bolideBeliefSwitcher['human']):
                    if (record.humanOpinions is None):
                        # Not available, set to NaN
                        bolideBelief.append(np.nan)
                    elif (len(record.humanOpinions) > 1):
                        raise Exception("Currently only one opinion per bolide detection is supported")
                    else:
                        for human in record.humanOpinions:
                            try:
                                bolideBelief.append(human.belief)
                            except:
                                print('ha!')
            
                elif (selection == bolideBeliefSwitcher['triage'] or selection == bolideBeliefSwitcher['validation']):
                    if record.machineOpinions is None:
                        # Not available, set to NaN
                        bolideBelief.append(np.nan)
                    else:
                        opinionIdx = [i for i, machineOpinion in enumerate(record.machineOpinions) if machineOpinion.source == beliefSource]
                        if len(opinionIdx) >= 1:
                            bolideBelief.append(np.mean([record.machineOpinions[i].bolideBelief for i in opinionIdx]))
                        else:
                            raise Exception('Error finding machine opinion')
                  # elif len(record.machineOpinions) > 1:
                  #     raise Exception("Currently only one opinion per bolide detection is supported")
                  # else:
                  #     for machine in record.machineOpinions:
                  #         if machine.bolideBelief is None:
                  #             bolideBelief.append(0.0)
                  #         else:
                  #             bolideBelief.append(machine.bolideBelief)

        bolideBelief = np.array(bolideBelief)

        return bolideBelief


    #******************************************************************************
    # create_ML_data_set 
    #
    # Takes the bolideDispositionProfileList with computed features and generates a machine learning training set with an
    # optional seperate test set.
    #
    # Will scale the features.
    #
    # This method is also used to generate the X matrix for us in compute_machine_opinions. When used there we
    # do not want to permute the indices but keep the order so that we can populate the machineOpinions in 
    # self.bolideDispositionProfileList. This mode is used by setting computeMachineOpinionsMode = True
    #
    # Inputs:
    #   featuresToUse               -- [str list] List of features to use
    #                                   must be in bFeatures.FeaturesClass().__dict__.keys()
    #   testRatio                   -- [float] Ratio of seperate as test set
    #   randomSeed                  -- [int] Random seed for sampling and permutation
    #   plotFeatures                -- [bool] If True then scatter plot all the features
    #   computeMachineOpinionsMode  -- [bool] If True then set up X matrix for compute_machine_opinions 
    #   satellite                   -- [str] Which satellite to draw training set from
    #                                       {bolide_detections.validSatellites, or  'all'}
    #   useOnlyIntersectDays        -- [bool] If True then only use days that have both website bolides and pipeline bolides.
    #                                   Otherwise, use all data in statistics
    #   columnTransformer           -- [sklearn ColumnTransformer object] 
    #                                   The already fit transformer to normalize the features
    #                                   if None then create a new transformer
    #   beliefSource                -- {'human', 'machine'}
    #                                   What source to use to determine truth
    #   bolidesFromWebsite          -- [WebsiteBolideEvent list] created by:
    #                                   bolide_dispositions.pull_dispositions_from_website 
    #                                   Used if beliefSource = 'human'
    #
    # Outputs:
    #   xTrainNorm  -- [np.ndarray float] Normalized training features
    #   xTestNorm   -- [np.ndarray float] Normalized testing features
    #   yTrain      -- [np.ndarray bool] Training responses [yBolideTrain]
    #   yTest       -- [np.ndarray bool] Testing reponses [yBolideTest]
    #   sampleWeightsTrain -- [np.ndarray float] Sample weights for training
    #   responseLabels  -- [str list] Names of the reponses in the y matrices
    #   IDsTrain    -- [int64] IDs for all detections in xTrainNorm
    #   bolideBeliefArrayTrain -- [float] Detection score for all training data in xTrainNorm
    #
    #   self.featuresToUse     -- [str list] Features used in classifier
    #                               If None then use those stored in self.featuresToUse
    #   self.columnTransformer -- [sklearn ColumnTransformer object] The transformer used
    #
    #******************************************************************************
    def create_ML_data_set (self, 
            featuresToUse=None, 
            testRatio=0.33, 
            randomSeed=43, 
            plotFeatures=False, 
            computeMachineOpinionsMode=False, 
            satellite='all', 
            useOnlyIntersectDays=False, 
            columnTransformer=None,
            beliefSource=None,
            bolidesFromWebsite=None):

        assert bd.validSatellites.count(satellite) == 1 or satellite == 'all', \
                "<satellite> must be one of {bolide_detections.validSatellites, or  'all'}"

        # Confidence threshold when determining truth
        truthThreshold = 0.5

        if (not self.featuresComputed):
            print('Fetures not yet computed, cannot create training set')
            return np.full(6, None)

        if (featuresToUse is None):
            if (self.featuresToUse is None):
                raise Exception('self.featuresToUse is not yet stored.')
        else:
            # Check for valid features
            nProblems = bFeatures.FeaturesClass.check_for_valid_features(featuresToUse)
            if nProblems > 0:
                raise Exception ('Unknown feature labels in <featuresToUse>')
            self.featuresToUse = featuresToUse
 
        #**************************
        # Only use days that have both website data and pipeline data
        # This is to correct the performance statistics to days that were vetted and data was available.
        if useOnlyIntersectDays:
            assert bolidesFromWebsite is not None, 'If useOnlyIntersectDays=True then bolidesFromWebsite must be passed'
            [bolideDispositionProfileListLocal , bolidesFromWebsiteLocal] = intersect_days_bolideDispositionProfileList_and_bolidesFromWebsite(
                self.bolideDispositionProfileList, bolidesFromWebsite, satellite=satellite)
            n_tot_entries = len(bolideDispositionProfileListLocal)

            # Location of the preserved bolides in the full list
            IDs = np.array([profile.ID for profile in bolideDispositionProfileListLocal])
            # The below takes way to long to run for a training data set with millions of rejections
            IdIndicesOrigList = np.array([idx for idx, profile in enumerate(self.bolideDispositionProfileList) if np.isin(profile.ID, IDs)])
            # TODO: find a faster way (I need a Maptab ismember for numpy!)
            # Note: intersect1d does not preserve the order!
           #IDsOrig = np.array([profile.ID for profile in self.bolideDispositionProfileList])
           #IDsSorted, IdIndicesOrigList, IdIndicesLocalList = np.intersect1d(IDsOrig, IDs, assume_unique=True, return_indices=True)
           ## Now sort the Ids in the original array order

        else:
            bolideDispositionProfileListLocal = self.bolideDispositionProfileList
            IDs = np.array([profile.ID for profile in bolideDispositionProfileListLocal])
            bolidesFromWebsiteLocal = bolidesFromWebsite
            n_tot_entries = self.n_tot_entries
            IdIndicesOrigList = np.arange(len(self.bolideDispositionProfileList))


        #**************************
        # Create feature matrix
        X = np.ndarray(shape=(n_tot_entries,0), dtype=float)
        for feature in self.featuresToUse:
            featureArray = np.array([profile.features.__getattribute__(feature)  for profile in
                bolideDispositionProfileListLocal]).reshape(n_tot_entries,1)
            X = np.append(X, featureArray, axis=1)
 
        nFeatures = X.shape[1]
 
        responseLabels = [
                       'bolide',           
                       ]
        # If no beliefs then return all False for yBolide
        bolideBeliefArray = self.generate_bolide_belief_array(beliefSource, bolidesFromWebsite=bolidesFromWebsiteLocal)
        bolideBeliefArray = bolideBeliefArray[IdIndicesOrigList]
        yBolide = bolideBeliefArray >= truthThreshold

        #***
        # Select satellite to draw from
        goesSatellite = [disposition.features.goesSatellite for disposition in bolideDispositionProfileListLocal]
        if satellite == 'all':
            datumsToUse = np.arange(X.shape[0])
        else:
            datumsToUse = np.nonzero([sat == satellite for sat in goesSatellite])[0]
        X = X[datumsToUse,:]
        IDs = IDs[datumsToUse]
        yBolide = yBolide[datumsToUse]
        bolideBeliefArray = bolideBeliefArray[datumsToUse]

        #***
        # Compute sample weights
        # Weight based on total energy
        # Set median weight to 1.0 but really emphasize high energy bolides so take the square
        maxWeight = 20.0 # Otherwise, would get huge weights on some targets
        totEnergy = [profile.features.totEnergy for profile in bolideDispositionProfileListLocal]
        totEnergy = np.asarray(totEnergy)
        totEnergy = totEnergy[datumsToUse]
        trueBolideIdx = np.nonzero(yBolide)[0]
        if (len(trueBolideIdx) == 0):
            medianWeight = 1.0
        else:
            medianWeight = np.median(totEnergy[trueBolideIdx])
        sampleWeights = (totEnergy / medianWeight)**2.0
        sampleWeights = np.asarray([np.min([x, maxWeight]) for x in sampleWeights])
        # However, for all true negatives, set weight to 0.5
        sampleWeights[np.nonzero(np.logical_not(yBolide))[0]] = 0.5

        #***
        # Remove NaNs and infs from data set
        # If computeMachineOpinionsMode then replace NaNs and Infs, otherwise, remove. 
        # In computeMachineOpinionsMode we want to keep the array length the same as bolideDispositionProfileListLocal
        # Remove NaNs
        badColsNaNs = np.nonzero(np.logical_or(np.any(np.isnan(X),1), np.isnan(yBolide)))[0]
        badColsInfs = np.nonzero(np.logical_or(np.any(np.isinf(X),1), np.isinf(yBolide)))[0]
        badCols = np.union1d(badColsNaNs, badColsInfs)
        if computeMachineOpinionsMode:
            X[badCols,:] = 0.0
            sampleWeights[badCols] = 0.0
            yBolide[badCols] = False
        else:
            X = np.delete(X, badCols,0)
            IDs = np.delete(IDs, badCols,0)
            sampleWeights = np.delete(sampleWeights, badCols,0)
            yBolide = np.delete(yBolide, badCols)
            bolideBeliefArray = np.delete(bolideBeliefArray, badCols)
            
        nDatums = len(X[:,1])

        #***
        # If no detections are left after the cuts then return
        if X.shape[0] == 0:
            return np.full(6, None)

        #***
        # split train / test

        # If computeMachineOpinionsMode then do not do any data scrambling or break off test set
        if (computeMachineOpinionsMode):
            train_indices = np.arange(nDatums)
            test_indices = []

        else:
 
            np.random.seed(randomSeed)
            
            # First shuffle all indices to begin with so there is no order of input sort
            shuffled_indices    = np.random.permutation(nDatums)
            X = X[shuffled_indices,:]
            IDs = IDs[shuffled_indices]
            sampleWeights = sampleWeights[shuffled_indices]
            yBolide = yBolide[shuffled_indices]
            bolideBeliefArray = bolideBeliefArray[shuffled_indices]
            
            # Now split off the test set
            shuffled_indices    = np.random.permutation(nDatums)
            test_set_size       = int(nDatums * testRatio)
            test_indices        = shuffled_indices[:test_set_size]
            train_indices       = shuffled_indices[test_set_size:]
 
        xTrain      = X[train_indices,:]
        IDsTrain    = IDs[train_indices]
        xTest       = X[test_indices,:]
        yBolideTrain    = yBolide[train_indices]
        bolideBeliefArrayTrain    = bolideBeliefArray[train_indices]
        sampleWeightsTrain  = sampleWeights[train_indices]
        yBolideTest     = yBolide[test_indices]
        bolideBeliefArrayTest     = bolideBeliefArray[test_indices]
 
        yTrain  = np.asarray(yBolideTrain)
        yTest   = np.asarray(yBolideTest)
 
 
        #***
        # Scale the features
       #if (columnTransformer is None):
       #    from sklearn.compose import ColumnTransformer
       #    scalerList = []
       #    for idx, feature in enumerate(featuresToUse):
       #        scalerList.append(('scaler_{}'.format(idx), bFeatures.FeaturesClass.featureScaleMethod[feature], [idx]))
       #    self.columnTransformer = ColumnTransformer(scalerList)
       #    # Do not use the test data when computing the tranformation normalizations
       #    xTrainNorm = self.columnTransformer.fit_transform(xTrain)
       #else:
       #    self.columnTransformer = columnTransformer
       #    xTrainNorm = self.columnTransformer.transform(xTrain)
 
       #if (not test_indices == [] and len(test_indices) > 0):
       #    xTestNorm     = self.columnTransformer.transform(xTest)
       #else:
       #    xTestNorm = []

        xTrainNorm, self.columnTransformer = bFeatures.scale_features(xTrain, columnTransformer, featuresToUse)
        xTestNorm, self.columnTransformer = bFeatures.scale_features(xTest, self.columnTransformer)
 
 
        # Plot the distribution of features
        if plotFeatures:
            plotLogScale = ['totEnergy', 'nGroups', 'timeDuration', 'latLonLinelets', 'splinelets']
            fig, axes = plt.subplots(nrows=nFeatures, ncols=2, figsize=(10, 15))
 
            for i in np.arange(nFeatures):
                axes[i,0].plot(train_indices, xTrain[:, i], '.b', label='Train')
               #axes[i,0].plot(test_indices,  xTest[:, i], '.r', label='Test', alpha=0.5)
                axes[i,0].set_title('Unscaled {}'.format(featuresToUse[i]))
                axes[i,1].plot(train_indices, xTrainNorm[:, i], '.b', label='Train')
               #axes[i,1].plot(test_indices,  xTestNorm[:,i], '.r', label='Test', alpha=0.5)
                axes[i,1].set_title('Scaled {}'.format(featuresToUse[i]))
 
            plt.tight_layout(pad=0.0)
 
        return [xTrainNorm, xTestNorm, yTrain, yTest, sampleWeightsTrain, responseLabels, IDsTrain, bolideBeliefArrayTrain]

    #*********************************************************************************************************
    def generate_labeled_image_data_set(self, labeled_image_path, beliefSource='human', bolidesFromWebsite=None,
            image_size=None):
        """ Generates a labeled image data set using the bolideDispositionProfileList set of bolide candidates and the
        associated cutout image files. The data set format is for use with torchvision.datasets.ImageFolder.

        The images at <image_path> will be in this directory structure:
        root/dog/xxx.png
        root/dog/xxy.png

        root/cat/123.png
        root/cat/nsdf3.png

        labeled_image_path : str
            path to where to store the output images
        beliefSource : str
            Where to obtain the truth {'machine', 'human'}
        bolidesFromWebsite : [WebsiteBolideEvent list] created by:
            bolide_dispositions.pull_dispositions_from_website
            If passed and beliefSource == 'human' then use this data to determein truth.
        image_size : int or None
            If not None then resize the images to image_size x image_size

        """

        raise Exception('This method should no longer be used')

        # Clear out labeled_image_path
        if os.path.exists(labeled_image_path):
            shutil.rmtree(labeled_image_path)
        os.mkdir(labeled_image_path)

        bolide_path = os.path.join(labeled_image_path, 'bolide')
        not_bolide_path = os.path.join(labeled_image_path, 'notBolide')
        os.mkdir(bolide_path)
        os.mkdir(not_bolide_path)

        # Determine bolide truth
        bolideBeliefArray = self.generate_bolide_belief_array(beliefSource, bolidesFromWebsite)
        truthThreshold = 0.5
        yBolide = bolideBeliefArray >= truthThreshold


        # Copy images into the appropriate directories
        # For now, just use the 60_sec_integ image
        for idx in tqdm(range(self.n_tot_entries), 'Copying over images into labeled directories...'):
            bolide = self.bolideDispositionProfileList[idx]
            if not bolide.cutoutFeatures.success:
                continue
            # Find the filename for the 60-sec-integ figure
            fig_idx = int(np.nonzero(['60_sec_integ' in string for string in bolide.cutoutFeatures.figure_filenames])[0])
            year = str(bolide.features.bolideTime.year)
            MMDD = str(bolide.features.bolideTime.month).zfill(2) + str(bolide.features.bolideTime.day).zfill(2)
            orig_filename_w_path = os.path.join(self.database_path, year, MMDD, bolide.cutoutFeatures.figure_filenames[fig_idx])
            filename = os.path.basename(orig_filename_w_path)

            # If we wish to resize then do that now
            if image_size is not None:
                with Image.open(orig_filename_w_path) as img:
                    # Resize
                    img = img.resize((image_size, image_size))
                    # Save out the resized image
                    if yBolide[idx]:
                        img.save(os.path.join(bolide_path, filename), "PNG", optimize=True)
                    else:
                        img.save(os.path.join(not_bolide_path, filename), "PNG", optimize=True)
            else:
                #just copy over the original file
                if yBolide[idx]:
                    # Copy to bolide directory
                    shutil.copyfile(orig_filename_w_path, os.path.join(bolide_path, filename))
                else:
                    # Copy to not bolide directory
                    shutil.copyfile(orig_filename_w_path, os.path.join(not_bolide_path, filename))


        return labeled_image_path

    #*********************************************************************************************************
    def train_val_test_split(self, ratio=(0.8, 0.1, 0.1), clean_bad_datums=True, random_seed=42):
        """ Splits the bolideDispositionProfileList into a training, validation and test sets.
    
        clean_bad_datums is to clean bolideDispositionProfileList of any entries that do not have conforming data for
        the data sets. Note: this removes datums!

        Parameters
        ----------
        ratio : tuple(3)
            (train, validate, test) split ratios (adds up to 1.0)
        clean_bad_datums : bool
            If True then clean the data set of bad entries before splitting
        random_seed : int
            Random seed for permuting datums
            If None then set randomly

        Returns
        -------
        train_IDs : list of int64
            IDs in bolideDispositionProfileList of the training data
        val_IDs : list of int64
        test_IDs : list of int64

        """

        assert np.sum(ratio) == 1.0, 'Train, val, test split ratio must add up to 1.0.'

        # Clean bolideDispositionProfileList of any datums that are not correct
        if clean_bad_datums:
            IDs_to_remove = []
            for bolide in self.bolideDispositionProfileList:
                if not bolide.cutoutFeatures.success:
                    IDs_to_remove.append(bolide.ID)
            # Pop cannot be vectorized : (
            for ID in IDs_to_remove:
                bolideDetectionListIDs = [b.ID for b in self.bolideDispositionProfileList]
                idx = bolideDetectionListIDs.index(ID)
                self.bolideDispositionProfileList.pop(idx)

        
        nDatums = self.n_tot_entries

        # Rand perm the indices then split off into the three sets
        if random_seed is None:
            np.random.seed()
        else:
            np.random.seed(random_seed)
        shuffled_indices = np.random.permutation(nDatums)        
        
        train_start_idx = 0
        val_start_idx = int(nDatums * ratio[0])
        test_start_idx = val_start_idx + int(nDatums * ratio[1])

        train_indices = shuffled_indices[train_start_idx:val_start_idx]
        val_indices = shuffled_indices[val_start_idx:test_start_idx]
        test_indices = shuffled_indices[test_start_idx:]

        assert len(train_indices) + len(val_indices) + len(test_indices) == nDatums, 'Bookkeeping error in train/val/test split'

        # sefl.IDs is a dynamic dependent property, so, only create it once for speed
        IDs = self.IDs

        return IDs[train_indices], IDs[val_indices], IDs[test_indices]

    #******************************************************************************
    def return_IDs(self, indices):
        """
        Returns the IDs associated with the bolides with the indices in bolideDispositionProfileList

        Parameters
        ----------
        indices : int list
            List of indices to return the IDs for

        Returns
        -------
        IDs : int list

        """

        bolideDetectionListIDs = np.array([b.ID for b in self.bolideDispositionProfileList])
        IDs = bolideDetectionListIDs[np.array(indices)].tolist()

        return IDs
        
    #******************************************************************************
    def return_indices(self, IDs):
        """
        Returns the indices in bolideDispositionProfileList associated with the bolide IDs

        This preserves order

        Parameters
        ----------
        IDs : int list
            List of IDs to return the indices for

        Returns
        -------
        indices : int list

        """

        bolideDetectionListIDs  = self.IDs.tolist()
        # np.nonzero does not preserve order to use this method to retrieve the indices
        indices = np.array([bolideDetectionListIDs.index(ID) for ID in IDs])

        return indices
        

    #******************************************************************************
    # save_bolideDispositionProfileList 
    #
    # This is used to save the information to reconstruct a BolideDispositions object.
    # Computing features can be slow so use this to compute the features once and then
    # save them in bolideDispositionProfileList then reload them to generate a new
    # BolideDispositions object.
    #
    # Inputs:
    #   filename    -- [str] name of pickle file to save trained classifier dict to 
    #
    # Outputs:
    #   A file containing the following:
    #   bolideDispositionProfileList -- [list] of BolideDispositionProfile objects
    # 
    #******************************************************************************
    def save_bolideDispositionProfileList (self, filename):

        with open(filename, 'wb') as fp:
            pickle.dump(self.bolideDispositionProfileList , fp)

        fp.close()

    #******************************************************************************
    # save_trained_classifier
    #
    # Saves the data needed to deploy a trained classifier in a dictionary.
    #
    # Inputs:
    #   filename    -- [str] name of pickle file to save trained classifier dict to 
    #   trainedClassifier  -- [sklearn classifier object] The trained classifier
    #                               Must have a 'predict_proba' method
    #
    # Outputs:
    #   A file containing the following dict:
    #       trainedClassifierDict  -- [dict] Contains the trained classifer info
    #           'trainedClassifier'  -- [sklearn classifier object] The trained classifier
    #                               Must have a 'predict' method
    #           'featuresToUse'     -- [str list] Features used in classifier
    #           'columnTransformer' -- [sklearn ColumnTransformer object] The already fit transformer to normalize the features
    # 
    #******************************************************************************
    def save_trained_classifier(self, filename, trainedClassifier):
    
        if (self.featuresToUse is None or self.columnTransformer is None):
            raise Exception('It looks like the ML data set has not yet been created')

        # Form the dict
        trainedClassifierDict = {
                'trainedClassifier': trainedClassifier,
                'featuresToUse': self.featuresToUse,
                'columnTransformer': self.columnTransformer} 
    
        # Save the classifier
        with open(filename, 'wb') as fp:
            pickle.dump(trainedClassifierDict, fp)

        fp.close()

    #*********************************************************************************************************
    # compute_machine_opinions
    #
    # Takes a trained classfier and computes the machine predictions.
    #
    #
    # Inputs:
    #   trainedClassifier   -- [obj] A Trained classfier with a .predict_proba method
    #   featuresToUse               -- [str list] List of features to use
    #                                   must be in bFeatures.FeaturesClass().__dict__.keys()
    #   columnTransformer           -- [sklearn ColumnTransformer object] 
    #                                   The already fit transformer to normalize the features
    #                                   if None then create a new transformer
    #   method              -- [str] The type of method used for the classifer
    #   source              -- [str] What step in the detection pipeline related to this opinion
    #                                   {e.g. 'triage', 'validation'}
    #   ID                  -- [str] An identification string to label this trained classifer
    #   comments            -- [str] Any other comments
    #
    # Outputs:
    #   self.bolideDispositionProfileList[:].machineOpinions
    #
    #*********************************************************************************************************
    def compute_machine_opinions (self, trainedClassifier, featuresToUse, columnTransformer,  method=None, source=None, ID=None, comments=None):

        # Set up the feature array for ALL detections
        [xAll, _, _, _, _, responseLabels, _, _] = \
            self.create_ML_data_set (featuresToUse=featuresToUse, testRatio=0.0, randomSeed=43, plotFeatures=False,
                    computeMachineOpinionsMode=True, columnTransformer=columnTransformer, beliefSource=None)

        # Predict each class
        predictScore = trainedClassifier.predict_proba(xAll)

        # Get prediction for each belief class
        bolideBeliefIndex = [i for i, name in enumerate(responseLabels) if name == 'bolide']
        if (bolideBeliefIndex == []):
            bolideBelief = np.zeros(self.n_tot_entries)
        else:
            # Don't forget the 1 offset for a null prediction in the predict_proba array
            bolideBelief = predictScore[:,bolideBeliefIndex[0]+1]

        # TODO: FIX ME!
       #glintBeliefIndex = [i for i, name in enumerate(responseLabels) if name == 'glint']
       #if (glintBeliefIndex == []):
       #    glintBelief = np.zeros(self.n_tot_entries)
       #else:
       #    glintBelief = predictScore[:,glintBeliefIndex[0]+1]
        glintBelief = np.zeros(self.n_tot_entries)


        for iProfile in np.arange(self.n_tot_entries):
            self.bolideDispositionProfileList[iProfile].machineOpinions = \
                [MachineOpinion(bolideBelief=bolideBelief[iProfile], glintBelief=glintBelief[iProfile], 
                    method=method, source=source, ID=ID, comments=comments)]
        

    #******************************************************************************
    def plot_GLM_events_on_globe (self, beliefSource, lowConfidenceThreshold=0.5, 
            highConfidenceThreshold=0.95, doPlotAllDetections=False, doPlotDensityMaps=True, title=None):
        """
        
        Scatter plots all GLM events from both instruments superimposed on the Earth
       
        Modified from Paul Register and Geert Berentsen code, thanks!
        ... and now uses the Bolides package from Anthony Ozerov, thanks!
       
        Parameters
        ----------
        beliefSource                    -- [str] Which source to use for belief {'human', 'triage', 'validation'}
        lowConfidenceThreshold          -- [float] belief threshold when plotting low confidence bolide candidates ( >= )
        highConfidenceThreshold         -- [float] belief threshold when plotting high confidence bolide candidates ( >= )
        doPlotAllDetections             -- [bool] If True then plot all detections, not just high or low confidence candidates.
        doPlotDensityMaps               -- [bool] If True then plot density heat maps of detections
        title                           -- [str] Title of plot
       
        Returns
        -------
        m           -- Matplotlib figure axis handle
        """
     
     

        if doPlotAllDetections and len(self.bolideDispositionProfileList) > 1e5:
            print('Requesting to plot all detections, and there are {} detections.'.format(len(self.bolideDispositionProfileList)))
            print('This might take a long time to process!')
    
        #Default title
        if title is None:
            scatter_title = 'GOES GLM Pipeline Bolide Candidates'
        else:
            scatter_title = title
    
        allBeliefs = self.generate_bolide_belief_array(beliefSource)
    
        # High confidence bolides
        highConfidenceBolides = np.full(self.n_tot_entries, False)
        highConfidenceBolides[np.nonzero(allBeliefs>=highConfidenceThreshold)[0]] = True
        highConfidenceBolides = highConfidenceBolides.tolist()
    
        # Low confidence bolides
        lowConfidenceBolides = np.full(self.n_tot_entries, False)
        lowConfidenceBolides[np.nonzero(allBeliefs>=lowConfidenceThreshold)[0]] = True
        lowConfidenceBolides = np.logical_and(lowConfidenceBolides, np.logical_not(highConfidenceBolides))
        lowConfidenceBolides = lowConfidenceBolides.tolist()
    
        # Everything else
        allOtherDetections = np.full(self.n_tot_entries, False)
        allOtherDetections[np.nonzero(allBeliefs<lowConfidenceThreshold)[0]] = True
        allOtherDetections = allOtherDetections.tolist()
        
    
        totEnergy = np.array([disposition.features.totEnergy for disposition in self.bolideDispositionProfileList])
    
        # These cannot be vectorized (AFAIK, but I'm sure there is a Python trick)
        avgLat      = np.full(self.n_tot_entries, np.nan)
        avgLon      = np.full(self.n_tot_entries, np.nan)
        # Time referenced to J2000: bd.BolideDetection.epoch
        startTime   = np.full(self.n_tot_entries, 0.0)
        endTime     = np.full(self.n_tot_entries, 0.0)
        for i in tqdm(range(len(self.bolideDispositionProfileList)), 'Compiling avgLat, avgLon and times...'):
            profile = self.bolideDispositionProfileList[i]
            [avgLat[i], avgLon[i]] = [profile.features.avgLat, profile.features.avgLon]
            startTime[i] = profile.features.startTime
            endTime[i]   = profile.features.endTime
    
        #***
        goesSatellite = [disposition.features.goesSatellite for disposition in self.bolideDispositionProfileList]
        g16Here = np.array([g == 'G16' for g in goesSatellite])
        g17Here = np.array([g == 'G17' for g in goesSatellite])
        g18Here = np.array([g == 'G18' for g in goesSatellite])
        g19Here = np.array([g == 'G19' for g in goesSatellite])

        #***
        # Determine stereo detections
        # We determine if it's a stereo detection if the stereoFeatures are populated
        stereoHere = np.full(self.n_tot_entries, False)
        for idx, b in enumerate(self.bolideDispositionProfileList):
            if b.stereoFeatures.sat_east is not None and b.stereoFeatures.sat_west is not None:
                stereoHere[idx] = True
            else:
                stereoHere[idx] = False
        # If stereo then do not also flag as one or the other
        g16Here[stereoHere] = False
        g17Here[stereoHere] = False
        g18Here[stereoHere] = False
        g19Here[stereoHere] = False

        # sanity check
        assert np.count_nonzero(stereoHere) + \
                np.count_nonzero(g16Here) + \
                np.count_nonzero(g17Here) + \
                np.count_nonzero(g18Here) + \
                np.count_nonzero(g19Here) == self.n_tot_entries, \
        "Bookkeeping error when computing location of bolides"

        sz = 2 # minimum point size on scatter plot
        fc = 5 # scaling factor for point size based on energy

        fig, ax = generate_plot(figsize=(20,10))

        # Plot everything else first since they should be stacked on the bottom
        if doPlotAllDetections:
            if np.any(allOtherDetections):
                plot_scatter(avgLat[allOtherDetections], avgLon[allOtherDetections], boundary=['goes-e', 'goes-w'],
                    marker="o", color="lightgrey", alpha=0.2, edgecolor=None, s=sz, zorder=900, label='Other', fig=fig, ax=ax)

        # Low confidence are dimmed
        if (np.any(lowConfidenceBolides)):
            g16PlotHereLow = np.logical_and(g16Here, lowConfidenceBolides).tolist()
            if np.any(g16PlotHereLow):
                plot_scatter(avgLat[g16PlotHereLow], avgLon[g16PlotHereLow], boundary=['goes-e', 'goes-w'],
                    marker="o", color="r", alpha=0.2, edgecolor=None,
                    s=sz+fc*np.log(totEnergy[g16PlotHereLow]/min(totEnergy[g16PlotHereLow])), zorder=902, label='G16 Low Confidence', 
                    fig=fig, ax=ax)
            g17PlotHereLow = np.logical_and(g17Here, lowConfidenceBolides).tolist()
            if np.any(g17PlotHereLow):
                plot_scatter(avgLat[g17PlotHereLow], avgLon[g17PlotHereLow], boundary=['goes-e', 'goes-w'],
                    marker="o", color="dodgerblue", alpha=0.2, edgecolor=None,
                    s=sz+fc*np.log(totEnergy[g17PlotHereLow]/min(totEnergy[g17PlotHereLow])), zorder=902, label='G17 Low Confidence', 
                    fig=fig, ax=ax)
            g18PlotHereLow = np.logical_and(g18Here, lowConfidenceBolides).tolist()
            if np.any(g18PlotHereLow):
                plot_scatter(avgLat[g18PlotHereLow], avgLon[g18PlotHereLow], boundary=['goes-e', 'goes-w'],
                    marker="o", color="mediumblue", alpha=0.2, edgecolor=None,
                    s=sz+fc*np.log(totEnergy[g18PlotHereLow]/min(totEnergy[g18PlotHereLow])), zorder=902, label='G18 Low Confidence', 
                    fig=fig, ax=ax)
            g19PlotHereLow = np.logical_and(g19Here, lowConfidenceBolides).tolist()
            if np.any(g19PlotHereLow):
                plot_scatter(avgLat[g19PlotHereLow], avgLon[g19PlotHereLow], boundary=['goes-e', 'goes-w'],
                    marker="o", color="r", alpha=0.2, edgecolor=None,
                    s=sz+fc*np.log(totEnergy[g19PlotHereLow]/min(totEnergy[g19PlotHereLow])), zorder=902, label='G19 Low Confidence', 
                    fig=fig, ax=ax)
            stereoPlotHereLow = np.logical_and(stereoHere, lowConfidenceBolides).tolist()
            if np.any(stereoPlotHereLow):
                plot_scatter(avgLat[stereoPlotHereLow], avgLon[stereoPlotHereLow], boundary=['goes-e', 'goes-w'],
                    marker="o", color="c", alpha=0.2, edgecolor=None,
                    s=sz+fc*np.log(totEnergy[stereoPlotHereLow]/min(totEnergy[stereoPlotHereLow])), zorder=902, label='Stereo Low Confidence', 
                    fig=fig, ax=ax)

        # High confidence are full alpha
        if (np.any(highConfidenceBolides)):
            g16PlotHereHigh = np.logical_and(g16Here, highConfidenceBolides).tolist()
            if np.any(g16PlotHereHigh):
                plot_scatter(avgLat[g16PlotHereHigh], avgLon[g16PlotHereHigh], boundary=['goes-e', 'goes-w'],
                    marker="o", color="r", alpha=1.0, edgecolor=None,
                    s=sz+fc*np.log(totEnergy[g16PlotHereHigh]/min(totEnergy[g16PlotHereHigh])), zorder=902, 
                    label='G16 High Confidence', fig=fig, ax=ax)
            g17PlotHereHigh = np.logical_and(g17Here, highConfidenceBolides).tolist()
            if np.any(g17PlotHereHigh):
                plot_scatter(avgLat[g17PlotHereHigh], avgLon[g17PlotHereHigh], boundary=['goes-e', 'goes-w'],
                    marker="o", color="dodgerblue", alpha=1.0, edgecolor=None,
                    s=sz+fc*np.log(totEnergy[g17PlotHereHigh]/min(totEnergy[g17PlotHereHigh])), zorder=902, 
                    label='G17 High Confidence', fig=fig, ax=ax)
            g18PlotHereHigh = np.logical_and(g18Here, highConfidenceBolides).tolist()
            if np.any(g18PlotHereHigh):
                plot_scatter(avgLat[g18PlotHereHigh], avgLon[g18PlotHereHigh], boundary=['goes-e', 'goes-w'],
                    marker="o", color="mediumblue", alpha=1.0, edgecolor=None,
                    s=sz+fc*np.log(totEnergy[g18PlotHereHigh]/min(totEnergy[g18PlotHereHigh])), zorder=902, 
                    label='G18 High Confidence', fig=fig, ax=ax)
            g19PlotHereHigh = np.logical_and(g19Here, highConfidenceBolides).tolist()
            if np.any(g19PlotHereHigh):
                plot_scatter(avgLat[g19PlotHereHigh], avgLon[g19PlotHereHigh], boundary=['goes-e', 'goes-w'],
                    marker="o", color="r", alpha=1.0, edgecolor=None,
                    s=sz+fc*np.log(totEnergy[g19PlotHereHigh]/min(totEnergy[g19PlotHereHigh])), zorder=902, 
                    label='G19 High Confidence', fig=fig, ax=ax)
            stereoPlotHereHigh = np.logical_and(stereoHere, highConfidenceBolides).tolist()
            if np.any(stereoPlotHereHigh):
                plot_scatter(avgLat[stereoPlotHereHigh], avgLon[stereoPlotHereHigh], boundary=['goes-e', 'goes-w'],
                    marker="o", color="c", alpha=1.0, edgecolor=None,
                    s=sz+fc*np.log(totEnergy[stereoPlotHereHigh]/min(totEnergy[stereoPlotHereHigh])), zorder=902, 
                    label='Stereo High Confidence', fig=fig, ax=ax)

        plt.title(scatter_title, fontsize='x-large')
        plt.legend(fontsize='x-large', markerscale=1.0, loc='upper right')
        plt.show()



        #***
        if doPlotDensityMaps:

            if (np.any(lowConfidenceBolides)):
                # Low confidence detections
                fig, ex = plot_density(avgLat[lowConfidenceBolides], avgLon[lowConfidenceBolides], bandwidth=2,
                        lat_resolution=200, lon_resolution=100, boundary=['goes-e', 'goes-w'],
                        figsize=(20,10), title = 'Density Distribution of Low Confidence Detections ( >= {} )'.format(lowConfidenceThreshold))
                plt.show()
    
            if (np.any(highConfidenceBolides)):
                # High confidence detections
                fig, ex = plot_density(avgLat[highConfidenceBolides], avgLon[highConfidenceBolides], bandwidth=2, 
                        lat_resolution=200, lon_resolution=100, boundary=['goes-e', 'goes-w'],
                        figsize=(20,10), title = 'Density Distribution of High Confidence Detections ( >= {} )'.format(highConfidenceThreshold))
                plt.show()
    

        return
    
    
    #******************************************************************************
    def plot_disposition_statistics(self, beliefSource, bolidesFromWebsite=None, bolideBeliefThreshold=0.5, 
            startDate=None, endDate=None, acrossSatellites=True):
        """
        Plots a bunch of summary statistics of all the dispositioned bolide detections,
      
        The bolide detections are now stored in a ZODB database, we therefore want to be intelligent about what
        information we use here.
      
        Parameters
        ----------
        beliefSource        : [str] 
            Which source to use for belief {'human', 'machine'}
        bolidesFromWebsite  : [WebsiteBolideEvent list] 
            created by: bolide_dispositions.pull_dispositions_from_website 
            Can be used if beliefSource='human'
        bolideBeliefThreshold : [float] 
            Threshold for belief to be a bolide ( >= )
        startDate           : [str] 
            The starting date for analysis
            Is ISO format: 'YYYY-MM-DD'
            None or '' means no start date
        endDate             : [str] 
            The ending date for analysis
            Is ISO format: 'YYYY-MM-DD'
            None or '' means no end date
        acrossSatellites : bool
            If True then search for bolide belief across all satellites and not just if there is a match for the same
            satellite on the website.
     
        """

        #**************************
        # Only use data from selected days
        [bolideDispositionProfileListReduced, bolidesToKeep] = select_bolideDispositionProfileList_from_dates (self.bolideDispositionProfileList,
                startDate=startDate, endDate=endDate)

        allBeliefs = self.generate_bolide_belief_array(beliefSource, bolidesFromWebsite=bolidesFromWebsite, acrossSatellites=acrossSatellites)
        allBeliefs = allBeliefs[bolidesToKeep]
    
        # Collect the information we want
        # stored in bolideDispositionProfileListReduced
        goesSatelliteArray = [disposition.features.goesSatellite for disposition in bolideDispositionProfileListReduced]
        bolideTimeArray = np.array([disposition.features.bolideTime for disposition in bolideDispositionProfileListReduced])


        for satellite in bd.validSatellites:
            self._plot_disposition_statistics_this_satellite(goesSatelliteArray, bolideTimeArray, allBeliefs, beliefSource,
                bolideBeliefThreshold, satellite)
        

        return

    #******************************************************************************
    @staticmethod
    def _plot_disposition_statistics_this_satellite(goesSatelliteArray, bolideTimeArray, allBeliefs, beliefSource,
            bolideBeliefThreshold, satellite):
        """
        This is a helper function to plot the performance statistics for a specific satellite

        Only generate a plot of there are detections on this satellite.

        """

        colorSwitcher = {'G16': 'b', 'G17': 'g', 'G18': 'm', 'G19': 'c'}

        assert satellite in colorSwitcher, 'Unknown satellite'

        thisSatHere = [g == satellite for g in goesSatelliteArray]

        # do nothing if there are no detections on this satellite
        if not np.any(thisSatHere):
            return

        thisSatBeliefs = allBeliefs[thisSatHere]
        nBeliefs = len(thisSatBeliefs)


        thisSatBolideTime = bolideTimeArray[thisSatHere]

        #***
        # Set up figure
        fig, ax = plt.subplots(4,1)
        fig.set_figwidth(7.0)
        fig.set_figheight(9.0)

        if (beliefSource == 'human'):
            nBins = 10
        elif (beliefSource == 'machine'):
            nBins = 100
        else:
            raise Exception('Unknown belief source')

        color = colorSwitcher[satellite]
    
        # Plot a histogram of beliefs
        [bins, _, _] = ax[0].hist(thisSatBeliefs, bins=nBins, log=True, facecolor=color, alpha=0.5, rwidth=0.5,  
                label='{} tot: {}'.format(satellite, np.count_nonzero(thisSatHere)))

        ax[0].plot([bolideBeliefThreshold, bolideBeliefThreshold], [0, np.max(bins)], '-r',
                label='Threshold={:.2f}'.format(bolideBeliefThreshold))
        ax[0].grid(axis='y', which='both')
        ax[0].set_xticks([0, 0.5, 1.0])
        ax[0].legend()
        ax[0].set_title('Distribution of All {} Bolide {} Candidate Beliefs; Num Tot = {}'.format(satellite, beliefSource, nBeliefs))
    
        # Dispositions versus time
        locator1 = mdates.AutoDateLocator()
       #productTimeDayOfYr = [time.timetuple().tm_yday for time in thisSatBolideTime]

        # Count number of days in data set
        numberOfDays = (np.max(thisSatBolideTime) - np.min(thisSatBolideTime)).days
        # One day per bin, but at least 100 bins
        timeBins = np.max([100, numberOfDays])

            
       #mdateTimeOrigin = mdates.date2num(thisSatBolideTime)
        mdateTimeOrigin = mdates.date2num(bolideTimeArray)
        minBin = np.min(mdateTimeOrigin[thisSatHere])
        maxBin = np.max(mdateTimeOrigin[thisSatHere])


        ax[1].hist(mdateTimeOrigin[thisSatHere], bins=timeBins, range=(minBin, maxBin), log=False, facecolor=color, alpha=0.5, label=satellite)

        ax[1].grid(axis='y')
        ax[1].legend()
        ax[1].xaxis.set_major_locator(locator1)
       #ax[1].xaxis.set_major_formatter(mdates.AutoDateFormatter(locator1))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
        ax[1].set_xlabel('Date of Event')
        ax[1].set_title('Distribution of All Bolide Clusters in Time')
    
        # Bolide Rejections versus time
        locator2 = mdates.AutoDateLocator()
        falseHere = np.logical_and(np.array(thisSatHere), allBeliefs < bolideBeliefThreshold)
        ax[2].hist(mdateTimeOrigin[falseHere], bins=timeBins, range=(minBin, maxBin), log=False, facecolor=color, alpha=0.5, 
            label='{} tot: {}'.format(satellite, np.count_nonzero(falseHere)))

        ax[2].grid(axis='y')
        ax[2].legend()
        ax[2].xaxis.set_major_locator(locator2)
       #ax[2].xaxis.set_major_formatter(mdates.AutoDateFormatter(locator2))
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
        ax[2].set_xlabel('Date of Event')
        ax[2].set_title('Distribution of All Bolide Rejections in Time (Belief < {})'.format(bolideBeliefThreshold))
    
    
        # Bolide Candidates versus time
        locator3 = mdates.AutoDateLocator()
        trueHere = np.logical_and(np.array(thisSatHere), allBeliefs >= bolideBeliefThreshold)
        ax[3].hist(mdateTimeOrigin[trueHere], bins=timeBins, range=(minBin, maxBin), log=False, facecolor=color, alpha=0.5, 
            label='{} tot: {}'.format(satellite, np.count_nonzero(trueHere)))

        ax[3].grid(axis='y')
        ax[3].legend()
        ax[3].xaxis.set_major_locator(locator3)
       #ax[3].xaxis.set_major_formatter(mdates.AutoDateFormatter(locator3))
        ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
        ax[3].set_xlabel('Date of Event')
        ax[3].set_title('Distribution of All Bolide Candidates in Time (Belief >= {})'.format(bolideBeliefThreshold))
    
        plt.tight_layout()
        plt.show()
    
        return

    #******************************************************************************
    def find_bolide_hot_spots(self, n_peaks=1, beliefSource='machine', confidenceThreshold=0.2,
            bracketingSigmaThreshold=4.0, radius_threshold=4.0, max_radius=100.0):
        """ Generates a bolide detection heat map and then finds the hot spots in the heat map.

        This will compute a radius aroudn each hot spot centroid that encampasses radius_threshold amount of the hot
        spot.

        Parameters
        ----------
        n_peaks : int
            Number of peaks to find
        beliefSource : str
            Which source to use for belief {'human', 'machine'}
        confidenceThreshold : float
            Confidence threshold to include bolides in heat map
        bracketingSigmaThreshold : float
            How many times above the RMS of the heat map to declare a peak region
            (For bracketing purposes)
        radius_threshold : float
            What fraction of the hot spot to be covered by the circle
            in range [0, 100] percent
        max_radius : float
            Maximum radius for the hot spot encpompassing circle

        The property self.hotSpots is populated with the found hot spots.

        Returns
        -------
        hotSpots : HotSpots class
            .lat_peaks : float array
                The found latitude peaks
            .lon_peaks : float array
                The found longitude peaks
            .radius : float array
                The radius the encompases the hot spots

        """

        allBeliefs = self.generate_bolide_belief_array(beliefSource)

        # Finds bolides above confidence threshold
        highConfDetections = allBeliefs >= confidenceThreshold

        # Generate list of latitude and longitude
        avgLat = np.array([p.features.avgLat for p in self.bolideDispositionProfileList])
        avgLon = np.array([p.features.avgLon for p in self.bolideDispositionProfileList])
        lat = avgLat[highConfDetections]
        lon = avgLon[highConfDetections]
        
        # Find the hot spots
        lon_peaks, lat_peaks, radius = geoUtil.find_hot_spots(lon, lat, n_peaks=n_peaks,
                bracketingSigmaThreshold=bracketingSigmaThreshold, radius_threshold=radius_threshold)

        hotSpots = bFeatures.HotSpots(lon_peaks, lat_peaks, radius)

        self.hotSpots = hotSpots

        return hotSpots

    #******************************************************************************
    def generate_hot_spot_removal_optimization(self, bolidesFromWebsite, hotSpots=None, targetThreshold=0.5,
            max_radius=100.0, recomputeHotSpotFeature=False, satellite=None):
        """  
        This method will will optimize the deemphasis parameters for
        each hot spot in order to minimize false positives (while maximizing recall) due to the hot spots.

        This method require ground truth via bolidesFromWebsite. If ground truth is not available then use
        radius foudn in find_bolide_hot_spots, which is a more crude method.

        Parameters
        ----------
        bolidesFromWebsite  -- [WebsiteBolideEvent list] created by:
            bolide_dispositions.pull_dispositions_from_website 
        hotSpots : HotSpots class
            If passed then use the listed hot spots, if not passed then use the hotSpots at self.hotSpots
        targetThreshold : float
            The target detection threshold we optimize the masking regions for
        max_radius : float
            The maximum radius of the hard wall cyclinder deemphasis region in degrees lat/lon
        recomputeHotSpotFeature : bool
            If True then compute the hot spot feature, otherwise, use those stored in
            detection.features.hotSpot
        satellite : [str] 
            Which GLM to compute removal for 'G**'

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The fitted deemphasis box parameters

        """

        # Local import to not have a cicular import
        from bolide_detection_performance import find_bolide_match_from_website

        # Make a copy to not manipulate the original
        bolidesFromWebsiteLocal = copy.deepcopy(bolidesFromWebsite)
       #bolideDispositionProfileListLocal = copy.deepcopy(self.bolideDispositionProfileList)
        bolideDispositionProfileListLocal = self.bolideDispositionProfileList

        if hotSpots is None:
            # Hot spots not passed, use those stored in the object
            assert isinstance(self.hotSpots, bFeatures.HotSpots), 'Hot spots have not been found, call find_bolide_hot_spots'
            hotSpots = self.hotSpots
        else:
            raise Exception('call self.find_bolide_hot_spots first.')

        # Compute distance to hot spots using the hot spot feature
        if recomputeHotSpotFeature:
            detections = [b.bolideDetection for b in bolideDispositionProfileListLocal]
            if (np.all([d is None for d in detections])):
                raise Exception('The bolideDetection information needs to be copied over (construct with  copyOverBolideDetection=False)')
            hotSpotFeature = bFeatures.hot_spot_feature(detections, hotSpots=hotSpots, plot_figure=False, ax=None)
        else:
            hotSpotFeature = [b.features.hotSpot for b in bolideDispositionProfileListLocal]
        
        #****************
        # The deemphasis function for each hot spot in the static method 'deemphasisFcn'



        #***
        # First get the bolide truths from the website data
        # Determine how many website matches were found, ignoring confidenceThreshold
        matchIndex = find_bolide_match_from_website(bolideDispositionProfileListLocal , bolidesFromWebsiteLocal, satellite,
                confidenceThreshold=0.0, beliefSource='machine')
        # 0 or above means a match
        bolideTruth = np.full(len(matchIndex), False)
        trueHere = np.nonzero(np.greater_equal(matchIndex, 0))
        bolideTruth[trueHere] = True

        #***
        # Measure the original Positive Likelihood Ratio as a baseline
        bolideBelief = [copy.copy(bolide.machineOpinions[0].bolideBelief) for bolide in bolideDispositionProfileListLocal]
        orig_bolideBelief = copy.copy(bolideBelief)
       #orig_precisions, orig_recalls, thresholdsTrain = precision_recall_curve(bolideTruth, orig_bolideBelief)
       #orig_auc = auc(orig_recalls, orig_precisions)

        # Positive Likelihood Ratio = (TPR/FPR)
        fpr, tpr, thresholds = roc_curve(bolideTruth, orig_bolideBelief)
        targetHere = np.argmin(np.abs(thresholds - targetThreshold))
        orig_PLR = tpr[targetHere] / fpr[targetHere]

        #***
        # Compute a new PLR using a overly aggresive deemphasis in order to compute the lower bound to the PLR value range
        X = [max_radius, 1.0, max_radius, 1.0]
        # Set the gamma regulaizer to zero in order to return the negative PLR
        gamma = 0.0
        # If gamma == 0.0 then the merit function is the negative of the PLR
        new_PLR = -self._deemph_minimize_fcn(X, hotSpotFeature, orig_bolideBelief, bolideTruth, targetThreshold, gamma)
        

        #*********
        # Optimize deemphasis regions based on PLR
        # Set gamma so that at the maximum radius bound the weighted merit function is below the PLR difference above
        # Delta_PLR = gamma * n_hot_spots * max_radius
        # gamma = Delta_PLR / (n_hot_spots * max_radius)
        n_hot_spots = len(hotSpots.lon_peaks)
        gamma = (new_PLR - orig_PLR) / (n_hot_spots * max_radius)

        #*******
        # Try sequential 1 dimensional PLR optimization
       #X = [20.0, 1.0]
       #hotSpotFeature1Dim = [h[0] for h in hotSpotFeature]
       ## If gamma == 0.0 then the merit function is megative the PLR
       #new_PLR_1D = -self._deemph_minimize_fcn(X, hotSpotFeature1Dim, orig_bolideBelief, bolideTruth, targetThreshold, gamma)
        
       #for idx in np.arange(n_hot_spots):
       #    X = [5.0, 0.5]
       #    boundBox = ((0.5, 100.0),(0.1, 1.0))
       #    hotSpotFeature1Dim = [h[idx] for h in hotSpotFeature]
       #    result = minimize(self._deemph_minimize_fcn, X, args=(hotSpotFeature1Dim, orig_bolideBelief, bolideTruth,
       #        targetThreshold, gamma), bounds=boundBox, options={'disp':True})
        
       #    print(result)

        # Plot merit function vs radius
        print('Generating Diagnostic Figure...')
        merit_array = []
        PLR_array = []
        radius_array = np.linspace(1.0, max_radius, num=100)
        for radius in radius_array:
            X = [radius, 1.0, radius, 1.0]
            PLR_array.append(-self._deemph_minimize_fcn(X, hotSpotFeature, orig_bolideBelief, bolideTruth,
                targetThreshold, 0.0))
            merit_array.append(self._deemph_minimize_fcn(X, hotSpotFeature, orig_bolideBelief, bolideTruth,
                targetThreshold, gamma))
        plt.plot(radius_array, PLR_array, '.b', label='PLR')
        # Set an offset to the merit function so it is plotted well on top of the PLR plot
        merit_offset = np.max(PLR_array) - np.max(merit_array)
        plt.plot(radius_array, np.array(merit_array)+merit_offset , '.r', label='Optimizer Merit Function')
        plt.title('Positive Likelihood Ratio Merit Function vs. Hard Wall Radius')
        plt.xlabel('Hard Cylindrical Wall Radius')
        plt.ylabel('PLR = TPR/FPR')
        plt.legend()
        plt.grid()
        plt.show()



        #*****************
        # Optimize the fit
        # Set the initial guess to a high radius near the maximum and minimum alpha
        # Set bounds to radius = [0, 100] km
        # Set alpha bounds to [0,1]
        # Set the initial Radius to a large number, just under the maximum bound.
        # Set the alpha to the minimum bound
        print('Optimizing the fit...')
        X = []
        boundBox = []
        min_alpha = 0.1
        for idx,_ in enumerate(hotSpots.lon_peaks):
            X.extend([max_radius - 1.0, min_alpha])
            boundBox.extend(((0.5, max_radius),(min_alpha, 1.0)))
        result = minimize(self._deemph_minimize_fcn, X, args=(hotSpotFeature, orig_bolideBelief, bolideTruth,
            targetThreshold, gamma), bounds=boundBox, options={'disp':True}, method='Nelder-Mead')


        return result

    #******************************************************************************
    # Simple hard wall
    @staticmethod
    def deemphasisFcn(radius, alpha, bolideDistToHotSpot): 
        """ This is the deemphasis function that computes how much the detection score should be biased down based on
        proximity to hot spots.

        Right now, the deemphasis region is a simple cylinder centered on the hot spot with hard walls with a constant
        deemphasis value, alpha.

        This function operates on each individual bolide candidate.

        All input arrays should be the same length equal to the number of hot spots.

        Parameters
        ----------
        radius : float array(nHotSpots)
            The radius of the circular deemphasis region in km
        alpha : float array(nHotSpots)
            The deemphasis amoutn to the detection score
            Can be in the range [0,1], 1 meaning total masking
        bolideDistToHotSpot : float array(nHotSpots)
            Distance a bolide candidate is to a hot spot in km
            This is the hot spot feature

        Returns
        -------
        deemphScore : float
            The amount to deemphasize the bolide detection candidate

        """

        # If only a single hot spot then convert to lists
        # This is one thing Matlab has on Python: everything is already list, no need to convert!
        if np.array(radius).shape == ():
            radius = [radius]
            alpha = [alpha]
            bolideDistToHotSpot = [bolideDistToHotSpot]

        # Look at each hot spot and see if any is within the masking region
        # We assume each masking region does not overlap
        for r, a, b in zip(radius, alpha, bolideDistToHotSpot):
            if b <= r:
                return a

        # Nothing within any masking region, return no deemphasis
        return 0.0

        #***
        # 2D Gaussian PDF
        # NOTE: The for-loop below doies not work properly with lambda function the ele,ent 0 lambda function is
        # overwritten with the latter one. 
        # TODO: figure out a way to generate a list of lambda function in a for-loop
      # deemphasisFcn = []
      # covMat = lambda sigma_lon,sigma_lat,lonLatMix: np.array([[sigma_lon, lonLatMix],
      #                                                           [lonLatMix, sigma_lat]])
      ##for spotIdx,(lon,lat) in enumerate(zip(hotSpots.lon_peaks, hotSpots.lat_peaks)):
      ##    mu = [lon, lat]
      ##    deemphasisFcn.append(lambda alpha, sigma_lon, sigma_lat, lonLatMix, lon, lat: 
      ##                        alpha*multivariate_normal(mu, covMat(sigma_lon,sigma_lat,lonLatMix)).pdf([lon,lat]))

      # deemphasisFcn = [[],[]]
      # deemphasisFcn[0] = lambda alpha, sigma_lon, sigma_lat, lonLatMix, lon, lat: \
      #                         alpha*multivariate_normal((hotSpots.lon_peaks[0], hotSpots.lat_peaks[0]), covMat(sigma_lon,sigma_lat,lonLatMix)).pdf([lon,lat])
      # deemphasisFcn[1] = lambda alpha, sigma_lon, sigma_lat, lonLatMix, lon, lat: \
      #                         alpha*multivariate_normal((hotSpots.lon_peaks[1], hotSpots.lat_peaks[1]), covMat(sigma_lon,sigma_lat,lonLatMix)).pdf([lon,lat])


    #******************************************************************************
    def _deemph_minimize_fcn(self, X, hotSpotFeature, orig_bolideBelief, bolideTruth, targetThreshold, gamma):
        """

        Performs these steps:

        1) Deemphasizes the bolide beloef based on the deemphasis parameters given in X. 
        2) Then compute the Positive Likelihood Ratio (PLR) at the given bolide belide confidence threshold.
        3) Applies the gamma regularization term to penalize large cylinders
        4) Returns the negative of the regularized PLR as the minimizer merit function.

        Parameters
        ----------
        X : the variables to optimize
            Composed of pairs of radius and alpha (radius0, alpha0, radius1, alpha1, etc...)
            Length equal to twice the number of hot spots

        """

        bolideBelief = copy.copy(orig_bolideBelief)

        # Divide the X in radius and alpha arrays
        # but only if there is more than one hot spot
        if np.shape(X) != (2,):
            dealt_X = np.reshape(X, (int(len(X)/2), 2), order='F')
        else:
            dealt_X = X

        # Note that zip could be used here but zip makes a copy and does not pass by reference!
        # So to change the bolideBelief array we cannot use zip to deal out
        for idx, bolideDistToHotSpot in enumerate(hotSpotFeature):
            bolideBelief[idx] -= self.deemphasisFcn(dealt_X[0], dealt_X[1], bolideDistToHotSpot)
        # New bolide belief must be between 0.0 and 1.0
        bolideBelief = [np.max([0.0, b]) for b in bolideBelief]
        bolideBelief = [np.min([1.0, b]) for b in bolideBelief]

        #***
        # Positive Likelihood Ratio = (TPR/FPR)
        fpr, tpr, thresholds = roc_curve(bolideTruth, bolideBelief)
        # Find the TPR and FPR at the target confidence threshold
        targetHere = np.argmin(np.abs(thresholds - targetThreshold))
        PLR = tpr[targetHere] / fpr[targetHere]

        # Add in regularization term to penalize the radius so it does not grow too big
        gamma_term = 0.0
        if (not isinstance(dealt_X[0], float)) and len(dealt_X[0]) > 1:
            for radius in dealt_X[0]:
                gamma_term += gamma * radius
        else:
            gamma_term += gamma * dealt_X[0]
        merit = PLR - gamma_term

        return -merit
    
    #******************************************************************************
    # Plots a histogram of detections phase folded to specific phase.
    #
    # Inputs:
    #   beliefSource            -- [str] Which source to use for belief {'human', 'machine'}
    #   bolideBeliefThreshold   -- [float] Threshold for belief to be a bolide ( >= )
    #   period                  -- [float] The period to phase-fold the data at in hours
    #   
    #******************************************************************************
    def phase_fold_detections(self, beliefSource, bolideBeliefThreshold=0.5, period=24):
        
        # Collect the information we want
        # stored in self.bolideDispositionProfileList
        goesSatellite = [disposition.features.goesSatellite for disposition in self.bolideDispositionProfileList]
        bolideTime  = np.array([disposition.features.bolideTime for disposition in self.bolideDispositionProfileList])
        longitude   = np.array([disposition.features.avgLon for disposition in self.bolideDispositionProfileList])

        # Convert to mean solar time
      # bolideTime = convert_time_to_local(bolideTime, longitude, local_type='meanSolar')
    
        allBeliefs = self.generate_bolide_belief_array(beliefSource)
    
        nBeliefs = len(allBeliefs)
    
        raise Exception('Set up for G18 and G19')
        g16Here = [g == 'G16' for g in goesSatellite]
        g17Here = [g == 'G17' for g in goesSatellite]
        g16Beliefs = allBeliefs[g16Here]
        g17Beliefs = allBeliefs[g17Here]
    
        #***
        # Set up figure
        fig, ax = plt.subplots(1,1)
        fig.set_figwidth(7.0)
        fig.set_figheight(5.0)

        #***
        # Phase-fold the data at the given period
        if period <= 24:
            # We are folding within one day so ony keep the "time" part of the bolideTime
            # Set the date to 2021/1/1
            bolideTime = np.array([datetime.datetime.combine(datetime.date(2021,1,1),b.time()) for b in bolideTime])

        if period != 24:
            raise Exception('Right now this only folds at 24 hours exactly')


        mdateTimeOrigin = mdates.date2num(bolideTime)
        
        # Histogram of time for each bolide
        timeBins = 24
       #locator1 = mdates.AutoDateLocator()
        locator1 = mdates.HourLocator()
        g16TrueHere = np.logical_and(np.array(g16Here), allBeliefs >= bolideBeliefThreshold)
        if (np.any(g16TrueHere)):
            ax.hist(mdateTimeOrigin[g16TrueHere], bins=timeBins, log=False, facecolor='b', alpha=0.5, 
                label='G16 tot: {}'.format(np.count_nonzero(g16TrueHere)))
        g17TrueHere = np.logical_and(np.array(g17Here), allBeliefs >= bolideBeliefThreshold)
        if (np.any(g17TrueHere)):
            ax.hist(mdateTimeOrigin[g17TrueHere], bins=timeBins, log=False, facecolor='g', alpha=0.5, 
                label='G17 tot: {}'.format(np.count_nonzero(g17TrueHere)))
        ax.grid(axis='y')
        ax.legend()
        ax.xaxis.set_major_locator(locator1)
       #ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator1))
       #ax.xaxis.set_major_formatter(mdates.DateFormatter('%H%M%S'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.set_xlabel('Hour of Event')
        ax.set_title('Distribution of All Bolide Candidates in Time (Belief >= {})'.format(bolideBeliefThreshold))
    
    
        plt.tight_layout()
        plt.show()
    
        pass



# END class BolideDispositions
# ************************************************************************************************************
# ************************************************************************************************************
# ************************************************************************************************************

# *****************************************************************************
# rebalance_data_set (X, y, responseRatio)
#
# Rebalances a training set so that there is the specified ratio of positive to negative responses y.
# 
# Inputs:
#   X   -- [nd.array float matrix[nDatums,nFeatures]] the training features
#   y   -- [nd.array bool(nDatums)] The logical reponses
#   reponseRatio    -- [float] ratio of trues to negatives (0.1 => 10% are trues)
#
# Outputs:
#   X   -- [nd.array float matrix] the training features rebalanced
#   y   -- [nd.array bool(nDatums)] The logical reponses rebalanced
#   
# *****************************************************************************
def rebalance_data_set (X, y, responseRatio):

    if (not len(X[:,1]) == len(y)):
        raise Exception ('X and y do not appear to have the correct number of datums')

    # Tack y on to the end of X so we keep the sort order
    X = np.append(X,np.reshape(y, (len(y),1)), axis=1)

    # Get number of negatives to keep
    nPositives = np.count_nonzero(y)
    nNegatives = round(nPositives / responseRatio) - nPositives
    if (nNegatives > len(X[:,1])):
        # Not enough negatives, need to cut back on the positives
        raise Exception ('Write this condition')

    # Get the negative datums in the X feature matrix
    XPositive = X[np.nonzero(y)[0], :]
    XNegative = X[np.nonzero(np.logical_not(y))[0], :]
    shuffled_indices = np.random.permutation(len(XNegative[:,1]))
    XNegative = XNegative[shuffled_indices[0:nNegatives],:]

    X = np.append(XNegative, XPositive, axis=0)

    # Reshuffle
    shuffled_indices    = np.random.permutation(len(X[:,1]))
    X = X[shuffled_indices,:]

    return [X[:,0:-1], X[:,-1]]



# *****************************************************************************
# Parse an argument list.
#
# INPUTS
#     arg_list : A list of strings, each containing a command line argument.
#                NOTE that the first element of this list should NOT be the
#                program file name. Instead of passing sys.argv, pass
#                arg_list = sys.argv[1:]
#
# OUTPUTS
#     args     : A Namespace containing the extracted arguments.
#
# *****************************************************************************
def parse_arguments(arg_list):

    parser = argparse.ArgumentParser(description='Generate several plots from a compiled bolideDispositionProfileList.')
    parser.add_argument('inFile', metavar='inFile', type=str, nargs=1,
                        help='Input file path containing the compiled dispositions.')

    args = parser.parse_args(arg_list)

    return args


#*************************************************************************************************************
# This __main__ function is used to test the methods in this module

if __name__ == "__main__":

    pass
