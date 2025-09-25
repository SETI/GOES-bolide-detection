# Bolide_database Module
#
# Contains methods to manage the bolide detection and rejection databases.
#

import os
import sys
import time
import numpy as np
import ZODB, ZODB.FileStorage, zc.zlibstorage
from zc.lockfile import LockError 
import BTrees.OOBTree
import transaction
import time_utilities as timeUtil
import bolide_detections as bd
from datetime import datetime, date
import io_utilities as ioUtil

databaseFileBasename = 'bolide_database'

#*************************************************************************************************************
# class BolideDatabase
#
# This ZODB-based database contains a collection of bolide detections. 
#
# Uses a zlibstorage compressed database to save space.
#
# For a very large database, retreiving all the keys can be slow and require a lot of memory. 
# So, we store all the keys in the <detection_keys> and <rejection_keys> attributes.
#
# BTrees.LOBTree uses a int64 (signed) as the key type.
# 
# 
# Attributes:
#   databasePath    -- [str]  full path to database (must end with '.fs')
#   open_status    -- [int]
#           0 : Successful open
#           1 : Couldn't lock database file
#           2 : Any other database opening error
#   detections      -- [LOBTree] collection of detected bolides
#   detection_keys  -- [int64 np.ndarray] detected bolide IDs
#   n_tot_detections -- [int] running tally of all detections
#   rejections      -- [LOBTree] collection of rejected bolide clusters
#   rejection_keys  -- [int64 np.ndarray] detected bolide IDs
#   all_keys        -- [int64 np.ndarray] all bolide IDs
#   n_tot_rejections -- [int] running tally of all detections
#   n_tot_entries    -- [dependent property] n_tot_detections + n_tot_rejections
#   provenance  -- from [run_glm_pipeline_batch.Provenance] Configuration of pipeline that generated dataset
#                   But converted to a dict
#   storage     -- [ZODB.FileStorage] 
#   db          -- [ZODB.DB]
#   connection  -- [ZODB.Connection]
#
#*************************************************************************************************************
class BolideDatabase:

    def __init__(self, databasePath, provenance=None, purgeDatabase=False, wait_time_sec=0.0, **kwargs):
        """ Initialize a new database object. 
        This will either load in an already existing ZODB database or create a new one

        Parameters:
        -----------
        databasePath    : str 
            full path to database (must end with '.fs')
            If database file does not exist then a new database is created.
        provenance      : run_glm_pipeline_batch.Provenance 
            Configuration and provenance information for run that generated this data
            If None then do not set the attribute. 
        purgeDatabase   : bool 
            If true then purge (delete) all data in current database
        wait_time_sec  : float 
            The number of seconds to wait for the database to become available
            This can be used, for example, if another process is using the database file and this process waits in
            incrementes of 5.0 seconds until the other process is finished. 
            If The database file does not exist then this parameter should have no effect.
            If the database cannot be opening in this amount of time then an exception is raised
        **kwargs        -- passed to the db.open function
                        For example, the specify a at=datetime for readonly
        """

        assert os.path.splitext(databasePath)[1] == '.fs', "Must pass ZODB filename with extension '.fs'"

        self.databasePath = databasePath


        self.storage    = None
        self.db         = None
        self.connection = None

        still_waiting = True
        wait_sec = 5.0
        sleep_count_sec = 0.0
        while still_waiting:
            self.open_status = self.open(**kwargs)
            if self.open_status == 0:
                still_waiting = False
            else:
                # Pause wait_sec seconds and try again
                time.sleep(wait_sec)
                sleep_count_sec += wait_sec
            
            if sleep_count_sec > wait_time_sec:
                raise Exception('Timeout trying to open database')
        
        # Record the provenance in the database
        if provenance is not None:
            self.connection.root.provenance = BTrees.IOBTree.IOBTree()
            self.connection.root.provenance[1] = provenance.copy_to_dict()
        elif not hasattr(self.connection.root, 'provenance'):
            self.connection.root.provenance = None

        #***
        if purgeDatabase:
            # Purge current bolides
            # We want the bolides to be part of a ZODB database, so use BTrees
            self.connection.root.detections = BTrees.LOBTree.LOBTree()
            self.connection.root.rejections = BTrees.LOBTree.LOBTree()
            self.connection.root.detection_keys = np.array([], dtype=int)
            self.connection.root.rejection_keys = np.array([], dtype=int)
            self.connection.root.n_tot_detections = int(0)
            self.connection.root.n_tot_rejections = int(0)
        
        # If no bolides yet loaded then create empty BTrees
        if not hasattr(self.connection.root, 'detections'):
            # New database
            self.connection.root.detections = BTrees.LOBTree.LOBTree()
            self.connection.root.detection_keys = np.array([], dtype=int)
            self.connection.root.n_tot_detections = int(0)
        if not hasattr(self.connection.root, 'rejections'):
            # New database
            self.connection.root.rejections = BTrees.LOBTree.LOBTree()
            self.connection.root.rejection_keys = np.array([], dtype=int)
            self.connection.root.n_tot_rejections = int(0)

        self.commit()

    @property
    def detections(self):
        return self.connection.root.detections

    @property
    def rejections(self):
        return self.connection.root.rejections

    @property
    def detection_keys(self):
        return self.connection.root.detection_keys

    @property
    def rejection_keys(self):
        return self.connection.root.rejection_keys

    @property
    def all_keys(self):
        return np.r_[self.detection_keys, self.rejection_keys]

    @property
    def n_tot_detections(self):
        return self.connection.root.n_tot_detections

    @n_tot_detections.setter
    def n_tot_detections(self, n):
        self.connection.root.n_tot_detections = n

    @property
    def n_tot_rejections(self):
        return self.connection.root.n_tot_rejections

    @n_tot_rejections.setter
    def n_tot_rejections(self, n):
        self.connection.root.n_tot_rejections = n

    @property
    def provenance(self):
        return self.connection.root.provenance[1]

    @property
    def is_open(self):
        """ Reports if connection is open
        """
        return (self.connection is not None and self.connection._storage is not None)

    @property
    def n_tot_entries(self):
        """ The total number of detections and rejections

        Computing the total number of detections or rejections can be slow and memory intensive to save a running tally and report that here
        """
        return self.n_tot_detections + self.n_tot_rejections

    def commit(self):
        """ Commit changes to database
        """
        transaction.commit()

    def abort(self):
        """ Abort changes to database
        """
        transaction.abort()

    def cacheMinimize(self):
        """ Minimize cache sizes for all connections

        Call this if the resident memory is too high.

        TODO: figure out why setting a low cache size is not limiting memory usage
        """
        self.db.cacheMinimize()

    def open(self, **kwargs):
        """ Opens the database and creates a connection

        **kwargs    -- passed to the db.open function
                        For example, the specify a at=datetime for readonly

        Returns
        -------
        success : int
            0 : Successful open
            1 : Couldn't lock database file
            2 : Any other error
        """
        if (self.is_open):
            print('ZODB Connection already open')
            return 0

        try:
            self.storage = ZODB.FileStorage.FileStorage(self.databasePath)
        except LockError:
            return 1
        except:
            return 2

            
        compressed_storage = zc.zlibstorage.ZlibStorage(self.storage)
        self.db = ZODB.DB(compressed_storage, large_record_size=1<<27)
        self.connection = self.db.open(**kwargs)

        return 0

    def close(self):
        """Close a connecton, database and storage.

        Also packs the database
        """
        if (not self.is_open):
            print('ZODB Connection already closed')
            return

        self.connection.close()
        self.db.pack()
        self.db.close()
        self.storage.close()


    def add(self, bolideDetectionList, dataType='detections', just_generate_IDs=False):
        """ Adds bolide detections to a ZODB database of bolide detections or rejections

        If an ID is not already present for each bolide, a unique ID is generated and stored in the passed
        bolideDetectionList. Note: for speed, a copy is NOT made, it modifies the passed bolideDetectionList.

        The keys in the database are the IDs so we cannot have duplicates.  If the ID of a 
        new bolide to add is a duplicate then the three digit counter is incremented by 
        one for the detection ID to be added.  This will change the bolideDetectionList by incrementing the ID.
        
        Parameters
        ----------
        bolideDetectionList -- [list of bolide_detections.bolideDetection objects] The bolide detections 
        dataType            -- [str] Add to 'detections' or 'rejections'
        just_generate_IDs   -- [bool] If True then only generate a set of unique IDs for the passed bolideDetectionList
                                Do not actually add them to the database
        
        """
        
        assert ['detections', 'rejections'].index(dataType) >= 0, 'Unknown dataType'
        
        if len(bolideDetectionList) == 0:
            return
        
        # Use detection ID as key name [Int64]
        # Add list elements to BTree
        for bolide in bolideDetectionList:

            # If bolide.ID is not to format then it is probably a uint32 
            # then the ID was generated with generate_random_IDs and so we want to regenerate
            # with legit IDs.
            if not check_valid_ID(bolide.ID):
                # Generate ID
                bolide.ID = self._generate_ID(bolide)
        
        # Now check if any of the new bolides to add has a duplicate ID
        bolideDetectionList = self._check_duplicates(bolideDetectionList, all_current_IDs=self.all_keys)

        # If just generating IDs then return now with no changes to the database
        if just_generate_IDs:
            raise Exception('We should not be calling this anymore')
            return
        
        # Form a dictionary of all bolides to add
        bolideDict = {int(bolide.ID): bolide for bolide in bolideDetectionList}
        IDs = [int(b.ID) for b in bolideDetectionList]
        # Even though a LOBTree wants a int64 as the ID, it will not accept a np.int64. 
        # We must convert to a garden variety int!
        if dataType == 'detections':
            self.detections.update(bolideDict)
            self.connection.root.detection_keys = np.r_[self.connection.root.detection_keys, IDs]
            self.n_tot_detections += int(len(IDs))
        elif dataType == 'rejections':
            self.rejections.update(bolideDict)
            self.connection.root.rejection_keys = np.r_[self.connection.root.rejection_keys, IDs]
            self.n_tot_rejections += int(len(IDs))

        # Commit changes
        self.commit()

        return

    def add_cutout_features(self, cutoutFeatures):
        """ Adds cutout features generated by gen_goes_cutout to a detection. Note: will NOT add to rejections.

        cutoutFeatures[key].ID is the detection ID, so omr that we determine the correct detection entry to add to

        Parameters
        ----------
        cutoutFeatures : Dict of cutputOutputClass, Key being the detection ID

        """
        if not self.is_open:
            self.open()

        for key in cutoutFeatures:
            self.detections[cutoutFeatures[key].ID].add_cutout_features(cutoutFeatures[key])

        # "If you modify a non-persistent mutable value of a persistent-object 
        # attribute, you need to mark the persistent object as changed yourself by setting _p_changed to True"
        self.detections._p_changed = True
        self.commit()

    @staticmethod
    def _generate_ID(bolide):
        """
        Generates an identifier for a bolide detection with counter set to 001

        NOTE: This does NOT check for duplicates, that happens in _check_duplicates
     
        IIYYYYDDDHHMMSSs###
       
        II    = instrument number {16, 17, 18,...}
        YYYY  = year: e.g., 2015 
        DDD   = day of year: 001-366 
        HH    = UTC hour of day: 00-23 
        MM    = minute of hour: 00-59 
        SSs   = second of minute: 00-59 (60 indicates leap second and third “s” is tenth of second)
        ###   = index counter for each event that matches all previous digits.
       
        Example: 1620202622335217023
       
        which would be a detection on G16, with date stamp 20202622335217 and detection number 023 at that time. 
       
        This ID is an int64 with 19 digits (max int64 = 9,223,372,036,854,775,807), and yet, is human readable and conforms to
        the dating system used by GLM. 
        
        BTrees.LOBTree uses a int64 (signed) as the key type, so int64 is the maximum ID we can generate.
 
       
        Parameters
        ----------
        bolide.goesSatellite  -- [str] 'G**', The 'G' is stripped off and the 2 digit number is used
        bolide.bolideTime -- [datetime class] date and time of detection defined as mid time of all groups in detection
       
        Returns
        --------
            ID         -- [np.int64] Unique detection ID
        """
        
        # Satellite number
        satelliteIdx = int(bolide.goesSatellite[1:])
 
        satTimeInt = timeUtil.generate_glm_timestamp(bolide.bolideTime) + \
                            satelliteIdx * 100000000000000
 
        # Set the ID counter to 1
        ID = np.int64((satTimeInt * 1000 + 1))
 
        return ID

    @staticmethod
    def _check_duplicates(bolideDetectionList, all_current_IDs=None):
        """

        Checks for and resolves a duplicate ID duplicates by advancing the counter using an efficient method

        TODO: there's probably an even faster method. Find it!

        This will compare all detection IDs in bolideDetectionList to all those already in the database, and those in
        bolideDetectionList.

        Thsi si a static method so it can be used to generate unique ID before the database is populated (as in
        detect_bolide.py). Because of this, you must manually pass in any existing IDs with all_current_IDs.

        Start with the counter at 1. If it's a duplicate then add one to the highest counter already present.

        NOTE: For speed this method does not duplicate 

        Parameters
        ----------
        bolideDetectionList : list of bolide_detections.bolideDetection objects
            The bolide detections to check duplicate IDs
        all_current_IDs : list of int64
            All the current IDs to also compare the new IDs to
            This code will not change these IDs, which are assumed to already be unique.

        Returns
        -------
        bolideDetectionList -- [list of bolide_detections.bolideDetection objects] The bolide detections with 
                                all unique IDs
        """

        if all_current_IDs is not None:
            assert np.all([isinstance(ID, np.int64) for ID in all_current_IDs]), 'all_current_IDs must be a list of np.int64'

        # Add the IDs from bolideDetectionList to the keys so we check over all existing and to be added IDs to see if
        # there are any duplicates.
        new_IDs = [np.int64(b.ID) for b in bolideDetectionList]
        if all_current_IDs is None:
           all_current_IDs = np.array([], dtype=np.int64)
        n_current_IDs = len(all_current_IDs)
        all_IDs = np.concatenate((all_current_IDs, new_IDs), dtype=np.int64)

        for idx, ID in enumerate(new_IDs):
            duplicates = np.in1d(all_IDs, ID)
            # Since the ID is in all_IDs, we have a duplicate if there are two or more matches
            if (np.count_nonzero(duplicates) > 1):
                # At least one duplicate, If all duplicates are in the new_IDs then we do not want to advance the first
                # counter. Check now
                if (np.count_nonzero(np.in1d(all_current_IDs, ID)) == 0):
                    # All duplicates are in the new IDs. Skip the first instance
                    if new_IDs[0:idx].count(ID) == 0:
                        # This is the first duplicate instance, skip it
                        continue
                # Find the highest counter duplicate and add one to that to get the counter for the duplicate
                # This is slow, so only do it if there is at least one duplicate
                duplicates = np.in1d([int(key / 1000) for key in all_IDs], int(ID / 1000))
                # Add one to the current highest count ID
                bolideDetectionList[idx].ID = np.int64((np.max(all_IDs[duplicates]) + 1))
                # Check if we hit the counter limit
                if (bolideDetectionList[idx].ID % 1000 == 0):
                    raise Exception('Hit bolide ID counter limit!')
                # Adjust the ID in all_IDs to account for the new ID
                all_IDs[n_current_IDs + idx] = bolideDetectionList[idx].ID

        return bolideDetectionList


    def extract_bolideDetectionList(self, dataType='detections', IDs='all', deep_copy=True):
        """ Converts a bolide database into a list of bolideDetection objects
        
        Parameters
        ----------
        dataType    -- [str] extract 'detections' or 'rejections'
        IDs         -- [int64 list] List of detection (or rejection) IDs to extract ('all' => extract all)
        deep_copy   -- [bool] If True then perform deepcopy when returning the bolideDetectionList
                                Deep copying is slow so only do if necessary
        
        Returns
        -------
        bolideDetectionList -- [list of bolide_detections.bolideDetection objects] The bolide detections 
        """
        
        assert ['detections', 'rejections'].index(dataType) >= 0, 'Unknown dataType'

        # TODO: Simplify this
        if (type(IDs) == str and IDs == 'all'):
            if dataType == 'detections':
                IDs = self.detection_keys.copy()
            if dataType == 'rejections':
                IDs = self.rejection_keys.copy()

        if dataType == 'detections':
            keys = self.detection_keys.copy()
            IDs = IDs[np.isin(IDs, keys)]
            if deep_copy:
                bolideDetectionList = [self.detections[int(i)].copy() for i in IDs]
            else:
                bolideDetectionList = [self.detections[int(i)] for i in IDs]
        if dataType == 'rejections':
            keys = self.rejection_keys.copy()
            IDs = IDs[np.isin(IDs, keys)]
            if deep_copy:
                bolideDetectionList = [self.rejections[int(i)].copy() for i in IDs]
            else:
                bolideDetectionList = [self.rejections[int(i)] for i in IDs]
        
        # Commit changes
        # TODO: figure out why transaction must be committed when no changes occured.
       #self.commit()
        # Nothing to be changed to the database, abort any changes
        self.abort()

        return bolideDetectionList


#*************************************************************************************************************
def write_detection_list_to_csv_file(csvFilePath, bolideDetectionList, purgeCSVFile=False, GJ_data=False):
    """ 
    Record bolide detections in CSV format. Note that we record only the essential
    information necessary to reconstruct the bolideDetection object from the GLM
    L2 data file. If csvFilePath exists then data is appended to file.
    Event IDs is recorded if available, otherwise Group IDs are recorded.

    If there are no entries yet in the file for purgeCSVFile = True then a header is written

    Parameters
    ----------
    csvFilePath : [str] 
        full path to CSV file (must end with '.csv')
    bolideDetectionList : [list of bolideDetection objects] 
        The bolide detections 
    purgeCSVFile : [bool] 
        If true then purge (delete) all data in current CSV
    GJ_data : bool
        If True then append the extra data needed for the Boggs Gigantic Jets (GJ) study
    """

    assert os.path.splitext(csvFilePath)[1] == '.csv', "Must pass CSV filename with extension '.csv'"

    if purgeCSVFile:
        # Overwrite CSV file
        if os.path.isfile(csvFilePath):
            os.remove(csvFilePath)
    
    # If the file does not yet exist (or was just purged) then no header is present yet. Write it now.
    if not os.path.isfile(csvFilePath):
        if (len(bolideDetectionList) >= 1):
            bolideDetectionList[0].write_csv_header(csvFilePath, GJ_data=GJ_data)

    for detection in bolideDetectionList:
        detection.append_csv(csvFilePath, GJ_data=GJ_data)

#*************************************************************************************************************
def write_website_list_to_csv_file(csvFilePath, bolidesFromWebsite, purgeCSVFile=True):
    """ 
    Record bolide detections from Website in CSV format.

    If there are no entries yet in the file for purgeCSVFile = True then a header is written

    Parameters
    ----------
    csvFilePath         -- [str] full path to CSV file (must end with '.csv')
    bolidesFromWebsite  -- [WebsiteBolideEvent list] The bolide detections from website 
    purgeCSVFile        -- [bool] If true then purge (delete) all data in current database
    """

    assert os.path.splitext(csvFilePath)[1] == '.csv', "Must be a CSV filename with extension '.csv'"

    if purgeCSVFile:
        # Overwrite CSV file
        if os.path.isfile(csvFilePath):
            os.remove(csvFilePath)
    
    # If the file does not yet exist (or was just purged) then no header is present yet. Write it now.
    if not os.path.isfile(csvFilePath):
        if (len(bolidesFromWebsite) >= 1):
            bolidesFromWebsite[0].write_csv_header(csvFilePath)

    for detection in bolidesFromWebsite:
        detection.append_csv(csvFilePath)

#*************************************************************************************************************
def generate_random_IDs (bolideDetectionList):
    """ Generates a random set of uint32 IDs.

    This is for use when generating detections outside of the database, i.e. in detect_bolides.py or where we don't
    really care about the ID, we just want them all to be unique.

    By using uint32 we force the BolideDatabase.add function to regenerate the IDs if and when the bolideDetectionList
    is added to a database.

    Parameters
    ----------
    bolideDetectionList : [List of bolideDetection objects] The detected bolides to generate fake IDs for.

    Returns
    -------
    bolideDetectionList : [List of bolideDetection objects] The detected bolides wtih fake IDs generated.
        .ID
    

    """

    rng = np.random.default_rng(seed=42)
    rints = rng.integers(low=np.iinfo(np.uint32).min, high=np.iinfo(np.uint32).max,
            size=len(bolideDetectionList), dtype=np.uint32)
    for idx, detection in enumerate(bolideDetectionList):
        detection.ID = rints[idx]

    return bolideDetectionList

#*************************************************************************************************************
def check_valid_ID(ID):
    """ Checks the ID to see if it is a legitimate ID format. 

    This does not check for duplicates or if the dates are reasonable. You can pass a string or an int

    It just checks for the format to be:

    1) np.int64 compatible (e.g. can be a string of 19 numeric characters long
    2) 19 digits total
    3) First two digits are in the valid GOES satellite range

    Parameters
    ----------
    ID : any type
        The ID to check

    Returns
    -------
    isValid : bool
        True if the ID conforms to the format
    """
    
    if ID is None:
        return False

    # detect_bolides generates random uint32 IDs, these are not legit
    if isinstance(ID, np.uint32):
        return False

    # Try to type as an np.int64. If succeeds then is a valid int64
    try: 
        np.int64(ID)
    except:
        return False

    return True
