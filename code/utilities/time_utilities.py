# Module containing utilities to manipulate time.

import numpy as np
from datetime import datetime, timedelta
import re
import os

#*************************************************************************************************************
def convert_time_to_local(time, longitude, local_type='meanSolar'):

    """ Converts the time from UTC time (as is in the netCDF file) to local time.

    This will either work on a single value or a numpy array of values

    There are several options for local time:

    1) Sidereal time

    2) Apparent solar time

    3) Mean solar time

    Mean solar time is simple. UTC is very nearly mean solar time for the prime meridian so all that is needed to
    move to local mean solar time is add one hour for every 15 degrees longitude.

    Parameters
    ----------
    time        : datetime numpy.ndarray
        The UTC time
    longitude   : float numpy.ndarray
        The longitude to convert to local time
    local_type : str
        The type of local time to convert to.
        Options: {'meanSolar'}

    Returns
    -------
        localTime   : datetime array
            In the requested local time

    """

    if len(longitude) != len(time):
        raise Exception('time and longitude arrays must be equal length')

    localTime = np.full(np.shape(time), datetime(1,1,2))
    for idx in np.arange(len(time)):
        if local_type == 'meanSolar':
            # 1 hour for every 15 degrees longitude
            localTime[idx] = time[idx] + timedelta(hours=longitude[idx]/15.0)
        
        else:
            raise Exception('local_type not supported')

    return localTime

#*************************************************************************************************************
# Check time offset units, given the apparent discrepancy between the L2 PUG
# and the GLM data files.
#
# NOTE: This function is probably not necessary for any purpose with GLM data.
#
# INPUTS
#     offsetList :  A list of floating point time offsets. The list may contain
#                   any number of duplicates and may be in any order. It is
#                   assumed that the majority of unique offsets in this list are
#                   one sampling interval apart.
#
# OUTPUTS
#     unitStr   : A string (either 'seconds' or 'miliseconds') giving the
#                 inferred time units.
#
# NOTES
#
#*************************************************************************************************************
def infer_time_offset_units(offsetList):

    raise Exception('This function is not maintained. remove?')

    INTEGRATION_TIME_SEC = 0.002 # Sampling interval in seconds

    # Filter out any values that are not of type 'float'.
    validInd = [isinstance(i, float) for i in offsetList]
    offsetList = list(compress(offsetList, validInd))

    uniqueOffsets = list(set(offsetList))
    uniqueOffsets.sort()
    firstDifference = [j-i for i, j in zip(uniqueOffsets[:-1], uniqueOffsets[1:])]
    medianTimeStep = np.nanmedian(firstDifference)
    if np.abs(medianTimeStep - INTEGRATION_TIME_SEC) < np.abs(medianTimeStep - 1000 * INTEGRATION_TIME_SEC) :
        unitStr = 'seconds'
    else:
        unitStr = 'milliseconds'
    return unitStr


#*************************************************************************************************************
# Convert each element of offsetList to a time offset in milliseconds, if
# necessary.
#*************************************************************************************************************
def force_miliseconds(offsetList):
    unitStr = infer_time_offset_units(offsetList)
    if unitStr == 'milliseconds':
        pass  # No need to convert
    elif unitStr == 'seconds':
        offsetList = [o * 1000.0 for o in offsetList]
    return offsetList

#*************************************************************************************************************
def generate_glm_timestamp(timestamp):
    """ Converts a datetime object to a GLM netCDF time stamp

    GLM netCDF timestamp format:

    YYYYDDDHHMMSSs
    
    YYYY  = year: e.g., 2015 
    DDD   = day of year: 001-366 
    HH    = UTC hour of day: 00-23 
    MM    = minute of hour: 00-59 
    SSs   = second of minute: 00-59 (60 indicates leap second and third “s” is tenth of second)
    
    Parameters
    ----------
    timestamp   : [datetime] 
        a datetime object

    Returns
    -------
    glm_timestamp : int
        A GLM format timestamp

    """

    # Date number
    year            = timestamp.year
    dayOfYear       = timestamp.timetuple().tm_yday
    hour            = timestamp.hour
    minute          = timestamp.minute
    second          = timestamp.second
    tenthOfSecond   = timestamp.microsecond // 100000
 
    # Check if all the above (satellite and date) already line up with an existing detection
    glm_timestamp = tenthOfSecond + second * 10 + minute * 1000 + hour * 100000 + \
                    dayOfYear * 10000000 + year * 10000000000

    return glm_timestamp

#*************************************************************************************************************
def convert_glm_timestamp_to_datetime(timestamp):
    """ Converts a GLM file timestamp to a datetime object.

    Date stamp in format:
    YYYYDDDHHMMSSs
    
    YYYY  = year: e.g., 2015 
    DDD   = day of year: 001-366 
    HH    = UTC hour of day: 00-23 
    MM    = minute of hour: 00-59 
    SSs   = second of minute: 00-59 (60 indicates leap second and third “s” is tenth of second)

    
    Parameters
    ----------
    timestampArray   : [int or str] GLM time stamp

    Returns
    -------
    glmDatetime : datetime.datetime


    """

    assert isinstance(timestamp, int) or isinstance(timestamp, str) or isinstance(timestamp, np.int64), \
                'timestamp must be an int or a str'
    
    # Convert to string
    timestamp = str(timestamp)
    
    assert len(timestamp) == 14, 'timestamp must be 14 digits'
    
    year          = int(timestamp[0:4])
    dayOfYear     = int(timestamp[4:7])
    month         = (datetime(year, 1, 1) + timedelta(dayOfYear - 1)).month
    day           = (datetime(year, 1, 1) + timedelta(dayOfYear - 1)).day
    hour          = int(timestamp[7:9])
    minute        = int(timestamp[9:11])
    second        = int(timestamp[11:13])
    tenthOfSecond = int(timestamp[13:14])
    microsecond   = int(tenthOfSecond * 1e5)

    return datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=microsecond)
        
#*************************************************************************************************************
def extract_total_seconds_from_glm_timestamp(timestampArray):
    """ Converts a GLM netCDF time stamp to total seconds since bolide epoch (BolideDetection.epoch). 
    For use when comparing two timestamps.

    Can either take a single int or string or a list of ints or strings. Then returns either a scalar or a list

    Date stamp in format:
    YYYYDDDHHMMSSs
    
    YYYY  = year: e.g., 2015 
    DDD   = day of year: 001-366 
    HH    = UTC hour of day: 00-23 
    MM    = minute of hour: 00-59 
    SSs   = second of minute: 00-59 (60 indicates leap second and third “s” is tenth of second)

    Parameters
    ----------
    timestampArray   : [int or list of int] date stamp

    Returns
    -------
    totalSeconds : [float] total number of seconds since BolideDetection.epoch
    """

    # Import this locally so we do not get a circular import
    from bolide_detections import BolideDetection

    if not isinstance(timestampArray, list) and not isinstance(timestampArray, np.ndarray):
        wasScalar = True
        timestampArray = [timestampArray]
    else:
        wasScalar = False

    timeSinceEpochList = []
    for timestamp in timestampArray:
        glmDatetime = convert_glm_timestamp_to_datetime(timestamp)

        timeSinceEpoch = glmDatetime - BolideDetection.epoch

        timeSinceEpochList.append(timeSinceEpoch.total_seconds())

    if wasScalar:
        return timeSinceEpochList[0]
    else:
        return np.array(timeSinceEpochList)

#*************************************************************************************************************
def convert_GLM_date_str_to_ISO(string):
    """ Takes a date string from a GLM netCDF file and converts it to ISO text format.

    Example:
        `nc4Data.time_coverage_start = '2020-03-10T02:23:18.2Z'

        is converted to:

        '2020-03-10T02:23:18.200'

    which can then be used to generate a datetime.datetime object via:

    datetime.datetime.fromisoformat('2020-03-10T02:23:18.200')

    Parameters
    ----------
    string  : str
        A GLM netCDF date string

    Returns
    -------
    ISOString : str
        String converted to ISO format

    """

    if not isinstance(string, str):
        raise Exception('This function operates on strings')

    # Remove the 'Z' and replace with '00'
    try:
        idx = string.index('Z')
        if idx != len(string)-1:
            raise Exception('String format not as expected')
        ISOString = string[0:idx] + '00'
    except:
        raise Exception('String format not as expected')

    return ISOString

#*************************************************************************************************************
def replace_char_in_string(string, idx, a):
    """ Replaces a single character in a string to a specified replacement character at index idx.

    Strings in ython are immutable, so we need to convert the string to a list, replaces the element then convert back
    to a string.

    Parameters
    ----------
    string  : str
        The string to modify
    idx     : int
        Index of character in string to replace
    a       : str
        The single character to replace

    Returns
    -------
    newString   : str
        With the character replaced

    """

    # Convert string to list so that we can manipulate it
    strList = list(string)
    # Change the element
    strList[idx] = a
    # Convert back to a string
    newString = ''.join(aList)

    return newString

#*************************************************************************************************************
def extract_ISO_date_from_directory(path, data_source):
    """ Extracts an ISO date ('YYYY-MM-DD') from a directory path

    Note: If path is a directory and not a specific file, then the trailing '/' MUST be present.

    Parameters
    ----------
    path : str
        A directory path

        For GeoNEX contains
        L2: './G*/YYYY/dayOfYear/*'
        L0: './G*/YYYY/MMDD/*'

        For daily bundles contains './G1*/2022/OR_GLM-L2-LCFA_G1*_sYYYMMDD.nc.tgz'
    data_source : str
        one of: bolide_io.input_config.valid_data_sources

    Returns
    -------
    ISODate : str
        Date in ISO format: 'YYYY-MM-DD'

    """

    if data_source == 'geonex':
        # Get the day from the /YYYY/dayOfYear and convert to ISO format:  'YYYY-MM-DD'
        yearResult = re.search('/20../', path)
        if yearResult:
            year = int(path[yearResult.start()+1:yearResult.end()-1])
            # We now need to see if the day and month is referenced by dayOfYear (3 digits) or MMDD (4 digits)

            dayOfYearResult = re.search(yearResult.group(0) + '.../', path)
            MMDDResult = re.search(yearResult.group(0) + '..../', path)
            if dayOfYearResult:
                dayOfYear = int(path[dayOfYearResult.end()-4:dayOfYearResult.end()-1])
                dateObj = (datetime(year, 1, 1) + timedelta(dayOfYear - 1))
                month   = dateObj.month
                day     = dateObj.day
        
                # Make sure the use format: YYYY-MM-DD (so, pad the single digit months and days with a zero)
                return str(year)+'-{:02}-'.format(month)+'{:02}'.format(day)

            elif MMDDResult:
                MMDD    = path[MMDDResult.end()-5:MMDDResult.end()-1]
                month   = int(MMDD[0:2])
                day     = int(MMDD[2:4])
        
                # Make sure the use format: YYYY-MM-DD (so, pad the single digit months and days with a zero)
                return str(year)+'-{:02}-'.format(month)+'{:02}'.format(day)
            
            else:
                return None
        else:
            return None

    elif data_source == 'daily_bundle':
        # Get the day from the ./G1*/YYYY/OR_GLM-L2-LCFA_G1*_sYYYYMMDD.nc.tgz and convert to ISO format:  'YYYY-MM-DD'
        yearResult = re.search('/20../', path)
        if yearResult:
            year = int(path[yearResult.start()+1:yearResult.end()-1])
            # There should be no subdirectories for daily bundles data
            # Get the day from the filename
            filename = os.path.basename(path)
            dateResult = re.search('s.........nc.tgz', path)
            if dateResult:
                yearFromFile = int(dateResult.group(0)[1:5])
                if yearFromFile != year:
                    print('Daily bundle file year does not agree with directory year')
                    return None

                month   = int(dateResult.group(0)[5:7])
                day     = int(dateResult.group(0)[7:9])

                return str(year)+'-{:02}-'.format(month)+'{:02}'.format(day)
            else: 
                # Check if this is a temp ramdisk file
                # If so, the format is: /tmp/ramdisk/glm_tmp/G1*/YYYY/MMDD/OR_GLM-L2-LCFA_G17_sYYYYDDD*.nc
                tmpFileResult = re.search('OR_GLM-L2-LCFA_G1._s.*_e.*_c.*.nc', path)
                if tmpFileResult:
                    # temp data file, grab day from directory name
                    month   = int(path[yearResult.start()+6:yearResult.start()+8])
                    day     = int(path[yearResult.start()+8:yearResult.start()+10])
                    return str(year)+'-{:02}-'.format(month)+'{:02}'.format(day)
                else:
                    return None
        else:
            return None
    elif data_source == 'files':
        """
        # Get the day from directory: ./G1*/YYYY/MMDD/OR_GLM-L2-LCFA_G1*_s*_e*_c*.nc'
        yearResult = re.search('/20../', path)
        if yearResult:
            year = int(path[yearResult.start()+1:yearResult.end()-1])
            month   = int(path[yearResult.start()+6:yearResult.start()+8])
            day     = int(path[yearResult.start()+8:yearResult.start()+10])
            return str(year)+'-{:02}-'.format(month)+'{:02}'.format(day)
        else:
            return None
        """

        # Get the date from the filename
        tmpFileResult = re.search('OR_GLM-L2-LCFA_G1._s.*_e.*_c.*.nc', path)
        filename = path[tmpFileResult.start():tmpFileResult.end()]
        # Find the start timestamp
        timestampResult = re.search('_s.*', filename)
        timestamp = int(filename[timestampResult.start()+2:timestampResult.start()+16])

        return convert_glm_timestamp_to_datetime(timestamp).isoformat()
    else:
        return None

