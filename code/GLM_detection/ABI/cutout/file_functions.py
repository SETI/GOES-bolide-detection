#!/usr/bin/python

import sys
import os
import time
from os import listdir
import glob
import fnmatch
import datetime
import csv
import numpy as np
#import requests
import fnmatch

from ABI_image_processing_functions import latlon_to_grid

verbose=False

def sortfiles(str1):
    t = str1.split('_')
    snum=t[len(t)-3].replace('s','')
    return int(snum)

def get_monthday(eventtime):
    time = datetime.datetime.strptime(eventtime[0:11],"%Y%j%H%M")
    return time.strftime("%m%d")

def get_scans(lat,lon):
    """ JCS: I wonder what this function does...

    It looks like it determines if the data is within the field of view of each instrument.

    RadC := CONUS scene
    RadF := Full Disk
    """
    
    #TODO:update this to do based on files, not static data values
    #grab G16, G17, G18 and G19 file for the day
    
    G16 = []
    G17 = []
    G18 = []
    G19 = []
    
    # JCS: These look like latitude and longtiude values
    # JCS: looks like these are the field of view limits of the satellites
    G16F = [0.0,-75.0,0.0,-75.0,81.3282,-81.3282,-156.2995,6.2995]
    G16C = [0.0,-75.0,30.083003,-87.096958,56.761450,14.571340,-152.109282,-52.946879]
    G17F = [0.0,-137.0,0.0,-137.0,81.3282,-81.3282,141.7005,-55.7005]
    G17C = [0.0,-137.0,29.967,-137.000,53.500062,14.571340,175.623576,-89.623576]
    # JCS: G18 same as G17?
    G18F = [0.0,-137.0,0.0,-137.0,81.3282,-81.3282,141.7005,-55.7005]
    G18C = [0.0,-137.0,29.967,-137.000,53.500062,14.571340,175.623576,-89.623576]
    # JCS: G19 same as G16?
    G19F = [0.0,-75.0,0.0,-75.0,81.3282,-81.3282,-156.2995,6.2995]
    G19C = [0.0,-75.0,30.083003,-87.096958,56.761450,14.571340,-152.109282,-52.946879]
    GtestF = [0.0,-89.5,0.0,-89.5,81.3282,-81.3282,-170.7995,-8.2005]

    xoffset_16C = -0.101346
    xscale_factor_16C = 2.8e-05
    yoffset_16C = 0.128226
    yscale_factor_16C = -2.8e-05
    semi_minor_axis_rpol_16C = 6356752.31414
    semi_major_axis_req_16C = 6356752.31414
    perspective_point_height_16C = 35786023.0
    longitude_of_projection_origin_gamma_16C = -1.30899694
    H_16C = 42142775.31414
    e_16C = 0.0

    xoffset_17C = -0.069986
    xscale_factor_17C = 2.8e-05
    yoffset_17C = 0.128226
    yscale_factor_17C = -2.8e-05
    semi_minor_axis_rpol_17C = 6356752.31414
    semi_major_axis_req_17C = 6356752.31414
    perspective_point_height_17C = 35786023.0
    longitude_of_projection_origin_gamma_17C = -2.39110108
    H_17C = 42142775.31414
    e_17C = 0.0

    # JCS: I will assume G18 has similar values to G17
    xoffset_18C = -0.069986
    xscale_factor_18C = 2.8e-05
    yoffset_18C = 0.128226
    yscale_factor_18C = -2.8e-05
    semi_minor_axis_rpol_18C = 6356752.31414
    semi_major_axis_req_18C = 6356752.31414
    perspective_point_height_18C = 35786023.0
    longitude_of_projection_origin_gamma_18C = -2.39110108
    H_18C = 42142775.31414
    e_18C = 0.0

    # JCS: I will assume G19 has similar values to G16
    xoffset_19C = -0.101346
    xscale_factor_19C = 2.8e-05
    yoffset_19C = 0.128226
    yscale_factor_19C = -2.8e-05
    semi_minor_axis_rpol_19C = 6356752.31414
    semi_major_axis_req_19C = 6356752.31414
    perspective_point_height_19C = 35786023.0
    longitude_of_projection_origin_gamma_19C = -1.30899694
    H_19C = 42142775.31414
    e_19C = 0.0

    aa=0#90
    bb=0#180

    lat_n = float(lat) + 5 + aa
    lat_s = float(lat) - 5 + aa
    lon_w = float(lon) - 5 + bb
    lon_e = float(lon) + 5 + bb

    G16GLM = [66.56,-66.56,-141.56,-8.44]
    G17GLM = [66.56,-66.56,-203.56,-70.44]
    # JCS: G18 same as G17?
    G18GLM = [66.56,-66.56,-203.56,-70.44]
    # JCS: G19 same as G16?
    G19GLM = [66.56,-66.56,-141.56,-8.44]
 
    #***
    #check 16 Full Disk
    if lat_n <= (G16F[4]+aa) and lat_s >= (G16F[5]+aa) and lon_w >= (G16F[6]+bb) and lon_e <= (G16F[7]+bb) and float(lat)<=G16GLM[0] and float(lat)>=G16GLM[1] and float(lon)>=G16GLM[2] and float(lon)<=G16GLM[3]:
        G16F=True
    else:
        G16F=False
        
    if G16F:
        #check 16 CONUS
        [gridx_nw,gridy_nw] = latlon_to_grid(lat_n,lon_w,semi_minor_axis_rpol_16C,semi_major_axis_req_16C,H_16C,longitude_of_projection_origin_gamma_16C,xoffset_16C,xscale_factor_16C,yoffset_16C,yscale_factor_16C,e_16C)
        [gridx_ne,gridy_ne] = latlon_to_grid(lat_n,lon_e,semi_minor_axis_rpol_16C,semi_major_axis_req_16C,H_16C,longitude_of_projection_origin_gamma_16C,xoffset_16C,xscale_factor_16C,yoffset_16C,yscale_factor_16C,e_16C)
        [gridx_se,gridy_se] = latlon_to_grid(lat_s,lon_e,semi_minor_axis_rpol_16C,semi_major_axis_req_16C,H_16C,longitude_of_projection_origin_gamma_16C,xoffset_16C,xscale_factor_16C,yoffset_16C,yscale_factor_16C,e_16C)
        [gridx_sw,gridy_sw] = latlon_to_grid(lat_s,lon_w,semi_minor_axis_rpol_16C,semi_major_axis_req_16C,H_16C,longitude_of_projection_origin_gamma_16C,xoffset_16C,xscale_factor_16C,yoffset_16C,yscale_factor_16C,e_16C)
      
        #check that all points are within range (0->3000,0->5000)
        if (gridx_nw >= 0 and gridx_nw <= 5000 and gridx_ne >= 0 and gridx_ne <= 5000 and gridx_se >= 0 and gridx_se <= 5000 and gridx_sw >= 0 and gridx_sw <= 5000 and gridy_nw >= 0 and gridy_nw <= 3000 and gridy_ne >= 0 and gridy_ne <= 3000 and gridy_se >= 0 and gridy_se <= 3000 and gridy_sw >= 0 and gridy_sw <= 3000) :
            G16C=True
        else:
            G16C=False
    else:
        G16C=False

    #***
    #check 17 Full Disk
    if lat_n <= (G17F[4]+aa) and lat_s >= (G17F[5]+aa) and (lon_w >= (G17F[6]+bb) or lon_w <= (G17F[7]+bb)) and (lon_e >= (G17F[6]+bb) or lon_e <= (G17F[7]+bb)) and float(lat)<=G17GLM[0] and float(lat)>=G17GLM[1] and ((float(lon)>=G17GLM[2] and float(lon)<=G17GLM[3]) or (float(lon)>=156.44 and float(lon)<=180.0)):
        G17F=True
    else:
        G17F=False

    if G17F:
        #check 17 PACUS    
        [gridx_nw,gridy_nw] = latlon_to_grid(lat_n,lon_w,semi_minor_axis_rpol_17C,semi_major_axis_req_17C,H_17C,longitude_of_projection_origin_gamma_17C,xoffset_17C,xscale_factor_17C,yoffset_17C,yscale_factor_17C,e_17C)
        [gridx_ne,gridy_ne] = latlon_to_grid(lat_n,lon_e,semi_minor_axis_rpol_17C,semi_major_axis_req_17C,H_17C,longitude_of_projection_origin_gamma_17C,xoffset_17C,xscale_factor_17C,yoffset_17C,yscale_factor_17C,e_17C)
        [gridx_se,gridy_se] = latlon_to_grid(lat_s,lon_e,semi_minor_axis_rpol_17C,semi_major_axis_req_17C,H_17C,longitude_of_projection_origin_gamma_17C,xoffset_17C,xscale_factor_17C,yoffset_17C,yscale_factor_17C,e_17C)
        [gridx_sw,gridy_sw] = latlon_to_grid(lat_s,lon_w,semi_minor_axis_rpol_17C,semi_major_axis_req_17C,H_17C,longitude_of_projection_origin_gamma_17C,xoffset_17C,xscale_factor_17C,yoffset_17C,yscale_factor_17C,e_17C)

        #check that all points are within range (0->3000,0->5000)
        if (gridx_nw >= 0 and gridx_nw <= 5000 and gridx_ne >= 0 and gridx_ne <= 5000 and gridx_se >= 0 and gridx_se <= 5000 and gridx_sw >= 0 and gridx_sw <= 5000 and gridy_nw >= 0 and gridy_nw <= 3000 and gridy_ne >= 0 and gridy_ne <= 3000 and gridy_se >= 0 and gridy_se <= 3000 and gridy_sw >= 0 and gridy_sw <= 3000) :
            G17C=True
        else:
            G17C=False
    else:
        G17C=False

    #***
    #check 18 Full Disk
    if lat_n <= (G18F[4]+aa) and lat_s >= (G18F[5]+aa) and (lon_w >= (G18F[6]+bb) or lon_w <= (G18F[7]+bb)) and (lon_e >= (G18F[6]+bb) or lon_e <= (G18F[7]+bb)) and float(lat)<=G18GLM[0] and float(lat)>=G18GLM[1] and ((float(lon)>=G18GLM[2] and float(lon)<=G18GLM[3]) or (float(lon)>=156.44 and float(lon)<=180.0)):
        G18F=True
    else:
        G18F=False

    if G18F:
        #check 18 PACUS    
        [gridx_nw,gridy_nw] = latlon_to_grid(lat_n,lon_w,semi_minor_axis_rpol_18C,semi_major_axis_req_18C,H_18C,longitude_of_projection_origin_gamma_18C,xoffset_18C,xscale_factor_18C,yoffset_18C,yscale_factor_18C,e_18C)
        [gridx_ne,gridy_ne] = latlon_to_grid(lat_n,lon_e,semi_minor_axis_rpol_18C,semi_major_axis_req_18C,H_18C,longitude_of_projection_origin_gamma_18C,xoffset_18C,xscale_factor_18C,yoffset_18C,yscale_factor_18C,e_18C)
        [gridx_se,gridy_se] = latlon_to_grid(lat_s,lon_e,semi_minor_axis_rpol_18C,semi_major_axis_req_18C,H_18C,longitude_of_projection_origin_gamma_18C,xoffset_18C,xscale_factor_18C,yoffset_18C,yscale_factor_18C,e_18C)
        [gridx_sw,gridy_sw] = latlon_to_grid(lat_s,lon_w,semi_minor_axis_rpol_18C,semi_major_axis_req_18C,H_18C,longitude_of_projection_origin_gamma_18C,xoffset_18C,xscale_factor_18C,yoffset_18C,yscale_factor_18C,e_18C)

        #check that all points are within range (0->3000,0->5000)
        if (gridx_nw >= 0 and gridx_nw <= 5000 and gridx_ne >= 0 and gridx_ne <= 5000 and gridx_se >= 0 and gridx_se <= 5000 and gridx_sw >= 0 and gridx_sw <= 5000 and gridy_nw >= 0 and gridy_nw <= 3000 and gridy_ne >= 0 and gridy_ne <= 3000 and gridy_se >= 0 and gridy_se <= 3000 and gridy_sw >= 0 and gridy_sw <= 3000) :
            G18C=True
        else:
            G18C=False
    else:
        G18C=False

    #***
    #check 19 Full Disk
    if lat_n <= (G19F[4]+aa) and lat_s >= (G19F[5]+aa) and lon_w >= (G19F[6]+bb) and lon_e <= (G19F[7]+bb) and float(lat)<=G19GLM[0] and float(lat)>=G19GLM[1] and float(lon)>=G19GLM[2] and float(lon)<=G19GLM[3]:
        G19F=True
    else:
        G19F=False
        
    if G19F:
        #check 19 CONUS
        [gridx_nw,gridy_nw] = latlon_to_grid(lat_n,lon_w,semi_minor_axis_rpol_19C,semi_major_axis_req_19C,H_19C,longitude_of_projection_origin_gamma_19C,xoffset_19C,xscale_factor_19C,yoffset_19C,yscale_factor_19C,e_19C)
        [gridx_ne,gridy_ne] = latlon_to_grid(lat_n,lon_e,semi_minor_axis_rpol_19C,semi_major_axis_req_19C,H_19C,longitude_of_projection_origin_gamma_19C,xoffset_19C,xscale_factor_19C,yoffset_19C,yscale_factor_19C,e_19C)
        [gridx_se,gridy_se] = latlon_to_grid(lat_s,lon_e,semi_minor_axis_rpol_19C,semi_major_axis_req_19C,H_19C,longitude_of_projection_origin_gamma_19C,xoffset_19C,xscale_factor_19C,yoffset_19C,yscale_factor_19C,e_19C)
        [gridx_sw,gridy_sw] = latlon_to_grid(lat_s,lon_w,semi_minor_axis_rpol_19C,semi_major_axis_req_19C,H_19C,longitude_of_projection_origin_gamma_19C,xoffset_19C,xscale_factor_19C,yoffset_19C,yscale_factor_19C,e_19C)
      
        #check that all points are within range (0->3000,0->5000)
        if (gridx_nw >= 0 and gridx_nw <= 5000 and gridx_ne >= 0 and gridx_ne <= 5000 and gridx_se >= 0 and gridx_se <= 5000 and gridx_sw >= 0 and gridx_sw <= 5000 and gridy_nw >= 0 and gridy_nw <= 3000 and gridy_ne >= 0 and gridy_ne <= 3000 and gridy_se >= 0 and gridy_se <= 3000 and gridy_sw >= 0 and gridy_sw <= 3000) :
            G19C=True
        else:
            G19C=False
    else:
        G19C=False

    return [G16F,G16C,G17F,G17C,G18F,G18C,G19F,G19C]

def get_closest_glm(_eventtime, config, window_mins, detection_satellite):
    """ Returns the nearest GLM data files within a window for a given event time.

    On occasion, the GLM data for the detection satellite exists, but not for the other satellite. In these situations,
    prioritize the detection satellite data, and allow for the other satellite data being missing.

    Parameters
    ----------
    _eventtime : str
        Time of start of GLM netCDF file in filename time format
    config : cutoutConfiguration
    window_mins : str
        The window in minutes
    detection_satellite : str
        The satellite where the detection was made {'G16', 'G17', 'G18', etc...}

    Returns
    -------
    G16closests : str list
        The G16 GLM files within the given window
    G17closests : str list
    G18closests : str list
    G19closests : str list

    """

    raise Exception('This function is no longer used')

    G16files = []
    G17files = []
    G18files = []
    G19files = []

    eventtime = datetime.datetime.strptime(_eventtime[0:11],"%Y%j%H%M")
    stime = eventtime + datetime.timedelta(hours=-1)#minutes=-30)
    etime = eventtime + datetime.timedelta(hours=1)#minutes=30)
    _stime = stime.strftime("%Y%j%H%M")
    _etime = etime.strftime("%Y%j%H%M")
    sstime = eventtime + datetime.timedelta(hours=-2)#minutes=-30)
    eetime = eventtime + datetime.timedelta(hours=2)#minutes=30)
    _sstime = sstime.strftime("%Y%j%H%M")
    _eetime = eetime.strftime("%Y%j%H%M")

    _year = int(_eventtime[0:4])
    _hr = int(_eventtime[7:9])
    _syear = int(_stime[0:4])
    _sday = int(_stime[4:7])
    _shr = int(_stime[7:9])
    _eyear = int(_etime[0:4])
    _eday = int(_etime[4:7])
    _ehr = int(_etime[7:9])
    _ssyear = int(_sstime[0:4])
    _ssday = int(_sstime[4:7])
    _sshr = int(_sstime[7:9])
    _eeyear = int(_eetime[0:4])
    _eeday = int(_eetime[4:7])
    _eehr = int(_eetime[7:9])
    
    _monthday = get_monthday(_eventtime)
    _smonthday = get_monthday(_stime)
    _emonthday = get_monthday(_etime)
    _ssmonthday = get_monthday(_sstime)
    _eemonthday = get_monthday(_eetime)

    _monthday = _eventtime[4:7]
    _smonthday = _stime[4:7]
    _emonthday = _etime[4:7]
    _ssmonthday = _sstime[4:7]
    _eemonthday = _eetime[4:7]
    


    _G16dir = config.G16DIR
    _G17dir = config.G17DIR
    _G18dir = config.G18DIR
    _G19dir = config.G19DIR


    off_eventtime = int(_eventtime)+1#int(_eventtime[0:14])+1

    _G16targetname = 'OR_GLM-L2-LCFA_G16_s'+str(off_eventtime)+'_A_A'
    _G17targetname = 'OR_GLM-L2-LCFA_G17_s'+str(off_eventtime)+'_A_A'
    _G18targetname = 'OR_GLM-L2-LCFA_G18_s'+str(off_eventtime)+'_A_A'
    _G19targetname = 'OR_GLM-L2-LCFA_G19_s'+str(off_eventtime)+'_A_A'
    

    #just add all files within a 2hr period, then filter

    # TODO: Clean up this repeat code 
    G16files.extend(glob.glob(os.path.join(_G16dir,str(_ssyear)+'/'+str(_ssmonthday)+'/'+str(_sshr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G16_s'+str(_ssyear)+str(_ssday).zfill(3)+str(_sshr).zfill(2)+"*.nc")))
    G16files.extend(glob.glob(os.path.join(_G16dir,str(_syear)+'/'+str(_smonthday)+'/'+str(_shr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G16_s'+str(_syear)+str(_sday).zfill(3)+str(_shr).zfill(2)+"*.nc")))
    G16files.extend(glob.glob(os.path.join(_G16dir,str(_year)+'/'+str(_monthday)+'/'+str(_hr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G16_s'+_eventtime[0:9]+"*.nc")))
    G16files.extend(glob.glob(os.path.join(_G16dir,str(_eyear)+'/'+str(_emonthday)+'/'+str(_ehr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G16_s'+str(_eyear)+str(_eday).zfill(3)+str(_ehr).zfill(2)+"*.nc")))
    G16files.extend(glob.glob(os.path.join(_G16dir,str(_eeyear)+'/'+str(_eemonthday)+'/'+str(_eehr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G16_s'+str(_eeyear)+str(_eeday).zfill(3)+str(_eehr).zfill(2)+"*.nc")))

    G17files.extend(glob.glob(os.path.join(_G17dir,str(_ssyear)+'/'+str(_ssmonthday)+'/'+str(_sshr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G17_s'+str(_ssyear)+str(_ssday).zfill(3)+str(_sshr).zfill(2)+"*.nc")))
    G17files.extend(glob.glob(os.path.join(_G17dir,str(_syear)+'/'+str(_smonthday)+'/'+str(_shr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G17_s'+str(_syear)+str(_sday).zfill(3)+str(_shr).zfill(2)+"*.nc")))
    G17files.extend(glob.glob(os.path.join(_G17dir,str(_year)+'/'+str(_monthday)+'/'+str(_hr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G17_s'+_eventtime[0:9]+"*.nc")))
    G17files.extend(glob.glob(os.path.join(_G17dir,str(_eyear)+'/'+str(_emonthday)+'/'+str(_ehr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G17_s'+str(_eyear)+str(_eday).zfill(3)+str(_ehr).zfill(2)+"*.nc")))
    G17files.extend(glob.glob(os.path.join(_G17dir,str(_eeyear)+'/'+str(_eemonthday)+'/'+str(_eehr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G17_s'+str(_eeyear)+str(_eeday).zfill(3)+str(_eehr).zfill(2)+"*.nc")))

    G18files.extend(glob.glob(os.path.join(_G18dir,str(_ssyear)+'/'+str(_ssmonthday)+'/'+str(_sshr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G18_s'+str(_ssyear)+str(_ssday).zfill(3)+str(_sshr).zfill(2)+"*.nc")))
    G18files.extend(glob.glob(os.path.join(_G18dir,str(_syear)+'/'+str(_smonthday)+'/'+str(_shr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G18_s'+str(_syear)+str(_sday).zfill(3)+str(_shr).zfill(2)+"*.nc")))
    G18files.extend(glob.glob(os.path.join(_G18dir,str(_year)+'/'+str(_monthday)+'/'+str(_hr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G18_s'+_eventtime[0:9]+"*.nc")))
    G18files.extend(glob.glob(os.path.join(_G18dir,str(_eyear)+'/'+str(_emonthday)+'/'+str(_ehr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G18_s'+str(_eyear)+str(_eday).zfill(3)+str(_ehr).zfill(2)+"*.nc")))
    G18files.extend(glob.glob(os.path.join(_G18dir,str(_eeyear)+'/'+str(_eemonthday)+'/'+str(_eehr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G18_s'+str(_eeyear)+str(_eeday).zfill(3)+str(_eehr).zfill(2)+"*.nc")))

    G19files.extend(glob.glob(os.path.join(_G19dir,str(_ssyear)+'/'+str(_ssmonthday)+'/'+str(_sshr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G19_s'+str(_ssyear)+str(_ssday).zfill(3)+str(_sshr).zfill(2)+"*.nc")))
    G19files.extend(glob.glob(os.path.join(_G19dir,str(_syear)+'/'+str(_smonthday)+'/'+str(_shr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G19_s'+str(_syear)+str(_sday).zfill(3)+str(_shr).zfill(2)+"*.nc")))
    G19files.extend(glob.glob(os.path.join(_G19dir,str(_year)+'/'+str(_monthday)+'/'+str(_hr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G19_s'+_eventtime[0:9]+"*.nc")))
    G19files.extend(glob.glob(os.path.join(_G19dir,str(_eyear)+'/'+str(_emonthday)+'/'+str(_ehr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G19_s'+str(_eyear)+str(_eday).zfill(3)+str(_ehr).zfill(2)+"*.nc")))
    G19files.extend(glob.glob(os.path.join(_G19dir,str(_eeyear)+'/'+str(_eemonthday)+'/'+str(_eehr).zfill(2)+'/'+'OR_GLM-L2-LCFA_G19_s'+str(_eeyear)+str(_eeday).zfill(3)+str(_eehr).zfill(2)+"*.nc")))

    # We want to track which is the detection satellite.
    # We can tollerate missing GLM data for the other satellite, but not the detection satellite.
    detected_in_G16 = False
    detected_in_G17 = False
    detected_in_G18 = False
    detected_in_G19 = False
    if detection_satellite == 'G16':
        detected_in_G16 = True
    elif detection_satellite == 'G17':
        detected_in_G17 = True
    elif detection_satellite == 'G18':
        detected_in_G18 = True
    elif detection_satellite == 'G19':
        detected_in_G19 = True
    else:
        raise Exception('Unknown satellite')

    G16closests = _get_closest_glm_files(G16files, _eventtime, window_mins, detected_in_G16)
    G17closests = _get_closest_glm_files(G17files, _eventtime, window_mins, detected_in_G17)
    G18closests = _get_closest_glm_files(G18files, _eventtime, window_mins, detected_in_G18)
    G19closests = _get_closest_glm_files(G19files, _eventtime, window_mins, detected_in_G19)

    return [G16closests,G17closests,G18closests,G19closests]

def get_closest_abi(_eventtime,lat,lon,local, config):
    """ Returns the closest ABI filenames to the given event.
    One file on either side of the bolide event in time

    Instead of searching all band data, this just searches for the nearest config.single_fig_band data files.
    One can then assume the other band files are also available.

    Returns
    -------
    closest_abi_files : Dict
        'G16' : list [fileBefore, fileAfter] or None if no close-by files
        'G17' : list [fileBefore, fileAfter]
        'G18' : list [fileBefore, fileAfter]
        'G19' : list [fileBefore, fileAfter]

    """
    G16files = []
    G17files = []
    G18files = []
    G19files = []

    #parsetime
    #2020168133738.695
    eventtime = datetime.datetime.strptime(_eventtime[0:11],"%Y%j%H%M")
    stime = eventtime + datetime.timedelta(hours=-1)#minutes=-30)
    etime = eventtime + datetime.timedelta(hours=1)#minutes=30)
    _stime = stime.strftime("%Y%j%H%M")
    _etime = etime.strftime("%Y%j%H%M")
    
    _syear = int(_stime[0:4])
    _sday = int(_stime[4:7])
    _shr = int(_stime[7:9])
    _smin = int(_stime[9:11])
    _eyear = int(_etime[0:4])
    _eday = int(_etime[4:7])
    _ehr = int(_etime[7:9])
    _emin = int(_etime[9:11])

    targetfilename = "G_"+_eventtime[0:14]+"_A_A"
    
    # Determine which satellite ABI data can view the bolide event
    [G16F,G16C,G17F,G17C,G18F,G18C,G19F,G19C] = get_scans(lat,lon)
    #add all files to an array (g16, g17, g18 and g19 separately)

    
    if local==1:
        _dirpre = ''
        if G16F:
            G16files.extend(glob.glob(_dirpre+'OR_ABI-L1b-RadF-M*C'+str(config.single_fig_band).zfill(2)+'_G16_*.nc'))
        if G16C:
            G16files.extend(glob.glob(_dirpre+'OR_ABI-L1b-RadC-M*C'+str(config.single_fig_band).zfill(2)+'_G16_*.nc'))
        if G17F:
            G17files.extend(glob.glob(_dirpre+'OR_ABI-L1b-RadF-M*C'+str(config.single_fig_band).zfill(2)+'_G17_*.nc'))
        if G17C:
            G17files.extend(glob.glob(_dirpre+'OR_ABI-L1b-RadC-M*C'+str(config.single_fig_band).zfill(2)+'_G17_*.nc'))
        if G18F:
            G18files.extend(glob.glob(_dirpre+'OR_ABI-L1b-RadF-M*C'+str(config.single_fig_band).zfill(2)+'_G18_*.nc'))
        if G18C:
            G18files.extend(glob.glob(_dirpre+'OR_ABI-L1b-RadC-M*C'+str(config.single_fig_band).zfill(2)+'_G18_*.nc'))
        if G19F:
            G19files.extend(glob.glob(_dirpre+'OR_ABI-L1b-RadF-M*C'+str(config.single_fig_band).zfill(2)+'_G16_*.nc'))
        if G19C:
            G19files.extend(glob.glob(_dirpre+'OR_ABI-L1b-RadC-M*C'+str(config.single_fig_band).zfill(2)+'_G16_*.nc'))
    else:
        for y in range(_syear,_eyear+1):
            sd = _sday if (y==_syear) else 0
            ed = _eday if (y==_eyear) else 365
            for d in range(sd,ed+1):
                shr = _shr if (d==_sday) else 0
                ehr = _ehr if (d==_eday) else 23
                for h in range(shr,ehr+1):
                    if G16F and config.G16ABIFDIR is not None:
                        _dirpre = config.G16ABIFDIR
                        _prefix = 'OR_ABI-L1b-RadF-M*C'+str(config.single_fig_band).zfill(2)+'_G16_'
                    
                        _dir = os.path.join(_dirpre,str(y)+'/'+ str(d).zfill(3)+'/'+str(h).zfill(2)+'/')
                        arr = glob.glob(_dir+_prefix+'*.nc')
                        G16files.extend(arr)
                    if G16C and config.G16ABICDIR is not None:
                        _dirpre = config.G16ABICDIR
                        _prefix = 'OR_ABI-L1b-RadC-M*C'+str(config.single_fig_band).zfill(2)+'_G16_'
                    
                        _dir = os.path.join(_dirpre,str(y)+'/'+ str(d).zfill(3)+'/'+str(h).zfill(2)+'/')
                        arr = glob.glob(_dir+_prefix+'*.nc')
                        G16files.extend(arr)
                    if G17F and config.G17ABIFDIR is not None:
                        _dirpre = config.G17ABIFDIR
                        _prefix = 'OR_ABI-L1b-RadF-M*C'+str(config.single_fig_band).zfill(2)+'_G17_'
                    
                        _dir = os.path.join(_dirpre,str(y)+'/'+ str(d).zfill(3)+'/'+str(h).zfill(2)+'/')
                        arr = glob.glob(_dir+_prefix+'*.nc')
                        G17files.extend(arr)
                    if G17C and config.G17ABICDIR is not None:
                        _dirpre = config.G17ABICDIR
                        _prefix = 'OR_ABI-L1b-RadC-M*C'+str(config.single_fig_band).zfill(2)+'_G17_'
                    
                        _dir = os.path.join(_dirpre,str(y)+'/'+ str(d).zfill(3)+'/'+str(h).zfill(2)+'/')
                        arr = glob.glob(_dir+_prefix+'*.nc')
                        G17files.extend(arr)
                    if G18F and config.G18ABIFDIR is not None:
                        _dirpre = config.G18ABIFDIR
                        _prefix = 'OR_ABI-L1b-RadF-M*C'+str(config.single_fig_band).zfill(2)+'_G18_'
                    
                        _dir = os.path.join(_dirpre,str(y)+'/'+ str(d).zfill(3)+'/'+str(h).zfill(2)+'/')
                        arr = glob.glob(_dir+_prefix+'*.nc')
                        G18files.extend(arr)
                    if G18C and config.G18ABICDIR is not None:
                        _dirpre = config.G18ABICDIR
                        _prefix = 'OR_ABI-L1b-RadC-M*C'+str(config.single_fig_band).zfill(2)+'_G18_'
                    
                        _dir = os.path.join(_dirpre,str(y)+'/'+ str(d).zfill(3)+'/'+str(h).zfill(2)+'/')
                        arr = glob.glob(_dir+_prefix+'*.nc')
                        G18files.extend(arr)
                    if G19F and config.G19ABIFDIR is not None:
                        _dirpre = config.G19ABIFDIR
                        _prefix = 'OR_ABI-L1b-RadF-M*C'+str(config.single_fig_band).zfill(2)+'_G19_'
                    
                        _dir = os.path.join(_dirpre,str(y)+'/'+ str(d).zfill(3)+'/'+str(h).zfill(2)+'/')
                        arr = glob.glob(_dir+_prefix+'*.nc')
                        G19files.extend(arr)
                    if G19C and config.G19ABICDIR is not None:
                        _dirpre = config.G19ABICDIR
                        _prefix = 'OR_ABI-L1b-RadC-M*C'+str(config.single_fig_band).zfill(2)+'_G19_'
                    
                        _dir = os.path.join(_dirpre,str(y)+'/'+ str(d).zfill(3)+'/'+str(h).zfill(2)+'/')
                        arr = glob.glob(_dir+_prefix+'*.nc')
                        G19files.extend(arr)
    
    
    closest_abi_files = {}
    if len(G17files)>0:
        G17files.append(targetfilename)
        G17files.sort(key=sortfiles)
        ind = G17files.index(targetfilename)
        # Account for possibility that targetfilename is the first or last available file
        closest_abi_files['G17'] = ['NA', 'NA']
        if ind > 0:
            closest_abi_files['G17'][0] = G17files[ind-1]
        if ind < len(G17files)-1:
            closest_abi_files['G17'][1] = G17files[ind+1]
    else:
        closest_abi_files['G17'] = None

    if len(G18files)>0:
        G18files.append(targetfilename)
        G18files.sort(key=sortfiles)
        ind = G18files.index(targetfilename)
        # Account for possibility that targetfilename is the first or last available file
        closest_abi_files['G18'] = ['NA', 'NA']
        if ind > 0:
            closest_abi_files['G18'][0] = G18files[ind-1]
        if ind < len(G18files)-1:
            closest_abi_files['G18'][1] = G18files[ind+1]
    else:
        closest_abi_files['G18'] = None

    if len(G16files)>0:
        G16files.append(targetfilename)
        G16files.sort(key=sortfiles)
        ind = G16files.index(targetfilename)
        # Account for possibility that targetfilename is the first or last available file
        closest_abi_files['G16'] = ['NA', 'NA']
        if ind > 0:
            closest_abi_files['G16'][0] = G16files[ind-1]
        if ind < len(G16files)-1:
            closest_abi_files['G16'][1] = G16files[ind+1]
    else:
        closest_abi_files['G16'] = None

    if len(G19files)>0:
        G19files.append(targetfilename)
        G19files.sort(key=sortfiles)
        ind = G19files.index(targetfilename)
        # Account for possibility that targetfilename is the first or last available file
        closest_abi_files['G19'] = ['NA', 'NA']
        if ind > 0:
            closest_abi_files['G19'][0] = G19files[ind-1]
        if ind < len(G19files)-1:
            closest_abi_files['G19'][1] = G19files[ind+1]
    else:
        closest_abi_files['G19'] = None

    return closest_abi_files    

def _get_closest_glm_files(fileList, _eventtime, window_mins, this_is_detection_satellite):
    """ Helper function to return the closest files based on an event time and the given window length

    Parameters
    ----------
    fileList : str list
        List of files to search through
    _eventtime : str
        Time of start of GLM netCDF file in filename time format
    window_mins : str
        The window in minutes
    this_is_detection_satellite : bool
        If True then throw an error if data files not availabel for this satellite.
        Otherwise, just return a warning.

    Returns
    -------
    closestFiles = str list
        The closest files

    """

    raise Exception('This function is no longer used')


    # Each GLM file is 20 seconds, so set the start and end indices in the sorted list accordingly
    # The center file is closer to the front, so subtract or add 1 accordingly
    # For 10 minutes, that means 14 before and 16 after
    # For 60 minutes, that means 89 before and 91 after
    start_idx = round(window_mins/2.0 * 60 / 20 - 1)
    end_idx = round(window_mins/2.0 * 60 / 20 + 1)

    closestFiles = []
    if len(fileList)>0:
        fileList.sort(key=sortfiles)
        #check that prev file is targetfile
        tfile = fnmatch.filter(fileList,"*"+_eventtime+"*.nc")
        if len(tfile)>0:
            ind = fileList.index(tfile[0])

            st = ind-start_idx
            if st < 0:
                st=0
            ed = ind+end_idx
            if ed > (len(fileList)-1):
                ed=(len(fileList)-1)
            
            closestFiles.extend(fileList[st:ed])
        elif this_is_detection_satellite:
            raise FileNotFoundError("MISSING GLM FILES FOR DETECTION SATELLITE!")
        else:
            raise Warning("Missing GLM files for other satellite.")
    elif this_is_detection_satellite:
        raise FileNotFoundError("MISSING GLM FILES FOR DETECTION SATELLITE!")
    else:
        raise Warning("Missing GLM files for other satellite.")

    return closestFiles


