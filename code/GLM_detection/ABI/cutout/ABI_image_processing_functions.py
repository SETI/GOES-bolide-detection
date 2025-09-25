#!/usr/bin/python 

import numpy as np
import h5py
import cv2
import glob
from numba import jit
from copy import copy, deepcopy
from traceback import print_exc

class ABIDataAvailabilityError(Exception):
    # Define a custom exception type that prints the error message automatically
    # This way, we do not need to use print_exc(), which wioll also print the Traceback
    def __init__(self, message):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        print(message)
            

def latlon_to_grid(lat,lon,semi_minor_axis_rpol,semi_major_axis_req,H,longitude_of_projection_origin_gamma,xoffset,xscale_factor,yoffset,yscale_factor,e):
    """ This function presumably converts lat/lon coordinates to x/y coordinates on an image.

    No documentation provided, written by Nina McCurdy.

    """
    #to radians
    lat_rad = lat*(np.pi/180);
    lon_rad = lon*(np.pi/180);
    lat_c = np.arctan((pow(semi_minor_axis_rpol,2)/pow(semi_major_axis_req,2))*np.tan(lat_rad));
    r_c = semi_minor_axis_rpol/np.sqrt(1 - pow(e,2)*pow(np.cos(lat_c),2));
    sx = H - r_c*np.cos(lat_c)*np.cos(lon_rad - longitude_of_projection_origin_gamma);
    sy = -r_c*np.cos(lat_c)*np.sin(lon_rad - longitude_of_projection_origin_gamma);
    sz = r_c*np.sin(lat_c);
    y = np.arctan(sz/sx);
    x = np.arcsin(-sy/np.sqrt(pow(sx,2)+pow(sy,2)+pow(sz,2)));
    xgrid = (x-xoffset)/xscale_factor;
    ygrid = (y-yoffset)/yscale_factor;
    return [xgrid,ygrid]

def latlon_to_grid_specs(lat,lon,specs):
    semi_minor_axis_rpol = specs[0]
    semi_major_axis_req = specs[1]
    H = specs[2]
    longitude_of_projection_origin_gamma = specs[3]
    xoffset = specs[4]
    xscale_factor = specs[5]
    yoffset = specs[6]
    yscale_factor = specs[7]
    e = specs[8]
    #to radians
    lat_rad = lat*(np.pi/180);
    lon_rad = lon*(np.pi/180);
    lat_c = np.arctan((pow(semi_minor_axis_rpol,2)/pow(semi_major_axis_req,2))*np.tan(lat_rad))
    r_c = semi_minor_axis_rpol/np.sqrt(1 - pow(e,2)*pow(np.cos(lat_c),2))
    sx = H - r_c*np.cos(lat_c)*np.cos(lon_rad - longitude_of_projection_origin_gamma)
    sy = -r_c*np.cos(lat_c)*np.sin(lon_rad - longitude_of_projection_origin_gamma)
    sz = r_c*np.sin(lat_c)
    
    lh = H*(H-sx)
    rh = pow(sy,2)+(pow(semi_major_axis_req,2)/pow(semi_minor_axis_rpol,2))*pow(sz,2)
    
    y = np.arctan(sz/sx)
    x = np.arcsin(-sy/np.sqrt(pow(sx,2)+pow(sy,2)+pow(sz,2)))
    xgrid = (x-xoffset)/xscale_factor
    ygrid = (y-yoffset)/yscale_factor
    
    fxgrid=[]
    fygrid=[]
    
    for j in range(0,len(xgrid)):
        if (lh[j]>=rh[j]):
            fxgrid.append(xgrid[j]*2)
            fygrid.append((-ygrid[j]+10848)*2)
            
    return [fxgrid,fygrid]


# Note: this one function is a large fraction of the total CPU time for the cutout generation
#@jit(nopython=False, parallel=True)
def generate_ABI_composites(config, detection_ID, band_2_filename,avgLat,avgLon, detection_satellite):
    """ 
    1. Reads in the ABI data files.

    2. Generates quasi-RGB composite image

    3. Clips the ABI data to the specified boxes in degrees for the cutputs and
        to a set grid diemnsion for the multi-band data

    Parameters
    ----------
    config
    detection_ID : int64
        Detection ID
    band_2_filename : str
        Path to the band 2 filename, all other band data is assuemd to be in the same directory and have similar naming
        convenstions

    Returns
    -------
    multi_band_composite : np.array of size (dy, dx, len(config.bands_to_read))
        The ABI data for all bands desired and clipped to the desired viewing box
        Note: GOES ABI bands are 1-based! So, Band 1 is index 0 in this array.
    RGB_composite : 
        The ABI data for quasi-RGB and clipped to the desired viewing box
    multi_band_composite_zoom : 
        The ABI data for all bands desired and clipped to the desired zoomed viewing box
    RBG_composite_zoom : 
        The ABI data for quasi-RGB and clipped to the desired zoomed viewing box
    bounds : dict 
        The viewing box limits in ABI pixels [y0,y1,x0,x1]
        keys = ('RGB', 'RGB_zoom', 'multi_band', 'multi_band_zoom')
    raw_band_1_data :
        The raw Band 1 "blue" data returned for use as a reference
    detection_satellite : bool
        If True then we are pulling the data for the detection satellite
    
    """

    try: 
        multi_band_composite, raw_band_1_data = read_ABI_file_for_all_bands(config, detection_ID, band_2_filename)
    except ABIDataAvailabilityError as e:
        return [None, e, None, None, None, None]
    except Exception as e:
        # Unexpected exception
        if detection_satellite:
            # This is the detection satellite, the ABI must be loaded, but it did not, so throw the error
            print_exc()
            raise Exception('This is the detection satellite data, the ABI data must properly parse to generate cutout figure')
           #raise ABIDataAvailabilityError('ABI data not available')
        else:
            return [None, e, None, None, None, None]



    # Generate the quasi-RGB composite from the multi-band data
    # Day composite image of bands [2,3,1] [Red, Green, Blue]
    # Night composite image of bands [7,11,13]
    # The default Band 2 is the "Red" band
    # "Veggie" band 3
    # "Blue" band 1:
    # "Shortwave Window" band 7:
    # "Cloud-Top Phase" band 11:
    # "Clean" IR Longwave Window Band 13:
    # Note: Python is 0-based indexing!
    RGB_composite = scale_and_set_up_quasi_RGB_image(multi_band_composite[:,:,[1,2,0]], multi_band_composite[:,:,[6,10,12]])

    #calculate bounding box - based on Blue band (10848pxls)
    #print("calculating viewing box")
    # bounds = [y0,y1,x0,x1]
   #bounds = calc_viewing_box(raw_band_1_data,avgLat,avgLon,config.dlat,config.dlon)
   #bounds_zoom = calc_viewing_box(raw_band_1_data,avgLat,avgLon,config.dlat_zoom,config.dlon_zoom)
    bounds = {}
    bounds['RGB'] = calc_viewing_box(raw_band_1_data,avgLat,avgLon,dlat=config.dlat,dlon=config.dlon)
    bounds['multi_band'] = calc_viewing_box(raw_band_1_data, avgLat, avgLon, dlat=None, dlon=None, multi_band_window_size=config.image_60_sec_pixel_size)
    bounds['RGB_zoom'] = calc_viewing_box(raw_band_1_data, avgLat, avgLon, dlat=config.dlat_zoom, dlon=config.dlon_zoom)
    bounds['multi_band_zoom'] = calc_viewing_box(raw_band_1_data, avgLat, avgLon, dlat=None, dlon=None, multi_band_window_size=config.image_60_sec_zoom_pixel_size)

    # Clip the ABI data to the bounding box
    multi_band_composite_zoom = deepcopy(multi_band_composite)
    multi_band_composite = multi_band_composite[bounds['multi_band'][0]:bounds['multi_band'][1], bounds['multi_band'][2]:bounds['multi_band'][3],:]
    multi_band_composite_zoom = multi_band_composite_zoom[bounds['multi_band_zoom'][0]:bounds['multi_band_zoom'][1], bounds['multi_band_zoom'][2]:bounds['multi_band_zoom'][3],:]

    RGB_composite_zoom = deepcopy(RGB_composite)
    RGB_composite = RGB_composite[bounds['RGB'][0]:bounds['RGB'][1], bounds['RGB'][2]:bounds['RGB'][3],:]
    RGB_composite_zoom = RGB_composite_zoom[bounds['RGB_zoom'][0]:bounds['RGB_zoom'][1], bounds['RGB_zoom'][2]:bounds['RGB_zoom'][3],:]

    return [multi_band_composite, RGB_composite, multi_band_composite_zoom, RGB_composite_zoom, bounds, raw_band_1_data]

def read_ABI_files_and_return_quasi_RGB(config, band_2_filename):
    """ Reads in the ABI files and returns quasi-geo-color RGB composites for both day and night.

    Parameters
    ----------
    config
    band_2_filename : str
        Path to the band 2 filename, all other band data is assuemd to be in the same directory and have similar naming
        convenstions

    Returns
    -------
    composite : np.array of size (dy, dx, 3)
        Where dy and dx are the dimensions of the Blue band 1 (1 km resolution)
        Day composite image of bands [2,3,1]
    ncomposite
        Night composite image of bands [7,11,13]
    _bdset : 
        raw band 1 data

    """

    raise Exception('This function is no longer used')


def read_ABI_file_for_all_bands(config, detection_ID, band_2_filename):
    """ This will read in the ABI data for ALL bands and store in a multi-channel array

    The different bands have different resolutions. We use the "blue" band 2 as the reference, which has a 1km resolution.
    This means band 1 must be in the list config.bands_to_read.

    parameters
    ----------
    config
        .bands_to_read : list of ints (1-based)
            A list of integer indices of the bands to read
            Note: GOES ABI bands are 1-based!
            'all' means read all 16 bands
            Default is bands_to_read = (1,2,3,7,11,13)
    detection_ID : int64
        Detection ID
    band_2_filename : str
        Path to the band 2 filename, all other band data is assuemd to be in the same directory and have similar naming
        conventions
    raw_band_1_data : 
        raw band 1 data

    Returns
    -------
    multiBandComposite : np.array of size (dy, dx, len(config.bands_to_read))
        returned multi-band ABI image
        Note: GOES ABI bands are 1-based! So, Band 1 is index 0 in this array.
    raw_band_1_data :
        The raw Band 1 "blue" data returned for use as a reference
    """

    bands_to_read = copy(config.bands_to_read)
    if bands_to_read == 'all':
        bands_to_read = np.arange(1,17).tolist()

    assert np.min(bands_to_read) > 0 and np.max(bands_to_read) < 17, 'ABI bands must be in the range [1:16]'

    assert bands_to_read.count(1) > 0, 'Band 1 must be present'

    # Load in the raw band data

    bandData = {}
    for iBand in bands_to_read:
        if iBand == 2:
            filename = band_2_filename
            files = glob.glob(band_2_filename)
        else:
            band_string = 'C{:02}'.format(iBand)
            files = glob.glob(band_2_filename[0:band_2_filename.find('_e')].replace("C02",band_string)+"*")
        if len(files) == 1:
            bandData[iBand] = h5py.File(files[0],'r')
        else:
            raise ABIDataAvailabilityError('****WARNING**** ABI Band {} data not available for ID {}'.format(iBand, detection_ID))

    # Get the reference band 1 dimensions
    B = np.array(bandData[1]['Rad'])
    dy = B.shape[0]
    dx = B.shape[1]

    # Create a 16 band matrix. The unloaded bands are all 0.0
    multiBandComposite = np.full((dy,dx,16), 0.0, dtype=np.uint16)

    # Copy each band data to the composite array
    # rescale the image if not same resolution as the reference band 1
    for iBand in bands_to_read:
        B = np.array(bandData[iBand]['Rad'])
        if  dy != B.shape[0] or dx != B.shape[1]:
            multiBandComposite[...,iBand-1] = cv2.resize(B,(dx,dy),interpolation = cv2.INTER_AREA)
        else:
            multiBandComposite[...,iBand-1] = B

        # Save the Band 1 raw data, close all the others
        if iBand == 1:
            raw_band_1_data = bandData[iBand]
        else:
            bandData[iBand].close()
    
    return multiBandComposite, raw_band_1_data


#@jit(nopython=True, parallel=True)
def scale_and_set_up_quasi_RGB_image(composite, ncomposite):
    """ Does a bunch of stuff to set up the ABI band data as a quasi-RGB composite.

    TODO: This is all numpy code, so compile it with Numba

    Parameters
    ----------
    composite : np.array of size (dy, dx, 3)
        The daytime composite image
    ncomposite : np.array of size (dy, dx, 3)
        The nighttime composite image

    Returns
    RGB_composite :
        The multiband ABI data in RGB format
    """
      
    #***
    # Technically, numba-izing this part speeds up the code, but not by much
    # and appears to not when the cutout tool is run in parallel
  # composite, ncomposite = numba_part_1(composite, ncomposite)

    #scale all day bands to 16-bit
    composite[...,0] = composite[...,0].astype(np.uint16)*(65535/4095)
    composite[...,1] = composite[...,1].astype(np.uint16)*(65535/1023)
    composite[...,2] = composite[...,2].astype(np.uint16)*(65535/1023)
 
    #scale all night bands to 16-bit
    ncomposite[...,0] = ncomposite[...,0].astype(np.uint16)*(65535/16383)
    ncomposite[...,1] = ncomposite[...,1].astype(np.uint16)*(65535/4095)
    ncomposite[...,2] = ncomposite[...,2].astype(np.uint16)*(65535/4095)
 
    #invert scale night composite
    ncomposite = ncomposite/65535
    ncomposite = np.clip(ncomposite,0,1)
    ncomposite = 1 - ncomposite
    ncomposite = ncomposite/1.4
 
    #scale nbands to the same range and then take max and stretch result
    ncomposite[...,0] = np.interp(ncomposite[...,0],(0.686,0.718),(0,1.0))
    ncomposite[...,1] = np.interp(ncomposite[...,1],(0.36,0.7),(0,1.0))
    ncomposite[...,2] = np.interp(ncomposite[...,2],(0.3,0.675),(0,1.0))
 
    ncomposite[...,0] = np.maximum(ncomposite[...,0],np.maximum(ncomposite[...,1],ncomposite[...,2]))
    ncomposite[...,1] = ncomposite[...,0]
    ncomposite[...,2] = ncomposite[...,0]
    #ncomposite = np.interp(ncomposite,(0.1,1.0),(0.0,0.6))
    ncomposite = np.interp(ncomposite,(0.1,1.0),(0.0,0.4))#(0.0,0.5))
    #***
 
    maskon=True
    if maskon:
                
        # numba does not play with np.place, so spell it out
       #np.place(composite[...,0],composite[...,0]==65535,0)
       #np.place(composite[...,1],composite[...,1]==65535,0)
       #np.place(composite[...,2],composite[...,2]==65535,0)

        # Why does numba not like this global search and replace indexing?
        composite[composite==65535] = 0

        # OK, fine, just do it long-hand
       #compositeFlat = composite.flatten()
       #for i in np.arange(len(compositeFlat)):
       #    if compositeFlat[i] == 65535:
       #        compositeFlat[i] = 0
       #composite = compositeFlat.reshape(composite.shape)
 
    #***
    # Technically, numba-izing this part speeds up the code, but not by much
    # and appears to not when the cutout tool is run in parallel
  # composite, ncomposite, logicArray = numba_part_2(composite, ncomposite)

    #scale to 0-1
    composite = composite/65535
 
    #simulate green band
    composite[:,:,1] = 0.45*composite[:,:,0] + 0.10*composite[:,:,1] + 0.45*composite[:,:,2]
 
    #stretch a little bit
    composite = np.interp(composite,(0.02,0.6),(0.0,1.0))
 
    #apply sqrt enhancement
    composite = np.sqrt(composite)*composite
 
    #apply gamma enhancement
    gamma=2.2
    composite = np.power(composite,1/gamma)
 
    #to keep high-res daytime clouds
    rm3 = composite[...,0]<0.2
    vm3 = composite[...,1]<0.2
    bm3 = composite[...,2]<0.2
 
    #look for pixels with out color
 
   #rm3b = abs(composite[...,0]-(composite[...,0]+composite[...,1]+composite[...,2])/3)>0.1
   #vm3b = abs(composite[...,0]-(composite[...,0]+composite[...,1]+composite[...,2])/3)>0.1
   #bm3b = abs(composite[...,0]-(composite[...,0]+composite[...,1]+composite[...,2])/3)>0.1
      
    
   #rvbm = np.dstack([rm3*vm3*bm3,rm3*vm3*bm3,rm3*vm3*bm3])
    logicArray = np.logical_and(np.logical_and(rm3,vm3), bm3)
    #***

   #return composite, ncomposite, logicArray

    # Numba does not seem to like the use of dstack, so pull it out of the JIT code
    rvbm = np.dstack([logicArray, logicArray, logicArray])
    
    ncomposite = np.where(rvbm,ncomposite,composite*0.95)
 
    #incoporate night composite
    # pick which day or night composite is brighter for each RGB band seperately and use that for the RGB_composite
    RGB_composite = np.dstack([np.maximum(composite[...,0],ncomposite[...,0]),np.maximum(composite[...,1],ncomposite[...,1]),np.maximum(composite[...,2],ncomposite[...,2])])
 
    return RGB_composite


@jit(nopython=True, parallel=True)
def numba_part_1(composite, ncomposite):
    """ This numba code technically speeds up scale_and_set_up_quasi_RGB_image
    But not by muhbc and when running the cutout tool in parallel it actually slows things down a little, 
    Presumably because of the extra overhead to do the JIT compilation for each subtask separately.
    """

    #scale all day bands to 16-bit
    composite[...,0] = composite[...,0].astype(np.uint16)*(65535/4095)
    composite[...,1] = composite[...,1].astype(np.uint16)*(65535/1023)
    composite[...,2] = composite[...,2].astype(np.uint16)*(65535/1023)
 
    #scale all night bands to 16-bit
    ncomposite[...,0] = ncomposite[...,0].astype(np.uint16)*(65535/16383)
    ncomposite[...,1] = ncomposite[...,1].astype(np.uint16)*(65535/4095)
    ncomposite[...,2] = ncomposite[...,2].astype(np.uint16)*(65535/4095)
 
    #invert scale night composite
    ncomposite = ncomposite/65535
    ncomposite = np.clip(ncomposite,0,1)
    ncomposite = 1 - ncomposite
    ncomposite = ncomposite/1.4
 
    #scale nbands to the same range and then take max and stretch result
    ncomposite[...,0] = np.interp(ncomposite[...,0],(0.686,0.718),(0,1.0))
    ncomposite[...,1] = np.interp(ncomposite[...,1],(0.36,0.7),(0,1.0))
    ncomposite[...,2] = np.interp(ncomposite[...,2],(0.3,0.675),(0,1.0))
 
    ncomposite[...,0] = np.maximum(ncomposite[...,0],np.maximum(ncomposite[...,1],ncomposite[...,2]))
    ncomposite[...,1] = ncomposite[...,0]
    ncomposite[...,2] = ncomposite[...,0]
    #ncomposite = np.interp(ncomposite,(0.1,1.0),(0.0,0.6))
    ncomposite = np.interp(ncomposite,(0.1,1.0),(0.0,0.4))#(0.0,0.5))

    return composite, ncomposite

@jit(nopython=True, parallel=True)
def numba_part_2(composite, ncomposite):
    """ This numba code technically speeds up scale_and_set_up_quasi_RGB_image
    But not by muhbc and when running the cutout tool in parallel it actually slows things down a little, 
    Presumably because of the extra overhead to do the JIT compilation for each subtask separately.
    """
    #scale to 0-1
    composite = composite/65535
 
    #simulate green band
    composite[:,:,1] = 0.45*composite[:,:,0] + 0.10*composite[:,:,1] + 0.45*composite[:,:,2]
 
    #stretch a little bit
    composite = np.interp(composite,(0.02,0.6),(0.0,1.0))
 
    #apply sqrt enhancement
    composite = np.sqrt(composite)*composite
 
    #apply gamma enhancement
    gamma=2.2
    composite = np.power(composite,1/gamma)
 
    #to keep high-res daytime clouds
    rm3 = composite[...,0]<0.2
    vm3 = composite[...,1]<0.2
    bm3 = composite[...,2]<0.2
 
    #look for pixels with out color
 
   #rm3b = abs(composite[...,0]-(composite[...,0]+composite[...,1]+composite[...,2])/3)>0.1
   #vm3b = abs(composite[...,0]-(composite[...,0]+composite[...,1]+composite[...,2])/3)>0.1
   #bm3b = abs(composite[...,0]-(composite[...,0]+composite[...,1]+composite[...,2])/3)>0.1
      
    
   #rvbm = np.dstack([rm3*vm3*bm3,rm3*vm3*bm3,rm3*vm3*bm3])
    logicArray = np.logical_and(np.logical_and(rm3,vm3), bm3)

    return composite, ncomposite, logicArray


def project_glm_data(raw_band_1_data, glm_event_list, bolideLat, bolideLon, dlat, dlon):
    """ 
    "Project" the GLM lat/lon data onto the x/y coordinates of the ABI image figures.

    No documentation was provided with the original function, which was written by Nina McCurdy.

    Parameters
    ----------
    raw_band_1_data :
        The raw Band 1 "blue" data returned for use as a projection reference
    glm_event_list : GlmEvent list
        A list of all GLM events objects within the specified window
    bolideLat : float
        Avgerage latitude of bolide
    bolideLon : float
        Avgerage longitude of bolide
    dlat : float
        Size of ABI cutout to project GLM data onto +/- in degrees
    dlon : float
        Size of ABI cutout to project GLM data onto +/- in degrees
    
    Returns
    -------
    projected_glm_data : dict
        Keys: 'x', 'y', 'lon', 'lat', 'energyJoules'

    """
    x = raw_band_1_data['x']
    y = raw_band_1_data['y']
    proj = raw_band_1_data['goes_imager_projection']
    Rad=np.array(raw_band_1_data['Rad'])


    xoffset = x.attrs['add_offset']
    xscale_factor = x.attrs['scale_factor']
    yoffset = y.attrs['add_offset']
    yscale_factor = y.attrs['scale_factor']

    semi_minor_axis_rpol=proj.attrs['semi_minor_axis']
    semi_major_axis_req=proj.attrs['semi_minor_axis']
    perspective_point_height=proj.attrs['perspective_point_height']
    longitude_of_projection_origin_gamma=proj.attrs['longitude_of_projection_origin']
    longitude_of_projection_origin_gamma *= np.pi/180
    H = perspective_point_height + semi_major_axis_req
    e = np.sqrt((pow(semi_major_axis_req,2)-pow(semi_minor_axis_rpol,2))/pow(semi_major_axis_req,2))

    projected_glm_data = {}
    # Only keep GLM events that are within a dlat/dlon box around (bolideLat, bolideLon)
    # This expansion factor was in the original code, Presumably to make sure we do not clip the corners when
    # projecting, it's not clear why it was 2.0, Setting to 1.5, which seems plenty large.
    expans_fact = 1.5
    
    event_lats = np.array([e.latitudeDegreesNorth for e in glm_event_list], dtype=np.float32)
    event_lons = np.array([e.longitudeDegreesEast for e in glm_event_list], dtype=np.float32)
    event_energyJoules = np.array([e.energyJoules for e in glm_event_list], dtype=np.float32)

    events_to_keep = np.nonzero(np.logical_and(np.abs(bolideLat-event_lats) <= dlat*expans_fact, np.abs(bolideLon-event_lons) <= dlon*expans_fact))[0]

    projected_glm_data['lat'] = event_lats[events_to_keep]
    projected_glm_data['lon'] = event_lons[events_to_keep]
    projected_glm_data['energyJoules'] = event_energyJoules[events_to_keep]
    [projected_glm_data['x'], projected_glm_data['y']] = latlon_to_grid(projected_glm_data['lat'], projected_glm_data['lon'],
            semi_minor_axis_rpol,semi_major_axis_req,H,longitude_of_projection_origin_gamma,xoffset,xscale_factor,yoffset,yscale_factor,e)

    # This correction was in the original code. 
    # It looks like if instead of using the Blue band we use the high resolution red band, we have to make this
    # correction to the coordinates. Not clear why that would be the case. I thought latlon_to_grid would accoutn for
    # this. Anyway, keeping in the code, but throwing an error is it is supposed to run.
    if Rad.shape[0]==21696:
        raise Exception('Why is this code here?')
        projected_glm_data['x'] = projected_glm_data['x']/2.0
        projected_glm_data['y'] = projected_glm_data['y']/2.0

    return projected_glm_data

def calc_viewing_box(data,lat,lon,dlat=None,dlon=None,multi_band_window_size=None):
    """ Determines the box of pixels determined by a box in lat/lon around which to retrieve the ABI data 

    This can also returns a fixed grid of pixels for the multi-band data. This data must be a  of a set dientionslaity for
    use in a CNN. So, we do not base it on lat/lon, which varies over the field of view.

    If multi_band_window_size is not None then a fixed pixel grid is returned, otherwise, the cutout based on dlat and
    dlon is returned.

    The returned bounding box is in whatever units the overlays.py data is in. I think it is in pixels of the ABI data.

    Parameters
    ----------
    data : h5py._hl.files.File
        The raw band 1 netCDF data
    lat : float
        average latitude for bolide candidate
    lon : float
        average longitude for bolide candidate
    dlat : float
        +/- degrees for cutout window
    dlon : float
        +/- degrees for cutout window
    multi_band_window_size : int
        The size of the multi-band data array in pixels.

    Returns
    -------
    bounds : list
        bounding box in ABI pixels:
        [y0,y1,x0,x1]

    """

    if multi_band_window_size is None:
        assert dlat is not None and dlon is not None, 'If multi_band_window_size is None then dlat and dlon must be passed:'
    else:
        assert dlat is None and dlon is None, 'If multi_band_window_size is passed then dlat and dlon must be None:'


    #print(data)
    x = data['x']
    y = data['y']
    proj = data['goes_imager_projection']
    Rad=np.array(data['Rad'])
    xoffset = x.attrs['add_offset']
    xscale_factor = x.attrs['scale_factor']
    yoffset = y.attrs['add_offset']
    yscale_factor = y.attrs['scale_factor']
 
    semi_minor_axis_rpol=proj.attrs['semi_minor_axis']
    semi_major_axis_req=proj.attrs['semi_minor_axis']
    perspective_point_height=proj.attrs['perspective_point_height']
    longitude_of_projection_origin_gamma=proj.attrs['longitude_of_projection_origin']
    longitude_of_projection_origin_gamma *= np.pi/180
    H = perspective_point_height + semi_major_axis_req
    e = np.sqrt((pow(semi_major_axis_req,2)-pow(semi_minor_axis_rpol,2))/pow(semi_major_axis_req,2))
 
    comments=False
    if comments:
      print("xoffset = "+str(xoffset))
      print("xscale_factor = "+str(xscale_factor))
      print("yoffset = "+str(yoffset))
      print("yscale_factor = "+str(yscale_factor))
      print("semi_minor_axis_rpol = "+str(semi_minor_axis_rpol))
      print("semi_major_axis_req = "+str(semi_major_axis_req))
      print("perspective_point_height = "+str(perspective_point_height))
      print("longitude_of_projection_origin_gamma = "+str(longitude_of_projection_origin_gamma))
      print("H = "+str(H))
      print("e = "+str(e))
 
 
    [gridx,gridy] = latlon_to_grid(lat,lon,semi_minor_axis_rpol,semi_major_axis_req,H,longitude_of_projection_origin_gamma,xoffset,xscale_factor,yoffset,yscale_factor,e)            


    if multi_band_window_size is None:
        # This uses dlat and dlon
        [x0,y0] = latlon_to_grid((lat+dlat),(lon-dlon),semi_minor_axis_rpol,semi_major_axis_req,H,longitude_of_projection_origin_gamma,xoffset,xscale_factor,yoffset,yscale_factor,e)
        
        [x1,y1] = latlon_to_grid((lat+dlat),(lon+dlon),semi_minor_axis_rpol,semi_major_axis_req,H,longitude_of_projection_origin_gamma,xoffset,xscale_factor,yoffset,yscale_factor,e)
        
        [x2,y2] = latlon_to_grid((lat-dlat),(lon+dlon),semi_minor_axis_rpol,semi_major_axis_req,H,longitude_of_projection_origin_gamma,xoffset,xscale_factor,yoffset,yscale_factor,e)
        
        [x3,y3] = latlon_to_grid((lat-dlat),(lon-dlon),semi_minor_axis_rpol,semi_major_axis_req,H,longitude_of_projection_origin_gamma,xoffset,xscale_factor,yoffset,yscale_factor,e)
        
        topedge = x1-x0
        bottomedge = x2-x3
        leftedge = y2-y1
        rightedge = y3-y0
        
        longestdim = topedge if topedge > bottomedge else bottomedge
        longestdim = longestdim if longestdim > leftedge else leftedge
        longestdim = longestdim if longestdim > rightedge else rightedge
        
        # half_window_pixels is the half the pixel width of the cutout images (square)
        half_window_pixels = int(longestdim/2.0)
    else:
        # This uses multi_band_window_size
        half_window_pixels = int(multi_band_window_size / 2.0)


    x0 = gridx-half_window_pixels
    x1 = gridx+half_window_pixels
    y0 = gridy-half_window_pixels
    y1 = gridy+half_window_pixels
 
    if ((int(x1)-int(x0))%2)!=0:
        #print("uneven x")
      x0+=1
    if ((int(y1)-int(y0))%2)!=0:
        #print("uneven y")
      y0+=1
 
    #***
    #check that box doesn't run off end of image
    if x0<0:
      #add remainder to other side
      x1=x1-x0
      x0=0
    elif x1>Rad.shape[1]:
      x0=x0-(x1-Rad.shape[1])
      x1=Rad.shape[1]
 
    if y0<0:
      #add remainder to other side
      y1=y1-y0
      y0=0
    elif y1>Rad.shape[0]:
      y0=y0-(y1-Rad.shape[0])
      y1=Rad.shape[0]
    
    #y0 = gridy-500
    #y1 = gridy+500
    #x0 = gridx-500
    #x1 = gridx+500
    
    #***
    # TODO: Do we need this still in here?
    #if rdset, divide bounds by 2:
    if Rad.shape[0]==21696 or Rad.shape[0]==6000:
        raise Exception('Why is this code here?')
        y0=y0/2.0
        y1=y1/2.0
        x0=x0/2.0
        x1=x1/2.0
      
    return [int(y0),int(y1),int(x0),int(x1)]

