# This class is used to compute the GLM x/ pixels form a latitude and longitude

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.optimize import minimize_scalar
from numba import jit


class LatLon2Pix:

    def __init__(self, lookup_table_path, lookup_table_path_inverted=None):
        """ Initializes a lat/lon to GLM pixel object

        Parameters
        ----------
        lookup_table_path : str
            Path to the lookup table file.
        lookup_table_path_inverted : str
            Path to the lookup table file for the inverted yaw orientation
            Only relevent for G17.

        """

        # Load the CSV file
        self.table = pd.read_csv(lookup_table_path, header=0)
        if lookup_table_path_inverted is not None and not lookup_table_path_inverted == '':
            self.table_inverted = pd.read_csv(lookup_table_path_inverted, header=0)
        else:
            self.table_inverted = None

        self.x = self.table['x'].to_numpy()
        self.y = self.table['y'].to_numpy()
        self.lat = self.table['meanLatitude'].to_numpy()
        self.lon = self.table['meanLongitude'].to_numpy()

        # Remove NaNs
        goodHere = np.nonzero(np.logical_not(np.logical_or(np.isnan(self.lat), np.isnan(self.lon))))[0]
        self.x = self.x[goodHere]
        self.y = self.y[goodHere]
        self.lat = self.lat[goodHere]
        self.lon = self.lon[goodHere]

        # The inverted data
        if self.table_inverted is not None:
            self.x_inv = self.table_inverted['x'].to_numpy()
            self.y_inv = self.table_inverted['y'].to_numpy()
            self.lat_inv = self.table_inverted['meanLatitude'].to_numpy()
            self.lon_inv = self.table_inverted['meanLongitude'].to_numpy()

            # Remove NaNs
            goodHere = np.nonzero(np.logical_not(np.logical_or(np.isnan(self.lat_inv), np.isnan(self.lon_inv))))[0]
            self.x_inv = self.x_inv[goodHere]
            self.y_inv = self.y_inv[goodHere]
            self.lat_inv = self.lat_inv[goodHere]
            self.lon_inv = self.lon_inv[goodHere]


    def latLon2pix(self, lat, lon, yaw_flip_flag=0):
        """ Converts a Latitude and Longitude to GLM pixels.

        Using the lookup table it performs a nearest neighbor search to find the pixel associated with each lat/lon
        pair.

        No sub-pixel searches, just returns the nearest whole value pixel.
        Returns -1 where a nearest pixel could not be found

        Parameters
        ----------
        lat : float list or numpy.array
            Latitude values to look up
        lon : float list or numpy.array
            Longitude values to look up
        yaw_flip_flag : int
            orientation for GOES spacecraft.
            Only relevent for G17. For G16 must be set to 0
            As defined in the GOES GLM PUG
            0 : upright
            1 : neither
            2 : inverted

        Returns
        -------
        x   : numpy.array of ints
            x-pixel coord for each lat/lon requested
        y   : numpy.array of ints
            y-pixel coord for each lat/lon requested
        """

        if yaw_flip_flag == 2:
            assert self.table_inverted is not None, \
            'If yaw_flip_flag == 2 then the inverted table must be available.'

        assert yaw_flip_flag != 1, 'yaw_flip_flag cannot be 1'
        
        # Convert lat/lon arrays into numpy arrays
        if not isinstance(lat, np.ndarray):
            if isinstance(lat, list):
                lat = np.array(lat)
            else:
                lat = np.array([lat])
        if not isinstance(lon, np.ndarray):
            if isinstance(lon, list):
                lon = np.array(lon)
            else:
                lon = np.array([lon])

        assert len(lat) == len(lon), 'lat and lon must be same length'

        if yaw_flip_flag == 2:
            lat_array = self.lat_inv
            lon_array = self.lon_inv
            x_array = self.x_inv
            y_array = self.y_inv
        else:
            lat_array = self.lat
            lon_array = self.lon
            x_array = self.x
            y_array = self.y

        x_return = np.full(len(lat), int(-1))
        y_return = np.full(len(lat), int(-1))
        # Find the table index that is nearest to the requested latitude and longitude
        # Take the least squares of lat and lon
        for idx, (lat1, lon1) in enumerate(zip(lat, lon)):
            
            #********
            # Brute force
          # D = np.sqrt((lat_array - lat1)**2 + (lon_array - lon1)**2)
          # nearest_idx = np.nanargmin(D)

            #********
            # Numba brute force
            nearest_idx = nearest_table_entry(lat_array, lon_array, lat1, lon1)
            
            #********
            # Numba bracket (seems to be slower than brute force numba)
          # nearest_idx = bracket_nearest_table_entry(lat_array, lon_array, lat1, lon1)
            
            #********
            #********
            x_return[idx] = x_array[nearest_idx]
            y_return[idx] = y_array[nearest_idx]
            

        return x_return, y_return

    def find_pixel_boundaries(self, latArray, lonArray, yaw_flip_flag=0):
        """ Given an array of latitudes and longitudes will return an array of the nearest pixel edges. These vertex
        coordinates can be used to create a grid of pixels about the givens lat/lon points.

        Parameters
        ----------
        latArray : np.array
            The lattitude list
        lonArray : np.array
            The longitude list

        Returns
        -------
        lat_boundary : np.array
            List of pixels edges in latitude
        lon_boundary : np.array
            List of pixels edges in longitude

        """

        # Get the range limits for the data and expand a little 
        latMin = latArray.min() - 0.06
        latMax = latArray.max() + 0.06
        lonMin = lonArray.min() - 0.06
        lonMax = lonArray.max() + 0.06

        # Scan from minimum to maximum latitudes at each longitude and find the vertices
        lat_range = np.linspace(latMin, latMax, 1200)
        lon_range = np.linspace(lonMin, lonMax, 1200)
        latMesh, lonMesh = np.meshgrid(lat_range, lon_range)
        # Find pixels for each point in grid
        latMeshFlat = latMesh.flatten()
        lonMeshFlat = lonMesh.flatten()
        x, y = self.latLon2pix(latMeshFlat, lonMeshFlat, yaw_flip_flag=yaw_flip_flag)

        # Find the pixel boundaries as when the pixel index changes
        # Add 1 to the index to line up with the indices of the original array
        x_boundary_idx = np.nonzero(np.abs(np.diff(x))>0)[0] + 1
        y_boundary_idx = np.nonzero(np.abs(np.diff(y))>0)[0] + 1
        # Each found index is a boundary of a pixel
        both_boundary_idx = np.union1d(x_boundary_idx, y_boundary_idx)

        # Convert back to lat/lon
        lat_boundary = latMeshFlat[both_boundary_idx]
        lon_boundary = lonMeshFlat[both_boundary_idx]
        # Each pair of lat/lon is a boundary

        return lat_boundary, lon_boundary

# Fast table lookup using Numba
# Function is compiled to machine code when called the first time
@jit(nopython=True, parallel=True) # Set "nopython" mode for best performance, equivalent to @njit
def nearest_table_entry(lat_array, lon_array, lat1, lon1):

    D = np.sqrt((lat_array - lat1)**2 + (lon_array - lon1)**2)
                  
    return np.argmin(D)

# This seems to be slower than the brute force method above.
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def bracket_nearest_table_entry(lat_array, lon_array, lat1, lon1):

    # Bracket solution first
    # First find element close in latitude (to within 0.5 degrees)
    lat_array_close = np.nonzero(np.abs(lat_array - lat1) < 0.5)[0]
    lon_array_close = np.nonzero(np.abs(lon_array - lon1) < 0.5)[0]
    array_close_idx = np.intersect1d(lat_array_close, lon_array_close)

    # Now narrow in on the best table entry
    D = np.sqrt((lat_array[array_close_idx] - lat1)**2 + (lon_array[array_close_idx] - lon1)**2)

    return array_close_idx[np.argmin(D)]
