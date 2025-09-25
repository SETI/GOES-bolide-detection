#*************************************************************************************************************
# geometry_utilities.py
#
# A collection of simple geometry tools.
#
# tools include:
#   DistanceFromLatLonPoints -- Converts latitude and longitude in degrees to a linear distance in kilometers 
#                               along a great-circle path between two points.
#   NewLatLonFromBearingDistanceAndLatLon -- Returns a new Lat/lon in degrees based on an initial lat/lon, 
#                               a bearing and a distance in degrees.
#   BearingFromLatLon       -- Returns a bearing from two lat/long coordinates
#   ScaleLongitude          -- Rescales the longitude value to lie within the range -180...180. C.Rumpf
#   RadiusAlongEllipse      -- Returns the angular distance from the center of an ellipse-like form to the ellipse 
#                               given the two axis lengths in degrees and a bearing.
#   FindFurthestLatLonPoints --  Returns the indices of the furthest two points in a lat/lon array 
#
#
#*************************************************************************************************************
import numpy as np
from scipy.stats import gaussian_kde, median_abs_deviation
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

from shapely.geometry import LineString, Point
import geopandas as gpd
   
from numba import jit, prange, set_num_threads

from bolides.plotting import plot_density, plot_scatter, generate_plot

equatorialEarthRadiusKm = 6378.137
polarEarthRadiusKm = 6356.752
faiEarthRadiusKm = 6371.0

deg2rad = np.pi / 180.0
    

#*************************************************************************************************************
@jit(nopython=True, parallel=False)
def DistanceFromLatLonPoints(lat0_deg, lon0_deg, latArray_deg, lonArray_deg, sphereRadius=faiEarthRadiusKm):
    """
    Converts latitude and longitude in degrees to a central angle along a great-circle path between two
    points.
  
    lat0_deg and lon0_deg must be scalars.
    You can either pass two scalars or vectors for latArray_deg and lonArray_deg.
  
    This formula uses the ‘haversine’ formula to calculate the great-circle distance between two points – that is, the
    shortest distance over a sphere.
  
    The value returned can be either in degrees or a linear distance depending on the value of <sphereRadius>
      For example: 
          to return the distance in central angle degrees, set sphereRadius = 180.0 / np.pi
          to return in linear distance in km along the Earth's surface, set sphereRadius = 6371.0 [DEFAULT]
  
    Parameters
    ----------
    lat0_deg    : [float] 
        Reference latitude in degrees
    lon0_deg    : [float] 
        Reference longitude in degrees
    latArray_deg    : [float list] 
        the latitude coords to measure distance to
    lonArray_deg    : [float list] 
        the longitude coords to measure distance to
    sphereRadius    : [float] 
        See above
      
  
    Returns
    -------
    distance    : [float list(len(latArray_deg)] 
        The distance to each point relative to the reference point in units 
        specified by sphereRadius [DEFAULT km on Earth]
  
    """

    # Convert to radians
    lat1 = latArray_deg * deg2rad
    lat0 = lat0_deg * deg2rad
    lon1 = lonArray_deg * deg2rad
    lon0 = lon0_deg * deg2rad
    
    delta_lat = lat1 - lat0
    delta_lon = lon1 - lon0
    
    # The ‘haversine’ formula
    a = (np.sin(delta_lat * 0.5)**2.0) + np.cos(lat0) * np.cos(lat1) * (np.sin(delta_lon * 0.5)**2.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
    distance = sphereRadius * c
    
    return distance

#*************************************************************************************************************
@jit(nopython=True, parallel=True)
def DistanceFromLatLonPoints_parallel(lat0_deg, lon0_deg, lat1_deg, lon1_deg, sphereRadius=faiEarthRadiusKm, numba_threads=4):
    """
    Converts latitude and longitude in degrees to a central angle along a great-circle path between two
    points.

    This is a numba parallelized version of DistanceFromLatLonPoints.
  
    You can either pass two scalars or vectors for lat0_deg, lon0_deg, lat1_deg, lon1_deg.
  
    This formula uses the ‘haversine’ formula to calculate the great-circle distance between two points – that is, the
    shortest distance over a sphere.
  
    The value returned can be either in degrees or a linear distance depending on the value of <sphereRadius>
      For example: 
          to return the distance in central angle degrees, set sphereRadius = 180.0 / np.pi
          to return in linear distance in km along the Earth's surface, set sphereRadius = 6371.0 [DEFAULT]
  
    Parameters
    ----------
    lat0_deg    : [float] 
        Reference latitude in degrees
    lon0_deg    : [float] 
        Reference longitude in degrees
    latArray_deg    : [float list] 
        the latitude coords to measure distance to
    lonArray_deg    : [float list] 
        the longitude coords to measure distance to
    sphereRadius    : [float] 
        See above
    numba_threads : int
        Sets the number of parallel threads to use in numba jit parallelization
        0 => use all currently available threads via get_num_threads
      
  
    Returns
    -------
    distance    : [float list(len(latArray_deg)] 
        The distance to each point relative to the reference point in units 
        specified by sphereRadius [DEFAULT km on Earth]
  
    """

    raise Exception('Is this funtion used?')

    if numba_threads > 0:
        set_num_threads(numba_threads)
    
    # Convert to radians
    lat1 = lat1_deg * deg2rad
    lat0 = lat0_deg * deg2rad
    lon1 = lon1_deg * deg2rad
    lon0 = lon0_deg * deg2rad
    
    a = np.zeros((len(lat0),len(lat1)))
    for i in prange(len(lat0)):
        for j in prange(i + 1, len(lat1)):  # Notice the range starts from i+1 to avoid repetition
            a_ele = (np.sin((lat1[j] - lat0[i]) * 0.5)**2.0) + np.cos(lat0[i]) * np.cos(lat1[j]) * (np.sin((lon1[j] - lon0[i]) * 0.5)**2.0)
            a[i,j] = a_ele
            a[j,i] = a_ele

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))

    distance = sphereRadius * c
    
    return distance

#*************************************************************************************************************
def dist_point_to_line(P1, P2, X):
    """ shortest distance from a point to a line segment on the surface of the Earth

    The line is specified by two points on the line.

    Point and line specified by Latitude and Longitude end points

    Not that this function compute the distance form a point to a line *segment* geodesic, not a great circle around the Earth.
    
    Parameters
    ----------
    P1 : tuple (lon,lat)
        The first point on the line
    P2 : tuple (lon,lat)
        The second point on the line
    X : tuple (lon,lat)
        The test point

    Returns
    -------
    d : distance in kilometers

    """
    line = gpd.GeoSeries([LineString([(-117.93, 22.73), (-109.12, 22.9)])], crs='epsg:4326')
    line = gpd.GeoSeries([LineString([P1, P2])], crs='epsg:4326')
    point = gpd.GeoSeries([Point(X)], crs='epsg:4326')
   
    d = line.shortest_line(point)

    dist = DistanceFromLatLonPoints(d[0].xy[1][0], d[0].xy[0][0], d[0].xy[1][1], d[0].xy[0][1])

    return dist

#*************************************************************************************************************
# def NewLatLonFromBearingDistanceAndLatLon(latDeg, lonDeg, bearingDeg, angularDistDeg):
#
# Returns a new Lat/lon in degrees based on an initial lat/lon, a bearing and a distance in degrees.
#
# Inputs:
#   latDeg          -- Starting latitude in degrees
#   lonDeg          -- Starting longitude in degrees
#   bearingDeg      -- bearing in degrees
#   angularDistDeg  -- angular distance in degrees
#
# Outputs:
#   lat2Deg         -- destination latitude in degrees
#   lon2Deg         -- destination longitude in degrees
#
# Source: http://www.edwilliams.org/avform.htm#LL 
# Via Clemens Rumpf
#
#*************************************************************************************************************
def NewLatLonFromBearingDistanceAndLatLon(latDeg, lonDeg, bearingDeg, angularDistDeg):
    
    lat1 = latDeg * deg2rad
    lon1 = lonDeg * deg2rad
    bearingRad = bearingDeg * deg2rad
    
    lat2 = np.arcsin( np.sin(lat1) * np.cos(angularDistDeg) + np.cos(lat1) * np.sin(angularDistDeg) * np.cos(bearingRad) )
    lon2 = lon1 + np.arctan2( np.sin(bearingRad) * np.sin(angularDistDeg) * np.cos(lat1), np.cos(angularDistDeg) - np.sin(lat1) * np.sin(lat2))
    
    lat2Deg = lat2 / deg2rad
    lon2Deg = lon2 / deg2rad
    
    lon2Deg = ScaleLongitude(lon2Deg)
    
    return [lat2Deg, lon2Deg]

#*************************************************************************************************************
# def BearingFromLatLon (lat1Deg, lon1Deg, lat2Deg, lon2Deg)
#
# Returns a bearing from two lat/long coordinates
#
# Source: http://www.edwilliams.org/avform.htm#LL 
# 
# Inputs:
#   lat1Deg -- Initial Latititude degrees
#   lon1Deg -- Initial Longitude degrees
#   lat2Deg -- Final Latititude degrees
#   lon2Deg -- Final Longitude degrees
#
# Outputs:
#   bearing -- In absolute degrees
#
#*************************************************************************************************************
def BearingFromLatLon (lat1Deg, lon1Deg, lat2Deg, lon2Deg):

    lat1 = lat1Deg * deg2rad
    lon1 = lon1Deg * deg2rad
    lat2 = lat2Deg * deg2rad
    lon2 = lon2Deg * deg2rad


    bearing =  np.mod(np.arctan2( np.sin(lon1-lon2)*np.cos(lat2), 
                    np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon1-lon2)), 2*np.pi)

    return bearing / deg2rad

#*************************************************************************************************************
def wrap_longitude(lon_deg):
    """
    Wraps the longitude value to lie within the range -180...180.

    Parameters
    ----------
    lon_deg : float array or np.ndarray
        longitude in degrees

    Returns
    -------
    lon_deg_new : float or np.ndarray
        re-scaled longitude in degrees
    """

    if lon_deg is None or len(lon_deg) == 0:
        return lon_deg

    # This first formaula is not entirely correct
   #lon_deg_new = (lon_deg + 540.0)%360.0 - 180.0

    if type(lon_deg) == list:
        if len(lon_deg) == 0:
            return None
        lon_deg = np.array(lon_deg)
        val_type = type(lon_deg[0])
    if type(lon_deg) == np.ndarray:
        if len(lon_deg) == 0:
            return None
        val_type = type(lon_deg[0])
    else:
        val_type = type(lon_deg)

   #lon_deg_new = (lon_deg % 360 + 540) % 360 - 180
    # See ATAPJ-147: There is a float32 rounding error when passing an array of float32 numbers to the below function. We
    # need to force the function to operate on float64 values. Then retype back to the original type.
    lon_deg_new  = np.mod((np.mod(lon_deg, 360, dtype='float64') + 540), 360, dtype='float64') - 180

    # now retype as the original type
    lon_deg_new = lon_deg_new.astype(val_type)

    return lon_deg_new

#*************************************************************************************************************
def unwrap_longitude(lon_deg, unwrap_direction='West', unwrap_cutoff=150):
    """ Unwraps the longitude so that it will extend past /pm 180 degrees. 

    There is a question of which way do we unwrap: east or west. Since GOES only ventures past -180 degrees West, we
    will default to wrap to the West. If we ever extend this to perhaps a ESA instrument then this will have to be
    revisited.

    We also need to specificy how far we unwrap. GOES-17 spans to only about +160 East. So, the default is to only
    unwrap to +150 degrees East. 

    Parameters
    ----------
    lon_deg : float array
        The longitude values to unwrap
    unwrap_direction : str {'West', 'East'}
        Which direction to unwrap
    unwrap_cutoff : float
        How many degrees is the limit we unwrap

    """

    assert unwrap_direction == 'West', 'This function can only unwrap to the West.'

    if lon_deg is None or len(lon_deg) == 0:
        return lon_deg

    # Since we are only unwrapping to the WEST, the operation is really simple,
    # Just subtract 360 from all values greater than unwrap_cutout

    # Ensure is a numpy array
    lon_deg = np.array(lon_deg)

    # Find which datums to wrap
    wrap_locs = np.greater(lon_deg, unwrap_cutoff)

    lon_deg_new = lon_deg.copy()
    lon_deg_new[wrap_locs] = lon_deg[wrap_locs] - 360.0

    return lon_deg_new


#*************************************************************************************************************
# def RadiusAlongEllipse (latRadiusDeg, lonRadiusDeg, bearingDeg):
#
# Returns the angular distance from the center of an ellipse-like form to the ellipse given the two axis lengths in degrees and a
# bearing.
#
# Note: all this is doing is using a sinusoidal mixing angle between two values in order to generate a smooth
# transition.
#
# The bearing is wrt to longitude axis -- which means relative to the latitude radius!
#
# Inputs:
#   latRadiusDeg    -- The latitude radius of the ellipse
#   lonRadiusDeg    -- The longitude radius of the ellipse
#   bearingDeg      -- Bearing is degrees wrt longitude axis
#
# Output:
#   distanceDeg     -- distance on ellipse from center along given bearing in degrees
#
#*************************************************************************************************************
def RadiusAlongEllipse (latRadiusDeg, lonRadiusDeg, bearingDeg):

    lat = latRadiusDeg * deg2rad
    lon = lonRadiusDeg * deg2rad
    phi = bearingDeg * deg2rad

    radius = ((lat - lon)/2)*(np.sin(2*phi + np.pi/2) + 1) + lon

    return radius / deg2rad

#******************************************************************************
def findLatLonBoxDistance(lat_array, lon_array):
    """ Returns the longest diagonal length of the smallest box that covers all lat/lon points.

    This is an over-approximation for the total distance of the lat/lon points.

    It uses the Haversine great-circle distance.

    Note: This only works for UNWRAPED longitudes. The unwrapping occurs when first pulling the data form the netCDF
    files, so we should never see wrapped longitudes, but be careful!

    Parameters
    ----------
    lat_array : float np.array 
        Array of Latitutude in degrees
    lon_array : float np.array
        Array of Longitude in degrees

    Returns
    -------
    box_distance : float
        Box distance of the lat/lon points

    """

    # Find the most extreme points in lat and lon
    # There are two diaginals.
    # However, simple geometry says they are both the same length
    return DistanceFromLatLonPoints(lat_array.max(), lon_array.max(), lat_array.min(), lon_array.min())

#******************************************************************************
# def FindFurthestLatLonPoints(lat_array, lon_array):
#
# Returns the indices of the furthest two points along a great-circle path.
#
# This is a non-trivial problem. The solution is to find the Convex Hull of all data points and then use the rotating
# callipers method to find the two furthest points on the hull.
#
# See: scipy.spatial.ConvexHull
#
# Inputs:
#   lat_array   -- [float list] Array of Latitutude in degrees
#   lon_array   -- [float list] Array of Longitude in degrees
#
# Outputs:
#   point0Index -- [int] Index of one of the furthest points in the input arrays
#   point1Index -- [int] Index of the other furthest point in the input arrays
#
#******************************************************************************


#******************************************************************************
# def FindFurthestLatLonPointsAlongAxis(lat_array, lon_array):
#
# Returns the indices of the furthest two points along north/south or east/west axis directions in a lat/lon array.
#
# Note: This returns the indices of the furthest two point along north/south or east/west lines. It does NOT return the
# indices of the two furthest points along a great-circle path.
#
# Inputs:
#   lat_array   -- [float list] Array of Latitutude in degrees
#   lon_array   -- [float list] Array of Longitude in degrees
#
# Outputs:
#   point0Index -- [int] Index of one of the furthest points in the input arrays
#   point1Index -- [int] Index of the other furthest point in the input arrays
#   horizonatalFlag -- [bool] If true then the furthest points are along the East/West axis.
#
#******************************************************************************

def FindFurthestLatLonPointsAlongAxis(lat_array, lon_array):

    most_west_lon = lon_array.min()
    most_east_lon = lon_array.max()
    most_south_lat = lat_array.min()
    most_north_lat = lat_array.max()
    west_ind = np.where(lon_array == most_west_lon)
    east_ind = np.where(lon_array == most_east_lon)
    north_ind = np.where(lat_array == most_north_lat)
    south_ind = np.where(lat_array == most_south_lat)

    north_south_dist = DistanceFromLatLonPoints(lat_array[north_ind[0][0]], lon_array[north_ind[0][0]],
                                             lat_array[south_ind[0][0]], lon_array[south_ind[0][0]])
    east_west_dist = DistanceFromLatLonPoints(lat_array[east_ind[0][0]], lon_array[east_ind[0][0]],
                                           lat_array[west_ind[0][0]], lon_array[west_ind[0][0]])

    if north_south_dist > east_west_dist:
        return [north_ind[0][0], south_ind[0][0], False]
    else:
        return [east_ind[0][0], west_ind[0][0], True]

#*************************************************************************************************************
def find_hot_spots(lon, lat, n_peaks=1, nbins=300, bandwidth=0.05, bracketingSigmaThreshold=4.0,
        radius_threshold=95, max_radius=100):
    """ Finds the hot spots in a Gaussian kernel density function from a heat map of a scatter plot.

    Longuitude wraps at /pm 180 degrees. That will mess up the heat map. GOES-17 and 18 ventures in to 
    the Eastern Hemisphere, we can wrap a little bit over to be past -180 degrees West.

    This function will also find the radius of the hot spot where the Gaussian kernel desntiy falls below 

    Inputs
    ------
    Lat : float array
    lon : float array
    n_peaks : int
        Number of peaks to find
    nbins : int
        Number of bins to use when generating the initial grid to find the initial guesses
        (This gives the resolution of the heat map)
    bandwidth : float
        bandwidth for gaussian_kernel
        (bw_method in scipy.stats.gaussian_kde)
    bracketingSigmaThreshold : float
        How many times above the RMS of the heat map to declare a peak region
        (For bracketing purposes)
    radius_threshold : float
        What percentage of the hot spot to be covered by the circle
        This gives what percentage drop in the KDE to be consdiered acceptable.
        in range [0, 100] percent
    max_radius : float
        Maximum radius for the hot spot encpompassing circle

    Returns
    -------
    lon_peaks : float array
        The found latitude peaks (np.nan if not found)
    lat_peaks : float array
        The found latitude peaks (np.nan if not found)
    radius : float array
        The found radius the encompases radius_threshold fraction of the hot spots

    """

    try:
        import geopy.distance as gDist
    except:
        raise Exception('find_hot_spots requires the geopy module')

    # Ensure are numpy arrays
    lat = np.array(lat)
    lon = np.array(lon)

    # Unwrap the longitude past -180 Deg West
    lon = unwrap_longitude(lon, unwrap_direction='West', unwrap_cutoff=150)

    lat_peaks = np.full(n_peaks, np.nan)
    lon_peaks = np.full(n_peaks, np.nan)

    #***
    # First generate the gaussian KDE model (heat map) of the scatter data
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents to bracket the peaks
    kde = gaussian_kde([lon,lat], bw_method=bandwidth)
    x, y = np.mgrid[np.min(lon):np.max(lon):nbins*1j, np.min(lat):np.max(lat):nbins*1j]
    x_flat = x.flatten()
    y_flat = y.flatten()
    values_flat = kde(np.vstack([x_flat, y_flat]))
    kde_grid_values = values_flat.reshape(nbins,nbins).copy()

    #***

    #***
    # Find the threshold values to define peaks
    # sigma = 1.4826 * mad(x)
    sigma = median_abs_deviation(kde_grid_values.flatten(), scale='normal')
    peak_threshold = sigma*bracketingSigmaThreshold

    #***
    # Iterate until we find n_peaks
    # We want to mask out already found peaks (i.e. knock down the tall tentpoles to find the shorter ones).
    kde_grid_values_masked = copy.copy(kde_grid_values)
    for iPeak in np.arange(n_peaks):

        # Find the peak initial guess
        peakLocIdx = np.nanargmax(values_flat)
        # Maximize the function around this peak
        x0 = x_flat[peakLocIdx]
        y0 = y_flat[peakLocIdx]
        x0_idx, y0_idx = np.unravel_index(peakLocIdx, (nbins, nbins))
        # Find the bounds
        # We bound the peak by finding all kde_grid_values_masked near the peak that are above bracketingSigmaThreshold * sigma
        # For simplicity, we bound using a rectangular box
        # Find first index before and after the peak that falls below the peak_threshold,since this is a smoothed Gaussian
        # kernel density function, we should be OK finding the first index below and above and not have to worry about noise.
        # We need to account for the condition that the bounding box is right at the edge of the full pixel array
        try:
            x_bound_upper_idx = np.nonzero(kde_grid_values_masked[x0_idx:,y0_idx]-peak_threshold < 0.0)[0][0] + x0_idx
        except:
            # If failed then use extreme upper bound of pixel array
            x_bound_upper_idx = nbins-1

        try:
            x_bound_lower_idx = np.nonzero(kde_grid_values_masked[:x0_idx,y0_idx]-peak_threshold < 0.0)[0][-1]
        except:
            x_bound_lower_idx = 0

        try:
            y_bound_upper_idx = np.nonzero(kde_grid_values_masked[x0_idx,y0_idx:]-peak_threshold < 0.0)[0][0] + y0_idx
        except:
            y_bound_upper_idx = nbins-1

        try:
            y_bound_lower_idx = np.nonzero(kde_grid_values_masked[x0_idx,:y0_idx]-peak_threshold < 0.0)[0][-1]
        except:
            y_bound_lower_idx = 0
        
        boundBox = ((x[x_bound_lower_idx,y0_idx], 
                     x[x_bound_upper_idx,y0_idx]), 
                    (y[x0_idx, y_bound_lower_idx],
                     y[x0_idx,y_bound_upper_idx]) )
        result = minimize(_find_peak_minimize_fcn, (x0,y0), args=(kde), bounds=boundBox)

        if result.success:
            lon_peaks[iPeak] = result.x[0]
            lat_peaks[iPeak] = result.x[1]


        # Mask the just found peak
        # Zero the kde_grid_values_masked within the bounding box
        bound_box_grid = np.mgrid[x_bound_lower_idx:x_bound_upper_idx, y_bound_lower_idx:y_bound_upper_idx]
        x_array = bound_box_grid[0,:].flatten()
        y_array = bound_box_grid[1,:].flatten()
        kde_grid_values_masked[x_array, y_array] = np.nan 
        values_flat = kde_grid_values_masked.flatten()


    #***
    # Find hot spot radii around each peak
    # This finds a radius around each hot spot that encompases all the hot spot data down to radius_threshold

    # The number of points to evaluate along the radius
    n_points = 100
   #t_arr = np.linspace(0, 2.0*np.pi, n_points)
    t_arr = np.linspace(0.0, 360.0, n_points)

    # This is a simple optimization. We just increase the radius until the the circle is large enough to cover the hot
    # spot up to radius_threshold percentage of the hot spot KDE.

    from math import radians
    from sklearn.neighbors import KernelDensity
    all_data_deg = np.vstack([lat, lon]).T
    radius = []
    for idx, center in enumerate(zip(lat_peaks, lon_peaks)):
        # Find data in the region about hot spot
        # This allows us to generaet a KDE with details just aroudn the hjot spot, not the entire GLM FOV.
        region_radius_km = 100.0
        data_this_peak = _find_points_within_circle(all_data_deg, center, region_radius_km)

        # Compute gaussian KDE about this hot spot
        # Create and fit a KDE, use a very fine bandwidth to get good details.
        bandwidth = 0.001 # degrees
        kde_params = {}
        kde_params['kernel'] = 'gaussian' # Default is gaussain so this is not necessary
        kde = KernelDensity(bandwidth=radians(bandwidth), metric="haversine", **kde_params)
        kde.fit(np.radians(data_this_peak))

        #***
        # Find peak KDE about this hot spot and compute threshold value
        kde_rand_samples = kde.sample(1000)
        density_per_steradian = np.exp(kde.score_samples(kde_rand_samples))
        # Compute the threshold percentage of the KDE
        threshold_score = np.percentile(density_per_steradian, 100-radius_threshold)


        # Scan radii and compute the fraction of the hot spot encompassed by the circle. 
        # Find the target encompassment.
        scores = []
        radius_samples = np.linspace(0.01, max_radius, 3000)
        for this_radius in tqdm(radius_samples, 'Finding hot spot radius for hot spot {} of {}'.format(idx+1, len(lat_peaks))):
            scores.append(_compute_radius_merit_fcn(this_radius, center, kde, t_arr, threshold_score))

        # The encompassing score will sometimes start high for a very small radius then drop before gradaully increasing as we
        # encompass the hot spot.
        # Because of this, we need to maek sure we get past the first peak
        # TODO: make sure I understand why it sometimes starts high
        # Find point it first drops below 0.80% of threshold score
        # np.argmin finds the first occurance
        below_idx = np.argmin(np.array(scores)-0.80)
        # now find first occurance at threshold past below_idx
        target_idx = np.argmax(scores[below_idx:])
        radius.append(radius_samples[below_idx+target_idx])

        pass

    #***
    # Highlight the found hot spots locations
    fig, ax = generate_plot(figsize=(20,10))
    fig, ex = plot_density(lat, lon, bandwidth=2,
            lat_resolution=200, lon_resolution=100, boundary=['goes-e', 'goes-w'],
            fig=fig, ax=ax)

    plot_scatter(lat_peaks, lon_peaks, boundary=['goes-e', 'goes-w'],
        marker="o", color="red", edgecolor=None, s=15, label='Hot Spots', fig=fig, ax=ax)

    plt.title('Density Distribution of Detections and Found Hot Spots', fontsize='x-large')
    plt.legend(fontsize='x-large', markerscale=1.0, loc='upper right')
    plt.show()

    return lon_peaks, lat_peaks, radius


def _find_peak_minimize_fcn(X, kde):
    """ 
    Function to minimize in scipy.minimize 

    Finds the peak of the hot spot

    This returns the value of the guassian_kde at the (lon,lat) position X

    """

    return -kde(X)
    

def _find_points_within_circle(data, center, radius):
    """
    Find all points within the circle defined by center and radius

    It does this by comouting the haversince distance to all points and keeps those within radius.

    Parmaters
    ---------
    data : float array(N, 2) (lat,lon)
        The latitudes and longitudes to assess in degrees
    center : (float, float)
        (latitude, longitude)
    radius : float
        in kilometers

    Returns
    -------
    data_keep : float array(N, 2) (lat,lon)
        The latitudes and longitudes to assess in degrees

    """

    dist = DistanceFromLatLonPoints(center[0], center[1], data[:,0], data[:,1])

    datums_to_keep = np.nonzero(dist <= radius)[0]

    data_keep = copy.copy(data)

    data_keep = data_keep[datums_to_keep,:]

    return data_keep

def _compute_radius_merit_fcn(radius, center, kde, t_arr, threshold_score):
    """
    Computes ratio of points on circle that are below threshold_score.

    """

    scores = _find_KDE_along_circle(radius, center, kde, t_arr)

    score = np.count_nonzero(scores < threshold_score) / len(t_arr)

    return score

def _find_KDE_along_circle(radius, center, kde, t_arr):
    """
    Finds the KDE scores along a circle.

    Parameters
    ----------
    radius : float
        The radius of the circle in kilometers
    center: float array(lat,lon)
        The center of the circle
    kde : sklearn.neighbors.KernelDensity


    Returns
    -------
    scores_along_circle : float array
        The KDE scores along the circle

    """

    on_circle = circle(center[0], center[1], radius, t_arr)

    data = np.vstack([np.radians(on_circle[0]), np.radians(on_circle[1])]).T

    scores = np.exp(kde.score_samples(data))

    return scores

#@jit(nopython=True, parallel=True)
def circle(lat, lon, r, t):
    """
    Parametric form of a circle

    Returns the points on a circle equidistant from a point on the Earth.

    Uses GeoPy

    Parameters
    ----------
    lat : float
        Center latitude 
    lon : float
        Center longitude
    r : float
        Radius in kilometers
    t : float array
        parametric value
        Bearing in degrees: 0 – North, 90 – East, 180 – South, 270 or -90 – West.

    Returns
    -------
    lats_on_circle : float array
        latitudes on circle
    lons_on_circle : float array
        longitudes on circle
    lat : float array
        Latitudes on circle
    lon : float array
        Longitudes on circle

    """


    import geopy.distance as gDist

    lats_on_circle = []
    lons_on_circle = []
    for bearing in t:
        point = gDist.distance(kilometers=r).destination((lat, lon), bearing=bearing)
        lats_on_circle.append(point.latitude)
        lons_on_circle.append(point.longitude)


   #x = x0 + r * np.cos(t)
   #y = y0 + r * np.sin(t)

    return [lats_on_circle, lons_on_circle]


#*************************************************************************************************************
def compute_centroid(x,y, values):
    """ Returns the x and y centroid given a distribution of values

    Uses simple flux weighted centroiding.

    Parameters
    ----------
    x   : float array
        The x coordinates for the data
    y   : float array
        The y coordinates for the data
    values : float array
        The amplitude values at each (x,y) pair

    Returns
    -------
    x_centr : float
    y_centr : float

    """

    sumValues = np.nansum(values)

    x_centr = np.sum(values * x) / sumValues
    y_centr = np.sum(values * y) / sumValues

    return x_centr, y_centr

