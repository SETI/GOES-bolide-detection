""" 
Module glint_point

Use to compute the glint point of a GEOS satellite.

This module requires the spice kernels specific to the GOES spacecraft.

Methods used in the module were provided by Clemens Rumpf with very little documentation.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as opt
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from geometry_utilities import equatorialEarthRadiusKm 

validSatellites = ('G16', 'G17', 'G18', 'G19')
# These are the default 'nominal_satellite_subpoint_lat/lon' for the spacecraft, if specific values are not
# available
NOMINAL_GLM16_LATLON = np.array([0.0, -75.2])
NOMINAL_GLM17_LATLON = np.array([0.0, -137.2])
NOMINAL_GLM18_LATLON = np.array([0.0, -137.2])
NOMINAL_GLM19_LATLON = np.array([0.0, -75.2])
NOMINAL_GLM_RADIUS = 35786.023 + equatorialEarthRadiusKm # radius from center of Earth assuming at equator 35786.023 + 6378.137 = 42164.0
    

"""
class GlintPoint

A class to compute the glint point for a GOES satallite and datetime.

Attributes
----------
kernelPath  : str
            Path to kernels to load
spice       : spiceypy module
            The spice module set up and ready to compute glint points

glm16_latlon : float array([2,]
            GLM 16 Latitude and Longitude
glm16_radius : float
            GLM 16 radius above center of Earth
glm16_spice_id : int
            GLM 16 spice id
glm17_latlon : float array([2,]
            GLM 17 Latitude and Longitude
glm17_radius : float
            GLM 17 radius above center of Earth
glm17_spice_id : int
            GLM 17 spice id
glm18_latlon : float array([2,]
            GLM 18 Latitude and Longitude
glm18_radius : float
            GLM 18 radius above center of Earth
glm18_spice_id : int
            GLM 18 spice id

glm16_pos_rec : 
                GLM-16 location in rectangular coordinates
glm17_pos_rec : 
                GLM-17 location in rectangular coordinates
glm18_pos_rec : 
                GLM-18 location in rectangular coordinates

"""
class GlintPoint:


    #*********************************************************************************************************
    def __init__(self, kernel_path, 
            glm16_latlon=NOMINAL_GLM16_LATLON, glm17_latlon=NOMINAL_GLM17_LATLON, glm18_latlon=NOMINAL_GLM18_LATLON, glm19_latlon=NOMINAL_GLM19_LATLON,
            glm16_radius=NOMINAL_GLM_RADIUS, glm17_radius=NOMINAL_GLM_RADIUS, glm18_radius=NOMINAL_GLM_RADIUS, glm19_radius=NOMINAL_GLM_RADIUS,
            multiProcessEnabled=False):
        """
        Constructor
        
        Constructs a glint point object to compute the glint point on the globe for a specific GOES spacecraft and datetime.
        
        The kernelPath must contain the following files:
            naif0012.tls
            earth_720101_070426.bpc
            earth_070425_370426_predict.bpc
            de430.bsp
            earth_assoc_itrf93.tf
            pck00010.tpc
            GLM1617.bsp
        
        Parameters
        ----------
        kernel_path : str
            Path to kernels to load.
        glm16_latlon : double list(2)
            [lat, lon] Nominal G16 satellite subpoint (if not given then the expected value is used)
        glm17_latlon : double list(2)
            [lat, lon] Nominal G17 satellite subpoint (if not given then the expected value is used)
        glm18_latlon : double list(2)
            [lat, lon] Nominal G18 satellite subpoint (if not given then the expected value is used)
        glm19_latlon : double list(2)
            [lat, lon] Nominal G19 satellite subpoint (if not given then the expected value is used)
        glm16_radius : double
            Nominal G16 satellite radius from center of Earth
        glm17_radius : double
            Nominal G17 satellite radius from center of Earth
        glm18_radius : double
            Nominal G18 satellite radius from center of Earth
        glm19_radius : double
            Nominal G19 satellite radius from center of Earth
        multiProcessEnabled : bool
            If true parallelize the glint_spot computation
        
        Returns
        -------
        
        """

        # Make a local copy of spice module so each GlintPoint instance has its own copy.
        import spiceypy as spice
        self.spice = spice
        # Clear the keeper. We will be loading all the kernels we need below
        self.spice.kclear()

        self.kernel_path = kernel_path
        if not os.path.isdir(self.kernel_path):
            raise Exception('kernel_path directory does not exist: {}'.format(self.kernel_path))

        self.spice.furnsh(os.path.join(kernel_path,'naif0012.tls'))
        self.spice.furnsh(os.path.join(kernel_path,'earth_720101_070426.bpc'))
        self.spice.furnsh(os.path.join(kernel_path,'earth_070425_370426_predict.bpc'))
        self.spice.furnsh(os.path.join(kernel_path,'de430.bsp'))
        self.spice.furnsh(os.path.join(kernel_path,'earth_assoc_itrf93.tf'))
        self.spice.furnsh(os.path.join(kernel_path,'pck00010.tpc'))
        self.spice.furnsh(os.path.join(kernel_path,'GLM1617.bsp'))

        self.glm16_latlon   = glm16_latlon
        self.glm16_radius   = glm16_radius # km from center of Earth assuming above equator
        self.glm16_spice_id = '-161616'

        self.glm17_latlon   = glm17_latlon
        self.glm17_radius   = glm17_radius # km
        self.glm17_spice_id = '-171717'

        self.glm18_latlon   = glm18_latlon
        self.glm18_radius   = glm18_radius # km
        # GOEs-18 is at the same location of GOES-17 when used as GOES-West
        self.glm18_spice_id = '-171717'

        self.glm19_latlon   = glm19_latlon
        self.glm19_radius   = glm19_radius # km 
        # GOEs-19 is at the same location of GOES-16 when used as GOES-East
        self.glm19_spice_id = '-161616'


        self.glm16_pos_rec = self.spice.latrec(self.glm16_radius, self.D2R(self.glm16_latlon[1]), self.D2R(self.glm16_latlon[0]))
        self.glm17_pos_rec = self.spice.latrec(self.glm17_radius, self.D2R(self.glm17_latlon[1]), self.D2R(self.glm17_latlon[0]))
        self.glm18_pos_rec = self.spice.latrec(self.glm18_radius, self.D2R(self.glm18_latlon[1]), self.D2R(self.glm18_latlon[0]))
        self.glm19_pos_rec = self.spice.latrec(self.glm19_radius, self.D2R(self.glm19_latlon[1]), self.D2R(self.glm19_latlon[0]))

        # This is for the process bar
        self.pbar = None
        self.completed_count = None # This records the number of parallel time points computed

        self.switcher = {
            'G16': (self.glm16_pos_rec, self.glm16_spice_id),
            'G17': (self.glm17_pos_rec, self.glm17_spice_id),
            'G18': (self.glm18_pos_rec, self.glm18_spice_id),
            'G19': (self.glm19_pos_rec, self.glm19_spice_id),
            'unknown': (None, None)
            }


        self.multiProcessEnabled = multiProcessEnabled

    #*********************************************************************************************************
    def glint_spot(self, satellite, DT, glm_latlon=None, glm_radius=None, verbosity=False):
        """
        Returns the glint spot for the given satellite at the given JulianDate

        If glint spot is occulted by the Earth then np.nan is returned for lat and lon.
        
        Parameters
        ----------
        satellite   : str list
            One of validSatellites. 
            Either a scaler to apply to all timestamps or a list for each timestamp
        DT          : datetime list
                        Datetime array of times to compute glint spot
        glm_latlon  : double list of list(2)
            [lat, lon] Nominal satellite subpoint for each DT (if not given then the expected value is used)
            You can mix and match G16 and G17 lat/Lons just be sure they correpospond properly to the satellite input
            array.
        glm_radius : double list
            [lat, lon] Nominal satellite radius (if not given then the expected value is used)
        verbosity   : Bool
                    If true, display progress bar
        
        Returns
        -------
        lat         : numpy.array list(shape(JD))
                        Latitude of glint spot in degrees
        lon         : numpy.array list(shape(JD))
                        Longitude of glint spot in degrees
        
        """

        self.verbosity = verbosity

        if isinstance(DT, datetime.datetime):
            DT = [DT]

        if isinstance(satellite, str):
            satellite = np.full(len(DT), satellite)

        if (glm_latlon is None):
            glm_latlon = np.full(len(DT), None)

        if (glm_radius is None):
            glm_radius = np.full(len(DT), None)

        if (self.verbosity):
            self.pbar = tqdm(total=len(DT), desc='Computing Glint Spot')

        etList = self.spice.datetime2et(DT)

        # Spiceypy is not vectorizable :(
        # ...but we can use MPI to parallelize
        rotated_vec_lat = []
        rotated_vec_lon = []
        self.completed_count = int(0) # This records the number of parallel time points computed

        # Python "can't pickle module objects" so the callable multi-process function needs to be a seperate
        # function, not a method within a class :(
        # TODO: the spice object is messing this all up. Figure out how to get this working in parallel
        glintPointDict = {  
                'pbar' : self.pbar,
                'switcher': self.switcher,
                'spice' : self.spice,
                'completed_count' : int(0),
                'verbosity' : verbosity}
        if (self.multiProcessEnabled):
            raise Exception ('Parallization of glint spot is not yet fully implemented')
            pool = mp.Pool()
            results = [pool.apply_async(_glint_spot_elemental, args=(glintPointDict, satellite[idx], etOne)) for idx, etOne in enumerate(etList)]
            outputs = [result.get() for result in results]
        else:
            for idx, etOne in enumerate(etList):
                rotated_vec_lat_elem, rotated_vec_lon_elem = self._glint_spot_elemental(satellite[idx],
                        etOne, glm_latlon[idx], glm_radius[idx])
                rotated_vec_lat.append(rotated_vec_lat_elem)
                rotated_vec_lon.append(rotated_vec_lon_elem)


        if (self.verbosity):
            self.pbar.close()

        return np.array(rotated_vec_lat), np.array(rotated_vec_lon)
        

    #*********************************************************************************************************
    @staticmethod
    def min_angles(rot_ang_rad, spice, glm_pos_rec, sun_pos_rec, rotation_axis, etOne):
        """ 
        Objective function to minimize. 

        But what is it computing? No documentation provided.

        Returns
        -------
        sqrt_angle_diff
        """

        # rotate glm sub spot by rot_ang_rad
        rotated_vec_rec = spice.vrotv(glm_pos_rec, rotation_axis, rot_ang_rad)
        rotated_vec_lonlat = spice.reclat(rotated_vec_rec)

        # get surface point vector for glint spot
        rotated_vec_surf_rec_all = spice.latsrf('ELLIPSOID', 'Earth', etOne, 'ITRF93', np.array([rotated_vec_lonlat[1], rotated_vec_lonlat[2]]))
        rotated_vec_surf_rec = rotated_vec_surf_rec_all[0]
        
        # calculate angle between surface point vector and glm vector
        glint_sat_rec = glm_pos_rec - rotated_vec_surf_rec
        glint_sat_angle = spice.vsep(rotated_vec_surf_rec, glint_sat_rec)
        
        # calculate angle between surface point vector and sun vector
        glint_sun_rec = sun_pos_rec - rotated_vec_surf_rec
        glint_sun_angle = spice.vsep(rotated_vec_surf_rec, glint_sun_rec)
        
        # subtract the square of each angle from each other and get it to zero
        sqrt_angle_diff = np.abs(glint_sat_angle**2.0 - glint_sun_angle**2.0)
        
        return sqrt_angle_diff

    #*********************************************************************************************************
    # Convert degrees to radians
    @staticmethod
    def D2R(deg):
        return deg * np.pi / 180.

    # convert radians to degrees
    @staticmethod
    def R2D(rad):
       return rad * 180. / np.pi

    #*********************************************************************************************************
    def generate_glint_point_table (self, satellite, start_DT, end_DT, output_dir_path):
        """
        Generates glint lookup table for every hour within the specified range.

        Parameters:
        -----------
        satellite   : str
            One of validSatellites 
        start_DT    : datetime.datetime
            Starting datetime for generating table
        end_DT      : datetime.datetime
            ending datetime for generating table
        output_dir_path : str
            Directory to save lookup table, None => do not save to file

        Returns:
        --------
        glint_df        : pandas dataframe
                        Contains: JulianDay, Latitude[degrees], Longitude[degrees]
        """

        from astropy.time import Time

        start_JD = Time(start_DT).jd
        end_JD   = Time(end_DT).jd

        # Cadence step size
        n_day_ticks = 24.0 # hourly
       #n_day_ticks = 720.0 # every 2 minutes

        jd_range = np.arange(start_JD, end_JD, 1/n_day_ticks)
        JD = Time(jd_range,format='jd')
        # Convert from Julian date to datetime
        DT = JD.to_datetime()

        glintLatArray, glintLonArray = self.glint_spot(satellite, DT, verbosity=True)

        glint_df = pd.DataFrame(data={'JulianDay':jd_range, 'Latitude':glintLatArray, 'Longitude':glintLonArray})

        if output_dir_path is not None:
            data_file_name = 'Glint_Spot_{0}_{1:s}_{2:d}.txt'.format(satellite, str(start_JD), int(n_day_ticks))
            txt_file_path = os.path.join(output_dir_path + data_file_name)
            glint_df.to_csv(txt_file_path, index=False)
        

        return glint_df

    #*********************************************************************************************************
    def plot_glint_spot(self, satellite, glint_df, output_dir_path):

        glint_lons = glint_df.Longitude.values
        glint_lats = glint_df.Latitude.values
        jd_range = glint_df.JulianDay.values
        
        fig, ax = plt.subplots()
        ax.scatter(glint_lons, glint_lats, marker='+', c=jd_range, label='glint spot', s=15, linewidths=0.1)
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        ax.set_title('Glint Spot Hourly from {0} starting on {1:s}'.format(satellite, str(jd_range[0])))
        ax.legend()

        if output_dir_path is not None:
            data_file_name = 'Glint_Spot_{0}_{1:s}.png'.format(satellite, str(jd_range[0]))
            fig_file_path = os.path.join(output_dir_path + data_file_name)
            plt.savefig(fig_file_path, dpi=180)

    #*********************************************************************************************************
    def _glint_spot_elemental (self, satellite, etOne, glm_latlon=None, glm_radius=None):
        """
        Performs the glint spot calculation for each time point. 
        Seperated in this helper function so that the glint_spot calculation can be parallelized.
        
        Parameters
        ----------
        satellite   : str
            one of validSatellites 
        etOne       : float
            seconds past J2000, TDB.
        glm_latlon : double list(2)
            [lat, lon] satellite subpoint to use, if not given then use nominal value in glintPointDict
        glm_radius : double list
            [lat, lon] Nominal satellite radius (if not given then the expected value is used)
        
        Returns
        -------
        rotated_vec_lat : float
                        Latitude of glint spot in degrees
        rotated_vec_lon : float
                        Longitude of glint spot in degrees
        
        """
        
        # Pick which satellite to use
        glm_pos_rec, glm_spice_id = self.switcher.get(satellite, self.switcher['unknown'])
        
        # Override the nominal glm pocition is a specific lat,lon and radius are passed
        if (glm_latlon is not None and glm_radius is not None):
            glm_pos_rec = self.spice.latrec(glm_radius, self.D2R(glm_latlon[1]), self.D2R(glm_latlon[0]))
        elif(glm_latlon is not None or glm_radius is not None):
            raise Exception ('Both glm_latlon and glm_radius must be passed or neither')
        
        if (glm_pos_rec is None or glm_spice_id is None):
            raise Exception("Unknown GOES Satellite")
        
        sun_pos_rec = self.spice.spkezp(10, etOne, 'ITRF93', 'LT+S', 399)[0]
        
        occult_res = self.spice.occult('Sun', 'ELLIPSOID', 'IAU_SUN', 'Earth', 'ELLIPSOID', 'ITRF93', 'LT+S', glm_spice_id, etOne)
        if occult_res != 0:
            # Glint point occulted by the Earth
            rotated_vec_lat = np.nan
            rotated_vec_lon = np.nan
        
        else:
        
            rot_vector = np.cross(glm_pos_rec, sun_pos_rec)
           
            # Also not vectorizable
            bounds = (-1.5707963267948966, 1.5707963267948966)
            # Brent's method seems to better find the minimum, but it sometimes finds a solution outside the desired
            # bounds, when that happens fall back to bounded minimizer
            res = opt.minimize_scalar(GlintPoint.min_angles, args=(self.spice, glm_pos_rec, sun_pos_rec, rot_vector, etOne), 
                                method='brent',  bracket=(-0.01, 0.01))
            rotation_angle = res.x
            # If rotation is not within desired bounds then use bounded minimizer
            if (rotation_angle < bounds[0] or rotation_angle > bounds[1]):
                res = opt.minimize_scalar(GlintPoint.min_angles, args=(self.spice, glm_pos_rec, sun_pos_rec, rot_vector, etOne),
                                method='bounded',  bounds=bounds)
                rotation_angle = res.x
           
            # Using opt.minimize is a lot slower
           #rot_ang_rad_0 = 0.0
           #res = opt.minimize(GlintPoint.min_angles, rot_ang_rad_0, args=(self.spice, glm_pos_rec, sun_pos_rec, rot_vector, etOne), 
           #        bounds=[bounds])
           #rotation_angle = res.x[0]
           
            rotated_vec_rec = self.spice.vrotv(glm_pos_rec, rot_vector, rotation_angle)
           
            rotated_vec_latlon_elemental = self.spice.reclat(rotated_vec_rec)
           
            rotated_vec_lat = GlintPoint.R2D(rotated_vec_latlon_elemental[2])
            rotated_vec_lon = GlintPoint.R2D(rotated_vec_latlon_elemental[1])
        
        self.completed_count += int(1)
        
        if (self.verbosity and self.completed_count > 0 and np.mod(self.completed_count, 50) == 0):
            self.pbar.update(50)
            pass
        
        return rotated_vec_lat, rotated_vec_lon


#*************************************************************************************************************
# This main function will perform a unit test comparing a generated glint spot table with a reference table 
if __name__ == "__main__":

    spice_kernel_path = '/Users/bohr/data/ATAP/GLM_bolide_detector/glints/spice_kernels'

   #ref_table = '/Users/bohr/data/ATAP/clemens_glint_code/results_from_clemens/Glint_Spot_G16_2458849.5_24.txt'
   #ref_table = '/Users/bohr/data/ATAP/clemens_glint_code/results/Glint_Spot_G16_2458849.5_24.txt'
    ref_table = '/Users/bohr/data/ATAP/clemens_glint_code/results/Glint_Spot_G17_2458849.5_24.txt'

    output_dir_path = None

    glintPoint = GlintPoint(spice_kernel_path, multiProcessEnabled=False)
    
    satellite = 'G17'
    # Start and end times
   #start_JD = 2458849.5 # Jan 1 2020
   #end_JD = 2459215.5 # Jan 1 2021
    start_DT = datetime.datetime(2020, 1, 1, hour=0, minute=0, second=0, microsecond=0)
    end_DT   = datetime.datetime(2021, 1, 1, hour=0, minute=0, second=0, microsecond=0)
    glint_df = glintPoint.generate_glint_point_table (satellite, start_DT, end_DT, output_dir_path)
    glintPoint.plot_glint_spot(satellite, glint_df, output_dir_path)

    glint_df_ref = pd.read_csv(ref_table)

    diff_df = glint_df_ref - glint_df
    diff_df.plot()
    plt.title('Difference between computed Glint Spot and Reference Table')
    plt.xlabel('Cadence Number')
    plt.ylabel('Error (Degrees or Days)')

    pass

