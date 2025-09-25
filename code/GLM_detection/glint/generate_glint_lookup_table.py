#author Clemens Rumpf clemens.rumpf@nasa.gov 2018
"""
Instructions from Clemans Rumpf:

Here is the glint spot calculation code. Here are necessary steps.

1) Install spiceypy
2) Go into /code/glint_spot.py
    a) Change directory strings throughout file to adjust it to your local paths (after this, the code should run)
    b) Line 99 change start and end julian dates
    c) Line 104 change the daily step number for which to calculate the glint spot
3) Run generate_glint_lookup_table.py

PS: There seems to be a bit of an artifact where the algorithm reports glint on the backside of the Earth as well. You
can easily clean that up by filtering out certain longitude ranges (will be obvious when you look at the result figures)
"""

from __future__ import print_function
import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import scipy.optimize as opt

#***
# Configuration:
#spice.furnsh("/Users/crumpf/Programming/GLM/Glint_Spot/configuration/glintMetaK.txt")
spice.furnsh('./data/kernels/naif0012.tls')
spice.furnsh('./data/kernels/earth_720101_070426.bpc')
spice.furnsh('./data/kernels/earth_070425_370426_predict.bpc')
spice.furnsh('./data/kernels/de430.bsp')
spice.furnsh('./data/kernels/earth_assoc_itrf93.tf')
spice.furnsh('./data/kernels/pck00010.tpc')
spice.furnsh('./data/kernels/GLM1617.bsp')

save_directory = './test/'

# ADJUST TIME FRAME HERE
start_JD = 2458849.5 # Jan 1 2020
end_JD = 2459215.5 # Jan 1 2021

# ADJUST STEPSIZE HERE
n_day_ticks = 24.0 #hourly
#n_day_ticks = 1440 #minute

glm_numbers = [16, 17]

def print_ver():
       """Prints the TOOLKIT version
       """
       print(spice.tkvrsn('TOOLKIT'))



def D2R(deg):
       return deg * np.pi / 180.

def R2D(rad):
       return rad * 180. / np.pi

def HalfLatLon(lat0_deg, lon0_deg, lat1_deg, lon1_deg):
       lat0_rad = D2R(lat0_deg)
       lon0_rad = D2R(lon0_deg)
       lat1_rad = D2R(lat1_deg)
       lon1_rad = D2R(lon1_deg)

       Bx = np.cos(lat1_rad) * np.cos(lon1_rad - lon0_rad)
       By = np.cos(lat1_rad) * np.sin(lon1_rad - lon0_rad)
       lat_mid_rad = np.arctan2(np.sin(lat0_rad) + np.sin(lat1_rad), np.sqrt((np.cos(lat0_rad) + Bx) * (np.cos(lat0_rad) + Bx) + By * By))
       lon_mid_rad = lon0_rad + np.arctan2(By, np.cos(lat0_rad) + Bx)

       lon_mid_deg = R2D(lon_mid_rad)
       lat_mid_deg = R2D(lat_mid_rad)
       return [lat_mid_deg, lon_mid_deg]



def FindSunSubLatLon(julian_day):
       etOne = spice.str2et(str(julian_day) + ' JD')

       sun_pos_tup = spice.spkezp(10, etOne, 'ITRF93', 'LT+S', 399)

       sun_pos_ECEF = sun_pos_tup[0]

       sun_pos_latlon = spice.reclat(sun_pos_ECEF)
       sun_lon_deg = R2D(sun_pos_latlon[1])
       sun_lat_deg = R2D(sun_pos_latlon[2])

       return [sun_lat_deg, sun_lon_deg]

def FindBearing(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
       lat1 = D2R(lat1_deg)
       lon1 = D2R(lon1_deg)
       lat2 = D2R(lat2_deg)
       lon2 = D2R(lon2_deg)
       y = np.sin(lon2-lon1) * np.cos(lat2)
       x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
       brng = np.arctan2(y, x)
       return R2D(brng)



def min_angles(rot_ang_rad, glm_pos_rec, sun_pos_rec, rotation_axis, etOne):

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


def plot_glint_spot(glm_number, n_day_ticks, txt_file_path = '', save_directory = '.'):

    glint_df = pd.read_csv(txt_file_path)

    glint_lons = glint_df.Longitude.values
    glint_lats = glint_df.Latitude.values
    jd_range = glint_df.JulianDay.values
    
    fig, ax = plt.subplots()
    ax.scatter(glint_lons, glint_lats, marker='+', c=jd_range, label='glint spot', s=15, linewidths=0.1)
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_title('Glint Spot Hourly from G{0:d} starting on {1:s}'.format(glm_number, str(jd_range[0])))
    ax.legend()
    plt.savefig(save_directory + 'Glint_Spot_From_G{0:d}_{1:s}_{2:d}.png'.format(int(glm_number), str(jd_range[0]), int(n_day_ticks)), dpi=180)
    plt.show()

for glm_number in glm_numbers:
       if glm_number == 17:
              glm_latlon = np.array([0.0, -137.2])
              glm_radius = 42164.0 # km
              glm_spice_id = '-171717'
       elif glm_number == 16:
              glm_latlon = np.array([0.0, -75.2])
              glm_radius = 42164.0  # km
              glm_spice_id = '-161616'
       else:
              print('Invalid GLM number selected')
              print('{0} is not a valid GLM number (choose 16 or 17 instead).'.format(glm_number))

       jd_range = np.arange(start_JD, end_JD, 1/n_day_ticks)

       glint_lats = np.zeros(len(jd_range))
       glint_lons = np.zeros(len(jd_range))

       glm_pos_rec = spice.latrec(glm_radius, D2R(glm_latlon[1]), D2R(glm_latlon[0]))

       del_inds = []

       datetime_0 = datetime.datetime.now()

       for i, jd in enumerate(jd_range):

              etOne = spice.str2et(str(jd) + ' JD')
              sun_pos_tup = spice.spkezp(10, etOne, 'ITRF93', 'LT+S', 399)
              sun_pos_rec = sun_pos_tup[0]

              sun_glm_rec = glm_pos_rec - sun_pos_rec

              occult_res = spice.occult('Sun', 'ELLIPSOID', 'IAU_SUN', 'Earth', 'ELLIPSOID', 'ITRF93', 'LT+S', glm_spice_id, etOne)
              if occult_res != 0:
                     del_inds.append(i)
                     continue

              rot_vector = np.cross(glm_pos_rec, sun_pos_rec)

              rot_ang_rad_0 = 0.0
              res = opt.minimize(min_angles, rot_ang_rad_0, args=(glm_pos_rec, sun_pos_rec, rot_vector, etOne), bounds=[(-1.5707963267948966, 1.5707963267948966)])
              rotation_angle = res.x[0]

              rotated_vec_rec = spice.vrotv(glm_pos_rec, rot_vector, rotation_angle)
              rotated_vec_latlon = spice.reclat(rotated_vec_rec)

              glint_lats[i] = R2D(rotated_vec_latlon[2])
              glint_lons[i] = R2D(rotated_vec_latlon[1])

       glint_lats[del_inds] = np.inf
       glint_lons[del_inds] = np.inf

       glint_df = pd.DataFrame(data={'JulianDay':jd_range, 'Latitude':glint_lats, 'Longitude':glint_lons})
       txt_file_path = save_directory + 'Glint_Spot_G{0:d}_{1:s}_{2:d}.txt'.format(int(glm_number), str(start_JD), int(n_day_ticks))
       glint_df.to_csv(txt_file_path, index=False)
       
       plot_glint_spot(glm_number, n_day_ticks, txt_file_path, save_directory)








