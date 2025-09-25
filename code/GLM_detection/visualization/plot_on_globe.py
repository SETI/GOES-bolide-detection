# This module will plot data on an Earth globe projection using the python Basemap package

import matplotlib.pyplot as plt
# TODO: replace Basemap with cartopy
#from mpl_toolkits.basemap import Basemap
import numpy as np
from scipy.stats import median_abs_deviation, gaussian_kde
import os

from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm


class GlobePlotter:
    #*************************************************************************************************************
    # Constructor
    #
    # Inputs:
    #   PROJ_LIB    -- [str] Basemap requires the environment variable PROJ_LIB to be properly set
    #                           TODO: figure out a more robust way to do this.
    #   figsize     -- [float,float] Horizontal and vertical figure size
    def __init__(self, PROJ_LIB, figsize=(9*1.618, 9), projection='kav7'):


        if not os.path.isdir(PROJ_LIB):
            raise Exception('PROJ_LIB must be passed for Basemap to function')
        os.environ["PROJ_LIB"] = PROJ_LIB

        # Use mpl_toolkits.basemap for the Earth
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_axes([0.02, 0.03, 0.96, 0.89])
        self.ax.set_facecolor('white')
        if projection=='kav7':
            self.m = Basemap(projection=projection, lon_0=-90, lat_0=0, resolution="l", fix_aspect=False)
        elif projection=='lcc':
            self.m = Basemap(projection=projection, lon_0=-90, lat_0=0, resolution="l", fix_aspect=False, 
                    llcrnrlat=-70,urcrnrlat=70,llcrnrlon=-210,urcrnrlon=0,lat_ts=20)
        else:
            self.m = Basemap(projection=projection,llcrnrlat=-70,urcrnrlat=70,llcrnrlon=-210,urcrnrlon=0,lat_ts=20,resolution='l')
        self.m.drawcountries(color='#7f8c8d', linewidth=0.8)
        self.m.drawstates(color='#bdc3c7', linewidth=0.5)
        self.m.drawcoastlines(color='#7f8c8d', linewidth=0.8)
        self.m.fillcontinents('#ecf0f1', zorder=0)
        

    #*************************************************************************************************************
    # plot_scatter_on_globe
    #
    # Plots a collection of scatter points on a globe
    #
    # inputs:
    #   lat -- [float array] The latititudes to plot
    #   lon -- [float array] The longitudes to plot
    #   **kwargs    -- [dict] extra arguments to pass to Basemap.scatter 
    #
    # Returns:
    #   ax  -- [matplotlib axis]
    #   x   -- [float array] The translated x coordinates
    #   y   -- [float array] The translated y coordinates
    #
    #*************************************************************************************************************
    def plot_scatter_on_globe (self, lat, lon, **kwargs):

        x, y = self.m(lon, lat)
        o1 = self.m.scatter(x,y, **kwargs)

        return self.ax, x, y

    #*************************************************************************************************************
    # plot_density_on_globe
    #
    # Converts a scatter of points to a 2D kernel desnity estimate using scipy.stats.kde.
    #
    # Inputs:
    #   lat         -- [float array] The latitudes to plot
    #   lon         -- [float array] The longitudes to plot
    #   nBins       -- [int] Number of bins for the mesh in lat and lon
    #   sigmaThreshold -- [float] The factor above the STD of the density before density color change begins
    #   **kwargs    -- [dict] extra arguments to pass to Basemap.pcolormesh 
    #
    #*************************************************************************************************************

    def plot_density_on_globe (self, lat, lon, nBins=300, sigmaThreshold=2.0, **kwargs):

       #bandwidth = 0.1
        bandwidth = 0.05

        x, y = self.m(lon, lat)

        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        nbins=nBins
        k = gaussian_kde([x,y], bw_method=bandwidth)
        xi, yi = np.mgrid[np.min(x):np.max(x):nbins*1j, np.min(y):np.max(y):nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # Normalize pdf density
        # PDF should integrate to 1.0
       #zi = zi / np.sum(zi)

        #***
        # Calibrate the density
        # Generate a grid of one point per square degree
        lonCal, latCal = np.mgrid[0:359:360*1j, -90:90:180*1j]
        lonCal = lonCal.flatten()
        latCal = latCal.flatten()
        xCal, yCal = self.m(lonCal, latCal)
        kCal = gaussian_kde([xCal,yCal], bw_method=bandwidth)
        ziCal = kCal(np.vstack([xi.flatten(), yi.flatten()]))
        # Divide density by 1 bolide per square degree to calibrate
        zi = zi / ziCal

        #***
        # Pick the max range for the colormap
        # We want to scale the density map so that only when the density is high do we get any meaningful color 
        # A meaningful change is if the density exceeds 2 factors above a robust standard deviation of the density
        # Compute a sigma from the median absolute deviation of the PDF
        # sigma = 1.4826 * mad(x)
        sigma = 1.4826 * median_abs_deviation(zi)
        threshold = sigma*sigmaThreshold

        levels = MaxNLocator(nbins=100).tick_values(threshold, zi.max())

        # TEMP: Force specific colorbar ranges
        minVal = 10
        maxVal = 100
       #minVal = 10
       #maxVal = 20
        levels = MaxNLocator(nbins=100).tick_values(minVal, maxVal)


        # Pick the desired colormap, sensible levels, and define a normalization
        # instance which takes data values and translates those into levels.
       #cmap = plt.get_cmap('hot')
        cmap = plt.get_cmap('viridis')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # Make the plot
        o1 = self.m.pcolormesh(xi,yi, zi.reshape(xi.shape), edgecolors='face', alpha=0.5, cmap=cmap, norm=norm, **kwargs)
        self.fig.colorbar(o1, ax=self.ax, label='Bolides per Square Degree', ticks=[10,20,30,40,50,60,70,80,90,100])
       #self.fig.colorbar(o1, ax=self.ax, label='Bolides per Square Degree', ticks=[10,11,12,13,14,15,16,17,18,19,20])
 
    #*************************************************************************************************************
    # plot_title_and_legend
    #
    # Inputs:
    #   title -- [str] Figure title, duh!
    #
    #*************************************************************************************************************
    def plot_title_and_legend (self, title=None):

        plt.title(title, fontsize='x-large')
        plt.legend(fontsize='x-large', markerscale=1.0, loc='upper right')

    #*************************************************************************************************************
    # save_fig
    #
    # Saves the figure in the format given by the extension
    #
    # Inputs:
    #   filename -- [str] full path and name of saved figure
    #
    #*************************************************************************************************************
    def save_fig(self, filname):

        self.fig.savefig('20210805_offset_vs_aper_mask_threshold.jpg')
        
