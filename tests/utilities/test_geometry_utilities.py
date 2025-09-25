# GLM pipeline utilities unit tests

import pytest
from numpy.testing import (
    assert_allclose,
)

import geometry_utilities as geoUtil

import numpy as np

def test_compute_centroid():

    # Simple test
    # The centroid should be (4, 5)
    y, x = np.meshgrid(np.linspace(0,9,10), np.linspace(0,9,10))
    values = np.zeros((10,10))
    values[4,5] = 2.0

    x_centr, y_centr = geoUtil.compute_centroid(x,y,values)

    assert x_centr == 4
    assert y_centr == 5

def test_find_hot_spots():

    np.random.seed(123)

    #***
    # Make a hot spot at a lat/lon point and see if we recover it
    minLat = -45
    maxLat = 45
    minLon = -150
    maxLon = -20


    # First make a random distribution of points over the entire FOV
    nSamples = 1000
    scatterPoints = np.random.random_sample((nSamples,2))
    # Scale random sample by lat/lon limits
    scatterLatLon = scatterPoints * (maxLon-minLon, maxLat-minLat) + (minLon, minLat)

    # Add in some Gaussian distibutions about the desired centroids
    hotSpots = []
    hotSpotCenters = [(-50,10), (-65, 30), (-100, -20), (-140, -35)]
    for iSpot in np.arange(len(hotSpotCenters)):
        # Generate a random point cloud.
        nGausSamples = 100
        mu = hotSpotCenters[iSpot] # Hot spot center
        A = np.random.rand(2,2)
        covMat = A.T @ A;
        hotSpots.append(np.random.multivariate_normal(mu, covMat, size=nGausSamples))

    # Combine two data sets
    dataSet = scatterLatLon
    for spot in hotSpots:
        dataSet = np.concatenate((dataSet, spot), 0)
    

    #***
    # Find the hot spots
    hotSpots = geoUtil.find_hot_spots(dataSet[:,0], dataSet[:,1], n_peaks=len(hotSpotCenters), PROJ_LIB=None)

    # Check the found hot spots are where we want them to be
    # There is an error when generating the hot spot gaussian distributions, so expect soem error in the centroids.
    # Confirm are within /pm 0.2 degrees lat and lon
    assert_allclose(hotSpotCenters[0], (hotSpots.lon_peaks[2], hotSpots.lat_peaks[2]), atol=0.2)
    assert_allclose(hotSpotCenters[1], (hotSpots.lon_peaks[0], hotSpots.lat_peaks[0]), atol=0.2)
    assert_allclose(hotSpotCenters[2], (hotSpots.lon_peaks[1], hotSpots.lat_peaks[1]), atol=0.2)
    assert_allclose(hotSpotCenters[3], (hotSpots.lon_peaks[3], hotSpots.lat_peaks[3]), atol=0.2)

    pass

