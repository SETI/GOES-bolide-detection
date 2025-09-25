# This module contains classes and ufnctions related to CNNs for bolide detection

import time
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import tempfile
import subprocess
import copy
import shutil
import pickle
import bz2
from tqdm import tqdm
from functools import partial
from traceback import print_exc
import cv2

from PIL import Image

import torch
from torch import nn, set_num_threads
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, RandomSampler, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights

import ray
from ray import tune
from ray import train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig

from ray.train import Checkpoint

from scipy import interpolate, sparse

import sklearn.metrics as skmetrics
from sklearn.metrics._base import _average_binary_score
from sklearn.utils.multiclass import type_of_target

import bolide_features as bFeatures
import bolide_detections as bd
import bolide_dispositions as bDisp
from generate_detection_validation_report import display_report_for_these_IDs

RAMDISK_PATH = '/tmp/ramdisk/cnn_training'

# Flag to use a fixed-time lgith curve instead of a fixed-length (in datums) light curve
USE_FIXED_TIME_LC = True

def copy_data_to_ramdisk(
        database_path=None, 
        cache_path=None,
        bolidesFromWebsite_file=None,
        random_forest_uncertainties_file=None, 
        ramdisk_path=RAMDISK_PATH):
    """ Copies the database file and website data file to ramdisk for faster loading

    Two options:
    1) Copy the raw pipeline image data to ramdisk.
        To enable: Set database_path to the path
    2) Copy the cached processed data to ramdisk.
        See cache_image_tensors_to_file.
        To enable: set cache_path to the path

    Parameters
    ----------
    database_path : str
        Path to the pipeline run and database files
        This should be the top level output directory for the pipeline run
    cache_path : str
        Path to the cached images and bolide data
        See cache_image_tensors_to_file.
    bolidesFromWebsite_file : str
        The filename and path to the website data file
    random_forest_uncertainties_file : str
        Path to the Random Forest uncertanties table pickle file
    ramdisk_path : str
        Path to the ramdisk location

    Returns
    -------
    two options:
        If database_ramdisk_file was passed:
            database_file : str
                Path to the ZODB database .fs file for this satellite on ramdisk
        elif cache_randisk_path was passed:
            Path to the cached data on ramdisk

    bolidesFromWebsite_ramdisk_file : str
        Path to the website data file on ramdisk
    random_forest_uncertainties_ramdisk_file : str
        Path to the random forest uncertainties table pickle file

    """

    assert bool(database_path != None) ^ bool(cache_path != None), 'Cannot set both database_path and cache_path'

    RAMDISK_PATH = ramdisk_path

    startTime = time.time()
    print('*****************')
    print('Copying data to ramdisk...')


    #********
    # Database and supporting files
    
    if database_path is not None:
        
        # Add trailing '/' to path
        if database_path[-1] != os.path.sep:
            database_path += os.path.sep
        
        # Remove leading '/' from database_path
        # Otherwise, os.path.join will not actually join the paths
        if database_path[0] == os.path.sep:
           data_path_dirname_no_leading_sep = database_path[1:]
        else:
            data_path_dirname_no_leading_sep = database_path 
        tmpFilePath = os.path.join(ramdisk_path, 'data', data_path_dirname_no_leading_sep)
        if os.path.isdir(tmpFilePath):
            # Clear out ramdisk path
            shutil.rmtree(tmpFilePath)
        os.makedirs(tmpFilePath, exist_ok=True)
        os.system('cp -r {}* {}'.format(database_path, tmpFilePath))
        database_ramdisk_file = os.path.join(tmpFilePath, 'bolide_database.fs')

    elif cache_path is not None:
        
        cache_ramdisk_path = os.path.join(ramdisk_path, os.path.split(cache_path)[1]) 
        os.makedirs(cache_ramdisk_path, exist_ok=True)
        os.system('cp -r {} {}'.format(cache_path, ramdisk_path))

    else:
        raise Exception('Either database_path or cache_path must be passed')

    #********
    # Website data
    os.system('cp {} {}'.format(bolidesFromWebsite_file, ramdisk_path))
    bolidesFromWebsite_basename = os.path.basename(bolidesFromWebsite_file)
    bolidesFromWebsite_ramdisk_file = os.path.join(ramdisk_path, bolidesFromWebsite_basename)

    #********
    # Random Forest uncertainties table pickle file
    if random_forest_uncertainties_file  is not None:
        os.system('cp -r {} {}'.format(random_forest_uncertainties_file, ramdisk_path))
        random_forest_uncertainties_basename = os.path.basename(random_forest_uncertainties_file)
        random_forest_uncertainties_ramdisk_file = os.path.join(ramdisk_path, random_forest_uncertainties_basename)
    else:
        random_forest_uncertainties_ramdisk_file = None


    print('Finished copying data to ramdisk.')
    endTime = time.time()
    totalTime = endTime - startTime
    print("Total copying time: {:.2f} minutes".format(totalTime/60))
    print('*****************')

    if database_path is not None:
        return database_ramdisk_file, bolidesFromWebsite_ramdisk_file, random_forest_uncertainties_ramdisk_file
    elif cache_path is not None:
        return cache_ramdisk_path, bolidesFromWebsite_ramdisk_file, random_forest_uncertainties_ramdisk_file

def clear_tmp_directory(paths_to_clear=['/tmp/ramdisk', '/tmp/cnn_image_cache'], verbosity=True):
    """ On the NAS GPU nodes, the /tmp directory points to a ramdisk. 
    This should be cleared out when a job on the node ends. But that is not working correctly, so, clear out the /tmp paths manually. 

    This is causing the memory on the NAS GPU nodes to be consumed. 
    When requesting a node with a lot of memory, the job is halted until one becomes available with enough free memory.

    Parameters
    ----------
    paths_to_clear : str list
        List of paths under /tmp to clear out
    verbosity : bool

    """

    if verbosity:
        print('Manually clearing out all files in /tmp directories...')

   ## Check that all paths begin with '/tmp/'
   #assert np.all([path[:5] == '/tmp/' for path in paths_to_clear]), "All paths to clear should begin with '/tmp/'"

    for path in paths_to_clear:
        if path is not None:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
        
    if verbosity:
        print('Finished clearing out all files in /tmp directories...')


    return

#*************************************************************************************************************
def cache_image_tensors_to_file(bolide_dispositions_object, 
        save_path='/tmp/ramdisk/cnn_image_cache', 
        transform=None,
        use_multi_band_data=True,
        save_objects=True, 
        columns_to_use=['detection_satellite_before'],
        verbosity=True
        ):
    """ Converts ABI band data to torch tensors for caching purposes.
    ...use_multi_band_data = True, then it just records to filenames of the stored data files.

    This will convert all cutout images to torch tensors in preparation for training

    This will clean bolide_dispositions_object of bolide candidates that do not contain the cutoutFeatures image data.
    So, bolide_dispositions_object IS MODIFIED IN THIS CODE! 

    Parameters
    ----------
    bolide_dispositions_object : bolide_dispositions.BolideDispositions
        The bolide dispositions object generated from the pipeline data set
    save_path : str
        Path to save tensors and data to. Path is first cleared of all files.
    transform : torchvision.transforms.transforms
        The tranformer to use when loading the cutout images
        Only used if use_multi_band_data=False
    use_multi_band_data : bool
        If True then use multi-band data, instead of quasi-RGB cutout images
    save_objects : bool
        If True then the bolide_dispositions_object and cached_filenames are also saved to save_path as
        'bolide_dispositions_object.p' and 'cached_filenames.p'
    columns_to_use : tuple of str
        List of the image columns to cache. If the column data is not available then this candidate is remove fomr the
        data set, so only select ones that are availabel in the image data files. 
        exmaple: columns_to_use=('detection_satellite_before', 'detection_satellite_after')
    verbosity : bool
        If True then display tqdm status bar

    Returns
    -------
    cached_filenames : Dict of Dicts of Dicts
        The cached image data
        The top level Dicts are the image columns (I.e before or after the bolide)
        The next level are the image types, where the keys are glmDataSet.img_types
        Then the bottom level keys are individual IDs and their values are the filenames for each ID
    bolide_dispositions_object : bolide_dispositions.BolideDispositions
        The bolide dispositions object with bad entries removed

    """

    # Handle both cases of using either quasi-RGB images or multi-band data

    cached_filenames = {}
    if use_multi_band_data:
        # This is if using multi-band data

        # Set up the cached_filenames dict to list all the cached filenames
       #cached_filenames = {'detection_satellite_before':{}, 'detection_satellite_after':{}}
        for column in columns_to_use:
            cached_filenames[column] = {}
        for column in cached_filenames.keys():
            for img_str in glmDataSet.img_types:
                cached_filenames[column][img_str] = {}
                save_path_this_group = os.path.join(save_path, column, img_str)
                # Clear the cache location
                if os.path.isdir(save_path_this_group):
                    # Clear out path
                    shutil.rmtree(save_path_this_group)
                try:
                    os.makedirs(save_path_this_group)
                except OSError:
                    raise Exception('Creation of the directory {} failed'.format(save_path_this_group))


        # Gather the data for this image type
        datumsToRemove = []
        for idx in tqdm(range(len(bolide_dispositions_object.bolideDispositionProfileList)),
                'Caching image files for bolide candidates...', disable=(not verbosity)):

            bolide  = bolide_dispositions_object.bolideDispositionProfileList[idx]
            if not bolide.cutoutFeatures.success or (use_multi_band_data and bolide.cutoutFeatures.image_pickle_filename is None):
                # Multi-band data does not exist for this candidate, remove from list
                datumsToRemove.append(idx)
                continue

            # Pull out the data for all image types for this bolide candidate
            # If database_path is set, then use that as the absolute path, otherwise use a relative path
            if bolide_dispositions_object.database_path is not None:
                database_path = os.path.join(os.path.dirname(bolide_dispositions_object.database_path), bolide.features.goesSatellite)
                year = str(bolide.features.bolideTime.year)
                MMDD = str(bolide.features.bolideTime.month).zfill(2) + str(bolide.features.bolideTime.day).zfill(2)
                pickle_data_filename = os.path.join(database_path, year, MMDD, os.path.basename(bolide.cutoutFeatures.image_pickle_filename))
            else:
                pickle_data_filename = bolide.cutoutFeatures.image_pickle_filename

            with bz2.open(pickle_data_filename, 'rb') as fp:
                multi_band_data_all_images = pickle.load(fp)
            fp.close()

            try:
                for column in cached_filenames.keys():
                    if multi_band_data_all_images[column] is not None:
                        for img_str in glmDataSet.img_types:
                        
                            raw_data = multi_band_data_all_images[column][img_str]
                        
                            save_filename = os.path.splitext(os.path.basename(pickle_data_filename))[0] + '.pt'
                            save_path_this_group = os.path.join(save_path, column, img_str)
                            cached_filenames[column][img_str][bolide.ID] = os.path.join(save_path_this_group, save_filename)
                            if img_str in ('60_sec_integ', '60_sec_integ_zoom'):
                                # Check if ABI data is square
                                img_size = raw_data[0].shape
                                if img_size[0] != img_size[1]:
                                    print('Image index {} is not square'.format(idx))
                                    datumsToRemove.append(idx)
                                    break
                                data = scale_ABI_and_GLM_data(raw_data)
                            else:
                                # This is just GLM sparse data
                                data = raw_data.transpose((2,0,1))
                                GLM_data_dims = [data[idx,...] for idx in range(data.shape[0])]
                                data = [sparse.coo_array(GLM_data_dims[idx]) for idx in range(data.shape[0])]
                            torch.save(data, cached_filenames[column][img_str][bolide.ID])
                    else:
                        # Data not available, remove target from list
                        print('Data not available for {}, removing from list'.format(column))
                        datumsToRemove.append(idx)
            except:
                print('Error processing target index {}, removing'.format(idx))
                datumsToRemove.append(idx)

        # Remove bad bolides
        if (len(datumsToRemove) > 0):
            print('Bolide validation: Removing {} of {} bolide candidates having invalid or incomplete data.'.format(len(datumsToRemove), len(bolide_dispositions_object.bolideDispositionProfileList)))
            for datumIdx in sorted(np.unique(datumsToRemove), reverse=True):
                try:
                    del bolide_dispositions_object.bolideDispositionProfileList[datumIdx]
                except:
                    raise Exception('Error deleting profile index {} for ID {}'.format(datumIdx, bolide.ID))


    else:
        # This is if using quasi-RGB images
        for img_str in glmDataSet.img_types:
            datumsToRemove = []

            # Gather the image filenames
            image_filenames = []
            for idx, bolide in enumerate(bolide_dispositions_object.bolideDispositionProfileList):
                if not bolide.cutoutFeatures.success or bolide.cutoutFeatures.figure_filenames is None:
                    # Cutout images do not exist for this candidate, remove from list
                    datumsToRemove.append(idx)
                    continue
            
                # Image file names list
                # Find the filename for figures containing img_str
                fig_idx = int(np.nonzero([img_str in string for string in bolide.cutoutFeatures.figure_filenames])[0])
                year = str(bolide.features.bolideTime.year)
                MMDD = str(bolide.features.bolideTime.month).zfill(2) + str(bolide.features.bolideTime.day).zfill(2)
            
                # Get the image filename. The bDispObj.database_path does not correctly work for extra_bolideDispositionProfileList
                # But, the difference should be just the last directory name for the satellite
                database_path = os.path.join(os.path.dirname(bolide_dispositions_object.database_path), bolide.features.goesSatellite)
                filename = os.path.join(database_path, year, MMDD, bolide.cutoutFeatures.figure_filenames[fig_idx])
                image_filenames.append(filename)
            
            # Remove bad bolides
            # Each image type can remove individual bad bolides, so, this happens inside the img_str for-loop
            if (len(datumsToRemove) > 0):
                for datumIdx in sorted(np.unique(datumsToRemove), reverse=True):
                    del bolide_dispositions_object.bolideDispositionProfileList[datumIdx]
            
            save_path_this_group = os.path.join(save_path, img_str)
            cached_filenames[img_str] = convert_images_to_tensors(image_filenames, save_path_this_group, transform=transform, img_str=img_str)

    # If no images to cache then return cached_filenames = None
    if len(bolide_dispositions_object.bolideDispositionProfileList) == 0:
        print('********WARNING: No images to cache!')
        cached_filenames = None
        return cached_filenames, bolide_dispositions_object

    if save_objects:
        bDispObj_filename = os.path.join(save_path, 'bolideDispositionProfileList.p')
        bolide_dispositions_object.save_bolideDispositionProfileList(bDispObj_filename)
        cached_filename = os.path.join(save_path, 'cached_filenames.p')
        with open(cached_filename, 'wb') as fp :
            pickle.dump(cached_filenames, fp)
        fp.close()


    return cached_filenames, bolide_dispositions_object


def convert_images_to_tensors(image_filenames, save_path, transform=None, img_str=None):
    """ Converts png images to torch vectors 
    then saves the tensors to file located at save_path. 
    This path is first cleared of any files.

    Parameters
    ----------
    image_filenames : list of str
        List of PNG filenames to convert
    save_path : str
        Path to save tensors to. Path is first cleared of all files.
    transform : torchvision.transforms.transforms
        The tranformer to use when loading the cutout images
    img_str : str
        A string to identify this processing for use in tqdm.
        Does not impact processing, just for verbosity

    Returns
    -------
    cached_filenames : Dict of str
        Dict of saved filenames, full path
        The key is the ID


    """

    # First clear the cache location
    if os.path.isdir(save_path):
        # Clear out ramdisk path
        shutil.rmtree(save_path)
    try:
        os.makedirs(save_path)
    except OSError:
        raise Exception('Creation of the directory {} failed'.format(save_path))

    # Now save out the tensors
    cached_filenames = {}
    for idx in tqdm(range(len(image_filenames)),'Caching image files for type {}...'.format(img_str), disable=(img_str is None)):
        filename = image_filenames[idx]
        image = glmDataSet.pil_loader(filename)
        if transform is not None:
            image = transform(image)
        save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
        ID = int(os.path.split(save_filename)[1][0:19])
        cached_filenames[ID] = os.path.join(save_path, save_filename)
        torch.save(image, cached_filenames[ID])

    return cached_filenames

def scale_ABI_and_GLM_data(raw_data):
    """ scales the ABI image data perfoming two operations:

    1. Convert uint16 data to uint8
    2. Resizes by a factor of 0.5.

    Then scales the GLM data to the new resized ABI data

    Returns
    -------
    data : torch.tensor list
        A list of two tensors:
            [0] : torch.tensor(uint8)
                Scaled ABI data
            [1] : torch.sparse_coo(float32)
                GLM data scaled to ABI image data

    """

    #***
    # ABI Data

    # Downsize ABI data
    resize_factor = 2
    img_size = raw_data[0].shape
    ABI_image = cv2.resize(raw_data[0], (int(img_size[0]/resize_factor), int(img_size[1]/resize_factor)), interpolation=cv2.INTER_AREA)

    # Scale between [0,255]
   #minVal = np.min(ABI_image, axis=(0,1))
    maxVal = np.max(ABI_image, axis=(0,1))
    for dim in range(img_size[2]):
       #ABI_image[...,dim] -= minVal[dim]
       #ABI_image[...,dim] = ABI_image[...,dim] / ((maxVal[dim]-minVal[dim]) / 255)
        ABI_image[...,dim] = ABI_image[...,dim] / (maxVal[dim] / 255.0)
    ABI_image = ABI_image.transpose((2,0,1))

    # Convert to uint8
    ABI_image = torch.tensor(ABI_image.astype(np.uint8))
    
    #***
    # GLM data

    # Downsize GLM data to sync with the ABI data
    # Since this is sparse data, we do not want to interpolate because that might create some artifacts.
    # Instead use Max Pooling
    GLM_data = raw_data[1].transpose((2,0,1))
    # Use torch's maxpool to resize (so, need to convert to a tensor temporarily)
    GLM_data = F.max_pool2d(torch.tensor(GLM_data), resize_factor).numpy()

    # Scale GLM data to ABI uint8 range
    # We take the log of the GLM data so that it is not all piled up near zero (see ATAPJ-147)
    # Scale to [log(8.0e-16), log(3.5e-10)] =  [-34.77,  -21.76] to the range [0, 255]
    max_GLM_log_scale = -21.76
    min_GLM_log_scale = -34.77
    GLM_not_zero_here = np.nonzero(GLM_data)
    log_GLM_data_nonzero = np.log(GLM_data[GLM_not_zero_here])

    log_GLM_data_scaled_nonzero = (log_GLM_data_nonzero - min_GLM_log_scale) / (max_GLM_log_scale-min_GLM_log_scale)
    # Clip everything outside the scale
    log_GLM_data_scaled_nonzero[log_GLM_data_scaled_nonzero<0.0] = 0.0 
    log_GLM_data_scaled_nonzero[log_GLM_data_scaled_nonzero>1.0] = 1.0 

    # Scale to [0, 255]
    log_GLM_data_scaled_nonzero = log_GLM_data_scaled_nonzero * 255.0

    log_GLM_data_scaled = np.zeros_like(GLM_data, dtype=np.float32)
    log_GLM_data_scaled[GLM_not_zero_here] = log_GLM_data_scaled_nonzero


    # Use scipy sparce arrays. These must be 2-dimensional arrays, so break out the channel dimension.
    GLM_data_dims = [log_GLM_data_scaled[idx,...] for idx in range(log_GLM_data_scaled.shape[0])]
    GLM_data_sparse = [sparse.coo_array(GLM_data_dims[idx]) for idx in range(log_GLM_data_scaled.shape[0])]

    data = []
    data.append(ABI_image)
    data.append(GLM_data_sparse)

    return data

def load_cached_objects(save_path='/tmp/ramdisk/cnn_image_cache', image_columns_to_use='all'):
    """ Loads in the cached bolideDispositionsObject and cached_filenames list generated via 
    cache_image_tensors_to_file.

    Parameters
    ----------
    save_path : str
        Path cached data saved to. Path is first cleared of all files.
    image_columns_to_use : str tuple or 'all'
        The names of the image columns to use. See glmDataSet.img_column_names for options.

    Returns
    -------
    cached_filenames : Dict of Dicts
        List of cached filenames where the keys are glmDataSet.img_types
        Then the second level keys are individual IDs and their values are the filenames for each ID
    bolide_dispositions_object : bolide_dispositions.BolideDispositions
        The bolide dispositions object with bad entries removed

    """ 

    
    bDispObj_filename = os.path.join(save_path, 'bolideDispositionProfileList.p')
    with open(bDispObj_filename, 'rb') as fp:
        bolideDispositionProfileList = pickle.load(fp)
    fp.close()
    bolide_dispositions_object = bDisp.BolideDispositions.from_bolideDispositionProfileList(bolideDispositionProfileList, useRamDisk=False)
        
    cached_filename = os.path.join(save_path, 'cached_filenames.p')
    with open(cached_filename, 'rb') as fp:
        cached_filenames_tmp = pickle.load(fp)
    fp.close()

    # Only keep cached images form columns requested
    cached_filenames = {}
    if image_columns_to_use != 'all':
        for column in image_columns_to_use:
            cached_filenames[column] = cached_filenames_tmp[column]

    return cached_filenames, bolide_dispositions_object

#*************************************************************************************************************
class simpleDropoutWExtraFeatures(nn.Module):
    """ 
    A simple CNN with some convolutional layers, some fully connected layers and dropout

    But adds in some extra features to the final fully connected layer.
    
    The output is a 2-dimenational classifier [Bolide, Not-Bolide]

    """
    def __init__(self, kernel_size=3, n_chan_1=32, n_chan_fc1=128, dropout_p=0.4, momentum=0.1, image_size=None, n_extra_features=0):

        raise Exception('This model is no longer maintained')

        assert kernel_size==3, 'Kernel_size must be 3'

        super().__init__()
        self.n_chan_1 = n_chan_1

        self.conv1 = nn.Conv2d(3, n_chan_1, kernel_size=kernel_size, padding=1)
        self.conv1_dropout = nn.Dropout2d(p=dropout_p)

        self.conv2 = nn.Conv2d(n_chan_1, n_chan_1 // 2, kernel_size=kernel_size, padding=1)
        self.conv2_dropout = nn.Dropout2d(p=dropout_p)

        self.conv3 = nn.Conv2d(n_chan_1 // 2, n_chan_1 // 4, kernel_size=kernel_size, padding=1)
        self.conv3_dropout = nn.Dropout2d(p=dropout_p)

        self.conv4 = nn.Conv2d(n_chan_1 // 4, n_chan_1 // 8, kernel_size=kernel_size, padding=1)
        self.conv4_dropout = nn.Dropout2d(p=dropout_p)

        self.flatten = nn.Flatten()

       #fc1_input_size = image_size**2 //4 //4 //4 * n_chan_1 // 2 //2 
        fc1_input_size = image_size**2 //4 //4 //4 //4* n_chan_1 // 2 //2 //2

        self.fc1 = nn.Linear(fc1_input_size, n_chan_fc1)

        # Add in the extra features in the middle fully connected layer
        self.batchNorm = nn.BatchNorm1d(n_chan_fc1+n_extra_features, momentum=momentum)
        self.fc2 = nn.Linear(n_chan_fc1+n_extra_features,64)

        self.fc3 = nn.Linear(64,2)

        self._init_weights()


    def forward(self, imgs, extra_features):
        out = F.max_pool2d(F.relu(self.conv1(imgs)),2)
        out = self.conv1_dropout(out)

        out = F.max_pool2d(F.relu(self.conv2(out)),2)
        out = self.conv2_dropout(out)

        out = F.max_pool2d(F.relu(self.conv3(out)),2)
        out = self.conv3_dropout(out)

        out = F.max_pool2d(F.relu(self.conv4(out)),2)
        out = self.conv4_dropout(out)

        out = self.flatten(out)

        out = F.relu(self.fc1(out))

        # Add in extra features
        out = torch.cat((out, extra_features), dim=1)
        # Use batch normalization to scale the extra_features to the image data
        out = self.batchNorm(out)

        out = F.relu(self.fc2(out))

        out = self.fc3(out)

        return out

    def _init_weights(self):
        """ This is based on the recommended weight initialization from "Deep Learning with Pytorch" by Eli Stevens et al.

        """
        for network in self.modules():
            for m in network.modules():
                if type(m) in {nn.Linear, nn.Conv2d}:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                        bound = 1 / math.sqrt(fan_out)
                        nn.init.normal_(m.bias, -bound, bound)

        return

#*************************************************************************************************************
# A model with different CNNs for multiple images

class CNN_block_batchNorm(nn.Module):
    """ This is a single CNN block used to analyze a single cutout image.

    It utilizes batch normalization and leaky ReLU

    """
    def __init__(self, 
            kernel_size=3, 
            n_input_chan=3,
            n_chan_1=32, 
            n_chan_fc=128, 
            momentum=0.1, 
            leakyReLU_negative_slope=0.01, 
            image_size=None,
            use_multi_band_data=True):
        """
        Parameters
        ----------
        n_input_chan : int
            The number of channels for the input "image" 
            Standard RGB is 3 channels
        use_multi_band_data : bool
            If True then use multi-band data, otherwise of quasi-RGB cutout images
        """

        assert kernel_size==3, 'Kernel_size must be 3'

        self.use_multi_band_data = use_multi_band_data

        super().__init__()

        # This is to normalize the input image channels, The ABI data and GLM data are of considerable different scales.
        # We just want to normalize the input channels, so disable momentum (set to None)
        self.input_batchNorm = nn.BatchNorm2d(num_features=n_input_chan, momentum=None)

        self.conv1 = nn.Conv2d(n_input_chan, n_chan_1, kernel_size=kernel_size, padding=1)
        self.conv1_batchNorm = nn.BatchNorm2d(num_features=n_chan_1, momentum=momentum)
        self.leakyReLU1 = nn.LeakyReLU(negative_slope=leakyReLU_negative_slope)

        self.conv2 = nn.Conv2d(n_chan_1, n_chan_1 // 2, kernel_size=kernel_size, padding=1)
        self.conv2_batchNorm = nn.BatchNorm2d(num_features=n_chan_1 // 2, momentum=momentum)
        self.leakyReLU2 = nn.LeakyReLU(negative_slope=leakyReLU_negative_slope)

        self.conv3 = nn.Conv2d(n_chan_1 // 2, n_chan_1 // 4, kernel_size=kernel_size, padding=1)
        self.conv3_batchNorm = nn.BatchNorm2d(num_features=n_chan_1 // 4, momentum=momentum)
        self.leakyReLU3 = nn.LeakyReLU(negative_slope=leakyReLU_negative_slope)

        self.conv4 = nn.Conv2d(n_chan_1 // 4, n_chan_1 // 8, kernel_size=kernel_size, padding=1)
        self.conv4_batchNorm = nn.BatchNorm2d(num_features=n_chan_1 // 8, momentum=momentum)
        self.leakyReLU4 = nn.LeakyReLU(negative_slope=leakyReLU_negative_slope)

        #***
        # Determining fc1_input_size is non-trivial.
        # Experimentally determine it here
       #fc1_input_size = image_size**2 //4 //4 //4 //4* n_chan_1 // 2 //2 //2
        test_data = np.zeros((1, n_input_chan, image_size, image_size))
        test_data = torch.tensor(test_data, dtype=torch.float32)
        out = self.conv1_batchNorm(self.conv1(test_data))
        out = F.max_pool2d(self.leakyReLU1(out),2)

        out = self.conv2_batchNorm(self.conv2(out))
        out = F.max_pool2d(self.leakyReLU2(out),2)

        out = self.conv3_batchNorm(self.conv3(out))
        out = F.max_pool2d(self.leakyReLU3(out),2)

        out = self.conv4_batchNorm(self.conv4(out))
        out = F.max_pool2d(self.leakyReLU4(out),2)

        out = nn.Flatten()(out)
        fc1_input_size = out.shape[1]
        #***

        
        self.fc = nn.Linear(fc1_input_size, n_chan_fc)
        self.leakyReLUfc = nn.LeakyReLU(negative_slope=leakyReLU_negative_slope)
        
        self.flatten = nn.Flatten()

        self._init_weights()

    def forward(self, imgs):

        # If we are using multi-band data, then use batch normalization to nornmalize the input channels
        if self.use_multi_band_data:
            out = self.input_batchNorm(imgs)
        else:
            out = imgs

        out = self.conv1_batchNorm(self.conv1(out))
        out = F.max_pool2d(self.leakyReLU1(out),2)

        out = self.conv2_batchNorm(self.conv2(out))
        out = F.max_pool2d(self.leakyReLU2(out),2)

        out = self.conv3_batchNorm(self.conv3(out))
        out = F.max_pool2d(self.leakyReLU3(out),2)

        out = self.conv4_batchNorm(self.conv4(out))
        out = F.max_pool2d(self.leakyReLU4(out),2)

        out = self.flatten(out)

        out = self.leakyReLUfc(self.fc(out))

        return out

    def _init_weights(self):
        """ This is based on the recommended weight initialization from "Deep Learning with Pytorch" by Eli Stevens et al.

        """
        for network in self.modules():
            for m in network.modules():
                if type(m) in {nn.Linear, nn.Conv2d}:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                        bound = 1 / math.sqrt(fan_out)
                        nn.init.normal_(m.bias, -bound, bound)

        return

class CNN_1D_block_batchNorm(nn.Module):
    """ This is for the 1-D light curve time series for the bolide candidate GLM data.

    """
    GLM_LC_DATA_LENGTH = 1024

    def __init__(self, 
            data_length=None,
            kernel_size=3, 
            n_chan_1=32, 
            n_chan_fc=128, 
            momentum=0.1, 
            leakyReLU_negative_slope=0.01):
        """
        Parameters
        ----------
        data_length : int
            The number of datums in the GLM time series.
        """

        super().__init__()

        # Default data length
        if data_length is None:
            data_length = self.GLM_LC_DATA_LENGTH

        self.conv1 = nn.Conv1d(1, n_chan_1, kernel_size, stride=1, padding=1)
        self.conv1_batchNorm = nn.BatchNorm1d(num_features=n_chan_1, momentum=momentum)
        self.leakyReLU1 = nn.LeakyReLU(negative_slope=leakyReLU_negative_slope)

        self.conv2 = nn.Conv1d(n_chan_1, n_chan_1 // 2, kernel_size=kernel_size, padding=1)
        self.conv2_batchNorm = nn.BatchNorm1d(num_features=n_chan_1 // 2, momentum=momentum)
        self.leakyReLU2 = nn.LeakyReLU(negative_slope=leakyReLU_negative_slope)

        self.conv3 = nn.Conv1d(n_chan_1 // 2, n_chan_1 // 4, kernel_size=kernel_size, padding=1)
        self.conv3_batchNorm = nn.BatchNorm1d(num_features=n_chan_1 // 4, momentum=momentum)
        self.leakyReLU3 = nn.LeakyReLU(negative_slope=leakyReLU_negative_slope)

        fc1_input_size = data_length //2 // 2 //2 * n_chan_1 // 4
        self.fc = nn.Linear(fc1_input_size, n_chan_fc)
        self.leakyReLUfc = nn.LeakyReLU(negative_slope=leakyReLU_negative_slope)
        
        self.flatten = nn.Flatten()

        self._init_weights()

    def forward(self, lc):
        
        # Input number of channels is 1, unsqueeze to add this dimension
        lc = torch.unsqueeze(lc,1)

        out = self.conv1_batchNorm(self.conv1(lc))
        out = F.max_pool1d(self.leakyReLU1(out),2)

        out = self.conv2_batchNorm(self.conv2(out))
        out = F.max_pool1d(self.leakyReLU2(out),2)

        out = self.conv3_batchNorm(self.conv3(out))
        out = F.max_pool1d(self.leakyReLU3(out),2)

        out = self.flatten(out)

        out = self.leakyReLUfc(self.fc(out))

        return out

    def _init_weights(self):
        """ This is based on the recommended weight initialization from "Deep Learning with Pytorch" by Eli Stevens et al.

        """
        for network in self.modules():
            for m in network.modules():
                if type(m) in {nn.Linear, nn.Conv1d}:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                        bound = 1 / math.sqrt(fan_out)
                        nn.init.normal_(m.bias, -bound, bound)

        return

    @staticmethod
    def pre_process_GLM_light_curves(bDispObj, GLM_LC_data_length, debug=False):
        """ Pre-processes the GLM light curves in the bolide disposition object

        This is a static method to be performed once in the CnnTrainer.

        Inteprolates each GLM light curves to a set number of cadences, scaling the time duration to fit the
        inrerpolated data length.

        Parameters
        ----------
        bDispObj : bolide_dispositions.BolideDispositions
            The bolide dispositions object generated from the pipeline data set
            This is used to generate the training data sets.
            If passing in the data_sets then this can be None
        GLM_LC_data_length : int
            The data length in cadences for the interpolated GLM light curve.
        debug : bool
            If True, then cycle through and plot each processed light curve

        Returns
        -------
        GLM_LC_dict : Dict
            A Dict of pre-processed light curves where the keyname is the detection ID

        """

        # This is used to fill long gaps with zeros
        # Far away means greater than 10 frames (1 frame = ~526 HZ, or 0.0019 s)
        frame_length = 0.0019
        far_away = 5*frame_length
                
        if debug:
            fig = plt.figure()

        GLM_LC_dict = {}

       #timeDur = []

        for detection in tqdm(bDispObj.bolideDispositionProfileList, 'Pre-processing GLM Light Curves...'):
            detection.bolideDetection.sort_groups()
            energy = detection.bolideDetection.energy
            time = detection.bolideDetection.group_times
            time = np.array([t.timestamp() for t in time])

            # Check that time is a "strictly increasing sequence"
            # Occasionally, we find a duplicate time. Only use the first entry.
            time, unique_idx = np.unique(time, return_index=True)
            energy = energy[unique_idx]

           #timeDur.append(time.max() - time.min())

            # We want to snap to zero all interpolated points that are far away from a real data point
            gap_here = np.nonzero(np.diff(time)>far_away)[0]
            if len(gap_here) > 0:
                last_real_idx = len(time)-1
                # For each gap, fill all points that are far_away with zeros
                for gap_idx in gap_here:
                    left_real_time = time[gap_idx]
                    right_real_time = time[gap_idx+1]
                    fill_times = np.arange(left_real_time+far_away, right_real_time-far_away, frame_length)
                    time = np.append(time, fill_times)
                    energy = np.append(energy, np.zeros(len(fill_times)))
                # Resort time and energy
                sort_order = np.argsort(time)
                time = time[sort_order]
                energy = energy[sort_order]
                if debug:
                    # Keep track of filled datums for debug plotting purposes
                    filled_datums = np.full(len(time), False)
                    filled_datums[np.arange(last_real_idx+1, len(time))] = True
                    filled_datums = filled_datums[sort_order]
                    filled_datums = np.nonzero(filled_datums)[0]

            # We want a uniform time series of GLM_LC_DATA_LENGTH
            x = np.linspace(time.min(), time.max(), GLM_LC_data_length)


            # Use PCHIP interpolation
            GLM_LC_dict[detection.ID] = torch.tensor(interpolate.pchip_interpolate(time, energy, x), dtype=torch.float32)

            if debug:
                fig.clf()
                plt.plot(time, energy, '*b', label='Original Data')
                if len(gap_here) > 0:
                    plt.plot(time[filled_datums], energy[filled_datums], '*r', label='Filled Gaps')
                plt.plot(x, GLM_LC_dict[detection.ID], '-m', label='Interpolated')
                plt.legend()
                plt.show()
               #input('Hit the Any key to view next figure')

            pass


        return GLM_LC_dict

    @staticmethod
    def pre_process_GLM_light_curves_const_time(bDispObj, GLM_LC_data_length, lc_short_time_sec=0.5,
            lc_long_time_sec=8.0, debug=False, verbosity=True):
        """ Pre-processes the GLM light curves in the bolide disposition object

        This is a static method to be performed once in the CnnTrainer.

        Preserves the time duration of the GLM light curve. Producing two scales. A zoomed out scale to cover even the
        longest bolide candidates and a zoomed in one to handle the details of a more typical shorter bolide candidate.
        Also includes a boolean array of gap locations

        Parameters
        ----------
        bDispObj : bolide_dispositions.BolideDispositions
            The bolide dispositions object generated from the pipeline data set
            This is used to generate the training data sets.
            If passing in the data_sets then this can be None
        GLM_LC_data_length : int
            The data length in cadences for the interpolated GLM light curve.
        lc_short_time_sec : float
            Thw length of the zoomed-in short time scale time series
        lc_long_time_sec : float
            Thw length of the zoomed-out short time scale time series
        debug : bool
            If True, then cycle through and plot each processed light curve
    verbosity : bool
        If True then display tqdm status bar

        Returns
        -------
        GLM_LC_dict : Dict of Dicts
            A Dict of pre-processed light curves where the keyname is the detection ID
            Each Dict is a dict containing these keys:
            'short_lc'
            'short_gaps'
            'long_lc'
            'long_gaps'

        """

        # This is used to fill long gaps with zeros
        # Far away means greater than 10 frames (1 frame = ~526 HZ, or 0.0019 s)
        frame_length = 0.0019
        far_away = 5*frame_length
                
        if debug:
            fig = plt.figure()

            # Randomly pick exmaples to plot
            # Do this by rolling the dices and get above a certain threshold
            sample_rate = 0.1

        GLM_LC_dict = {}

       #timeDur = []

        for detection in tqdm(bDispObj.bolideDispositionProfileList, 'Pre-processing GLM Light Curves...', disable=(not verbosity)):
            detection.bolideDetection.sort_groups()
            energy = detection.bolideDetection.energy
            time = detection.bolideDetection.group_times
            time = np.array([t.timestamp() for t in time])

            # Check that time is a "strictly increasing sequence"
            # Occasionally, we find a duplicate time. Only use the first entry.
            time, unique_idx = np.unique(time, return_index=True)
            energy = energy[unique_idx]

           #timeDur.append(time.max() - time.min())

            # We want to snap to zero all interpolated points that are far away from a real data point
            gap_here = np.nonzero(np.diff(time)>far_away)[0]
            if len(gap_here) > 0:
                last_real_idx = len(time)-1
                # For each gap, fill all points that are far_away with zeros
                for gap_idx in gap_here:
                    left_real_time = time[gap_idx]
                    right_real_time = time[gap_idx+1]
                    fill_times = np.arange(left_real_time+far_away, right_real_time-far_away, frame_length)
                    time = np.append(time, fill_times)
                    energy = np.append(energy, np.zeros(len(fill_times)))
                # Resort time and energy
                sort_order = np.argsort(time)
                time = time[sort_order]
                energy = energy[sort_order]
                # Keep track of filled datums
                filled_datums = np.full(len(time), False)
                filled_datums[np.arange(last_real_idx+1, len(time))] = True
                filled_datums = filled_datums[sort_order]
                filled_datums = np.nonzero(filled_datums)[0]
            else:
                filled_datums = np.full(len(time), False)

            # We want a uniform time series of GLM_LC_DATA_LENGTH
            # We do not knwo where in the timer series the actual bolid einformation is, so, have the time series be
            # centered around the peak in the flux
            peak_idx = energy.argmax()
            x_short = np.linspace(time[peak_idx] - (lc_short_time_sec/2.0), time[peak_idx] + (lc_short_time_sec/2.0), GLM_LC_data_length)
            x_long = np.linspace(time[peak_idx] - (lc_long_time_sec/2.0), time[peak_idx] + (lc_long_time_sec/2.0), GLM_LC_data_length)


            GLM_LC_dict[detection.ID] = {}

            # Use PCHIP interpolation
            P_short = interpolate.PchipInterpolator(time, energy, extrapolate=False)(x_short)
            P_short[np.isnan(P_short)] = 0.0
            GLM_LC_dict[detection.ID]['short_lc'] = torch.tensor(P_short, dtype=torch.float32)
            orig_gaps = np.full(len(time), False, dtype=bool)
            orig_gaps[filled_datums] = True
            # Fill in all data outside the orig_gaps range with gaps
            short_gaps = np.interp(x_short, time, orig_gaps, left=1.0, right=1.0)
            # Anything outside the original gap regions in the interpolated gaps should be not gapped (so, do not allow
            # gap creep due to interpolation)
            short_gaps[short_gaps < 1.0] = 0.0
            # PyTorch NNs wants float32, not booleans
            GLM_LC_dict[detection.ID]['short_gaps'] = torch.tensor(short_gaps, dtype=torch.float32)

            P_long = interpolate.PchipInterpolator(time, energy, extrapolate=False)(x_long)
            P_long[np.isnan(P_long)] = 0.0
            GLM_LC_dict[detection.ID]['long_lc'] = torch.tensor(P_long, dtype=torch.float32)
            orig_gaps = np.full(len(time), False, dtype=bool)
            orig_gaps[filled_datums] = True
            # Fill in all data outside the orig_gaps range with gaps
            long_gaps = np.interp(x_long, time, orig_gaps, left=1.0, right=1.0)
            # Anything outside the original gap regions in the interpolated gaps should be not gapped (so, do not allow
            # gap creep due to interpolation)
            long_gaps[long_gaps < 1.0] = 0.0
            GLM_LC_dict[detection.ID]['long_gaps'] = torch.tensor(long_gaps, dtype=torch.float32)


            if debug:
                # Only ploy if greater than lc_short_time_sec 
                if time.max() - time.min() < lc_short_time_sec:
                    continue

                if np.random.uniform() <= sample_rate:

                    fig, ax = plt.subplots(2,1, figsize=(10, 15))
                   
                    ax[0].plot(time, energy, '*b', label='Original Data')
                    if len(gap_here) > 0:
                        ax[0].plot(time[filled_datums], energy[filled_datums], '*r', label='Filled Gaps')
                    ax[0].plot(x_short, GLM_LC_dict[detection.ID]['short_lc'], '-m', label='Interpolated')
                   
                    short_gaps_bool = np.array(GLM_LC_dict[detection.ID]['short_gaps'], dtype=bool)
                    gap_val_to_plot = np.full(len(x_short[short_gaps_bool]), -1.0 * np.max(energy)*0.05)
                    ax[0].plot(x_short[short_gaps_bool], gap_val_to_plot, '.c', label='Gap Indicator')
                   
                    ax[0].set_xlim([x_short.min(), x_short.max()])
                    ax[0].set_title('Short Scale Interpolated Light Curve')
                    ax[0].set_xlabel('time [s]')
                    ax[0].legend()
                   
                    ax[1].plot(time, energy, '*b', label='Original Data')
                    if len(gap_here) > 0:
                        ax[1].plot(time[filled_datums], energy[filled_datums], '*r', label='Filled Gaps')
                    ax[1].plot(x_long, GLM_LC_dict[detection.ID]['long_lc'], '-m', label='Interpolated')
                   
                    long_gaps_bool = np.array(GLM_LC_dict[detection.ID]['long_gaps'], dtype=bool)
                    gap_val_to_plot = np.full(len(x_long[long_gaps_bool]), -1.0 * np.max(energy)*0.05)
                    ax[1].plot(x_long[long_gaps_bool], gap_val_to_plot, '.c', label='Gap Indicator')
                   
                    ax[1].set_title('Long Scale Interpolated Light Curve')
                    ax[1].set_xlabel('time [s]')
                    ax[1].legend()
                   
                    # Set block=False so that plt.show(0 is not a "blocking function" and the close command afterwards
                    # actually closes the figure
                    plt.show(block=False)
                   #input('Hit the Any key to view next figure')
                    pass
                    plt.close(fig)

            pass



        return GLM_LC_dict



class MultiImageModel(nn.Module):
    """ This is a multi-image CNN model with extra features

    Depending on which specific model is being set up, different configuation parameters are needed.
    This will check if only the required configuration parameters are passed. All others should be None.

    """
    
    # This Dict is a way to reference the variable name
    # See glmDataSet.img_types for the corresponding images
   #out_img_names = {1:'out_imgs1', 2:'out_imgs2', 3:'out_imgs3', 4:'out_imgs4'}

    def __init__(self, 
            kernel_size=None,
            n_input_chan=3,
            n_chan_1=None, 
            n_chan_fc1=None, 
            n_chan_fc2=None,
            CNN_momentum=None, 
            leakyReLU_negative_slope=None,
            extra_feature_momentum=None, 
            image_size=None, 
            n_extra_features=None,
            use_resnet=0,
            use_resnet_pretraining=None,
            images_to_use='all',
            image_columns_to_use='all',
            use_multi_band_data=True,
            GLM_LC_data_length=None):

        """

        Parameters
        ----------
        n_input_chan : int or dict
            The number of channels for the input images
            Standard RGB is 3 channels
            For multi-band data, the number of channels is different for each image
        image_size : int or dict
            If not use_multi_band_data, then all images are the same size
            Otherwise, this is a dict to define to square shape for each image type.
        use_resnet : int
            Use this resnet version for the image CNN
            0 means do not use, nont-zero means use this resnet (E.g. 50 means use resnet50)
        use_resnet_pretraining : bool
            If True then use the default pretrained model weights given by the torchvision
            If False then use the untrained model
        images_to_use : int list or 'all'
            List of images to use with the CNNs, or 'all' to use all images
            see MultiImageModel.out_img_names for list of images and order
            Note: this uses 1-based indexing!
        image_columns_to_use : str tuple or 'all'
            The names of the image columns to use. See glmDataSet.img_column_names for options.
        use_multi_band_data : bool
            If True then use multi-band data, otherwise of quasi-RGB cutout images
        GLM_LC_data_length : int
            The data length for the interpolated GLM light curve.
            Set to None if not using GLM light curve 

        """
    
        assert n_chan_fc2 is not None, 'n_chan_fc2 must be passed'
        assert leakyReLU_negative_slope is not None, 'leakyReLU_negative_slope must be passed'
        assert extra_feature_momentum is not None, 'extra_feature_momentum must be passed'
        assert n_extra_features is not None, 'n_extra_features must be passed'

        super().__init__()

        # Which images to use
        assert images_to_use == 'all' or images_to_use.count(0) == 0, 'images_to_use uses 1-based indexing!'
        if images_to_use == 'all':
            self.n_images_to_use = len(glmDataSet.img_types)
            self.images_to_use = np.arange(1, self.n_images_to_use+1).tolist()
        else:
            self.n_images_to_use = len(images_to_use)
            self.images_to_use = images_to_use

        # Which image columns to use
        if image_columns_to_use == 'all':
            self.n_image_columns_to_use = len(glmDataSet.img_column_names)
            self.image_columns_to_use = glmDataSet.img_column_names
        else:
            self.n_image_columns_to_use = len(image_columns_to_use)
            self.image_columns_to_use = image_columns_to_use
        assert np.all([col_name in glmDataSet.img_column_names for col_name in self.image_columns_to_use]), \
                "image_columns_to_use must be from list in glmDataSet.img_column_names"

        self.use_resnet = use_resnet
        self.use_resnet_pretraining = use_resnet_pretraining
        self.use_multi_band_data = use_multi_band_data
        self.GLM_LC_data_length = GLM_LC_data_length 

        self.n_chan_1 = n_chan_1

        #***************
        # The image CNNs
        if use_resnet == 0:
            # Check that required configuratuon parameters are given
            assert kernel_size==3, 'kernel_size must be 3'
            assert n_chan_1 is not None, 'n_chan_1 must be passed'
            assert n_chan_fc1 is not None, 'n_chan_fc1 must be passed'
            assert CNN_momentum is not None, 'CNN_momentum must be passed'
            assert image_size is not None, 'image_size must be passed'
            assert use_resnet_pretraining is None, 'use_resnet_pretraining must not be passed'

            # If passed an integer, convert to a dict
            if isinstance(image_size, int):
                data = [(idx, image_size) for idx in range(1,9)]
                image_size = {key: value for (key, value) in data}
            if isinstance(n_input_chan, int):
                data = [(idx, n_input_chan) for idx in range(1,9)]
                n_input_chan = {key: value for (key, value) in data}

            # Use the custom CNN
            self.CNN1 = CNN_block_batchNorm(kernel_size=kernel_size, n_input_chan=n_input_chan[1], n_chan_1=n_chan_1, n_chan_fc=n_chan_fc1, momentum=CNN_momentum, 
                    leakyReLU_negative_slope=leakyReLU_negative_slope, image_size=image_size[1], use_multi_band_data=self.use_multi_band_data)
            
            self.CNN2 = CNN_block_batchNorm(kernel_size=kernel_size, n_input_chan=n_input_chan[2], n_chan_1=n_chan_1, n_chan_fc=n_chan_fc1, momentum=CNN_momentum, 
                    leakyReLU_negative_slope=leakyReLU_negative_slope, image_size=image_size[2], use_multi_band_data=self.use_multi_band_data)
            
            self.CNN3 = CNN_block_batchNorm(kernel_size=kernel_size, n_input_chan=n_input_chan[3], n_chan_1=n_chan_1, n_chan_fc=n_chan_fc1, momentum=CNN_momentum, 
                    leakyReLU_negative_slope=leakyReLU_negative_slope, image_size=image_size[3], use_multi_band_data=self.use_multi_band_data)
            
            self.CNN4 = CNN_block_batchNorm(kernel_size=kernel_size, n_input_chan=n_input_chan[4], n_chan_1=n_chan_1, n_chan_fc=n_chan_fc1, momentum=CNN_momentum, 
                    leakyReLU_negative_slope=leakyReLU_negative_slope, image_size=image_size[4], use_multi_band_data=self.use_multi_band_data)

            if 'detection_satellite_after' in self.image_columns_to_use:
                self.CNN5 = CNN_block_batchNorm(kernel_size=kernel_size, n_input_chan=n_input_chan[5], n_chan_1=n_chan_1, n_chan_fc=n_chan_fc1, momentum=CNN_momentum, 
                        leakyReLU_negative_slope=leakyReLU_negative_slope, image_size=image_size[5], use_multi_band_data=self.use_multi_band_data)
                
                self.CNN6 = CNN_block_batchNorm(kernel_size=kernel_size, n_input_chan=n_input_chan[6], n_chan_1=n_chan_1, n_chan_fc=n_chan_fc1, momentum=CNN_momentum, 
                        leakyReLU_negative_slope=leakyReLU_negative_slope, image_size=image_size[6], use_multi_band_data=self.use_multi_band_data)
                
                self.CNN7 = CNN_block_batchNorm(kernel_size=kernel_size, n_input_chan=n_input_chan[7], n_chan_1=n_chan_1, n_chan_fc=n_chan_fc1, momentum=CNN_momentum, 
                        leakyReLU_negative_slope=leakyReLU_negative_slope, image_size=image_size[7], use_multi_band_data=self.use_multi_band_data)
                
                self.CNN8 = CNN_block_batchNorm(kernel_size=kernel_size, n_input_chan=n_input_chan[8], n_chan_1=n_chan_1, n_chan_fc=n_chan_fc1, momentum=CNN_momentum, 
                        leakyReLU_negative_slope=leakyReLU_negative_slope, image_size=image_size[8], use_multi_band_data=self.use_multi_band_data)
        elif use_resnet > 0:
            # Check that required configuration parameters are given
            assert len(self.image_columns_to_use) == 1 and self.image_columns_to_use[0] == 'detection_satellite_before', \
                    "Resnet models only works with the 'detection_satellite_before' image column"
            assert kernel_size is None, 'kernel_size must not be passed'
            assert n_chan_1 is None, 'n_chan_1 must not be passed'
            assert n_chan_fc1 is None, 'n_chan_fc1 must not be passed'
            assert CNN_momentum is None, 'CNN_momentum must notbe passed'
            assert image_size is None, 'image_size must not be passed'
            assert use_resnet_pretraining is not None, 'use_resnet_pretraining must be passed'

            assert use_multi_band_data == False, 'use_multi_band_data cannot be used with resnet'

            # Overwrite the FC1 n_chan input to accomodate the resnet
            n_chan_fc1 = 1000

            if use_resnet == 18:
                if self.use_resnet_pretraining:
                    # Use best available pre-trained weights
                    weights = ResNet18_Weights.DEFAULT
                    self.resnet_preprocess = weights.transforms()
                else:
                    weights = None
                    self.resnet_preprocess = None
                self.resnet1 = resnet18(weights=weights)
                self.resnet2 = resnet18(weights=weights)
                self.resnet3 = resnet18(weights=weights)
                self.resnet4 = resnet18(weights=weights)
                
            elif use_resnet == 50:
                if self.use_resnet_pretraining:
                    # Use best available pre-trained weights
                    weights = ResNet50_Weights.DEFAULT
                    self.resnet_preprocess = weights.transforms()
                else:
                    weights = None
                    self.resnet_preprocess = None
                self.resnet1 = resnet50(weights=weights)
                self.resnet2 = resnet50(weights=weights)
                self.resnet3 = resnet50(weights=weights)
                self.resnet4 = resnet50(weights=weights)
                
            elif use_resnet == 152:
                if self.use_resnet_pretraining:
                    # Use best available pre-trained weights
                    weights = ResNet152_Weights.DEFAULT
                    self.resnet_preprocess = weights.transforms()
                else:
                    weights = None
                    self.resnet_preprocess = None
                self.resnet1 = resnet152(weights=weights)
                self.resnet2 = resnet152(weights=weights)
                self.resnet3 = resnet152(weights=weights)
                self.resnet4 = resnet152(weights=weights)
                
            else:
                raise Exception('Unsupported ResNet model')
        else:
            raise Exception('Unknown ')

        #********************
        # GLM Light curve CNN
        if self.GLM_LC_data_length is not None:
            # There are 4 time series for the light curves (short_lc, short_gaps, long_lc, long_gaps)
            lc_cnn = CNN_1D_block_batchNorm(
                data_length=GLM_LC_data_length , 
                kernel_size=kernel_size, 
                n_chan_1=n_chan_1, 
                n_chan_fc=n_chan_fc1, 
                momentum=CNN_momentum, 
                leakyReLU_negative_slope=leakyReLU_negative_slope)

            self.short_lc_cnn = copy.deepcopy(lc_cnn)
            self.short_lc_gaps_cnn = copy.deepcopy(lc_cnn)
            self.long_lc_cnn = copy.deepcopy(lc_cnn)
            self.long_lc_gaps_cnn = copy.deepcopy(lc_cnn)
        else:
            self.lc_cnn = None


        # Add in the extra features to the output of the CNNs
        if self.GLM_LC_data_length is not None:
            self.batchNorm = nn.BatchNorm1d(n_chan_fc1*self.n_images_to_use*self.n_image_columns_to_use+n_chan_fc1*4+n_extra_features, momentum=extra_feature_momentum)
            self.fc1 = nn.Linear(n_chan_fc1*self.n_images_to_use*self.n_image_columns_to_use+n_chan_fc1*4+n_extra_features,n_chan_fc2)
        else:
            self.batchNorm = nn.BatchNorm1d(n_chan_fc1*self.n_images_to_use*self.n_image_columns_to_use+n_extra_features, momentum=extra_feature_momentum)
            self.fc1 = nn.Linear(n_chan_fc1*self.n_images_to_use*self.n_image_columns_to_use+n_extra_features,n_chan_fc2)

        self.leakyReLUfc = nn.LeakyReLU(negative_slope=leakyReLU_negative_slope)

        self.fc2 = nn.Linear(n_chan_fc2,2)

        # This softmax is to return confidence scores [0,1] unstead of raw logits
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def forward(self, images,  lc_data, extra_features):
        """ The main forward pass for the nueral network model

        Parameters:
        -----------
        images : Dict of torch.tensor
            The list of images where the key is the image index, starting with 1

        """

        out_imgs_dict = {}
        if self.use_resnet == 0:
            if 'detection_satellite_before' in self.image_columns_to_use:
                if self.images_to_use.count(1) == 1:
                    # If saving out the ONNX model, then the torch.onnx.export function appears to modify the images
                    # dict key names to tensors. The code below tests for this condition and set the key names back to
                    # an integer.
                    try:
                        out_imgs_dict[1] = self.CNN1(images[1])
                    except:
                        images_tmp = copy.copy(images)
                        images[1] = images_tmp[list(images_tmp.keys())[0]]
                        images[2] = images_tmp[list(images_tmp.keys())[1]]
                        images[3] = images_tmp[list(images_tmp.keys())[2]]
                        images[4] = images_tmp[list(images_tmp.keys())[3]]
                        out_imgs_dict[1] = self.CNN1(images[1])
                if self.images_to_use.count(2) == 1:
                    out_imgs_dict[2] = self.CNN2(images[2])
                if self.images_to_use.count(3) == 1:
                    out_imgs_dict[3] = self.CNN3(images[3])
                if self.images_to_use.count(4) == 1:
                    out_imgs_dict[4] = self.CNN4(images[4])
            if 'detection_satellite_after' in self.image_columns_to_use:
                if self.images_to_use.count(1) == 1:
                    out_imgs_dict[5] = self.CNN1(images[5])
                if self.images_to_use.count(2) == 1:
                    out_imgs_dict[6] = self.CNN2(images[6])
                if self.images_to_use.count(3) == 1:
                    out_imgs_dict[7] = self.CNN3(images[7])
                if self.images_to_use.count(4) == 1:
                    out_imgs_dict[8] = self.CNN4(images[8])
        elif self.use_resnet > 0:
            # pre-trained Resnet models expect specifically pre-processed images
            if self.use_resnet_pretraining:
                imgs1 = self.resnet_preprocess(imgs1)
                imgs2 = self.resnet_preprocess(imgs2)
                imgs3 = self.resnet_preprocess(imgs3)
                imgs4 = self.resnet_preprocess(imgs4)

            if self.images_to_use.count(1) == 1:
                out_imgs_dict[1] = self.resnet1(imgs1)
            if self.images_to_use.count(2) == 1:
                out_imgs_dict[2] = self.resnet2(imgs2)
            if self.images_to_use.count(3) == 1:
                out_imgs_dict[3] = self.resnet3(imgs3)
            if self.images_to_use.count(4) == 1:
                out_imgs_dict[4] = self.resnet4(imgs4)

        # GLM light curve
        if self.GLM_LC_data_length is not None:
            out_short_lc      = self.short_lc_cnn(lc_data['short_lc'])
            out_short_lc_gaps = self.short_lc_gaps_cnn(lc_data['short_gaps'])
            out_long_lc       = self.long_lc_cnn(lc_data['long_lc'])
            out_long_lc_gaps  = self.long_lc_gaps_cnn(lc_data['long_gaps'])
        else:
            out_lc = None

        # Create tuple of desired images
        out_imgs_list = [out_imgs_dict[key] for key in out_imgs_dict.keys()]
        if self.GLM_LC_data_length is not None:
            out_imgs_list.append(out_short_lc)
            out_imgs_list.append(out_short_lc_gaps)
            out_imgs_list.append(out_long_lc)
            out_imgs_list.append(out_long_lc_gaps)
        out_imgs_list.append(extra_features)
        out_imgs_tuple = tuple(out_imgs_list)

        # Add in extra features
       #out = torch.cat((out_imgs1, out_imgs2, out_imgs3, out_imgs4, extra_features, out_lc), dim=1)
        out = torch.cat(out_imgs_tuple, dim=1)

        # Use batch normalization to scale the extra_features to the image data
        out = self.batchNorm(out)

        out = self.leakyReLUfc(self.fc1(out))

        out = self.fc2(out)

        return out, self.head_softmax(out)

    def _init_weights(self):
        """ This is based on the recommended weight initialization from "Deep Learning with Pytorch" by Eli Stevens et al.

        """
        for network in self.modules():
            for m in network.modules():
                if type(m) in {nn.Linear, nn.Conv2d}:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                        bound = 1 / math.sqrt(fan_out)
                        nn.init.normal_(m.bias, -bound, bound)

        return

        
#*************************************************************************************************************

class glmDataSet(Dataset):
    # These are the image types to cache
    img_types = ('60_sec_integ', '60_sec_integ_zoom', 'GLM_projection',  'GLM_projection_zoomed_out')

    # These are the image column names
    img_column_names = ('detection_satellite_before', 'detection_satellite_after', 'other_satellite_before', 'other_satellite_after')

    def __init__(self, 
            bolide_dispositions_object=None, 
            data_IDs=None, 
            transform=None, 
            augmentations = ('rotate', 'flip'),
            balance_data=False,
            beliefSource='human',
            bolidesFromWebsite=None, 
            featuresToUse=None, 
            columnTransformer=None, 
            verbosity=True,
            cached_filenames=None,
            cached_lightcurves=None,
            use_multi_band_data=True,
            bands_to_use='all',
            from_cache_init=False):
        """ Custom torch.utils.data.Dataset to load in the cutout images and also the random forest features.

        The images are parsed and cached in memory as torch tensors so that we do not need to parse them every epoch.

        Parameters
        ----------
        bolide_dispositions_object : bolide_dispositions.BolideDispositions
            The bolide dispositions object generated from the pipeline data set
        data_IDs : list of int64
            IDs in bolide_dispositions_object.bolideDispositionProfileList of the datums to load
        transform : torchvision.transforms.transforms
            The tranformer to use when loading the cutout images
        augmentations : list of str
            The list of augmentations to perform on the images, which multiplies the data set
            None means no augmentation
        balance_data : bool
            If true then balance the data when using the DataLoader. This will set up the self.sampler method to return a 
            weighted sampling of datums using torch.utils.data.WeightedRandomSampler.
            If False then sampler is just torch.utils.data.RandomSampler
        beliefSource : str
            Where to obtain the truth {'machine', 'human'}
        bolidesFromWebsite : [WebsiteBolideEvent list] created by:
            bolide_dispositions.pull_dispositions_from_website
            If passed and beliefSource == 'human' then use this data to determine truth.
        featuresToUse : list of str
            Features to use in data set
            Note: if columnTransformer is passed then the order of these features must match the order expected in
            columnTransformer.
        columnTransformer   : sklearn ColumnTransformer object 
            The already fit transformer to normalize the extra features
            if None then create a new transformer and fit to the data
        verbosity : bool
            If True then display progress and info, otherwise, silent
        cached_filenames : Dict of Dicts
            List of cached filenames where the keys are img_types
            Then the second level keys are individual IDs and their values are the filenames for each ID
        cached_lightcurves : Dict of arrays
            Dict of pre-processed light curved using CNN_1D_block_batchNorm.pre_process_GLM_light_curves
        use_multi_band_data : bool
            If True then use multi-band data, instead of quasi-RGB cutout images
        bands_to_use : int tuple
            List of indices to bands to use in the ABI CNN image model.
            Only used if use_multi_band_data=True
            Note: This is in the order stored in the *_multi_bands.pbz2 files. The bands are dictated by the
            cutout_bands_to_read configuration tuple.
            For all bands use 'all'
        from_cache_init : bool
            If True then the object was initialized with the from_cache classmethod.
            In which case, there is nothing to do here
            
        """

        if from_cache_init:
            # Do nothing, everything is already set up
            return

        self.data_IDs = data_IDs
        self.n_datums = len(data_IDs)
        self.augmentations = augmentations
        self.transform = transform
        self.verbosity = verbosity
        self.use_multi_band_data = use_multi_band_data

        self.data_indices = bolide_dispositions_object.return_indices(self.data_IDs)

        # Store the machine opinions from the detector classifier
        self.machine_opinions = np.array([bolide_dispositions_object.bolideDispositionProfileList[idx].machineOpinions[0].bolideBelief for idx in self.data_indices])

        # Determine bolide truth
        bolideBeliefArray = bolide_dispositions_object.generate_bolide_belief_array(beliefSource, bolidesFromWebsite)
        truthThreshold = 0.5
        self.bolide_truth = bolideBeliefArray >= truthThreshold
        self.bolide_truth = self.bolide_truth[self.data_indices]

        # Check for valid features
        if featuresToUse is not None:
            nProblems = bFeatures.FeaturesClass.check_for_valid_features(featuresToUse, allow_special_features=True)
            if nProblems > 0:
                raise Exception ('Unknown feature labels in <featuresToUse>')
            bolide_dispositions_object.featuresToUse = featuresToUse
            self.featuresToUse = featuresToUse
        else:
            self.featuresToUse = None

        #***
        # We need to create the list of image files and features for each datum in the data set.

        # Images
        if 'detection_satellite_before' in cached_filenames:
            self.cached_filenames_1 = [os.path.join(RAMDISK_PATH, cached_filenames['detection_satellite_before'][glmDataSet.img_types[0]][ID]) for ID in self.data_IDs]
            self.cached_filenames_2 = [os.path.join(RAMDISK_PATH, cached_filenames['detection_satellite_before'][glmDataSet.img_types[1]][ID]) for ID in self.data_IDs]
            self.cached_filenames_3 = [os.path.join(RAMDISK_PATH, cached_filenames['detection_satellite_before'][glmDataSet.img_types[2]][ID]) for ID in self.data_IDs]
            self.cached_filenames_4 = [os.path.join(RAMDISK_PATH, cached_filenames['detection_satellite_before'][glmDataSet.img_types[3]][ID]) for ID in self.data_IDs]
        else:
            self.cached_filenames_1 = None
            self.cached_filenames_2 = None
            self.cached_filenames_3 = None
            self.cached_filenames_4 = None

        if 'detection_satellite_after' in cached_filenames:
            self.cached_filenames_5 = [os.path.join(RAMDISK_PATH, cached_filenames['detection_satellite_after'][glmDataSet.img_types[0]][ID]) for ID in self.data_IDs]
            self.cached_filenames_6 = [os.path.join(RAMDISK_PATH, cached_filenames['detection_satellite_after'][glmDataSet.img_types[1]][ID]) for ID in self.data_IDs]
            self.cached_filenames_7 = [os.path.join(RAMDISK_PATH, cached_filenames['detection_satellite_after'][glmDataSet.img_types[2]][ID]) for ID in self.data_IDs]
            self.cached_filenames_8 = [os.path.join(RAMDISK_PATH, cached_filenames['detection_satellite_after'][glmDataSet.img_types[3]][ID]) for ID in self.data_IDs]
        else:
            self.cached_filenames_5 = None
            self.cached_filenames_6 = None
            self.cached_filenames_7 = None
            self.cached_filenames_8 = None

        # The above assumes we are using cached images.
        # Check that the files are actually at the cache location
        assert os.path.isfile(self.cached_filenames_1[0]), 'This code only works when using cached images'

        # GLM Light Curves
        if cached_lightcurves is not None:
            self.light_curves = [cached_lightcurves[ID] for ID in self.data_IDs] 
        else:
            self.light_curves = None

        # Set up bands to use
        if bands_to_use == 'all':
            # We want all bands, so we need to determine which bands those are to set the tuple
            data = torch.load(self.cached_filenames_1[0])
            nBands = data[0].shape[0]
            self.bands_to_use = tuple(np.arange(nBands))
        else:
            self.bands_to_use = bands_to_use


        # Features list
        if featuresToUse is not None:
            feature_matrix = np.ndarray(shape=(self.n_datums,0), dtype=np.float32)
            for feature in self.featuresToUse:
                if feature == 'machine_opinion':
                    featureArray = self.machine_opinions
                    featureArray = featureArray[:, np.newaxis].astype(np.float32)
                else:
                    featureArray = np.array([profile.features.__getattribute__(feature) for profile in
                        bolide_dispositions_object.bolideDispositionProfileList], dtype=np.float32).reshape(bolide_dispositions_object.n_tot_entries,1)
                    featureArray = featureArray[self.data_indices,:]
                feature_matrix = np.append(feature_matrix, featureArray, axis=1)
            
            # If columnTransformer is None then the transformer is fitted to the feature_matrix
            feature_matrix, self.columnTransformer = bFeatures.scale_features(feature_matrix, columnTransformer, self.featuresToUse)
            self.feature_matrix = torch.tensor(feature_matrix)
        else:
            feature_matrix = None
            self.columnTransformer = None
            self.feature_matrix = torch.tensor([])

        # This augmentation technique will store all augmented images. It requires a lot of memory.
       #self.augment_data()

        # Set up weight for balancing the data set
        if balance_data:
            n_bolide = np.count_nonzero(self.bolide_truth)
            n_not_bolide = len(self.bolide_truth) - n_bolide

            # Weight all datums equally and have weights add up to one
            weight_bolide = (1/n_bolide) / (1/n_bolide + 1/n_not_bolide)
            weight_not_bolide = 1 / n_not_bolide / (1/n_bolide + 1/n_not_bolide)

            weights = np.full_like(self.bolide_truth, weight_not_bolide, dtype=float)
            weights[self.bolide_truth] = weight_bolide

            self.sampler = WeightedRandomSampler(weights=weights, num_samples=len(self.bolide_truth), replacement=True)
        else:
            # Just randomly sample all datums with equal probability
            self.sampler = RandomSampler(np.arange(len(self.bolide_truth)), replacement=False)

   #@classmethod
    def load_from_cache(self, init_data):
        """ Constructs a glmDataSet from a stored and cached data set.

        From a previous glmDataStruct use the generate_init_data_dict to create a dict with object attributes in it.
            
            data_set_dict = old_data_set.generate_init_data_dict()
            data_set = glmDataSet(from_cache_init=True)
            data_set.from_cache(data_set_dict)

            data_set is now identical to data_set_dict

        Parameters
        ----------
        init_data : Dict
            This contains all the information to initiate the glmDataSet object
            'data_indices'
            'augmentations'
            'transform'
            'verbosity'
            'machine_opinions'
            'bolide_truth'
            'featuresToUse'
            'cached_filenames_1'
            'cached_filenames_2'
            'cached_filenames_3'
            'cached_filenames_4'
            'feature_matrix'
            'columnTransformer'

        Returns
        -------
        glmDataSet object
            That is set up identical to the one used to generate the init_data

        """

        assert isinstance(init_data, dict), 'Must pass dict to glmDataSet.load_from_cache'

        for key in init_data.keys():
            setattr(self, key, init_data[key])

        return

    def generate_init_data_dict(self):
        """ This will create a dict that contains all the information needed to initialize the glmDataSet object

        Returns
        -------
        init_data : Dict
            This contains all the information to initiate the glmDataSet object

        """

        return self.__dict__

    def augment_data(self):
        """ Augments the images according to self.augmentations.

        This will make a copy of each instance and perform the listed augmentations. The expanded data set is then stored.
        This is NOT on-the-fly augmentation. 

        It will not augment the extra features. It just copies those over to the new instance.

        """

        raise Exception('This method is no longer maintained')

        if self.augmentations is None:
            return

        # Perform image augmentations
        n_augmentations = 0 # Gives the total number of duplicates for multiplying the feature matrix and truth array
        for augmentation in self.augmentations:
            if augmentation == 'rotate':
                self.cached_filenames_1 = self._rotate_images_to_file(self.cached_filenames_1, '60_sec_integ')
                self.cached_filenames_2 = self._rotate_images_to_file(self.cached_filenames_2, '60_sec_integ_zoom')
                self.cached_filenames_3 = self._rotate_images_to_file(self.cached_filenames_3, 'GLM_projection')
                self.cached_filenames_4 = self._rotate_images_to_file(self.cached_filenames_4, 'GLM_projection_zoomed_out')
                n_augmentations += 4

            if augmentation == 'flip':
                raise Exception('Flip not implemented yet')

        self.n_datums = int(self.n_datums*n_augmentations)

        # Multiply extra features
        if self.feature_matrix is not None:
            feature_matrix_save = copy.deepcopy(self.feature_matrix)
            self.feature_matrix = np.ndarray(shape=(0,feature_matrix_save.shape[1]), dtype=np.float32)
            for featureArray in feature_matrix_save:
                # Multiply each feature array by the number of augmentations
                # Need to get the dimensions correct to append to matrix
                featureArray = featureArray[np.newaxis,:]
                for idx in np.arange(n_augmentations):
                    self.feature_matrix = np.append(self.feature_matrix, featureArray, axis=0)

        # Multiply bolide truth array
        bolide_truth_save = copy.deepcopy(self.bolide_truth)
        self.bolide_truth = []
        for truth_val in bolide_truth_save:
            # Multiply each truth state by the number of augmentations
            for idx in np.arange(n_augmentations):
                self.bolide_truth.append(truth_val)
        self.bolide_truth = np.array(self.bolide_truth)

        # Multiply the machine opinion array
        machine_opinions_save = copy.deepcopy(self.machine_opinions)
        self.machine_opinions = []
        for opinion in machine_opinions_save:
            # Multiply each truth state by the number of augmentations
            for idx in np.arange(n_augmentations):
                self.machine_opinions.append(opinion)
        self.machine_opinions = np.array(self.machine_opinions)

        return

    def _rotate_images(self, images, img_str):
        """ Makes rotated copies of images and returns the larger list of the original images and the rotated images

        The returned images are in a list in this order:
        rotated_images = [original_image_1, rotated_90_1, rotated_180_1, rotated_270_1, original_image_2, rotated_90_2, ...]
                
        """

        raise Exception('This method is no longer maintained')

        # Make a copy of the original images
        images_temp = copy.deepcopy(images)

        rotated_images = []
        for idx in tqdm(range(len(images_temp)),'Rotating image files for augmentations of type {}...'.format(img_str), disable=(not self.verbosity)):
            image = images_temp[idx]
            # First store the original in the new list
            rotated_images.append(image)
            # now store the rotated images
            rotated_images.append(TF.rotate(image,90))
            rotated_images.append(TF.rotate(image,180))
            rotated_images.append(TF.rotate(image,270))

        return rotated_images
            
    def _rotate_images_to_file(self, image_filenames, img_str):
        """ Makes rotated copies of images and saves the rotated image tensors to the same directory as the original, 
        also adds them to the cached_filenames_* list

        The returned images are in a list in this order:
        rotated_images_filenames = [original_image_1, rotated_90_1, rotated_180_1, rotated_270_1, original_image_2, rotated_90_2, ...]
                
        """

        raise Exception('This method is no longer maintained')

        # Make a copy of the original images
        image_filenames_temp = copy.deepcopy(image_filenames)

        rotated_image_filenames = []
        for idx in tqdm(range(len(image_filenames_temp)),'Rotating image files for augmentations of type {}...'.format(img_str), disable=(not self.verbosity)):
            image = torch.load(image_filenames_temp[idx])
            # First store the original filename in the new list
            rotated_image_filenames.append(image_filenames_temp[idx])
            # now store the rotated images
            for rot_angle in [90, 180, 270]:
                rotated_image = TF.rotate(image,rot_angle)
                save_filename = os.path.splitext(image_filenames_temp[idx])[0] + '_{}'.format(rot_angle) + '.pt'
                rotated_image_filenames.append(save_filename)
                torch.save(image, save_filename)

        return rotated_image_filenames
            

    def augment_images(self, images):
        """ This will perform a random augmentation to the given images.

        This is designed to be used on-the-fly inside the training_loop. Torchvision is so much faster on GPUs so be sure the images are on a cuda device.

        Since we have multiple images, we augment them all within this function in order to ensure the compatible augmentation happen to all four. 
        For example, if rotating images, rotate all four images the same amount.

        It is recommended to make sure the images are already on a cuda device for speed.

        Parameters:
        -----------
        images : Dict of torch.tensor
            The list of images where the key is the image index, starting with 1

        Returns:
        -----------
        images : Dict of torch.tensor
            The list of images aith augmentations where the key is the image index, starting with 1

        """

        if self.augmentations is None:
            return images

        rotate_options = (0, 90, 180, 270)
        flip_options = ('N', 'H', 'V')

        # The batch size is the first dimensions in the image tensor
        batch_size = images[1].shape[0]

        for augmentation in self.augmentations:
            if augmentation == 'rotate':
                rand_select = np.random.randint(0, len(rotate_options), size=batch_size)
                rotate_angle = [rotate_options[rand_select[idx]] for idx in np.arange(batch_size)]
                
                # All images in an instance are rotated by the same angle
                for image_collection in images.values():
                    for idx, image in enumerate(image_collection):
                        image_collection[idx] = TF.rotate(image, rotate_angle[idx])

            elif augmentation == 'flip':
                rand_select = np.random.randint(0, len(flip_options), size=batch_size)
                flip_dir = [flip_options[rand_select[idx]] for idx in np.arange(batch_size)]
                
                # All images in an instance are flipped the same way
                for image_collection in images.values():
                    for idx, image in enumerate(image_collection):
                        if flip_dir[idx] == 'N':
                            # 'N' means no flip
                            pass
                        elif flip_dir[idx] == 'H':
                            image_collection[idx] = TF.hflip(image)
                        elif flip_dir[idx] == 'V':
                            image_collection[idx] = TF.vflip(image)

            elif augmentation == 'gaussian_blur':
                raise Exception('Gaussian Blur augmentation not yet implemented')
                #Need to decide how to handle the kernel_size if doing Gaussian Blur

        return images

    

    def __len__(self):
        return self.n_datums

    def __getitem__(self, idx):

        # load the image data
        image = {}

        if self.use_multi_band_data:

            # This is ABI and GLM data
            data = torch.load(self.cached_filenames_1[idx])
            # Form the full dense matrix
            # Only keep desired bands
            data[0] = data[0][self.bands_to_use, ...]
            # Convert the sparse scipy array of GLM data to a dense tensor
            data[1] = torch.tensor(np.stack([data[1][idx].todense() for idx in range(len(data[1]))], axis=0))
            image[1] = torch.concatenate(data, dim=0)

            # This is ABI and GLM data
            data = torch.load(self.cached_filenames_2[idx])
            # Form the full dense matrix
            # Only keep desired bands
            data[0] = data[0][self.bands_to_use, ...]
            # Convert the sparse scipy array to a dense tensor
            data[1] = torch.tensor(np.stack([data[1][idx].todense() for idx in range(len(data[1]))], axis=0))
            image[2] = torch.concatenate(data, dim=0)

            # This is just GLM data
            data = torch.load(self.cached_filenames_3[idx])
            image[3] = torch.tensor(np.stack([data[idx].todense() for idx in range(len(data))], axis=0))

            # This is just GLM data
            data = torch.load(self.cached_filenames_4[idx])
            image[4] = torch.tensor(np.stack([data[idx].todense() for idx in range(len(data))], axis=0))

            #****
            # These are the after images
            if self.cached_filenames_5 is not None:
                # This is ABI and GLM data
                data = torch.load(self.cached_filenames_5[idx])
                # Form the full dense matrix
                # Only keep desired bands
                data[0] = data[0][self.bands_to_use, ...]
                # Convert the sparse scipy array of GLM data to a dense tensor
                data[1] = torch.tensor(np.stack([data[1][idx].todense() for idx in range(len(data[1]))], axis=0))
                image[5] = torch.concatenate(data, dim=0)
                
                # This is ABI and GLM data
                data = torch.load(self.cached_filenames_6[idx])
                # Form the full dense matrix
                # Only keep desired bands
                data[0] = data[0][self.bands_to_use, ...]
                # Convert the sparse scipy array to a dense tensor
                data[1] = torch.tensor(np.stack([data[1][idx].todense() for idx in range(len(data[1]))], axis=0))
                image[6] = torch.concatenate(data, dim=0)
                
                # This is just GLM data
                data = torch.load(self.cached_filenames_7[idx])
                image[7] = torch.tensor(np.stack([data[idx].todense() for idx in range(len(data))], axis=0))
                
                # This is just GLM data
                data = torch.load(self.cached_filenames_8[idx])
                image[8] = torch.tensor(np.stack([data[idx].todense() for idx in range(len(data))], axis=0))

        else:
            # The images have already been converted to tensors and individually stored
            image[1] = torch.load(self.cached_filenames_1[idx])
            image[2] = torch.load(self.cached_filenames_2[idx])
            image[3] = torch.load(self.cached_filenames_3[idx])
            image[4] = torch.load(self.cached_filenames_4[idx])

        # GLM Light Curves
        if self.light_curves is not None:
            light_curve = self.light_curves[idx]
        else:
            light_curve = {}
        
        
        # Load the features
        if self.feature_matrix.__len__() > 0 :
            features = self.feature_matrix[idx,:]
        else:
            features = torch.tensor([])

        # PyTorch models expect 2 classes for a binary classifier
        if self.bolide_truth[idx]:
            bolide_truth = [0,1]
        else:
            bolide_truth = [1,0]
        bolide_truth = torch.tensor(bolide_truth, dtype=torch.float32)

        return image, light_curve, features, bolide_truth

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


#*************************************************************************************************************
#*************************************************************************************************************
#*************************************************************************************************************
# This is the main class to perform hyperparameter optimization using Ray Tune

class HyperTune():

    def __init__(self,
            image_cache_path='/tmp/ramdisk/cnn_image_cache',
            image_columns_to_use='all',
            use_multi_band_data=False,
            GLM_LC_data_length=None,
            bolidesFromWebsitePath=None, 
            random_seed=42,
            avg_precision_recall_limit=0.0,
            target_precision = 0.98,
            fbeta_score_threshold=0.5, 
            fbeta_score_beta=1.0,
            num_cpu_threads=None, 
            cuda_devices=None,
            random_forest_uncertainties_file=None
            ):

        """ Initiliazes the main onbjects and sets some parameters for the tuning runs.

        Parameters
        ----------
            bolide_dispositions_object : bolide_dispositions.BolideDispositions
                The bolide dispositions object generated from the pipeline data set
                This is used to generate the training data sets.
                If passing in the data_sets then this can be None
            cached_filenames : Dict of Dicts
                If passed then use this set of cached images instead of generating them
                List of cached filenames where the keys are glmDataSet.img_types
                Then the second level keys are individual IDs and there values are the filenames for each ID
        image_cache_path : str
            Path to where the preprocessed images converted to tensors are saved
            Also where the cached_filenames and bolide_dispositions_object pickle files were saved
        image_columns_to_use : str tuple or 'all'
            The names of the image columns to use. See glmDataSet.img_column_names for options.
        use_multi_band_data : bool
            If True then use multi-band data, otherwise of quasi-RGB cutout images
        GLM_LC_data_length : int
            The data length for the interpolated GLM light curve.
            Set to None if not using GLM light curve 
        bolidesFromWebsitePath : str
            Path to the bolidesFromWebsite pickle file
        random_seed : int
            Random seed for train/val/test split
            If None then set randomly
            This can be reset in the hyper_tune method if you want multiple tuing runs to be identical
        avg_precision_recall_limit : float [0.0, 1.0)
            When computing the average precision, this is the lower recall limit to use in the integration.
            Average precision is normalized by the recall evaluation range.
        target_precision : float [0.0, 1.0]
            Target precision to compute recall at for recall_at_precision_score
        fbeta_score_threshold : float
            The classifier threshold score to use when computing fbeta_score
        fbeta_score_beta : float
            The fbeta score beta value to use when weighting precision vs recall.
        num_cpu_threads : int
            Number of CPU threads to use per sample trial job, 1 GPU is always used.
            None means use all available CPU cores (including virtual cores)
        cuda_devices : string
            Sets which CUDA GPUs are visable when runing samples in parallel
            This must be a string, '0' means device 0, '0,1' means devices 0 and 1
            None means use all devices
        random_forest_uncertainties_file : str
            Path to pickle file containing the random forest uncertainty estimate data in a PrecisionRecallResults object

        """
        self.image_columns_to_use = image_columns_to_use

        # Force random seed to specific state
        if random_seed is None:
            np.random.seed()
            torch.manual_seed(np.random.randint(1e10))
        else:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # If number of threads not specified then set to total number available
        if num_cpu_threads is None:
            self.num_cpu_threads = len(os.sched_getaffinity(0))
        else:
            self.num_cpu_threads = num_cpu_threads
        
        # Specify which devices to use
        if cuda_devices is not None:
            assert isinstance(cuda_devices, str), 'cuda_devices must be a str'
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        self.random_forest_uncertainties_file = random_forest_uncertainties_file
 
        self.use_multi_band_data = use_multi_band_data
        self.GLM_LC_data_length = GLM_LC_data_length

        self.avg_precision_recall_limit = avg_precision_recall_limit
        self.target_precision = target_precision 
        self.fbeta_score_threshold = fbeta_score_threshold
        self.fbeta_score_beta = fbeta_score_beta
 
        # On the NAS torch.cuda.device_count() does not work, have to use nvidia-smi
        if os.getenv('isOnNas') == 'True':
            #TODO: This is so awkward!
            processOut = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            num_GPUs_avail = processOut.stdout.count('\n')
        else:
            num_GPUs_avail = torch.cuda.device_count()
        print('CUDA devices available: {}'.format(num_GPUs_avail))
 
        # This sets the total number of CPU cores available to ray
        # On bolide.nas.nasa.gov the error "Failed to register worker..." occurs if this is not set.
        ray.init(num_cpus=int(num_cpu_threads*num_GPUs_avail+1))
 
        # Generate the bolide dispositions object
       #bDispObj = bDisp.create_BolideDispositions_from_multiple_database_files(database_files, verbosity=True, useRamDisk=False)
        cached_filenames, bolide_dispositions_object = load_cached_objects(save_path=image_cache_path, image_columns_to_use=self.image_columns_to_use)
        
        self.bolidesFromWebsite = bd.unpickle_bolide_detections(bolidesFromWebsitePath)
 
        # Create a mother CnnTrainer object to setup the training data sets
        self.cnnObj = CnnTrainer(bolide_dispositions_object, 
                cached_filenames=cached_filenames,
                use_multi_band_data=use_multi_band_data,
                GLM_LC_data_length=GLM_LC_data_length,
                image_size=None, 
                gpu_index=None, 
                avg_precision_recall_limit=avg_precision_recall_limit,
                target_precision=target_precision,
                fbeta_score_threshold=fbeta_score_threshold,
                fbeta_score_beta=fbeta_score_beta,
                run_in_tune=True, 
                verbosity=True,
                random_forest_uncertainties_file = random_forest_uncertainties_file)

        # Pickle the bolideDispositionProfileList for use when computing uncertainties
        self.bolide_dispositions_object_file = os.path.join(image_cache_path, 'bolideDispositionProfileList.p')
       #bDispObj.save_bolideDispositionProfileList(self.bolide_dispositions_object_file)
 

    def hyper_tune(self, 
            config, 
            model_name,
            images_to_use='all',
            extra_feature_names=None, 
            bands_to_use='all',
            starting_model_and_optim_checkpoint=None,
            ratio=(0.8, 0.1, 0.1),
            num_samples=10, 
            max_num_epochs=10, 
            max_num_uncertainty_epochs=10, 
            batch_size=128, 
            augmentations=['rotate'],
            random_seed=None,
            ASHA_brackets=1,
            ASHA_reduction_factor=2,
            optim_metric='loss',
            PR_fig_filename=None, 
            output_save_path=None,
            n_uncert_samples=0,
            test_trainer=False
            ):
        """ Hyperparameter tuning using Ray Tune.
 
        Parameters
        ----------
        config : Dict
            The Ray Tune config parameters, gives which hyperparmaters to tune and their value ranges.
        model_name : str
            Name of model to use
            See CnnTrainer.set_model for options
        images_to_use : int list or 'all'
            List of images to use with the CNNs, or 'all' to use all images
            see MultiImageModel.out_img_names for list of images and order
            Note: this uses 1-based indexing!
        extra_feature_names : list of str
            The names of the extra features to add in.
            They must be from the set of features in bolide_features.FeatureClass
            If None then the features used in trained_classifier_path are used.
        bands_to_use : int tuple
            List of indices to bands to use in the ABI CNN image model.
            Only used if use_multi_band_data=True
            Note: This is in the order stored in the *_multi_bands.pbz2 files, which is a 0-based index. The actual
            bands these indices refer to are dictated by the cutout_bands_to_read configuration tuple.
            For all bands use 'all'
        starting_model_and_optim_checkpoint : dtr
            If given, then load the model and optimizer state from this checkpoint file as the starting point for all sample runs.
        ratio : tuple(3)
            (train, validate, test) split ratios (adds up to 1.0)
        num_samples : int
            Number of test instances of the config parameter space to evaluate
        max_num_epochs : int
            Maximum number of epochs for each sample instance to train
        max_num_uncertainty_epochs : int
            Maximum number of epochs when computing cross-validation uncertainty estimates
        batch_size : int
            Batch size for training
        augmentations : list of str
            List of augmentations to apply to the images
        random_seed : int
            Random seed for train/val/test split
            If None then keep as already set
        ASHA_brackets : int
            Number of brackets when performing Async Successive Halving
        ASHA_reduction_factor : int
            Reduction factor when performing Async Successive Halving
        optim_metric : str
            What metric to use for schedular to optimize on.
            It computes the metric on the validation data.
        PR_fig_filename : str
            filename for precision recall curve for best model
        output_save_path : str
            Path to save output best checkpoint and performance metrics
            If None then save to current working directory
        n_uncert_samples : int
            Number of cross-validation samples when computing the precision and recall uncertainties
            Set to 0 if not performing this measurement
        test_trainer : bool
            The actual trainer, hyper_tune_trainer, runs inside Ray Tune in subprocesses. The debugger does not follow
            these subprocesses. Setting this tri True will run hyper_tne_trainer using a single config instance then
            end.
 
        """
 
        startTime = time.time()

        assert max_num_epochs > 0 and max_num_uncertainty_epochs > 0, 'max_num_epochs and max_num_uncertainty_epochs must be greater than 0'
        assert max_num_epochs != 1 and max_num_uncertainty_epochs != 1, 'Odd things happen if max_num_epochs or max_num_uncertainty_epochs is equal to one, why?'

        # Force random seed to specific state
        # If none then use the random seed set in the constructer
        if random_seed is None:
            pass
        else:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
 
        if output_save_path is None:
            output_save_path = os.getcwd()

        self.model_name = model_name
        self.images_to_use = images_to_use
        self.extra_feature_names = extra_feature_names
        self.bands_to_use = bands_to_use
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.ratio = ratio
        self.max_num_epochs = max_num_epochs
        self.max_num_uncertainty_epochs = max_num_uncertainty_epochs
        self.starting_model_and_optim_checkpoint  = starting_model_and_optim_checkpoint 

        self.ASHA_brackets = ASHA_brackets
        self.ASHA_reduction_factor = ASHA_reduction_factor
 
        #***
        # Generate master train/val/test split data
        self.cnnObj.split_and_load_data(
                ratio=self.ratio, 
                batch_size=self.batch_size, 
                augmentations=augmentations, 
                random_seed=random_seed,
                bolidesFromWebsite=self.bolidesFromWebsite, 
                extra_feature_names=extra_feature_names, 
                bands_to_use=bands_to_use,
                trained_classifier_path=None)
        # Extract the data sets for the worker jobs
        data_sets = {}
        data_sets['train'] = self.cnnObj.data_train.generate_init_data_dict()
        data_sets['val'] = self.cnnObj.data_val.generate_init_data_dict()
        data_sets['test'] = self.cnnObj.data_test.generate_init_data_dict()
 
       #data_sets_ref = ray.put(data_sets)
 
        # options for validation optimization
        if optim_metric == 'loss':
            self.optim_mode = 'min'
        elif optim_metric == 'avg_prec':
            self.optim_mode = 'max'
        elif optim_metric == 'fbeta_score':
            self.optim_mode = 'max'
        elif optim_metric == 'recall_at_prec':
            self.optim_mode = 'max'
        else:
            raise Exception('Unknown optimization metric')
        self.optim_metric = optim_metric
 
        self.scheduler = ASHAScheduler(
            metric=self.optim_metric,
            mode=self.optim_mode,
            max_t=self.max_num_epochs,
            grace_period=np.min([10,self.max_num_epochs]),
            brackets=self.ASHA_brackets,
            reduction_factor=self.ASHA_reduction_factor)
        self.reporter = CLIReporter(
            metric_columns=["loss", "avg_prec", "fbeta_score", "recall_at_prec", "training_iteration"],
            max_report_frequency=60)
 
        #***
        # Run in test mode to run hyper_tune_trainer outside of Ray Tune
        if test_trainer:
            # Use a random config sample instance
            config_instance = {
                'learning_rate': config['learning_rate'].sample(),
                'weight_decay': config['weight_decay'].sample(),
                'n_chan_1': config['n_chan_1'].sample(),
                'n_chan_fc1': config['n_chan_fc1'].sample(),
                'n_chan_fc2': config['n_chan_fc2'].sample(),
                'CNN_momentum': config['CNN_momentum'].sample(),
                'leakyReLU_negative_slope': config['leakyReLU_negative_slope'].sample(),
                'extra_feature_momentum': config['extra_feature_momentum'].sample()
            }
            hyper_tune_trainer(config_instance, 
                           self.extra_feature_names, 
                           self.model_name,
                           self.use_multi_band_data,
                           self.GLM_LC_data_length,
                           self.bands_to_use,
                           self.images_to_use,
                           self.image_columns_to_use,
                           self.starting_model_and_optim_checkpoint,
                           self.max_num_epochs, 
                           batch_size, 
                           random_seed, 
                           self.num_cpu_threads, 
                           self.avg_precision_recall_limit, 
                           self.target_precision,
                           self.fbeta_score_threshold, 
                           self.fbeta_score_beta, 
                           data_sets=data_sets,
                           run_in_tune=False),

            return

        #***
        # Perform the tuning
        tuningStartTime = time.time()
        result = tune.run(
            lambda config: hyper_tune_trainer(config, 
                                              self.extra_feature_names, 
                                              self.model_name,
                                              self.use_multi_band_data,
                                              self.GLM_LC_data_length,
                                              self.bands_to_use,
                                              self.images_to_use,
                                              self.image_columns_to_use,
                                              self.starting_model_and_optim_checkpoint,
                                              self.max_num_epochs, 
                                              batch_size, 
                                              random_seed, 
                                              self.num_cpu_threads, 
                                              self.avg_precision_recall_limit, 
                                              self.target_precision,
                                              self.fbeta_score_threshold, 
                                              self.fbeta_score_beta, 
                                              data_sets=data_sets),
            resources_per_trial={"cpu": self.num_cpu_threads, "gpu": 1},
            config=config,
            num_samples=num_samples,
            scheduler=self.scheduler,
            progress_reporter=self.reporter,
            raise_on_failed_trial=False)
        tuningEndTime = time.time()
        totalTuningTime = tuningEndTime - tuningStartTime
        print('*****************')
        print("Total hyperparameter tuning time: {:.2f} minutes, {:.2f} hours".format(totalTuningTime/60, totalTuningTime/60/60))
        timePerSample = totalTuningTime / num_samples
        print("Average time per sample: {:.2f} seconds, {:.2f} minutes".format(timePerSample, timePerSample/60))
        print('*****************')

        #*************************
        # Recover the trial and epoch checkpoint with the best (lowest) loss
        # Search for the single lowest epoch loss over the entire training session for reach trial
        best_trial = result.get_best_trial(metric=self.optim_metric, mode=self.optim_mode, scope="all")

        #*************************
        # Precision and Recall Uncertainty measurements
        if n_uncert_samples > 0:
            try:
                print('*****************')
                uncertCalcStartTime = time.time()
                print('Computing precision and recall uncertainty for best model')
                cross_val_train_PR_data, cross_val_val_PR_data, cross_val_test_PR_data = self.cross_val_error_estimate(best_trial.config, n_uncert_samples=n_uncert_samples)
                uncertCalcEndTime = time.time()
                totalUncertCalcTime = uncertCalcEndTime - uncertCalcStartTime
                print('*****************')
                print("Total uncertainty calculation time: {:.2f} minutes, {:.2f} hours".format(totalUncertCalcTime/60, totalUncertCalcTime/60/60))
                print('*****************')
            except:
                print('*****************')
                print('Error running cross_val_error_estimate')
                print_exc()
                print('*****************')
                cross_val_train_PR_data = None
                cross_val_val_PR_data = None
                cross_val_test_PR_data = None
        else:
            cross_val_train_PR_data = None
            cross_val_val_PR_data = None
            cross_val_test_PR_data = None
 
        #*************************
        # Save the best checkpoint and model
        # Set up model based on best fit model
        if self.model_name == 'MultiImageModel':
            self.cnnObj.set_model(self.model_name, 
                    kernel_size=3, 
                    n_chan_1=best_trial.config['n_chan_1'],
                    n_chan_fc1=best_trial.config['n_chan_fc1'], 
                    n_chan_fc2=best_trial.config['n_chan_fc2'],
                    CNN_momentum=best_trial.config['CNN_momentum'], 
                    leakyReLU_negative_slope=best_trial.config['leakyReLU_negative_slope'], 
                    extra_feature_momentum=best_trial.config['extra_feature_momentum'],
                    images_to_use=self.images_to_use,
                    image_columns_to_use=self.image_columns_to_use,
                    use_multi_band_data=self.use_multi_band_data)
        elif self.model_name[0:6] == 'resnet':
            self.cnnObj.set_model(self.model_name, 
                    n_chan_fc2=best_trial.config['n_chan_fc2'],
                    leakyReLU_negative_slope=best_trial.config['leakyReLU_negative_slope'], 
                    extra_feature_momentum=best_trial.config['extra_feature_momentum'],
                    images_to_use=self.images_to_use,
                    image_columns_to_use=self.image_columns_to_use,
                    use_multi_band_data=self.use_multi_band_data)
        else:
            raise Exception('Unknown model')
        # Recover epoch checkpoint with the best (lowest) loss, not necessarily the last epoch.
        checkpoint = result.get_best_checkpoint(best_trial, metric=self.optim_metric, mode=self.optim_mode)
        best_checkpoint_dir = checkpoint.path
        # Load the best model from the Ray Tune run
        self.cnnObj.load_checkpoint(best_checkpoint_dir)
        print('*****************')
        output_save_path, best_checkpoint_dir = self.save_best_checkpoint_and_model(output_save_path, best_checkpoint_dir, purge_ray_results=True)
        print('*****************')
        print("Best trial config: {}".format(best_trial.config))
        #TODO: figure out how to get the best checkpoint loss value (not the last checkpoint loss value)
       #print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
        print('*****************')

        #*************************
        # Plot precision recall curve for best model
        print('Computing performance metrics for best model...')
 
        self.cnnObj.compute_precision_recall_curves(cross_val_train_PR_data, cross_val_val_PR_data, cross_val_test_PR_data)
        self.cnnObj.print_machine_opinions()
        self.cnnObj.print_model_opinions()

        # Get the optimization metric values for all trials and checkpoints

        self.get_all_checkpoint_optim_metric_vals(result)
 
        print('Plotting precision/recall curves for best model...')
        self.cnnObj.plot_precision_recall_curves(save_filename=os.path.join(output_save_path, PR_fig_filename), 
                    trial_checkpoint_metric_vals=self.trial_checkpoint_metric_vals, optim_metric=self.optim_metric, 
                    best_trial=best_trial, best_checkpoint_dir=best_checkpoint_dir)


        endTime = time.time()
        totalTime = endTime - startTime
        print('*****************')
        print("Total tuning time: {:.2f} minutes, {:.2f} hours".format(totalTime/60, totalTime/60/60))
 
        return

    def cross_val_error_estimate(self,
            config,
            n_uncert_samples=10,
            random_seed=None):
        """ Uses a cross-validation technique to train multiple models using different train and test splits. 
        It then analyzes the spread in precision vs recall curves to estimate the uncertainty on the curves.
 
        Parameters
        ----------
        config : Dict
            The configuration hyper-parameters. But, here, pass only the best set.
        n_uncert_samples : int
            Number of cross-validation samples
        random_seed : int
            Random seed for train/val/test split
            If None then keep as already set
 
 
        Returns
        -------
        cross_val_train_PR_data : list PrecisionRecallResults
            The precision recall curves for each of the n_uncert_samples model fits
        cross_val_val_PR_data : list PrecisionRecallResults
        cross_val_test_PR_data : list PrecisionRecallResults
 
        """
 
        # Force random seed to specific state
        # If none then use the random seed set in the constructer
        if random_seed is None:
            pass
        else:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
 
        #***
        # Perform cross-validation evaluation
        # We do this by just calling hyper_tune_trainer many times with the same configuration. 
        # We also do not pass data_sets, which results in split_and_load_data generating new random splits for each run.

        cached_filenames = copy.deepcopy(self.cnnObj.cached_filenames)

        # We want to use a scheduler without early stopping, so set grace_period to max_num_uncertainty_epochs
        scheduler = ASHAScheduler(
            metric=self.optim_metric,
            mode=self.optim_mode,
            max_t=self.max_num_uncertainty_epochs,
            grace_period=self.max_num_uncertainty_epochs,
            brackets=self.ASHA_brackets,
            reduction_factor=self.ASHA_reduction_factor)
 
        # We want a new random seed for each run
        result = tune.run(
            lambda config: hyper_tune_trainer(config, 
                                              self.extra_feature_names, 
                                              self.model_name,
                                              self.use_multi_band_data,
                                              self.GLM_LC_data_length,
                                              self.bands_to_use,
                                              self.images_to_use,
                                              self.image_columns_to_use,
                                              self.starting_model_and_optim_checkpoint,
                                              self.max_num_uncertainty_epochs, 
                                              self.batch_size, 
                                              None, 
                                              self.num_cpu_threads, 
                                              self.avg_precision_recall_limit, 
                                              self.target_precision,
                                              self.fbeta_score_threshold, 
                                              self.fbeta_score_beta, 
                                              data_sets=None,
                                              cached_filenames=cached_filenames,
                                              ratio=self.ratio,
                                              bolideDispositionProfileList=self.bolide_dispositions_object_file,
                                              bolidesFromWebsite=self.bolidesFromWebsite,
                                              augmentations=self.augmentations),
            resources_per_trial={"cpu": self.num_cpu_threads, "gpu": 1},
            config=config,
            num_samples=n_uncert_samples,
            scheduler=scheduler,
            progress_reporter=self.reporter,
            raise_on_failed_trial=False)
 
 
        #***
        # Find best checkpoint for each trial
        best_checkpoint = []
        for trial in result.trials:
            checkpoint = result.get_best_checkpoint(trial, metric=self.optim_metric, mode=self.optim_mode)
            best_checkpoint.append(checkpoint.path)
 
 
        #***
        # Compute precision recall curve for each run
        if self.model_name == 'MultiImageModel':
            self.cnnObj.set_model(self.model_name, 
                    kernel_size=3, 
                    n_chan_1=config['n_chan_1'],
                    n_chan_fc1=config['n_chan_fc1'], 
                    n_chan_fc2=config['n_chan_fc2'],
                    CNN_momentum=config['CNN_momentum'], 
                    leakyReLU_negative_slope=config['leakyReLU_negative_slope'], 
                    extra_feature_momentum=config['extra_feature_momentum'],
                    images_to_use=self.images_to_use,
                    image_columns_to_use=self.image_columns_to_use,
                    use_multi_band_data=self.use_multi_band_data)
        elif self.model_name[0:6] == 'resnet':
            self.cnnObj.set_model(self.model_name, 
                    n_chan_fc2=config['n_chan_fc2'],
                    leakyReLU_negative_slope=config['leakyReLU_negative_slope'], 
                    extra_feature_momentum=config['extra_feature_momentum'],
                    images_to_use=self.images_to_use,
                    image_columns_to_use=self.image_columns_to_use,
                    use_multi_band_data=self.use_multi_band_data)
        else:
            raise Exception('Unknown model')
 
        # Store the PR data for each run
        cross_val_train_PR_data = []
        cross_val_val_PR_data = []
        cross_val_test_PR_data = []
        for checkpoint_dir in best_checkpoint:
            self.cnnObj.load_checkpoint(checkpoint_dir)
 
            # Set up test set from saved data splits ID_sets.p
            # Here a new columnTransformer is created for each checkpoint. 
            # This could mean that the data normalization is different than when training the classifiers. 
            # However, because we are passing the data set splits with the ID_sets dict, the same normalization will
            # occur as above in hyper_tune_trainer
            ID_sets_filename = os.path.join(os.path.split(checkpoint_dir)[0], 'ID_sets.p')
            with open(ID_sets_filename, 'rb') as fp:
                ID_sets = pickle.load(fp)
            self.cnnObj.split_and_load_data(
                    ratio=ID_sets, 
                    batch_size=self.batch_size, 
                    augmentations=self.augmentations, 
                    random_seed=random_seed, 
                    bolidesFromWebsite=self.bolidesFromWebsite, 
                    extra_feature_names=self.extra_feature_names, 
                    bands_to_use=self.bands_to_use,
                    data_sets=None)
 
            self.cnnObj.load_checkpoint(checkpoint_dir)
 
            self.cnnObj.compute_precision_recall_curves()
 
            cross_val_train_PR_data.append(self.cnnObj.train_precision_recall)
            cross_val_val_PR_data.append(self.cnnObj.val_precision_recall)
            cross_val_test_PR_data.append(self.cnnObj.test_precision_recall)
 
        '''
        # Define scaling and run configs
        scaling_config = ScalingConfig(
                trainer_resources=n_tot_cpus,
                num_workers=n_uncert_samples,
                use_gpu=True,
                resources_per_worker={'CPU':num_cpu_threads, 'GPU':1}
                )
 
        train_loop_config={}
 
        run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))
 
        trainer = TorchTrainer(
            train_loop_per_worker= hyper_tune_trainer(
                                            config, 
                                            extra_feature_names, 
                                            model_name,
                                            starting_model_and_optim_checkpoint,
                                            max_num_uncertainty_epochs, 
                                            batch_size, 
                                            random_seed, 
                                            num_cpu_threads, 
                                            avg_precision_recall_limit, 
                                            target_precision,
                                            fbeta_score_threshold, 
                                            fbeta_score_beta, 
                                            data_sets=None,
                                            cached_filenames=cnnObj.cached_filenames,
                                            ratio=ratio,
                                            bolide_dispositions_object=cnnObj.bDispObj,
                                            bolidesFromWebsite=bolidesFromWebsite,
                                            run_in_tune=False),
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config)
 
        result = trainer.fit()
        '''
 
        return cross_val_train_PR_data, cross_val_val_PR_data, cross_val_test_PR_data 

    def save_best_checkpoint_and_model(self, save_path, best_checkpoint_dir, purge_ray_results=False):
        """ 
        This will save the best checkpoint and trial log information to the specified directory.
        This file can be used to resume a Ray Tune run.
 
        This will then also save out the full model, columnTransformer and extra_feature_names from the training data set. This latter save
        file can then be used to evaluate the fit model on new data.

        Only the best checkpoint given by the path is saved.
 
        Parameters
        ----------
        save_path : str
            Path to save the best checkpoint and trial log information
        best_checkpoint_dir : str
            The directory of the best checkpoint. One directory up will be where the trial log information is located that
            is copied over to save_path.
        top_level_run_dir : str
            The path to the top level. If purge_ray_results==True then everything in the directory is deleted
        purge_ray_results : bool
            If True then everything at top_level_run_dir is deleted.
 
        Returns
        -------
        save_path : str
            Top level path to the saved data
        best_checkpoint_save_path : str
            Path to the new copy of the best checkpoint
 
        """
 
        # Get the directory name of the best trial
        best_checkpoint_path, best_checkpoint_name = os.path.split(best_checkpoint_dir)
        top_level_run_dir, best_trial_dir_name = os.path.split(best_checkpoint_path)
 
        best_trial_dir = os.path.join(top_level_run_dir, best_trial_dir_name)
 
        # Add run top level name to save path
        run_name = os.path.basename(top_level_run_dir)
        save_path = os.path.join(save_path, run_name)
       #best_trial_save_path = os.path.join(save_path, best_trial_dir_name)
        best_checkpoint_save_path = os.path.join(save_path, best_checkpoint_name)
 
        # Clear out or create the output directories, if needed
        if os.path.isdir(save_path):
            # Clear out path
            shutil.rmtree(save_path)
        else:
            os.makedirs(save_path)
 
        #***
        # Copy over best checkpoint and other log information
 
        # Copy over checkpoint directory
        os.system('cp -r {} {}'.format(best_checkpoint_dir, save_path))
 
        # Copy over other log information for the run
        # Copy only log files, not directories
        os.system('find '+ top_level_run_dir +  ' -maxdepth 1 -type f -execdir cp "{}" '+ save_path + ' ";"')
        # Now copy log files for this best trial (but not all checkpoints)
        os.system('find '+ best_trial_dir +  ' -maxdepth 1 -type f -execdir cp "{}" '+ save_path + ' ";"')
 
        print('Best checkpoint saved to {}'.format(best_checkpoint_save_path))

        #***
        # Save the best model, columnTransformer and extra_feature_names
        best_model_save_name = os.path.join(save_path, 'best_model.pt')
        torch.save((self.cnnObj.model, 
                    self.cnnObj.data_train.columnTransformer, 
                    self.extra_feature_names, 
                    self.use_multi_band_data, 
                    self.bands_to_use, 
                    self.GLM_LC_data_length), 
                      best_model_save_name)
        print('Best model saved to {}'.format(best_model_save_name))
 
        if purge_ray_results:
            shutil.rmtree(top_level_run_dir)
 
        return save_path, best_checkpoint_save_path

    def get_all_checkpoint_optim_metric_vals(self, result):
        """
        Returns the self.optim_metric values for all checkpoints for all trials

        Parameters
        ----------
        result : ray.tune.analysis.experiment_analysis.ExperimentAnalysis
            The result of the Ray Tune run

        Returns
        -------
        self.trial_checkpoint_metric_vals : dict
            All trials and checkpoint optim_metrics values over all epochs        
            The keys are the trial_ids and the arrays are the optim_metric values for each checkpoint (i.e. epoch)
        """

       #self.trial_checkpoint_metric_vals = np.full((len(result.trials), (self.max_num_epochs)), np.nan)
        self.trial_checkpoint_metric_vals = dict()
        for i, trial in enumerate(result.trials):
            checkpoints_and_metrics = result._get_trial_checkpoints_with_metric(trial, metric=self.optim_metric)
            trial_id = os.path.split(os.path.split(checkpoints_and_metrics[0][0].path)[0])[1][7:18]
           #self.trial_checkpoint_metric_vals[i,:] = [cm[1] for cm in checkpoints_and_metrics]
            self.trial_checkpoint_metric_vals[trial_id] = [cm[1] for cm in checkpoints_and_metrics]

        return
 
        

# This function is the Ray Tune worker and cannot be in a class
def hyper_tune_trainer(
        config, 
        extra_feature_names, 
        model_name, 
        use_multi_band_data,
        GLM_LC_data_length,
        bands_to_use,
        images_to_use,
        image_columns_to_use,
        starting_model_and_optim_checkpoint,
        max_num_epochs, 
        batch_size,
        random_seed, 
        num_cpu_threads,
        avg_precision_recall_limit, 
        target_precision,
        fbeta_score_threshold, 
        fbeta_score_beta, 
        data_sets=None,
        cached_filenames=None,
        ratio=None,
        bolideDispositionProfileList=None,
        bolidesFromWebsite=None,
        augmentations=None,
        run_in_tune=True):
    """ This is the trainer function used in the hyper parameter tuning in hyper_tune.

    Parameters
    ----------
    config : Dict
        The Ray Tune config parameters, gives which hyperparmaters to tune and their value ranges.
    extra_feature_names : list of str
        The names of the extra features to add in.
        They must be from the set of features in bolide_features.FeatureClass
    model_name : str
        Name of model to use
        See CnnTrainer.set_model for options
    use_multi_band_data : bool
        If True then use multi-band data, otherwise of quasi-RGB cutout images
    GLM_LC_data_length : int
        The data length for the interpolated GLM light curve.
        Set to None if not using GLM light curve 
    bands_to_use : int tuple
        List of indices to bands to use in the ABI CNN image model.
        Only used if use_multi_band_data=True
        Note: This is in the order stored in the *_multi_bands.pbz2 files. The bands are dictated by the
        cutout_bands_to_read configuration tuple.
        For all bands use 'all'
    images_to_use : int list or 'all'
        List of images to use with the CNNs, or 'all' to use all images
        see MultiImageModel.out_img_names for list of images and order
        Note: this uses 1-based indexing!
    image_columns_to_use : str tuple or 'all'
        The names of the image columns to use. See glmDataSet.img_column_names for options.
    starting_model_and_optim_checkpoint : str
        If given, then load the model and optimizer state from this checkpoint file as the starting point for all sample runs.
    max_num_epochs : int
        Maximum number of epochs for each sample instance to train
    batch_size : int
        Batch size for training
    random_seed : int
        Random seed for train/val/test split
        If None then set randomly
    num_cpu_threads : int
        Number of CPU threads to use per sample trial job, 1 GPU is always used.
    avg_precision_recall_limit : float [0.0, 1.0)
        When computing the average precision, this is the lower recall limit to use in the integration.
        Average precision is normalized by the recall evaluation range.
    target_precision : float [0.0, 1.0]
        Target precision to compute recall at for recall_at_precision_score
    fbeta_score_threshold : float
        The classifier threshold score to use when computing fbeta_score
    fbeta_score_beta : float
        The fbeta score beta value to use when weighting precision vs recall.
    data_sets : dict
        The saved data sets generated in the mother cnnObj created in hyper_tune
        A dict of glmDataSet init_data dicts used to initialize the glmDataSets objects with cached data
        keys = {'train', 'val', 'test'}
        If not passed, then the data set splits are generated, using cached_filenames and ratio
    cached_filenames : Dict of Dicts
        If passed then use this set of cached images instead of generating them
        List of cached filenames where the keys are glmDataSet.img_types
        Then the second level keys are individual IDs and there values are the filenames for each ID
    ratio : tuple(3) or Dict
        (train, validate, test) split ratios (adds up to 1.0)
        If passed a Dict then use these IDs for each set
        Dict{"train": IDs, "val": IDs, "test": IDs}
    bolideDispositionProfileList : str
        path to bolideDispositionProfileList pickle file
        Ray Tune does not like objects passed to workers, so load in a pickle file
        The bolide dispositions object generated from the pipeline data set
        This is used to generate the training data sets.
        If passing in the data_sets then this must be None
    bolidesFromWebsite : [WebsiteBolideEvent list] created by:
        bolide_dispositions.pull_dispositions_from_website
        If passed and beliefSource == 'human' then use this data to determine truth.
        If passing in the data_sets then this can be None
    run_in_tune : bool
        If True, then we are running in ray.tune

    """

    if data_sets is None:
        assert cached_filenames is not None, 'cached_filenames must be passed if data_sets is None'
        assert ratio is not None, 'ratio must be passed if data_sets is None'
        assert bolideDispositionProfileList is not None, 'bolideDispositionProfileList must be passed if data_sets is None'
        assert bolidesFromWebsite is not None, 'bolidesFromWebsite must be passed if data_sets is None'

        # Construct BolideDispositions object
        bolide_dispositions_object = bDisp.BolideDispositions.from_bolideDispositionProfileList(bolideDispositionProfileList, useRamDisk=False)
    else:
        assert cached_filenames is None, 'cached_filenames must not be passed if data_sets is not None'
        assert ratio is None, 'ratio must not be passed is data_sets if not None'
        assert bolideDispositionProfileList  is None, 'bolideDispositionProfileList  must not be passed if data_sets is not None'
        assert bolidesFromWebsite is None, 'bolidesFromWebsite must not be passed if data_sets is not None'
        assert augmentations is None, 'augmentations must not be passed if data_sets is not None'
        # We are loading the data sets and not creating them, so set these to None to tell CnnTrainer to not try to
        # generate the data sets.
        bolide_dispositions_object = None

    # Force random seed to specific state
    if random_seed is None:
        np.random.seed()
        torch.manual_seed(np.random.randint(1e10))
    else:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Retrieve the training data sets from shared object store memory
   #data_sets = ray.get(data_sets_ref)

    set_num_threads(num_cpu_threads)

    # Create a new CnnTrainer object for this training run
    # and set the data sets to the sets generated in the mother cnnObj in hyper_tune
    cnnObj = CnnTrainer(bolide_dispositions_object=bolide_dispositions_object, 
            image_cache_path=None,
            cached_filenames=cached_filenames,
            use_multi_band_data=use_multi_band_data,
            GLM_LC_data_length=GLM_LC_data_length,
            gpu_index=None, 
            avg_precision_recall_limit=avg_precision_recall_limit,
            target_precision=target_precision,
            fbeta_score_threshold=fbeta_score_threshold,
            fbeta_score_beta=fbeta_score_beta,
            run_in_tune=run_in_tune, 
            verbosity=False)

    cnnObj.split_and_load_data(
            ratio=ratio, 
            batch_size=batch_size, 
            augmentations=None, 
            random_seed=random_seed, 
            bolidesFromWebsite=bolidesFromWebsite, 
            extra_feature_names=extra_feature_names, 
            bands_to_use=bands_to_use,
            trained_classifier_path=None, 
            data_sets=data_sets)

    # Set up model, optimizer and loss function
    if model_name == 'MultiImageModel':
        cnnObj.set_model(model_name, 
                kernel_size=3, 
                n_chan_1=config['n_chan_1'],
                n_chan_fc1=config['n_chan_fc1'], 
                n_chan_fc2=config['n_chan_fc2'],
                CNN_momentum=config['CNN_momentum'], 
                leakyReLU_negative_slope=config['leakyReLU_negative_slope'], 
                extra_feature_momentum=config['extra_feature_momentum'],
                images_to_use=images_to_use,
                image_columns_to_use=image_columns_to_use,
                use_multi_band_data=use_multi_band_data)
    elif model_name[0:6] == 'resnet':
        cnnObj.set_model(model_name, 
                n_chan_fc2=config['n_chan_fc2'],
                leakyReLU_negative_slope=config['leakyReLU_negative_slope'], 
                extra_feature_momentum=config['extra_feature_momentum'],
                images_to_use=images_to_use,
                image_columns_to_use=image_columns_to_use,
                use_multi_band_data=use_multi_band_data)
    else:
        raise Exception('Unknown model')

    cnnObj.set_optimizer_and_loss_fcn(optimizer_name='Adam', learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'], loss_fn_name='CrossEntropyLoss')

    # If given, load in the starting model and optimizer checkpoint
    if starting_model_and_optim_checkpoint is not None:
        cnnObj.load_checkpoint(starting_model_and_optim_checkpoint)

    # train!
    cnnObj.training_loop(n_epochs=max_num_epochs)

    # NOTE: If running in ray.tune then nothing after traiining_loop will be executed below!

    return


#*************************************************************************************************************
# Cross-Validation error estimates


#*************************************************************************************************************
#*************************************************************************************************************
#*************************************************************************************************************
class PrecisionRecallResults():
    """ This stores the precision and recall results of the application of a model on a data set
    """

    # This is the number of cadences in the interpolated cross_val_PR_data vectors
    n_PR_curve_cadences = 2000

    # These are used to generate the precision vs recall curves for the fitted model
    precision = None
    recall = None
    thresholds = None

    # These are used to compute the uncertainties on the precision and recall
    cross_val_PR_data = None
    interpolated_cross_val_PR_data = None
    prec_err_mean = None
    prec_err_low = None
    prec_err_high = None
    recall_err_mean = None
    recall_err_low = None
    recall_err_high = None
    err_thresholds = None

    avg_precision = None
    
    def __init__(self,
            precision=None, 
            recall=None, 
            thresholds=None,
            avg_precision=None):

        self.precision = precision
        self.recall = recall
        self.thresholds = thresholds
        self.avg_precision = avg_precision

    def compute_uncertainties(self, cross_val_PR_data=None):
        """ Computes the uncertainties for the precision and recall based on the data in the cross validation PR data

        Parameters
        ----------
        cross_val_PR_data : list PrecisionRecallResults
            The precision recall curves for each of the n_uncert_samples model fits
            If not passed then this method attempts to used stored data, if available.

        """

        if cross_val_PR_data is not None:
            self.cross_val_PR_data = cross_val_PR_data

        if self.cross_val_PR_data is None:
            # No data, nothing to compute
            return

        # Interpolate the PR data so they all have the same cadence.
        # Turns out sklearn.metrics.precision_recall_curve does not return the same length precision and recall curves.
        # The length of the returned vectors is dependent on the steps in the data.
        # So, use scipy.interpolate.interp1d(x,y, kind='nearest'). 
        # We do not want smoothing, we want the true steps to continue as in the original PR curves, just with sampling at the appropriate cadence.
        x_interpolated = np.arange(0.0, 1.0, 1.0/self.n_PR_curve_cadences)
        self.interpolated_cross_val_PR_data = []
        for PR_data in self.cross_val_PR_data:
            single_interp_results = PrecisionRecallResults()
            f_prec = interpolate.interp1d(PR_data.thresholds, PR_data.precision[:-1], kind='nearest', fill_value='extrapolate')
            single_interp_results.precision = f_prec(x_interpolated)
            f_recall = interpolate.interp1d(PR_data.thresholds, PR_data.recall[:-1], kind='nearest', fill_value='extrapolate')
            single_interp_results.recall = f_recall(x_interpolated)

            single_interp_results.thresholds = x_interpolated

            self.interpolated_cross_val_PR_data.append(single_interp_results)
        

        # Compute the middle 90th percentile bands 
        prctle_vals = [5, 95]
        cross_val_prec_err_mean = []
        cross_val_prec_err_low = []
        cross_val_prec_err_high = []
        cross_val_recall_err_mean = []
        cross_val_recall_err_low = []
        cross_val_recall_err_high = []
        for idx in np.arange(len(self.interpolated_cross_val_PR_data[0].precision)):
            precision_this_idx = [x.precision[idx] for x in self.interpolated_cross_val_PR_data]
            recall_this_idx = [x.recall[idx] for x in self.interpolated_cross_val_PR_data]
            
            prec_mean = np.mean(precision_this_idx)
            [prec_low, prec_high] = np.percentile(precision_this_idx, prctle_vals)
            recall_mean = np.mean(recall_this_idx)
            [recall_low, recall_high] = np.percentile(recall_this_idx, prctle_vals)

            cross_val_prec_err_mean.append(prec_mean)
            cross_val_prec_err_low.append(prec_low)
            cross_val_prec_err_high.append(prec_high)
            cross_val_recall_err_mean.append(recall_mean)
            cross_val_recall_err_low.append(recall_low)
            cross_val_recall_err_high.append(recall_high)

        self.prec_err_mean = np.array(cross_val_prec_err_mean)
        self.prec_err_low = np.array(cross_val_prec_err_low)
        self.prec_err_high = np.array(cross_val_prec_err_high)
        self.recall_err_mean = np.array(cross_val_recall_err_mean)
        self.recall_err_low = np.array(cross_val_recall_err_low)
        self.recall_err_high = np.array(cross_val_recall_err_high)
        self.err_thresholds = x_interpolated
        
        return


class CnnTrainer():
    """ This is the main class to handle CNN training
    """

    def __init__(self, 
            bolide_dispositions_object=None, 
            image_cache_path='/tmp/cnn_image_cache',
            cached_filenames=None,
            use_multi_band_data=True,
            GLM_LC_data_length=None,
            image_size=None, 
            gpu_index=None,
            avg_precision_recall_limit=0.0,
            target_precision = 0.98,
            fbeta_score_threshold=0.5,
            fbeta_score_beta=1.0,
            run_in_tune=False, 
            verbosity=True,
            random_forest_uncertainties_file=None,
            save_onnx_file=False
            ):
        """ class initilializer

        This will first set up diagnostic and configuration paerameters.

        It also will preprocess the images and cache to file.

        Parameters
        ----------
        bolide_dispositions_object : bolide_dispositions.BolideDispositions
            The bolide dispositions object generated from the pipeline data set
            This is used to generate the training data sets.
            If passing in the data_sets then this can be None
        image_cache_path : str
            Path to where to save the preprocessed images converted to tensors
            If None then no preprocessing of images is performed. This is used, for example, when running within Ray Tune and the preprocessing already occured.
        cached_filenames : Dict of Dicts
            If passed then use this set of cached images instead of generating them
            List of cached filenames where the keys are glmDataSet.img_types
            Then the second level keys are individual IDs and there values are the filenames for each ID
        use_multi_band_data : bool
            If True then use multi-band data, otherwise of quasi-RGB cutout images
        GLM_LC_data_length : int
            The data length for the interpolated GLM light curve.
            Set to None if not using GLM light curve 
        image_size : int
            Rescales the images to image_size x image_size pixels
            If None, then do not resize, use original image sizes.
        gpu_index : int
            Specify the GPU index to use for processing
            Only useful for a multi-GPU machine
            If None then use default for CUDA
        avg_precision_recall_limit : float [0.0, 1.0)
            When computing the average precision, this is the lower recall limit to use in the integration.
            Average precision is normalized by the recall evaluation range.
        target_precision : float [0.0, 1.0]
            Target precision to compute recall at for recall_at_precision_score
        fbeta_score_threshold : float
            The classifier threshild score to use when computing fbeta_score
        fbeta_score_beta : float
            The fbeta score beta value to use when weighting precision vs recall.
        run_in_tune : bool
            If True then we are running in Ray Tune and record report information
        verbosity : bool
        random_forest_uncertainties_file : str
            Path to pickle file containing the random forest uncertainty estimate data in a PrecisionRecallResults object
        save_onnx_file : bool
            If Ture then save out the model as a ONNX file and input data to a pickle file, then exit.

        """

        if use_multi_band_data:
            assert image_size is None, 'If use_multi_band_data then image_size must be none'

        # Be more efficient with GPU memory usage.
        # This does not seem to result in a performance penalty but does allow for more full utilization of the GPU
        # memory.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

        # If we are cleaning the data then the disposition object is modified, so make a copy.
        self.bDispObj = copy.deepcopy(bolide_dispositions_object)

        self.run_in_tune = run_in_tune
        self.use_multi_band_data = use_multi_band_data

        self.GLM_LC_data_length = GLM_LC_data_length 

        #***
        # These are loss function tuning parameters
        self.avg_precision_recall_limit = avg_precision_recall_limit
        assert fbeta_score_threshold >= 0.0 and fbeta_score_threshold <= 1.0, 'fbeta_score_threshold must be in the range [0.0, 1.0]'
        self.fbeta_score_threshold = fbeta_score_threshold
        self.fbeta_score_beta = fbeta_score_beta
        assert target_precision >= 0.0 and target_precision <= 1.0, 'target_precision must be in the range [0.0, 1.0]'
        self.target_precision = target_precision

        self.verbosity = verbosity

        self.random_forest_uncertainties_file = random_forest_uncertainties_file

        self.save_onnx_file = save_onnx_file

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if gpu_index is None:
            self.gpu_index = torch.cuda.current_device()
        else:
            self.gpu_index = gpu_index
        if self.device == 'cuda':
            if self.verbosity:
                print('Using CUDA')
            # Add in GPU index
            self.device = self.device + ':{}'.format(self.gpu_index)
        else: 
            if self.verbosity:
                print('Using CPU')

        #******
        # Cache image tensors
        self.image_size = image_size
        if cached_filenames is None and image_cache_path is not None:
            self.image_cache_path = image_cache_path
            if self.use_multi_band_data:
                # Not using RGB images, so do not use torchvision
                # The data is not stored as pytorch tensors, so, no tranformer
                self.transformer = None
            elif self.image_size is None:
                self.transformer = transforms.ToTensor()
            else:
                self.transformer = transforms.Compose([
                            transforms.Resize((self.image_size,self.image_size)),
                            transforms.ToTensor()
                            ])
            self.cached_filenames, self.bDispObj = cache_image_tensors_to_file(self.bDispObj, save_path=self.image_cache_path, 
                    transform=self.transformer, use_multi_band_data=self.use_multi_band_data)
        elif isinstance(cached_filenames, dict):
            self.cached_filenames = cached_filenames
        else:
            # If we get here, then we are using the stored data_sets
            self.cached_filenames = None

        #******
        # Pre-process GLM light curves
        # If bolide_dispositions_object (bDispObj) is None, then we are loading in the data sets with data_sets in the
        # split_and_load_data method. If so, then we are loading in the data and do not need to pre-process the light
        # curves -- they are already in the data sets.
        if self.GLM_LC_data_length is not None and self.bDispObj is not None:
            if USE_FIXED_TIME_LC:
                self.cached_lightcurves = CNN_1D_block_batchNorm.pre_process_GLM_light_curves_const_time(self.bDispObj, self.GLM_LC_data_length)
            else:
                raise Exception('This option is not currently available')
                self.cached_lightcurves = CNN_1D_block_batchNorm.pre_process_GLM_light_curves(self.bDispObj, self.GLM_LC_data_length)
        else:
            self.cached_lightcurves = None


        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.model = None
        self.optimizer = None
        self.loss_fn = None
            
        # This is to flag if the data sets were split in this object
        self.data_splits_generated = False

        # Precision and Recall computed in compute_precision_recall_curves
        self.precision_and_recall_computed = False

    def split_and_load_data(self, 
            ratio=(0.8, 0.1, 0.1), 
            batch_size=64,
            augmentations=None,
            random_seed=42, 
            bolidesFromWebsite=None,
            extra_feature_names=None, 
            bands_to_use='all',
            trained_classifier_path=None,
            data_sets=None):
        """ Sets up and loads the training, validation and test data sets using glmDataSet

        This routine can use the trained classifier file in order to load in the features used and the set scaling
        parameters for normalizing the features.

        The glmDataSet objects and other data are returned. They can be used, for example, when sharing data accross training runs in
        Ray Tune.

        If the optional data_sets are passed then instead of creating new glmDataSet objects, it will use the passed
        ones.

        Parameters
        ----------
        ratio : tuple(3) or Dict
            (train, validate, test) split ratios (adds up to 1.0)
            If passed a Dict then use these IDs for each set
            Dict{"train": IDs, "val": IDs, "test": IDs}
        batch_size : int
            Batch size for image DataLoader
        augmentations : list of str
            List of augmentations to apply to the train and validation images
        random_seed : int
            Random seed for train/val/test split
            If None then set randomly
        bolidesFromWebsite : [WebsiteBolideEvent list] created by:
            bolide_dispositions.pull_dispositions_from_website
            If passed and beliefSource == 'human' then use this data to determine truth.
        extra_feature_names : list of str
            The names of the extra features to add in.
            They must be from the set of features in bolide_features.FeatureClass
            If None then the features used in trained_classifier_path are used.
        bands_to_use : int tuple
            List of indices to bands to use in the ABI CNN image model.
            Only used if use_multi_band_data=True
            Note: This is in the order stored in the *_multi_bands.pbz2 files. The bands are dictated by the
            cutout_bands_to_read configuration tuple.
            For all bands use 'all'
        trained_classifier_path : str
            Path to the saved trained classifier object created by bolideDispositions.save_trained_classifier
            If passed then the extra features are those used in the saved classifier
        data_sets : Dict of glmDataSet init_data dicts
            A dict of glmDataSet init_data dicts used to initialize the glmDataSets objects with cached data
            keys = {'train', 'val', 'test'}
            If this is passed, then no splitting is done, the already split and cached data is loaded

        Returns
        -------
        """

        assert not(extra_feature_names is not None and trained_classifier_path is not None), \
                'extra_feature_names and trained_classifier_path cannot both be passed'

        if trained_classifier_path is not None:
            with open(trained_classifier_path, 'rb') as fp:
                trainedClassifierDict = pickle.load(fp)
            fp.close()
            self.extra_feature_names = trainedClassifierDict['featuresToUse']
            columnTransformer = trainedClassifierDict['columnTransformer']
        elif extra_feature_names is not None:
            self.extra_feature_names = extra_feature_names
            columnTransformer = None
        else:
            columnTransformer = None
            self.extra_feature_names = None

        if self.extra_feature_names is not None:
            self.n_extra_features = len(self.extra_feature_names)
        else:
            self.n_extra_features = 0

        self.bands_to_use = bands_to_use

        # If data_sets is not passed then generate the data sets
        # If passed then use the passed data sets
        if data_sets is None:

            self.data_splits_generated = True

            assert self.cached_filenames is not None, 'CnnTrainer did not cache images, we cannot generate data sets.'

            # The train/val/test split is handled as three arrays of indices to bDispObj.bolideDispositionProfileList 
            if isinstance(ratio, tuple):
                train_IDs, val_IDs, test_IDs = self.bDispObj.train_val_test_split(ratio=ratio, clean_bad_datums=True, random_seed=random_seed)
            elif isinstance(ratio, dict):
                train_IDs   = np.array(ratio['train'])
                val_IDs     = np.array(ratio['val'])
                test_IDs    = np.array(ratio['test'])
            else:
                raise Exception('unknown ratio type')
            
            
            self.data_train = glmDataSet(self.bDispObj, train_IDs, augmentations=augmentations,
                                            beliefSource='human', bolidesFromWebsite=bolidesFromWebsite, featuresToUse=self.extra_feature_names,
                                            columnTransformer=columnTransformer, verbosity=self.verbosity, 
                                            cached_filenames=self.cached_filenames,
                                            cached_lightcurves=self.cached_lightcurves,
                                            use_multi_band_data=self.use_multi_band_data, bands_to_use=self.bands_to_use,
                                            balance_data=True)
            # Use the same fitted columnTransformer for the other data sets
            columnTransformer = self.data_train.columnTransformer
            
            
            if len(val_IDs) > 0:
                self.data_val = glmDataSet(self.bDispObj, val_IDs, augmentations=augmentations, 
                                                beliefSource='human', bolidesFromWebsite=bolidesFromWebsite, featuresToUse=self.extra_feature_names,
                                                columnTransformer=columnTransformer, verbosity=self.verbosity, 
                                                cached_filenames=self.cached_filenames, 
                                                cached_lightcurves=self.cached_lightcurves,
                                                use_multi_band_data=self.use_multi_band_data, bands_to_use=self.bands_to_use,
                                                balance_data=False)
            else:
                self.data_val = None
            
            # Do not augment test data
            if len(test_IDs) > 0:
                self.data_test = glmDataSet(self.bDispObj, test_IDs, augmentations=None, 
                                                beliefSource='human', bolidesFromWebsite=bolidesFromWebsite, featuresToUse=self.extra_feature_names,
                                                columnTransformer=columnTransformer, verbosity=self.verbosity, 
                                                cached_filenames=self.cached_filenames, 
                                                cached_lightcurves=self.cached_lightcurves,
                                                use_multi_band_data=self.use_multi_band_data, bands_to_use=self.bands_to_use,
                                                balance_data=False)
            else:
                self.data_test = None

        else:
            # Load in the passed data_sets
            # TODO: figure out why @classmethod does not work for from_cache
            self.data_train = glmDataSet(from_cache_init=True)
            self.data_train.load_from_cache(data_sets['train'])
            self.data_val = glmDataSet(from_cache_init=True)
            self.data_val.load_from_cache(data_sets['val'])
            self.data_test = glmDataSet(from_cache_init=True)
            self.data_test.load_from_cache(data_sets['test'])
            self.data_splits_generated = True


        # Problems can occur if the last batch is of a smaller size, so for training drop the last batch
        # Also, we are using a specified sampler, so do not set shuffle parameter
        self.train_loader = torch.utils.data.DataLoader(self.data_train, batch_size=batch_size, drop_last=True,
                pin_memory=True, pin_memory_device=self.device, sampler=self.data_train.sampler)
        if self.data_val is not None:
            self.val_loader = torch.utils.data.DataLoader(self.data_val, batch_size=batch_size, shuffle=True, pin_memory=True, pin_memory_device=self.device)
        else:
            self.val_loader = None
        if self.data_test is not None:
            self.test_loader = torch.utils.data.DataLoader(self.data_test, batch_size=batch_size, shuffle=True, pin_memory=True, pin_memory_device=self.device)
        else:
            self.test_loader = None


        # Determine image size from one of the figures if no resizing
        # For quasi-RGB, all images are the same size
        # For multi-band images, the images can be different sizes and number of channels
        if self.image_size is None:
            images, _, _, _ = self.data_train.__getitem__(0)
            if self.use_multi_band_data:
                self.image_size = {}
                self.n_input_chan = {}
                for str_idx in np.arange(1,len(images)+1):
                    data_shape = images[str_idx].shape
                    # Check that image is square
                    assert data_shape[1] == data_shape[2], 'Images must be square'
                    self.image_size[str_idx] = data_shape[1]
                    self.n_input_chan[str_idx] = data_shape[0]
            else:
                # These are RGB images, all the same size
                self.image_size = images[1].shape[1]
                self.n_input_chan = 3


        self._extract_machine_opinions()

        return


    def set_model(self, model_name, **kwargs):
        """ Sets the CNN model to use.

        Each model has different confgiruation parameters. See the models for details.

        Parameters
        ----------
        model_name : str
            Name of model to use
        kwargs : Dict
            Arguments passed to the model constructor
            This is dependent on which model being used

        Returns
        -------
        self.model

        """
    
        if model_name == 'simpleDropout':
            self.model = simpleDropout(dropout_p=dropout_p, image_size=self.image_size)
        elif model_name == 'simpleDropoutWExtraFeatures':
            self.model = simpleDropoutWExtraFeatures(**kwargs)
        elif model_name == 'MultiImageModel':
            self.model = MultiImageModel(**kwargs,
                                            n_input_chan=self.n_input_chan,
                                            image_size=self.image_size, 
                                            n_extra_features=self.n_extra_features,
                                            use_resnet=0, GLM_LC_data_length=self.GLM_LC_data_length)
        elif model_name == 'resnet18':
            self.model = MultiImageModel(**kwargs,
                                            n_extra_features=self.n_extra_features,
                                            use_resnet=18,
                                            use_resnet_pretraining=False, GLM_LC_data_length=self.GLM_LC_data_length)

        elif model_name == 'resnet18_pretrained':
            self.model = MultiImageModel(**kwargs,
                                            n_extra_features=self.n_extra_features,
                                            use_resnet=18,
                                            use_resnet_pretraining=True, GLM_LC_data_length=self.GLM_LC_data_length)

        elif model_name == 'resnet50':
            self.model = MultiImageModel(**kwargs,
                                            n_extra_features=self.n_extra_features,
                                            use_resnet=50,
                                            use_resnet_pretraining=False, GLM_LC_data_length=self.GLM_LC_data_length)

        elif model_name == 'resnet50_pretrained':
            self.model = MultiImageModel(**kwargs,
                                            n_extra_features=self.n_extra_features,
                                            use_resnet=50,
                                            use_resnet_pretraining=True, GLM_LC_data_length=self.GLM_LC_data_length)

        elif model_name == 'resnet152':
            self.model = MultiImageModel(**kwargs,
                                            n_extra_features=self.n_extra_features,
                                            use_resnet=152,
                                            use_resnet_pretraining=False, GLM_LC_data_length=self.GLM_LC_data_length)

        elif model_name == 'resnet152_pretrained':
            self.model = MultiImageModel(**kwargs,
                                            n_extra_features=self.n_extra_features,
                                            use_resnet=152,
                                            use_resnet_pretraining=True, GLM_LC_data_length=self.GLM_LC_data_length)

        else:
            raise Exception('Unknown model')

        self.model.to(device=self.device)

    def set_optimizer_and_loss_fcn(self, optimizer_name, learning_rate=1e-3, weight_decay=0.0, loss_fn_name='CrossEntropyLoss'):
        """ Sets the optimizer and loss function to use.

        Parameters
        optimizer_name : str
            Name of optimizer to use from torch.optim
        learning_rate : float
            Optimizer learning rate
        weight_decay : float
            weight decay (L2 penalty)
        loss_fn_name : str
            Name of the torch.nn loss function to use in backpropagation
        """

        if optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise Exception('Unknown optimizer')

        if loss_fn_name == 'CrossEntropyLoss':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn_name == 'MSELoss':
            self.loss_fn = nn.MSELoss()
        else:
            raise Exception('Unknown loss function')

        return
 
    def training_loop(self, n_epochs, save_filename=None):
        """ Training loop with L2-Norm regularization

        Parameters
        ----------
        n_epochs : int
            Number of epochs in training
        save_filename : str
            filename and path for saving the figure
            If None then do not save figure and just display the figure.
        
        """

        if self.train_loader is None:
            raise Exception('training data not yet loaded in')

        if self.model is None:
            raise Exception('model not yet set')

        if self.optimizer is None:
            raise Exception('optimizer not yet set')

        startTime = time.time()

        for epoch in range(1, n_epochs+1):
            # We are training, so tell the model
            self.model.train()
            loss_train = 0.0
        
            for images, light_curves, features, labels in self.train_loader:
                for key in images.keys():
                    images[key] = images[key].to(device=self.device)
                for key in light_curves.keys():
                    light_curves[key] = light_curves[key].to(device=self.device)
                features = features.to(device=self.device)
                labels = labels.to(device=self.device)

                # Augment images
                images = self.data_train.augment_images(images)

                outputs, outputs_score = self.model(images, light_curves, features)

                # Save out ONNX model
                if self.save_onnx_file:
                    onnx_file_name = os.path.join(os.getcwd(), 'model.onnx')
                    torch.onnx.export(self.model, (images, light_curves, features), onnx_file_name)
                    # Save out model input data
                    data_file_name = os.path.join(os.getcwd(), 'onnx_model_data.p')
                    with open(data_file_name, 'wb') as fp:
                        pickle.dump((images, light_curves, features), fp)
                    fp.close()
                    raise Exception('Saved out ONNX file for model. Now exiting.')

                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_train += loss.item()


            loss_train = loss_train / len(self.train_loader)

            # Compute validation loss
            if self.val_loader is not None:
                with torch.no_grad():
                    # We are NOT training so tell the model we are just evaluating
                    self.model.eval()
                    loss_val = 0.0
                    val_labels = torch.tensor([]).to(device=self.device)
                    val_scores = torch.tensor([]).to(device=self.device)
                    for images, light_curves, features, labels in self.val_loader:
                        for key in images.keys():
                            images[key] = images[key].to(device=self.device)
                        for key in light_curves.keys():
                            light_curves[key] = light_curves[key].to(device=self.device)
                        features = features.to(device=self.device)
                        labels = labels.to(device=self.device)

                        # Augment images
                        images = self.data_val.augment_images(images)

                        outputs, outputs_score = self.model(images, light_curves, features)
                        loss = self.loss_fn(outputs, labels)
                        val_labels = torch.cat((val_labels, labels))
                        val_scores = torch.cat((val_scores, outputs_score))
                    
                        loss_val += loss.item()

                    loss_val = loss_val / len(self.val_loader)
                    val_scores = val_scores[:,1].cpu()
                    val_y_true = np.array(val_labels.cpu())[:,1]
                    val_avg_precision = custom_average_precision_score(val_y_true, val_scores, self.avg_precision_recall_limit)

                    fbeta_score_val = calc_fbeta_score(val_y_true, val_scores, threshold=self.fbeta_score_threshold, beta=self.fbeta_score_beta)
                    recall_at_prec_score_val = recall_at_precision_score(val_y_true, val_scores, target_precision=self.target_precision)

            else:
                loss_val = None
                val_avg_precision = None
                fbeta_score_val = None
                recall_at_prec_score_val = None
            

            if self.verbosity:
               #if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
                print('{} Epoch {}, Training loss {}, Validation loss {}, Validation Avg. Precision {}'.format(datetime.datetime.now(), epoch,
                    loss_train, loss_val, val_avg_precision))

                # Print precision and recall every 5th epoch, but not at the end, that's done below
                if epoch == 1 or epoch % 5 == 0:
                    self.compute_precision_recall_curves()
                    self.print_model_opinions()


            if self.run_in_tune:
                # Save *every* epoch checkpoint. 
                # In save_best_checkpoint we save the best checkpoint and purge all the others
               #with tune.checkpoint_dir(epoch) as checkpoint_dir:

                with tempfile.TemporaryDirectory() as tempdir:
                    self.save_checkpoint(tempdir)

                    metrics = {'loss':loss_val, 'avg_prec':val_avg_precision, 'fbeta_score':fbeta_score_val, 'recall_at_prec':recall_at_prec_score_val}
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))

                    # Save this in the real top-level checkpoint directory, not the temp directory
                    if epoch == 1 and self.data_splits_generated: 
                      # save_dir = tempdir

                        checkpoint = train.get_checkpoint()
                        with checkpoint.as_directory() as checkpoint_dir:
                            save_dir = os.path.split(checkpoint_dir)[0]
                            
                            ID_sets = {}
                            ID_sets['train'] = self.data_train.data_IDs
                            ID_sets['val'] = self.data_val.data_IDs
                            ID_sets['test'] = self.data_test.data_IDs
                            
                            save_ID_sets_filename = os.path.join(save_dir, 'ID_sets.p')
                            with open(save_ID_sets_filename, 'wb') as fp:
                                pickle.dump(ID_sets, fp)
                            fp.close()

            

              # checkpoint = train.get_checkpoint()
              # with checkpoint.as_directory() as checkpoint_dir:
              #     print('***************************')
              #     print('***************************')
              #     print('***************************')
              #     print('checkpoint_dir: {}'.format(checkpoint_dir))
              #     print('***************************')
              #     print('***************************')
              #     print('***************************')
              #     self.save_checkpoint(checkpoint_dir)

              #     # Note: because the worker ends immediately when tune.report reports the last epoch iteration, we have to place the code to save the
              #     # data set IDs here.
              #     # If computing error estimate then save the data splits
              #     # Find the trial run temp directory via the checkpoint directory
              #     # TODO: There must be another way to find this, I just don't know enough about Ray Tune
              #     if epoch == 1 and self.data_splits_generated: 
              #         save_dir = os.path.split(checkpoint_dir)[0]
              #         ID_sets = {}
              #         ID_sets['train'] = self.data_train.data_IDs
              #         ID_sets['val'] = self.data_val.data_IDs
              #         ID_sets['test'] = self.data_test.data_IDs
              #         
              #         save_ID_sets_filename = os.path.join(save_dir, 'ID_sets.p')
              #         with open(save_ID_sets_filename, 'wb') as fp:
              #             pickle.dump(ID_sets, fp)
              #         fp.close()

              # # Note each Tune workers ends immediately after the last tune.report at the maximum iteration is called. 
              # # No code after this line during the last iteratin will be executed!
              # tune.report(loss=loss_val, avg_prec=val_avg_precision, fbeta_score=fbeta_score_val, recall_at_prec=recall_at_prec_score_val)


        endTime = time.time()
        totalTime = endTime - startTime
        if self.verbosity:
            print("")
            print("Total training time: {:.2f} minutes, {:.2f} hours".format(totalTime/60, totalTime/60/60))
            timePerEpoch = totalTime / n_epochs
            print("Average time per epoch: {:.2f} seconds, {:.2f} minutes".format(timePerEpoch, timePerEpoch/60))
            
            # Plot the precision recall curves
            self.print_machine_opinions()
            self.print_model_opinions()
            self.plot_precision_recall_curves(save_filename=save_filename)

        return

    def save_checkpoint(self, checkpoint_dir, comment=''):
        """ Saves a model and oprimizer state as a checkpoint.

        Parameters
        ----------
        checkpoint_dir : str
            Directory path to save checkpoint file
        comment : str
            Optional string to append to end of checkpoint filename.
            Otherwise, filename is just "checkpoint.pt"

        """

        path = os.path.join(checkpoint_dir, "checkpoint"+comment+'.pt')

        torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)

        return

    def load_checkpoint(self, checkpoint_dir, comment=''):
        """ Loads a model and optimizer from a checkpoint

        If model or optimizer is not set yet then the model or optimizer checkpoint is not loaded

        Parameters
        ----------
        checkpoint_dir : str
            Directory path to saved PyTorch checkpoint file
        comment : str
            Optional string appended to end of checkpoint filename.
            Otherwise, filename is just "checkpoint"

        """

        # When we load the checkpoint data we appearently have to first load it to cpu memory. 
        # The loaded model is already in GPU memory, so appearently, PyTorch handles the transfer.
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"+comment+'.pt'), map_location=torch.device('cpu'))
        if self.model is not None:
            self.model.load_state_dict(model_state)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer_state)

        return


    def avg_precision(self):
        """ Computes the average precision, as the weighted mean of precisions achieved at each threshold, 
        with the increase in recall from the previous threshold used as the weight.

        It is computed for both the training and test data sets.

        """

        raise Exception('Elliminate this redundant method')

        # Compute training and test precision
        with torch.no_grad():
            # We are NOT training so tell the model we are just evaluating
            self.model.eval()
            #***
            # Evaluate model on all training data
            train_labels = []
            train_scores = torch.tensor([]).to(device=self.device)
            for images, light_curves, features, labels in self.train_loader:
                for key in images.keys():
                    images[key] = images[key].to(device=self.device)
                for key in light_curves.keys():
                    light_curves[key] = light_curves[key].to(device=self.device)
                features = features.to(device=self.device)

                outputs, outputs_score = self.model(images, light_curves, features)
                train_labels.extend(labels.tolist())
                train_scores = torch.cat((train_scores, outputs_score))

            train_scores = train_scores[:,1].cpu()

            y_true = np.array(train_labels)[:,1]
            train_avg_precision = custom_average_precision_score(y_true, train_scores, self.avg_precision_recall_limit)

            #***
            # Evaluate model on all test data
            test_labels = []
            test_scores = torch.tensor([]).to(device=self.device)
            for images, light_curves, features, labels in self.test_loader:
                for key in images.keys():
                    images[key] = images[key].to(device=self.device)
                for key in light_curves.keys():
                    light_curves[key] = light_curves[key].to(device=self.device)
                features = features.to(device=self.device)

                outputs, outputs_score = self.model(images, light_curves, features)
                test_labels.extend(labels.tolist())
                test_scores = torch.cat((test_scores, outputs_score))

            test_scores = test_scores[:,1].cpu()

            y_true = np.array(test_labels)[:,1]
            test_avg_precision = custom_average_precision_score(y_true, test_scores, self.avg_precision_recall_limit)

        print('*****************')
        print('Train Avg Precision = {}'.format(self.train_precision_recall.avg_precision))
        print('Test Avg Precision = {}'.format(self.test_precision_recall.avg_precision))
        print('*****************')

        return 

    def compute_precision_recall_curves(self, 
            cross_val_train_PR_data=None, 
            cross_val_val_PR_data=None, 
            cross_val_test_PR_data=None):
        """ Computes the precision and recall curve data and also average precision.

        Parameters
        ----------
        cross_val_train_PR_data : list PrecisionRecallResults
            The precision recall curves found in HyperTune.cross_val_error_estimate
        cross_val_val_PR_data : list PrecisionRecallResults
        cross_val_test_PR_data : list PrecisionRecallResults

        Returns
        -------
        self.train_precision_recall : PrecisionRecallResults
        self.val_precision_recall : PrecisionRecallResults
        self.test_precision_recall : PrecisionRecallResults

        """

        # Free up as much memory as we can becuase the below code will accumulate GPU memory
        torch.cuda.empty_cache()

        # Compute training and test precision and recall
        with torch.no_grad():
            # We are NOT training so tell the model we are just evaluating
            self.model.eval()
            #***
            # Evaluate model on all training data
            self.train_precision_recall = PrecisionRecallResults()
            train_labels = np.empty((0,2))
            train_scores = np.empty((0,2))
            for images, light_curves, features, labels in self.train_loader:
                for key in images.keys():
                    images[key] = images[key].to(device=self.device)
                for key in light_curves.keys():
                    light_curves[key] = light_curves[key].to(device=self.device)
                features = features.to(device=self.device)

                outputs, outputs_score = self.model(images, light_curves, features)
                train_labels = np.concatenate((train_labels, labels.cpu().numpy()))
                train_scores = np.concatenate((train_scores, outputs_score.cpu().numpy()))

            train_scores = train_scores[:,1]

            train_y_true = train_labels[:,1]

            self.train_precision_recall.precision, self.train_precision_recall.recall, self.train_precision_recall.thresholds = \
                    skmetrics.precision_recall_curve(train_y_true, train_scores)
            self.train_precision_recall.avg_precision = custom_average_precision_score(train_y_true, train_scores, self.avg_precision_recall_limit)

            # Evaluate model on all validation data
            self.val_precision_recall = PrecisionRecallResults()
            if self.val_loader is not None:
                val_labels = np.empty((0,2))
                val_scores = np.empty((0,2))
                for images, light_curves, features, labels in self.val_loader:
                    for key in images.keys():
                        images[key] = images[key].to(device=self.device)
                    for key in light_curves.keys():
                        light_curves[key] = light_curves[key].to(device=self.device)
                    features = features.to(device=self.device)

                    outputs, outputs_score = self.model(images, light_curves, features)
                    val_labels = np.concatenate((val_labels, labels.cpu().numpy()))
                    val_scores = np.concatenate((val_scores, outputs_score.cpu().numpy()))
                
                val_scores = val_scores[:,1]
                
                val_y_true = val_labels[:,1]

                self.val_precision_recall.precision, self.val_precision_recall.recall, self.val_precision_recall.thresholds = \
                        skmetrics.precision_recall_curve(val_y_true, val_scores)
                self.val_precision_recall.avg_precision = custom_average_precision_score(val_y_true, val_scores, self.avg_precision_recall_limit)

            # Evaluate model on all test data
            self.test_precision_recall = PrecisionRecallResults()
            if self.test_loader is not None:
                test_labels = np.empty((0,2))
                test_scores = np.empty((0,2))
                for images, light_curves, features, labels in self.test_loader:
                    for key in images.keys():
                        images[key] = images[key].to(device=self.device)
                    for key in light_curves.keys():
                        light_curves[key] = light_curves[key].to(device=self.device)
                    features = features.to(device=self.device)

                    outputs, outputs_score = self.model(images, light_curves, features)
                    test_labels = np.concatenate((test_labels, labels.cpu().numpy()))
                    test_scores = np.concatenate((test_scores, outputs_score.cpu().numpy()))
                
                test_scores = test_scores[:,1]
                
                test_y_true = test_labels[:,1]

                self.test_precision_recall.precision, self.test_precision_recall.recall, self.test_precision_recall.thresholds = \
                        skmetrics.precision_recall_curve(test_y_true, test_scores)
                self.test_precision_recall.avg_precision = custom_average_precision_score(test_y_true, test_scores, self.avg_precision_recall_limit)
    
        # Compute the uncertainties, if available
        self.train_precision_recall.compute_uncertainties(cross_val_train_PR_data)
        self.val_precision_recall.compute_uncertainties(cross_val_val_PR_data)
        self.test_precision_recall.compute_uncertainties(cross_val_test_PR_data)

        self.precision_and_recall_computed = True

        # Clear cache one more time so that the memory accumulation in this code is freed.
        torch.cuda.empty_cache()

        return


    def plot_precision_recall_curves(self, 
            save_filename=None, 
            trial_checkpoint_metric_vals=None, 
            optim_metric=None,
            best_trial=None,
        best_checkpoint_dir=None):
        """ Plots the precision recall curves for the train, validation and test sets and also for the detection pipeline

        if precision_and_recall_computed is False then this will compute them

        Parameters
        ----------
        save_filename : str
            filename and path for saving the figure
            If None then do not save figure and just disply the figure.
        trial_checkpoint_metric_vals : dict (OPTIONAL)
            All trials and checkpoint optim_metrics values over all epochs        
            The keys are the trial_ids and the arrays are the optim_metric values for each checkpoint (i.e. epoch)
        optim_metric : str (OPTIONAL)
            What metric to use for schedular to optimize on.
            It computes the metric on the validation data.
        best_trial :  (OPTIONAL)
            The best trial foudn in Ray Tune
        best_checkpoint_dir : str (OPTIONAL)
            Path to the best checkpoint found during Ray Tune

        """

        legend_fontsize = 'x-small'

        if not self.precision_and_recall_computed:
            self.compute_precision_recall_curves()


        #************************
        # plot all curves
      # fig, ax = plt.subplots(3,1, figsize=(7, 9), sharex=True)
        fig, ax = plt.subplots(3,1, figsize=(7, 9))
        # If uncertainties are calculated then plot those on a third axis
        if self.test_precision_recall.cross_val_PR_data is not None:
            plot_errors = True
        else:
            plot_errors = False

        #***
        # Precision vs recall curve
        ax[0].plot(self.machine_train_precision_recall.recall, self.machine_train_precision_recall.precision, '-.m',
                label='Forest Train    Data, avg. prec {:.3f}'.format(self.machine_train_precision_recall.avg_precision))
        if self.val_loader is not None:
            ax[0].plot(self.machine_val_precision_recall.recall, self.machine_val_precision_recall.precision, ':m',
                    label='Forest Val      Data, avg. prec {:.3f}'.format(self.machine_val_precision_recall.avg_precision))
        if self.test_loader is not None:
            ax[0].plot(self.machine_test_precision_recall.recall, self.machine_test_precision_recall.precision, '-m',
                    label='Forest Test     Data, avg. prec {:.3f}'.format(self.machine_test_precision_recall.avg_precision))
        ax[0].plot(self.train_precision_recall.recall, self.train_precision_recall.precision, '-.b',
                label='CNN    Train    Data, avg. prec. {:.3f}'.format(self.train_precision_recall.avg_precision))
        if self.val_loader is not None:
            ax[0].plot(self.val_precision_recall.recall, self.val_precision_recall.precision, ':b',
                    label='CNN    Val      Data, avg. prec. {:.3f}'.format(self.val_precision_recall.avg_precision))
        if self.test_loader is not None:
            ax[0].plot(self.test_precision_recall.recall, self.test_precision_recall.precision, '-b',
                    label='CNN    Test     Data, avg. prec. {:.3f}'.format(self.test_precision_recall.avg_precision))

        ax[0].set_xlim(0.0, 1.0)
        ax[0].set_ylim(0.4, 1.0)
        ax[0].grid()
        ax[0].legend(fontsize=legend_fontsize)
        ax[0].set_title('Precision Recall Curves for Forest and CNN')
        ax[0].set_ylabel('Precision')
        ax[0].set_xlabel('Recall')

        #***
        """
        # Validation precision and recall vs threshold
        if self.val_loader is not None:
            ax[1].plot(self.machine_val_precision_recall.thresholds, self.machine_val_precision_recall.precision[:-1], '-m', label='Forest Val Precision Data')
            ax[1].plot(self.val_precision_recall.thresholds, self.val_precision_recall.precision[:-1], '-b', label='CNN Val Precision Data')
            
            ax[1].plot(self.machine_val_precision_recall.thresholds, self.machine_val_precision_recall.recall[:-1], '-.m', label='Forest Val Recall Data')
            ax[1].plot(self.val_precision_recall.thresholds, self.val_precision_recall.recall[:-1], '-.b', label='CNN Val Recall Data')

        ax[1].set_xlim(0.0, 1.0)
        ax[1].set_ylim(0.0, 1.0)
        ax[1].grid()
        ax[1].legend(fontsize=legend_fontsize)
        ax[1].set_title('Validation Precision and Recall vs Threshold')
       #ax[1].set_xlabel('Classifier Score Threshold')

        #***
        # Test precision and recall vs threshold
        if self.test_loader is not None:
            ax[2].plot(self.machine_test_precision_recall.thresholds, self.machine_test_precision_recall.precision[:-1], '-m', label='Forest Test Precision Data')
            ax[2].plot(self.test_precision_recall.thresholds, self.test_precision_recall.precision[:-1], '-b', label='CNN Test Precision Data')
            
            ax[2].plot(self.machine_test_precision_recall.thresholds, self.machine_test_precision_recall.recall[:-1], '-.m', label='Forest Test Recall Data')
            ax[2].plot(self.test_precision_recall.thresholds, self.test_precision_recall.recall[:-1], '-.b', label='CNN Test Recall Data')

        ax[2].set_xlim(0.0, 1.0)
        ax[2].set_ylim(0.0, 1.0)
        ax[2].grid()
        ax[2].legend(fontsize=legend_fontsize)
        ax[2].set_title('Test Precision and Recall vs Threshold')
       #ax[2].set_xlabel('Classifier Score Threshold')
        """

        # If uncertainty data is available, then plot that
        # If not available then plot optim_metric vs epoch
        if plot_errors: 
            # If the random forest uncertainty estimate curves are available then plot those first
            if self.random_forest_uncertainties_file is not None:
                # Load the random forest data
                with open(self.random_forest_uncertainties_file, 'rb') as fp:
                    RF_test_precision_recall = pickle.load(fp)
                    fp.close()

                # Compute the uncertainties using the current computation technique
                RF_test_precision_recall.compute_uncertainties()
            else:
                RF_test_precision_recall = None

            # Plot the full zoomed-out curve
            self._plot_uncertainty_curves(ax[1], RF_test_precision_recall, 0.0)

            # Plot zoomed-in uncertainties just near the top.
            self._plot_uncertainty_curves(ax[2], RF_test_precision_recall, 0.875)

        elif trial_checkpoint_metric_vals is not None:
            assert optim_metric is not None, "If trial_checkpoint_metric_vals then optim_metric must als be passed"
            assert best_trial is not None, "If trial_checkpoint_metric_vals then best_trial must als be passed"
            assert best_checkpoint_dir is not None, "If trial_checkpoint_metric_vals then best_checkpoint_dir must als be passed"

            #best_trial.trial_id example = a75ff_00000
            best_checkpoint_idx = int(best_checkpoint_dir[-6:])
            best_trial_optim_metric_vals = trial_checkpoint_metric_vals[best_trial.trial_id]
            min_loss_plot_val = 0.01
            max_loss_plot_val = 100

            # Axis 1 is all trials
            for trial,vals in trial_checkpoint_metric_vals.items():
                ax[1].semilogy(vals, '-', linewidth=0.5, label=None)
            ax[1].semilogy(best_trial_optim_metric_vals, '-b.', linewidth=3, label='Best Trial')
            ax[1].semilogy(best_checkpoint_idx, best_trial_optim_metric_vals[best_checkpoint_idx], 'ro', label='Best Epoch')
            ax[1].set_ylim(min_loss_plot_val, max_loss_plot_val)
            ax[1].grid()
            ax[1].set_title('All trials {} values vs. Epoch'.format(optim_metric))
            ax[1].set_ylabel('{}'.format(optim_metric))
            ax[1].set_xlabel('Epoch')

            # Axis 2 is the best trial
            ax[2].semilogy(best_trial_optim_metric_vals, '-b.', linewidth=3, label='Best Trial = {}'.format(best_trial.trial_id))
            ax[2].semilogy(best_checkpoint_idx, best_trial_optim_metric_vals[best_checkpoint_idx], 'ro', label='Best Epoch = {}'.format(best_checkpoint_idx))
            ax[2].legend(fontsize=legend_fontsize)
           #ax[2].set_ylim(min_loss_plot_val, max_loss_plot_val)
            ax[2].grid()
            ax[2].set_title('Best trial {} values vs. Epoch'.format(optim_metric))
            ax[2].set_ylabel('{}'.format(optim_metric))
            ax[2].set_xlabel('Epoch')

        # Make the plot pretty
        plt.tight_layout(pad=0.05)

        if save_filename is not None:
            fig.savefig(save_filename, dpi=150);
            plt.close(fig)
            print('')
            print('Best model PR curve saved as {}'.format(save_filename))
        else:
            plt.show()

        return

    def _plot_uncertainty_curves(self, ax, RF_test_precision_recall, ylim_low):
        """
        This will plot the precision and recall uncertainty curves. We use a help function because we plot this twice,
        once full scale and another just zoomed into the top portion of the axis.
        """
        legend_fontsize = 'x-small'

        if RF_test_precision_recall is not None:

            ax.fill_between(RF_test_precision_recall.err_thresholds, RF_test_precision_recall.prec_err_low, RF_test_precision_recall.prec_err_high, 
                    alpha=0.2, color='m', linestyle='-')
            ax.fill_between(RF_test_precision_recall.err_thresholds, RF_test_precision_recall.recall_err_low, RF_test_precision_recall.recall_err_high, 
                    alpha=0.2, color='m', linestyle='-')
            # Plot the individual sample curves very lightly (low alpha)
            for PR_data in RF_test_precision_recall.cross_val_PR_data:
                ax.plot(PR_data.thresholds, PR_data.precision[:-1], alpha=0.02, c='m', ls='-')
                ax.plot(PR_data.thresholds, PR_data.recall[:-1], alpha=0.02, c='m', ls='-.')
            # Plot the mean curves
            ax.plot(RF_test_precision_recall.err_thresholds, RF_test_precision_recall.prec_err_mean, alpha=1.0, c='m', ls='-', label='Forest Test Precision Uncertainties')
            ax.plot(RF_test_precision_recall.err_thresholds, RF_test_precision_recall.recall_err_mean, alpha=1.0, c='m', ls='-.', label='Forest Test Recall Uncertainties')

        #***
        # The CNN uncertainty estimate
        ax.fill_between(self.test_precision_recall.err_thresholds, self.test_precision_recall.prec_err_low, self.test_precision_recall.prec_err_high, 
            alpha=0.2, color='blue', linestyle='-')
        ax.fill_between(self.test_precision_recall.err_thresholds, self.test_precision_recall.recall_err_low, self.test_precision_recall.recall_err_high, 
            alpha=0.2, color='blue', linestyle='-')
        # Plot the individual sample curves very lightly (low alpha)
        for PR_data in self.test_precision_recall.cross_val_PR_data:
            ax.plot(PR_data.thresholds, PR_data.precision[:-1], alpha=0.08, c='blue', ls='-')
            ax.plot(PR_data.thresholds, PR_data.recall[:-1], alpha=0.08, c='blue', ls='-.')
        # Plot the mean curves
        ax.plot(self.test_precision_recall.err_thresholds, self.test_precision_recall.prec_err_mean, alpha=1.0, c='blue', ls='-', label='Test Precision Uncertainties')
        ax.plot(self.test_precision_recall.err_thresholds, self.test_precision_recall.recall_err_mean, alpha=1.0, c='blue', ls='-.', label='Test Recall Uncertainties')

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(ylim_low, 1.0)
        ax.grid()
        ax.legend(fontsize=legend_fontsize)
        ax.set_title('Test Precision and Recall Middle 90% Uncertainty Estimates')
        ax.set_xlabel('Classifier Score Threshold')

        return


    def _extract_machine_opinions(self):
        """ 
        Then extracts the machine opinions for all IDs in each data set

        """

        # Compute detection pipeline precision and recall
        self.machine_train_precision_recall = PrecisionRecallResults()
        self.machine_train_precision_recall.avg_precision = custom_average_precision_score(self.data_train.bolide_truth, self.data_train.machine_opinions, self.avg_precision_recall_limit)
        self.machine_train_precision_recall.precision, self.machine_train_precision_recall.recall, self.machine_train_precision_recall.thresholds = \
                skmetrics.precision_recall_curve(self.data_train.bolide_truth, self.data_train.machine_opinions)

        self.machine_val_precision_recall = PrecisionRecallResults()
        if self.val_loader is not None:
            self.machine_val_precision_recall.avg_precision = custom_average_precision_score(self.data_val.bolide_truth, self.data_val.machine_opinions, self.avg_precision_recall_limit)
            self.machine_val_precision_recall.precision, self.machine_val_precision_recall.recall, self.machine_val_precision_recall.thresholds = \
                    skmetrics.precision_recall_curve(self.data_val.bolide_truth, self.data_val.machine_opinions)

        self.machine_test_precision_recall = PrecisionRecallResults()
        if self.test_loader is not None:
            self.machine_test_precision_recall.avg_precision = custom_average_precision_score(self.data_test.bolide_truth, self.data_test.machine_opinions, self.avg_precision_recall_limit)
            self.machine_test_precision_recall.precision, self.machine_test_precision_recall.recall, self.machine_test_precision_recall.thresholds = \
                    skmetrics.precision_recall_curve(self.data_test.bolide_truth, self.data_test.machine_opinions)

        return

    def print_machine_opinions(self):
        """
        Prints the training and test set average precision for the machine opinions
        """
        print('*****************')
        print('Detection Pipeline Train Avg Precision = {:.4f}'.format(self.machine_train_precision_recall.avg_precision))
        print('Detection Pipeline Val Avg Precision = {:.4f}'.format(self.machine_val_precision_recall.avg_precision))
        print('Detection Pipeline Test Avg Precision = {:.4f}'.format(self.machine_test_precision_recall.avg_precision))
        print('*****************')
        
        return

    def print_model_opinions(self):
        """
        Prints the training and test set average precision for the fitted model
        """
        print('*****************')
        print('Best Model Train Avg Precision = {:.4f}'.format(self.train_precision_recall.avg_precision))
        print('Best Model Val Avg Precision = {:.4f}'.format(self.val_precision_recall.avg_precision))
        print('Best Model Test Avg Precision = {:.4f}'.format(self.test_precision_recall.avg_precision))
        print('*****************')
        
        return

    # End CnnTrainer

#*************************************************************************************************************
class CnnEvaluator:
    """ This class will load a pre-trained classifer and has methods to evaluate it on data.
    """

    # This batch_size is not really important, just pick something small enough to fit in memory
    batch_size = 64

    def __init__(self, 
            model_path, 
            bolide_dispositions_object, 
            bolidesFromWebsite=None,
            image_cache_path='/tmp/cnn_image_cache',
            gpu_index=None, 
            force_cpu=False,
            verbosity=True
            ):
        """
        Parameters
        ----------
        model_path : str
            filename with full path to saved PyTorch model file generated by
            cnnTrainer.save_best_checkpoint_and_model
        bolide_dispositions_object : bolide_dispositions.BolideDispositions
            The bolide dispositions object generated from the pipeline data set
        bolidesFromWebsite : [WebsiteBolideEvent list] created by:
            bolide_dispositions.pull_dispositions_from_website
            If passed then use this data to determine truth.
        image_cache_path : str
            Path to where to save the preprocessed images converted to tensors
        gpu_index : int
            Specify the GPU index to use for processing
            Only useful for a multi-GPU machine
            If None then use default for CUDA
            If 'cpu' then use the CPU
        force_cpu : bool
            If True then force the use of the CPU for PyTorch, even if GPUs are available 
        verbosity : bool
            If True then display tqdm status bar

        """

        # The self.cached_filenames below should have the absolute path to the ramdisk data
        RAMDISK_PATH = ''

        cached_filenames = None

        if force_cpu:
            self.device = 'cpu'
            self.gpu_index = None
            if verbosity: print('Using CPU')
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if gpu_index is None:
                self.gpu_index = torch.cuda.current_device()
            else:
                self.gpu_index = gpu_index
            if self.device == 'cuda':
                if verbosity: print('Using CUDA')
                # Add in GPU index
                self.device = self.device + ':{}'.format(self.gpu_index)
            else: 
                if verbosity: print('Using CPU')

        self.load_model(model_path)

        self.bDispObj = copy.deepcopy(bolide_dispositions_object)

        if bolidesFromWebsite is not None:
            beliefSource = 'human'
        else:
            beliefSource = 'machine'

        #******
        # Cache image tensors
        if image_cache_path is not None:
            self.image_cache_path = image_cache_path
            # Do not scale the images, use the native size
            self.transformer = transforms.ToTensor()
            self.cached_filenames, self.bDispObj = cache_image_tensors_to_file(self.bDispObj,
                    save_path=self.image_cache_path, transform=self.transformer, verbosity=verbosity)
        elif isinstance(cached_filenames, dict):
            raise Exception('This is not set up yet')
            self.cached_filenames = cached_filenames
        else:
            self.cached_filenames = None
        
        # If there are no cached images then there is nothing to evaluate
        if self.cached_filenames == None:
            self.init_success = False
        else:
            #******
            # Pre-process GLM light curves
            if self.GLM_LC_data_length is not None:
                if USE_FIXED_TIME_LC:
                    self.cached_lightcurves = CNN_1D_block_batchNorm.pre_process_GLM_light_curves_const_time(self.bDispObj, self.GLM_LC_data_length, verbosity=verbosity)
                else:
                    raise Exception('This option is not currently available')
                    self.cached_lightcurves = CNN_1D_block_batchNorm.pre_process_GLM_light_curves(self.bDispObj, self.GLM_LC_data_length)
            else:
                self.cached_lightcurves = None
            
            
            # Load the data into the PyTorch tensor format
            # We do not have the truth data
            self.data = glmDataSet(self.bDispObj, self.bDispObj.IDs, augmentations=None, 
                                            beliefSource=beliefSource, bolidesFromWebsite=bolidesFromWebsite, featuresToUse=self.extra_feature_names,
                                            columnTransformer=self.columnTransformer, verbosity=True, 
                                            cached_filenames=self.cached_filenames, 
                                            cached_lightcurves=self.cached_lightcurves,
                                            use_multi_band_data=self.use_multi_band_data, bands_to_use=self.bands_to_use,
                                            balance_data=False)
            # We want to keep the order of the candidates, so set shuffle=False
            if self.device == 'cpu':
                self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=False,
                    pin_memory=False)
            else:
                self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=False,
                    pin_memory=True, pin_memory_device=self.device)

            self.init_success = True


    def load_model(self, model_path):
        """ Loads a model from a checkpoint

        Parameters
        ----------
        model_path : str
            filename with full path to saved PyTorch model file generated by
            cnnTrainer.save_best_checkpoint_and_model

        """

        self.model, self.columnTransformer, self.extra_feature_names, self.use_multi_band_data, self.bands_to_use, self.GLM_LC_data_length = \
                torch.load(model_path, map_location=torch.device('cpu'))
        self.model.to(device=self.device)

        return


    def evaluate(self, IDs):
        """ Evaluates the model on the given Bolide IDs

        Parameters
        ----------
        IDs : int64 list
            List of detectoion IDs to evaluate the model on
            or pass 'all' to evaluate on all

        Returns
        all_scores : float torch.tensor
            Returns a softmax sigmoid confidence score in the range [0,1]
        all_labels : bool torch.tensor shape(:,2)
            The labels returned by the data loader. If bolidesFromWebsite was passed when constructing this object, then
            that is used to determine the label. Otherwise, it's the machine score in the disposition object
        IDs : int64 list
            List of detection IDs evaluated

        """

        if IDs == 'all':
            IDs = self.bDispObj.IDs
        else:
            raise Exception("Only works on 'all' IDs")

        with torch.no_grad():
            # We are NOT training so tell the model we are just evaluating
            self.model.eval()
            
            # Evaluate model on all data
            all_labels = torch.tensor([])
            all_scores = torch.tensor([]).to(device=self.device)
            for images, light_curves, features, labels in self.data_loader:
                for key in images.keys():
                    images[key] = images[key].to(device=self.device)
                for key in light_curves.keys():
                    light_curves[key] = light_curves[key].to(device=self.device)
                features = features.to(device=self.device)
            
                _, outputs_score = self.model(images, light_curves, features)
                all_labels = torch.cat((all_labels, labels))
                all_scores = torch.cat((all_scores, outputs_score))
            
            self.scores = all_scores[:,1].cpu().numpy()
            self.labels = all_labels.numpy()
        
        return all_scores, all_labels, IDs

    def populate_validation_scores_in_bolideDetectionList(self, validation_input_config, bolideDetectionList):
        """
        When the CnnEvaluator is constructed is will remove bolide detection candidates that do not have valid data, so we
        need to match up the detection IDs when populating the validation scores.
        Non-evaluated detection validation scores are set to None
        """
        for detection in bolideDetectionList:
            detection.assessment.validation.low_threshold = validation_input_config.validation_low_threshold
            detection.assessment.validation.high_threshold = validation_input_config.validation_high_threshold

            detection.assessment.validation.method = os.path.basename(validation_input_config.validation_model_path)

            idx = np.nonzero(self.bDispObj.IDs == detection.ID)[0]
            if len(idx) == 1:
                detection.assessment.validation.score = self.scores[idx[0]]

                # Force candidacy if either the triaged forced candidacy or the validation config says we should
                if validation_input_config.min_num_groups_to_force_candidacy >= 0:
                    detection.assessment.validation.candidacy_forced = \
                        (detection.assessment.triage.candidacy_forced or
                            detection.features.nGroups >= validation_input_config.min_num_groups_to_force_candidacy)
                else:
                    detection.assessment.validation.candidacy_forced = detection.assessment.triage.candidacy_forced


            elif len(idx) > 1:
                raise Exception('Found multiple detection ID matches')
            else:
                detection.assessment.validation.score = None

        return bolideDetectionList

    def delete_image_pickle_files(self):
        """ This will delete the pickle files located at 

        self.bDispObj.bolideDispositionProfileList[:].cutoutFeatures.image_pickle_filename

        They are laergish files and we might not not want to use so much data storage after the evaluation has finished.

        """

        for bolide in self.bDispObj.bolideDispositionProfileList:
            if os.path.exists(bolide.cutoutFeatures.image_pickle_filename):
                os.remove(bolide.cutoutFeatures.image_pickle_filename)

        return

#*************************************************************************************************************
# Performance assessment tools

def assess_false_positives(
        model_path, 
        database_files, 
        bolidesFromWebsitePath, 
        score_threshold=0.5, 
        n_rand_to_show=None,
        outputPath=None):
    ''' 
    This tool is used to examine all the false positives for a trained classifier. 
    It will assess if a human agrees they are false positives.

    Parameters
    ----------
    model_path : str
        filename with full path to saved PyTorch model file generated by
        cnnTrainer.save_best_checkpoint_and_model
    database_files : str or list of str
        Path to the database file, or files, to load data from
        All database files must be from the same top level run
        Meaning, the same top level path with subdirectories like /G17/2022/1115/
    bolidesFromWebsitePath : str
        Path to the bolidesFromWebsite pickle file
    score_threshold : float
        Classifier confidence score threshold to be consdiered a bolide
    n_rand_to_show : int
        Maximum number of random detections to show
        If None then show all
    outputPath : str
        Path to where to save any selected validation reports

    Returns
    -------
    bolide_dispositions : HumanOpinion list
        The dispositions of the human viewing the validation reports

    '''

    if isinstance(database_files, str):
        database_files = [database_files]

    bDispObj = bDisp.create_BolideDispositions_from_multiple_database_files(database_files, copyOverBolideDetection=True, verbosity=True, useRamDisk=False)

    bolidesFromWebsite = bd.unpickle_bolide_detections(bolidesFromWebsitePath)

    # Create a CnnEvaluator object and load the model
    cnnEvaluator = CnnEvaluator(model_path, bDispObj, bolidesFromWebsite=bolidesFromWebsite)

    # Evaluate all candidates
    print('Evaluating model on all images...')
    all_scores, all_labels = cnnEvaluator.evaluate('all')
    print('Done evaluating all images.')
        
    # Identify all false positives above threshold
    # The first column int he all_labels truth array is if it's not a bolide; we want to find the candidates with high
    # scores that the truth says is not a bolide.
    FPs = np.nonzero(np.logical_and(all_scores > score_threshold, all_labels[:,0]))

    FP_IDs = bDispObj.IDs[FPs]
    FP_scores = np.array(all_scores[FPs].flatten())

    print('Number of false positives: {}'.format(len(FP_IDs)))
    if n_rand_to_show is not None:
        # Pick random detectison to show
        rng = np.random.default_rng()
        selected_idx = rng.permutation(len(FP_IDs))[0:n_rand_to_show]
        FP_IDs = FP_IDs[selected_idx]
        FP_scores = FP_scores[selected_idx]
        print('Showing {} random FPs'.format(n_rand_to_show))
    else:
        print('Showing all {} FPs'.format(len(FP_IDs)))

    bolideDetectionList = [disp.bolideDetection for disp in bDispObj.bolideDispositionProfileList]
    # We want the figure top path, which is above the directory which contains the database file
    detectionFigureTopPath = os.path.dirname(os.path.dirname(database_files[0]))

    bolide_dispositions = display_report_for_these_IDs(bolideDetectionList, detectionFigureTopPath, IDs=FP_IDs,
            scores=FP_scores, outputPath=outputPath, disposition_options=('TP', 'FP', 'UNC'))

    # Summary statistics
    dispositions = [opinion.disposition for opinion in bolide_dispositions]
    num_TP = dispositions.count('TP')
    num_FP = dispositions.count('FP')
    num_UNC = dispositions.count('UNC')

    print('Number True  Positives: {}'.format(num_TP))
    print('Number False Positives: {}'.format(num_FP))
    print('Number Uncertain      : {}'.format(num_UNC))

    # Save bolide_opinions
    print('****')
    bolide_dispositions_file = os.path.join(outputPath, 'bolide_dispositions.p')
    print('Saving bolide dispositions to file {}'.format(bolide_dispositions_file))
    with open(bolide_dispositions_file, 'wb') as fp:
        pickle.dump(bolide_dispositions, fp)
    
    # Check that all were disposed TP, FP or UNC
    num_errors = len(dispositions) - num_TP - num_FP - num_UNC
    if num_errors != 0:
        print('*************')
        print('*************')
        print('ERROR: {} were not disposed as TP, FP or UNC'.format(num_errors))
        print('*************')
        print('*************')


    return

#*************************************************************************************************************
# Hepler functions

def recall_at_precision_score(y_true, y_score, target_precision=0.98, method='max'):
    """ Returns the recall at the specified target_precision. 

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,) or (n_samples, n_classes)
        True binary labels or binary label indicators.

    y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by :term:`decision_function` on some classifiers).

    target_precision : float
        Target precision to compute the recall

    method : str
        Method to pick between multiple points with target_precision
        'min': Take the minumum recall
        'max': Take the maximum recall

    Returns
    -------
    recall_at_prec : float
        Recall value at target precision
    """

    assert target_precision >= 0.0 and target_precision <= 1.0, 'Precision must be in the range [0.0, 1.0]' 

    precision, recall, thresholds = skmetrics.precision_recall_curve(y_true, y_score)
    
    # Ignore last precision point, which is always 1.0
    precision = precision[:-1]
    recall = recall[:-1]

    # Find all points where precision crosses (up or down) target_precision
    precision_diff = precision - target_precision
    cross_indices_up = [idx for idx in np.arange(1,len(precision_diff)) if precision_diff[idx-1] < 0.0 and precision_diff[idx] >= 0.0]
    cross_indices_down = [idx for idx in np.arange(1,len(precision_diff)) if precision_diff[idx-1] > 0.0 and precision_diff[idx] <= 0.0]
    all_cross_indices = np.array(cross_indices_up + cross_indices_down)

    # If no cross points then return a recall of 0.0
    if len(all_cross_indices) == 0:
        return 0.0
        
    # Find recall for all these points as the average for before and after each cross (linearly interpolate)
    target_recalls = (recall[all_cross_indices-1] + recall[all_cross_indices]) / 2.0

    if method == 'min':
        # Take the minimum recall of all the cross points
        recall_at_prec = np.min(target_recalls)
    elif method == 'max':
        # Take the minimum recall of all the cross points
        recall_at_prec = np.max(target_recalls)
    else:
        raise Exception('Unkown method')

    return recall_at_prec

def calc_fbeta_score(y_true, y_score, threshold=0.5, beta=1.0):
    """ Returns an fbeta score at the specified threshold and beta.

    See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score

    If beta = 1.0 then this returns the f1_score.

    Beta < 1.0 lends more weight to precision.


    """

    # Everything above threshold is declared True
    y_pred = y_score >= threshold

    fbeta_score = skmetrics.fbeta_score(y_true, y_pred, beta=beta)
   
    return fbeta_score

def custom_average_precision_score(
    y_true, y_score, recall_limit=0.0, *, average="macro", pos_label=1, sample_weight=None
):
    """Compute average precision (AP) from prediction scores.

    Modified from Scikit-learn's average_precision_score.

    AP summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight:

    .. math::
        \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n` and :math:`R_n` are the precision and recall at the nth
    threshold [1]_. This implementation is not interpolated and is different
    from computing the area under the precision-recall curve with the
    trapezoidal rule, which uses linear interpolation and can be too
    optimistic.

    Note: this implementation is restricted to the binary classification task
    or multilabel classification task.

    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,) or (n_samples, n_classes)
        True binary labels or binary label indicators.

    y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by :term:`decision_function` on some classifiers).

    recall_limit : float
        The minimum recall to use when integrating.

    average : {'micro', 'samples', 'weighted', 'macro'} or None, \
            default='macro'
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    pos_label : int or str, default=1
        The label of the positive class. Only applied to binary ``y_true``.
        For multilabel-indicator ``y_true``, ``pos_label`` is fixed to 1.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    average_precision : float
        Average precision score.

    See Also
    --------
    roc_auc_score : Compute the area under the ROC curve.
    precision_recall_curve : Compute precision-recall pairs for different
        score thresholds.

    Notes
    -----
    .. versionchanged:: 0.19
      Instead of linearly interpolating between operating points, precisions
      are weighted by the change in recall since the last operating point.

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import average_precision_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> average_precision_score(y_true, y_scores)
    0.83...
    """
    assert recall_limit >= 0.0 and recall_limit < 1.0, 'Recall limit must be in the range [0.0, 1.0)'

    def _binary_uninterpolated_average_precision(
        y_true, y_score, pos_label=1, sample_weight=None
    ):
        precision, recall, _ = skmetrics.precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
        )
        # Find the starting recall index to use
        # Recall goes from 1.0 to 0.0
        # Recall decreases from 1.0 to 0.0. So, the argmin of recall_limit is the maximimum index.
        max_idx = np.argmin(np.abs(recall - recall_limit))
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        integral = -np.sum(np.diff(recall[:max_idx]) * np.array(precision[:max_idx])[:-1])
        # Sice we are no longer taking the integral over the entire range, 
        # we must normalize by the fraction of the curve we are integrating over.
        return integral / (1.0-recall_limit)

    y_type = type_of_target(y_true, input_name="y_true")
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError(
            "Parameter pos_label is fixed to 1 for "
            "multilabel-indicator y_true. Do not set "
            "pos_label or set pos_label to 1."
        )
    elif y_type == "binary":
        # Convert to Python primitive type to avoid NumPy type / Python str
        # comparison. See https://github.com/numpy/numpy/issues/6784
        present_labels = np.unique(y_true).tolist()
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
    average_precision = partial(
        _binary_uninterpolated_average_precision, pos_label=pos_label
    )
    return _average_binary_score(
        average_precision, y_true, y_score, average, sample_weight=sample_weight
    )

