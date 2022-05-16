#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 06:02:56 2022

@author: activexdiamond
"""

############################## Dependencies ##############################
##Path utils for image loading.
import os

from glob import glob
from pathlib import Path

##Maths, data conversion and image manipulation.
import pandas
import cv2

import random
import numpy
from sklearn.model_selection import train_test_split

##Visualization
from matplotlib import pyplot

############################## Custom Modules ##############################
import skull_stripping
import debugging

############################## Constants ##############################
##File paths.
IMAGE_RELATIVE_PATH = "/input/head_ct/images/*.png"
LABEL_RELATIVE_PATH = "/input/labels.csv"

##Preprocessing configs.
GRAYING_MODE = cv2.COLOR_BGR2GRAY

IMAGE_W = 128
IMAGE_H = 128
IMAGE_SIZE = (IMAGE_W, IMAGE_H)

SKLEARN_SHUFFLE_SEED = 42
RANDOM_SEED = 1337
TEST_RATIO = 0.2                    #How much of the dataset to use for testing. float ; [0, 1]
VALIDATION_RATIO = 0.5              #How much of the trainning dataset to use for validation. float ; [0, 1]

##Plotting and display configs.
PLOT_IMAGE_FIGURE_SIZE = (10, 10)
PLOTTED_IMAGE_COUNT = 9                #The number of raw-images to plot. int ; [0, 9]
PLOT_IMAGE_OFFSET = 0

PLOT_RATIO_BARS_H = 110

SKULL_STRIPS_TO_DISPLAY = 0

############################## Main Function ##############################
def process_dataset():
    ############################## Init State
    random.seed(RANDOM_SEED)

    ############################## Preprocessing - Fetch Image Paths 
    ##Fetch working dir.
    current_working_dir = os.getcwd()
    
    ##Fetch and sort images paths.
    image_paths = glob(current_working_dir + IMAGE_RELATIVE_PATH)
    image_paths = sorted(image_paths)
    dataset_len = len(image_paths)
    
    ##Fetch metadata and extract labels.
    metadata = pandas.read_csv(current_working_dir + LABEL_RELATIVE_PATH)
    labels = metadata[" hemorrhage"].tolist()
    
    ############################## Preprocessing - Loading & Transformations 
    image_shape = (len(image_paths), IMAGE_W, IMAGE_H)
    #raw_images = numpy.empty(image_shape)
    
    d_raw_images = []
    d_brain_images = []
    d_gray_images = []
    images = numpy.empty(image_shape)
    
    for i, path in enumerate(image_paths):
        ##Load image.
        image = cv2.imread(path)
        d_raw_images.append(image)
        ##Strip skull.
        print(f"Stripping skull from image at index: {i}")
        #The third param should be True if you wish to display plots to the user during processing.
        image = skull_stripping.strip_skull(image, True, i < SKULL_STRIPS_TO_DISPLAY)            
        d_brain_images.append(image)
        ##Convert to gray-scale.
        image = cv2.cvtColor(image, GRAYING_MODE)
        d_gray_images.append(image)
        ##Crop and resize
        images[i, :, :] = cv2.resize(image, (IMAGE_SIZE))
    print()
    
    ############################## Debug-Plot Images
    ##Display the images as matplotlib plots, to get an idea of what we're working with.
    ##This is the main purpose of the d_<x>_images arrays.
    debugging.image_plotter(images, labels, PLOT_IMAGE_FIGURE_SIZE, PLOTTED_IMAGE_COUNT,
                            "Final Result", PLOT_IMAGE_OFFSET)
    debugging.image_plotter(d_raw_images, labels, PLOT_IMAGE_FIGURE_SIZE, PLOTTED_IMAGE_COUNT,
                            "Raw", PLOT_IMAGE_OFFSET)
    debugging.image_plotter(d_brain_images, labels, PLOT_IMAGE_FIGURE_SIZE, PLOTTED_IMAGE_COUNT,
                            "Skull-Stripped", PLOT_IMAGE_OFFSET)
    debugging.image_plotter(d_gray_images, labels, PLOT_IMAGE_FIGURE_SIZE, PLOTTED_IMAGE_COUNT,
                            "Grayed", PLOT_IMAGE_OFFSET)
    
        
    ##Plot the proportion of normal and abnormal labels.
    pyplot.bar(labels, PLOT_RATIO_BARS_H)
    
    ############################## Preprocessing - Splits
    ##Shuffle then split our dataset into train/test/validation.
    train_images, test_images, train_labels, test_labels = train_test_split(
            images,
            labels,
            test_size=TEST_RATIO,
            random_state=SKLEARN_SHUFFLE_SEED)
    
    val_images, test_images, val_labels, test_labels = train_test_split(
            test_images,
            test_labels,
            test_size=VALIDATION_RATIO,
            random_state=SKLEARN_SHUFFLE_SEED)
    
    ############################## Debug-Echo Splits
    ##Some debug echoing of the used splits.
    print("Using the following splits:")
    print("\tTRAIN\tTEST\t\tVALIDATE")
    str = "\t{}\t\t{}\t\t{}\t\t\t(images)".format(len(train_images), len(val_images), len(test_images))
    print(str)
    str = "\t{}%\t{}%\t{}%\t\t(of the dataset)".format(
            len(train_images) / dataset_len * 100,
            len(val_images) / dataset_len * 100,
            len(test_images) / dataset_len * 100)
    print(str)
    
    #train (2), test (2), valid (2), debug (4)
    return (train_images, train_labels), (test_images, test_labels), (val_images, val_labels), (images, d_raw_images, d_brain_images, d_gray_images)
    
    
    
    
    