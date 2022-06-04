#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 19:29:26 2022
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
##Path utils for image loading.
import os
from glob import glob
from pathlib import Path

##Maths
import math
import numpy

##RNG
import random

## Data serialization and deserialization
import pandas
import deepdish

##Image manipulation.
import cv2
import skimage

##Dataset Mangement
from sklearn.model_selection import train_test_split

##Visualization
from matplotlib import pyplot

############################## Custom Modules ##############################
import preprocessing
import sfta
import debugging 

############################## Config ##############################
import config
import debug_config

def main():
    raw_images = preprocessing.load_images(config.IMAGE_RELATIVE_PATH)
    labels = preprocessing.load_labels(config.LABEL_RELATIVE_PATH)
    
    dataset_len = len(raw_images)
    image_shape = (dataset_len, config.IMAGE_W, config.IMAGE_H)

    d_raw_images = []
    d_brain_images = []
    d_gray_images = []
    images = numpy.empty(image_shape)
    features = []
    
    #for i in range(3):
    #    image = raw_images[i]
    for i, image in enumerate(raw_images):
        print(f"Stripping skull from image at index: {i}")
        #The third param should be True if you wish to display plots to the user during processing.
        image = preprocessing.strip_skull(image, True, i < debug_config.SKULL_STRIPS_TO_DISPLAY, i)            
        ##Convert to gray-scale.
        image = cv2.cvtColor(image, config.FINAL_GRAYING_MODE)
        ##Crop and resize.
        image = cv2.resize(image, (config.IMAGE_SIZE))
        
        feat = {}
        ## LBP
        feat["lbp"] = preprocessing.compute_lbp(image)
        
        ## HOG
        feat["hog"], feat["hog_image"] = skimage.feature.hog(image, visualize=True)
        
        ## SFTA
        feat["sfta"] = sfta.compute_sfta(image, 5)
        
        ## Third Order - GLCM
        feat["glcm"] = preprocessing.compute_glcm_stats(image)
        
        ## First Order - Stats    
        feat["fo"] = preprocessing.compute_fo_all(image)        
        
        ##Debug 
        debugging.plot_features(i, image, feat)
        
        ##Final step.
        features.append(feat)
        images[i, :, :] = image

    deepdish.io.save("output.h5", features)
    
    ############################## Preprocessing - Splits
    ##Shuffle then split our dataset into train/test/validation.
    train_images, test_images, train_labels, test_labels = train_test_split(
            images,
            labels,
            test_size=config.TEST_RATIO,
            random_state=config.SKLEARN_SHUFFLE_SEED)
    
    val_images, test_images, val_labels, test_labels = train_test_split(
            test_images,
            test_labels,
            test_size=config.VALIDATION_RATIO,
            random_state=config.SKLEARN_SHUFFLE_SEED)
    
    ############################## Debug-Plot Images
    ##Plot the proportion of normal and abnormal labels.
    pyplot.bar(labels, debug_config.PLOT_RATIO_BARS_H)
    
    ##Display the images as matplotlib plots, to get an idea of what we're working with.
    ##This is the main purpose of the d_<x>_images arrays.
    debugging.image_plotter(images, labels, debug_config.PLOT_IMAGE_FIGURE_SIZE, debug_config.PLOTTED_IMAGE_COUNT,
                            "Final Result", debug_config.PLOT_IMAGE_OFFSET)
    #debugging.image_plotter(d_raw_images, labels, debug_config.PLOT_IMAGE_FIGURE_SIZE, debug_config.PLOTTED_IMAGE_COUNT,
    #                        "Raw", debug_config.PLOT_IMAGE_OFFSET)
    #debugging.image_plotter(d_brain_images, labels, debug_config.PLOT_IMAGE_FIGURE_SIZE, debug_config.PLOTTED_IMAGE_COUNT,
    #                        "Skull-Stripped", debug_config.PLOT_IMAGE_OFFSET)
    #debugging.image_plotter(d_gray_images, labels, debug_config.PLOT_IMAGE_FIGURE_SIZE, debug_config.PLOTTED_IMAGE_COUNT,
    #                        "Grayed", debug_config.PLOT_IMAGE_OFFSET)
    
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
    
    
if __name__ == '__main__':
    main()