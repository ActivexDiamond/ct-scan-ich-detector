#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:19:31 2022
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
##Pathing.
import os

##Image manipulation and math.
import cv2
import numpy

import skimage.feature
from PIL import Image, ImageStat

##Visualization.
from matplotlib import pyplot

##Debugging - Timing
import timeit

############################## Custom Modules ##############################
from preprocessing import process_dataset
import skull_stripping
import sfta

############################## Config ##############################
import config

############################## Constants ##############################


############################## Debug Constants ##############################

##Debug view
PLOT_COLUMNS = 3
PLOT_ROWS = 5
PLOT_SIZE = PLOT_COLUMNS * PLOT_ROWS
FIG_SIZE = (20, 20)

##Debug image
DEBUG_PATH = os.getcwd() + "/input/head_ct/images/001.png"

debug = True

############################## Debug Helpers ##############################

    

    
#TODO: Refactor into a feature extraction-specific function. Implement overall pipeline elsewhere.
def main():
    """
    I.C.H. detector entry point.
    """
    ############################## Execute Preprocessing Pipeline ##############################    
    #processed_data = process_dataset()
    #(train_images, train_labels) = processed_data[0]
    #(test_images, test_labels) = processed_data[1]
    #(val_images, val_labels) = processed_data[2]
    #(images, d_raw_images, d_brain_images, d_gray_images) = processed_data[3]
    #dataset_len = len(images)
    
    ############################## Feature Extraction ##############################


if __name__ == "__main__":
    main()
    