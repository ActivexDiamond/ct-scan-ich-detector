#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:19:31 2022
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
import os

import cv2
import numpy

##Visualization
from matplotlib import pyplot


############################## Custom Modules ##############################
from preprocessing import process_dataset
import skull_stripping
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

def section_and_min_max(image, sections, x, y):
    rows, cols, depth = image.shape
    section_w = rows / sections
    section_h = cols / sections
    s_image = image[int(x * section_w):int((x + 1) * section_w), int(y * section_h):int((y + 1) * section_h), :]
    s_gray = cv2.cvtColor(s_image, cv2.COLOR_BGR2GRAY)
    (s_gbb_min, s_gbb_max, s_gbb_min_pos, s_gbb_max_pos) = cv2.minMaxLoc(s_gray)    
    
    s_gbb_image = s_image.copy()
    cv2.circle(s_gbb_image, s_gbb_min_pos, config.BRIGHTNESS_RADIUS, (0, 0, 255), 2)
    cv2.circle(s_gbb_image, s_gbb_max_pos, config.BRIGHTNESS_RADIUS, (255, 0, 0), 2)        
    
    features = (s_gbb_min, s_gbb_max, s_gbb_min_pos, s_gbb_max_pos)
    return features, s_image, s_gray, s_gbb_image
    

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
    image = cv2.imread(DEBUG_PATH)
    image = skull_stripping.strip_skull(image, True, False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ## Direct min/max.
    (db_min, db_max, db_min_pos, db_max_pos) = cv2.minMaxLoc(gray)    

    db_image = image.copy()
    cv2.circle(db_image, db_min_pos, config.BRIGHTNESS_RADIUS, (0, 0, 255), 2)
    cv2.circle(db_image, db_max_pos, config.BRIGHTNESS_RADIUS, (255, 0, 0), 2)
    
    ## Min/max with gaussian blur.
    blur_amount = (config.BRIGHTNESS_RADIUS, config.BRIGHTNESS_RADIUS)
    gray = cv2.GaussianBlur(gray, blur_amount, 0)
    (gbb_min, gbb_max, gbb_min_pos, gbb_max_pos) = cv2.minMaxLoc(gray)    

    gbb_image = image.copy()
    cv2.circle(gbb_image, gbb_min_pos, config.BRIGHTNESS_RADIUS, (0, 0, 255), 2)
    cv2.circle(gbb_image, gbb_max_pos, config.BRIGHTNESS_RADIUS, (255, 0, 0), 2)    
    
    
    sectioned_features = []
    sectioned_gbb_images = []
    ## Min/max with sections and gaussian blur.    
    for x in range(0, 3):
        for y in range(0, 3):
            print(x, y)
            features, _, _, _sectioned_gbb_images = section_and_min_max(image, config.SECTIONS, x, y)                                                                                
            sectioned_features.append(features)
            sectioned_gbb_images.append(_sectioned_gbb_images)
    
    f32_gray = numpy.float32(gray)
    harrison_image = cv2.cornerHarris(f32_gray, 2, 3, 0.04)
        
    ##Harrison Corner detection.
        
    if debug:
        figure = pyplot.figure(figsize=FIG_SIZE)
    
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 1)
        pyplot.imshow(harrison_image)
        pyplot.title("Harrison Corner Detection")
    
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 2)
        pyplot.imshow(db_image)
        pyplot.title("Direct Min/Max")    

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 3)
        pyplot.imshow(gbb_image)
        pyplot.title("Gaussian Min/Max")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 4)
        pyplot.imshow(sectioned_gbb_images[0])
        pyplot.title("Section (1, 1)")
            
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 5)
        pyplot.imshow(sectioned_gbb_images[1])
        pyplot.title("Section (1, 2)")
            
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 6)
        pyplot.imshow(sectioned_gbb_images[2])
        pyplot.title("Section (1, 3)")
            
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 7)
        pyplot.imshow(sectioned_gbb_images[3])
        pyplot.title("Section (2, 1)")            
            
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 8)
        pyplot.imshow(sectioned_gbb_images[4])
        pyplot.title("Section (2, 2)")            
            
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 9)
        pyplot.imshow(sectioned_gbb_images[5])
        pyplot.title("Section (2, 3)")            
            
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 10)
        pyplot.imshow(sectioned_gbb_images[6])
        pyplot.title("Section (3, 1)")            
            
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 11)
        pyplot.imshow(sectioned_gbb_images[7])
        pyplot.title("Section (3, 2)")            

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 12)
        pyplot.imshow(sectioned_gbb_images[8])
        pyplot.title("Section (3, 3)")            
            
        pyplot.show()

if __name__ == "__main__":
    main()
    