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

def compute_lbp(image):
    def compute_bit(x, y, center):
        w, h = image.shape
        if x < 0 or x >= w or y < 0 or y >= h:  
            return 0
        return 1 if image[x][y] >= center else 0
    
    def local_lbp(x, y):
        lbp = []
        center = image[x][y]
        #Hard-coded to improve performance.
        exps_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        #Start at the top right edge, then progress clockwise.
        lbp.append(compute_bit(x - 1,   y + 1, center)) 
        lbp.append(compute_bit(x,       y + 1, center)) 
        lbp.append(compute_bit(x + 1,   y + 1, center)) 
        lbp.append(compute_bit(x + 1,   y,     center)) 
        lbp.append(compute_bit(x + 1,   y - 1, center)) 
        lbp.append(compute_bit(x,       y - 1, center)) 
        lbp.append(compute_bit(x - 1,   y - 1, center)) 
        lbp.append(compute_bit(x - 1,   y,     center))
        for i in range(8):
            val += lbp[i] * exps_of_2[i]
            
        return val
        
    lbp_image = numpy.zeros((image.shape), numpy.uint8)
    w, h = image.shape
    for x in range(w):
        for y in range(h):
            lbp_image[x, y] = local_lbp(x, y)
            
    return lbp_image

def section_image(image, sections, x, y):
    rows, cols, depth = image.shape
    section_w = rows / sections
    section_h = cols / sections
    sectioned_image = image[int(x * section_w):int((x + 1) * section_w), int(y * section_h):int((y + 1) * section_h), :]
    return sectioned_image
    

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = compute_lbp(gray)
    
    if debug:
        figure = pyplot.figure(figsize=FIG_SIZE)

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 1)
        pyplot.imshow(gray, cmap="gray")
        pyplot.title("Grayed")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 2)
        pyplot.imshow(lbp_image, cmap="gray")
        pyplot.title("LBP")
            
        pyplot.show()

if __name__ == "__main__":
    main()
    