#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 22:50:38 2022

@author: activexdiamond
"""

############################## Dependencies ##############################
import os
import math

import cv2
import numpy
from matplotlib import pyplot

############################## Constants ##############################
##Dir config.
CWD = os.getcwd()
IMAGE_DIR = CWD + "/input/head_ct/images/"
IMAGE_FILE = "000.png"

##Preprocessing config.
GRAYING_MODE = cv2.COLOR_BGR2GRAY
BLUR_AMOUNT = (7, 7)
THRESHOLD = 200                 #200
MASK_VAL = 255
MASK_FLOOD_SEED = 0, 0

POOR_IMAGE_CHECK_SEED_OFFSET = 20                       #Must be greater than zero.

EPSILON_FOR_DP_APPROX = 0.03

RETRY_THRESHOLD_RATIO = 0.80                              #Percentage of black-to-white pixels the dictate the threshold result invalid,
RETRY_MIN_THRESHOLD = 10

THRESHOLD_DECREMENTS = [
    [245, 1],
    [200, 2],
    [RETRY_MIN_THRESHOLD, 1],
]

##Debug config.
PLOT_COLUMNS = 3
PLOT_ROWS = 5
PLOT_SIZE = PLOT_COLUMNS * PLOT_ROWS
FIG_SIZE = (20, 20)

############################## Main Function ##############################
def strip_skull(image, poorImageChecks=True, debug=True, _threshold=THRESHOLD):
    ############### Preprocessing
    ##Convert to high-saturation gray-scale.
    #print(image.shape)
    gray_image = cv2.cvtColor(image, GRAYING_MODE)
    ##Blur image to reduce "holes" left in the skull.
    blurred_image = cv2.GaussianBlur(gray_image, BLUR_AMOUNT, 0)
    
    ##Apply a threshold-based filter to acquire a contour_mask of the skull.
    _, threshold_mask = cv2.threshold(blurred_image, _threshold, MASK_VAL, cv2.THRESH_BINARY_INV)
    
    ##Fill the area outside of the skull with the same value as the contour_mask.
    cv2.floodFill(threshold_mask, None, MASK_FLOOD_SEED, 0)

    #This should only be called if the used image poorly fits the critera of strip_skull,
    #   is low quality, and/or has poorly definde edges.
    if poorImageChecks:
        offset = POOR_IMAGE_CHECK_SEED_OFFSET
        h, w = threshold_mask.shape
        cv2.floodFill(threshold_mask, None, (offset, offset), 0)
        cv2.floodFill(threshold_mask, None, (w - offset, offset), 0)
        cv2.floodFill(threshold_mask, None, (offset, h - offset), 0)
        cv2.floodFill(threshold_mask, None, (w - offset, h - offset), 0)
        
    ##Check if threshold was too high, and re-run with a lower one.
    #black_pixels = cv2.countNonZero(threshold_mask) / (threshold_mask.size)
    #white_pixels = 100 - black_pixels
    
    white_pixels = numpy.sum(threshold_mask == 255) / (threshold_mask.size)
    black_pixels = numpy.sum(threshold_mask == 0) / (threshold_mask.size)
    
    #print(f"Total: {total_pixels//1000}k\t\t White: {white_pixels//10000}k\t\t Black: {black_pixels//10000}k")
    #print(f"White: {white_pixels:.2f}\t\t Black: {black_pixels:.2f}")
    
    if black_pixels > RETRY_THRESHOLD_RATIO and _threshold > RETRY_MIN_THRESHOLD:
        new_threshold = _threshold
        for i in range(0, len(THRESHOLD_DECREMENTS)):
            #print(i, THRESHOLD_DECREMENTS[i], THRESHOLD_DECREMENTS[i][0], THRESHOLD_DECREMENTS[i][1])
            
            if _threshold >= THRESHOLD_DECREMENTS[i][0]:
                new_threshold -= THRESHOLD_DECREMENTS[i][1]
                break
        #print(f"Failed to create mask, retrying... Old threshold: {_threshold}\t\tNew threshold: {new_threshold}")
        return strip_skull(image, poorImageChecks, debug, new_threshold)
    else:
        print(f"Extracting skull with threshold set to: {_threshold}\t [White: {white_pixels*100:.2f}%\t\tBlack: {black_pixels*100:.2f}%]")
    
    #print(image.shape)
    #print(contour_mask.shape)

    ##Fill the inside area with the inverted value, to ensure no hemorrage is lost to the threshold filter.
    contours, hierarchy = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(threshold_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_mask = threshold_mask.copy()
    cv2.drawContours(contour_mask, contours, -1, 128, cv2.FILLED)
    
    ##Remove skull-adjaccent hemmorrage.
    test_mask = contour_mask.copy()
    ellipse_mask = contour_mask.copy()
    approx_mask = contour_mask.copy()
    convex_mask = contour_mask.copy()
    
    for i in range(0, len(contours)):
        cnt = contours[i]
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            if not math.isnan(ellipse[0][0]):
                cv2.ellipse(ellipse_mask, ellipse, (0, 0, 64), 0)
        
        eps = EPSILON_FOR_DP_APPROX * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        cv2.drawContours(approx_mask, [approx], 0, 64, 8)
        
        isConvex = cv2.isContourConvex(cnt)
        #print(f"At: `contour  == {i}` and `isConvex == {isConvex}`")
        hull = cv2.convexHull(cnt)
        cv2.drawContours(convex_mask, [hull], 0, 64, cv2.FILLED)
        cv2.drawContours(contour_mask, [hull], 0, 64, cv2.FILLED)
        

    #Sharpen contour_mask.
    mask_w, mask_h = contour_mask.shape
    mask_center = int(mask_w / 2), int(mask_h / 2)
    sharp_mask = contour_mask.copy()
    cv2.floodFill(sharp_mask, None, mask_center, 255)
    #cv2.floodFill(sharp_mask, None, MASK_FLOOD_SEED, 0)
    
    ##Invert contour_mask.
    sharp_mask = cv2.bitwise_not(sharp_mask)    
    
    sharp_mask = cv2.merge([sharp_mask, sharp_mask, sharp_mask])
    stripped = cv2.subtract(image, sharp_mask)
    absolute_stripped = cv2.absdiff(image, sharp_mask)
    
    
    ############### Show
    if debug:
        figure = pyplot.figure(figsize=FIG_SIZE)
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 1)
        pyplot.imshow(image)
        pyplot.title("Base")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 2)
        pyplot.imshow(gray_image)
        pyplot.title(f"Gray: cv2.COLOR_BGR2GRAY")

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 3)
        pyplot.imshow(ellipse_mask)
        pyplot.title("ellipse_mask")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 4)
        pyplot.imshow(blurred_image)
        pyplot.title(f"Gray+Blurred: {BLUR_AMOUNT}")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 5)
        pyplot.imshow(threshold_mask, cmap = "gray")
        pyplot.title(f"Threshold Mask:{_threshold}")

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 6)
        pyplot.imshow(approx_mask)
        pyplot.title("approx_mask")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 7)
        pyplot.imshow(contour_mask)
        pyplot.title("Contour Mask")

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 8)
        pyplot.imshow(sharp_mask)
        pyplot.title("Sharpened Mask")

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 9)
        pyplot.imshow(convex_mask)
        pyplot.title("convex_mask")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 10)
        pyplot.imshow(absolute_stripped)
        pyplot.title("Absolute Stripped")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 11)
        pyplot.imshow(stripped)
        pyplot.title("Stripped + HiContrast")
        
        pyplot.show()
        
    ##TODO: Check if stripping failed 
    return stripped

#Test the skull-stripping function.
#image = cv2.imread(os.getcwd() + "/input/head_ct/images/011.png")
#strip_skull(image, True, True)

##Skull-adjacent hemorrage.
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/008.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/011.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/013.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/032.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/040.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/043.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/061.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/075.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/093.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/097.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/098.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/162.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/163.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/164.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/165.png"), True, True)


##Low threshold.
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/014.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/044.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/049.png"), True, True) #Fails even at threshold=10
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/062.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/068.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/080.png"), True, True)
strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/095.png"), True, True) #Hard time grabbing all brian.
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/113.png"), True, True) #Fails even at threshold=10
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/140.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/141.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/160.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/161.png"), True, True)
# strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/194.png"), True, True)


##Other
#strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/015.png"), True, True)


#           yellow = white
#           purple = black