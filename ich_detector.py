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
#Used to time the execution of custom-written statistical extractors.
def time_fo_funcs(image):
    t_mean_runs = 100
    print("Timing execution of `compute_fo_iqr`")
    t_mean = timeit.timeit(lambda: compute_fo_iqr(image), number=t_mean_runs)
    print(f"Took: {t_mean / t_mean_runs}s")
    
    print("Timing execution of `compute_fo_all`")
    startTime = timeit.default_timer()
    _ = compute_fo_all(image)
    duration = timeit.default_timer() - startTime
    print(f"Took: {duration}s")
    
############################## Features - First Order - Gray-Level ##############################
#A few mor 1st-order gray level stats to be implemented later (possibly).
def compute_fo_mode(image):
    pass

def compute_fo_mad(image):
    pass

def compute_fo_iqr(image):
    pass

def compute_fo_kurtosis(image):
    pass

def compute_fo_skewness(image):
    pass

#Wraps the fetching of all 1st-order stats into a single function / dictionary.
def compute_fo_all(image):
    #Convert our NumPy image to a PIL-compatible one.
    pil_image = Image.fromarray(image)
    #Compute the 9 stats supported by PIL's Stat class.
    stats = ImageStat.Stat(pil_image)

    #Wrap all of 1st-order stats into a single dictionary for tidiness.
    features = {
        "extrema": stats.extrema,
        "count": stats.count,
        "sum": stats.sum,
        "sum2": stats.sum2,
        "mean": stats.mean,
        "median": stats.median,
        "rms": stats.rms,       #Root-mean-square, per image band.
        "var": stats.var,
        "stddev": stats.stddev,
    }
    return features
    
############################## Features - Second Order - GLCM ##############################
def compute_glcm_stats(image):
    distances = [5]
    angles = [0]
    
    glcm = skimage.feature.graycomatrix(image, distances, angles)
    features = {
        "contrast":       skimage.feature.graycoprops(glcm, "contrast"),
        "dissimilarity":  skimage.feature.graycoprops(glcm, "dissimilarity"),
        "homogeneity":    skimage.feature.graycoprops(glcm, "homogeneity"),
        "energy":         skimage.feature.graycoprops(glcm, "energy"),
        "correlation":    skimage.feature.graycoprops(glcm, "correlation"),
        "asm":            skimage.feature.graycoprops(glcm, "ASM"),
    }
    
    return features

############################## Features - LBP ##############################
#Local Binary Pattern
def compute_lbp(image):
    #Helper for computing a single bit.
    def compute_bit(x, y, center):
        w, h = image.shape
        if x < 0 or x >= w or y < 0 or y >= h:  
            return 0
        return 1 if image[x][y] >= center else 0
    #Function for computing LBP locally - for a single neighborhood.
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
    
    #Create a blank NumPy ndarray for our results.
    lbp_image = numpy.zeros((image.shape), numpy.uint8)
    w, h = image.shape
    #Iterate over each pixel in the image, computing their local LBP.
    for x in range(w):
        for y in range(h):
            lbp_image[x, y] = local_lbp(x, y)
            
    return lbp_image

#Sections an image into nxn sections and returns the section at (x, y) where;
#   x -> [0, w)
#   y -> [0, h)
def section_image(image, sections, x, y):
    rows, cols, depth = image.shape
    section_w = rows / sections
    section_h = cols / sections
    sectioned_image = image[int(x * section_w):int((x + 1) * section_w), int(y * section_h):int((y + 1) * section_h), :]
    return sectioned_image
    
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
    ## Init
    features = {}
    image = cv2.imread(DEBUG_PATH)
    image = skull_stripping.strip_skull(image, True, False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ## LBP
    features["lbp"] = compute_lbp(gray)
    
    ## HOG
    features["hog"], features["hog_image"] = skimage.feature.hog(gray, visualize=True)
    
    ## SFTA
    features["sfta"] = sfta.compute_sfta(image, 5)

    ## Third Order - GLCM
    features["glcm"] = compute_glcm_stats(gray)

    ## First Order - Stats    
    features["fo"] = compute_fo_all(gray)
    print("Printing all 1st Order features.")
    for k, v in features["fo"].items():
        print(k, v)
    print("Done!")
    
    if debug:
        figure = pyplot.figure(figsize=FIG_SIZE)

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 1)
        pyplot.imshow(gray, cmap="gray")
        pyplot.title("Grayed")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 2)
        pyplot.imshow(features["lbp"], cmap="gray")
        pyplot.title("LBP")

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 3)
        pyplot.plot(features["sfta"])
        pyplot.title("SFTA")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 4)
        pyplot.imshow(features["hog_image"], cmap="gray")
        pyplot.title("HOG Image")

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 5)
        d_display_hog_image = cv2.convertScaleAbs(features["hog_image"], alpha=3, beta=100)
        pyplot.imshow(d_display_hog_image, cmap="gray")
        pyplot.title("(Debug-Contrast - Viewing Only)")
        
        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 6)
        pyplot.plot(features["hog"])
        pyplot.title("HOG")

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 7)
        line = 1
        for k, v in features["glcm"].items():
            pyplot.text(0.01, 0.15 * line, f"{k} = {v}", size=20)
            line += 1
        pyplot.title("GLCM")

        figure.add_subplot(PLOT_ROWS, PLOT_COLUMNS, 8)
        line = 1
        for k, v in features["fo"].items():
            pyplot.text(0.01, 0.1 * line, f"{k} = {v}", size=15)
            line += 1
        pyplot.title("1st Order Gray Stats")

        pyplot.show()

if __name__ == "__main__":
    main()
    