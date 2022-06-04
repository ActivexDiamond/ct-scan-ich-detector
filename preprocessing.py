#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 06:02:56 2022
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

#CSV importing and other filetype manipulation.
import pandas

##Image manipulation.
import cv2
import skimage
import PIL
import PIL.ImageStat

##Dataset Mangement
from sklearn.model_selection import train_test_split

##Visualization
from matplotlib import pyplot

############################## Custom Modules ##############################
import debugging

############################## Config ##############################
import config
import debug_config

############################## Loading Helpers ##############################
#`path` should be relative to the CWD.
def load_images(path):
    images = []
    
    image_paths = glob(os.getcwd() + path)
    image_paths = sorted(image_paths)
    for i, path in enumerate(image_paths):
        ##Load image.
        images.append(cv2.imread(path))
    return images

#`path` should be relative to the CWD.
def load_labels(path):
    metadata = pandas.read_csv(os.getcwd() + path)
    labels = metadata[" hemorrhage"].tolist()
    return labels 

############################## Strip Skull ##############################
def strip_skull(image, poor_image_checks=True, debug=True, debug_name="Unnamed", _threshold=config.THRESHOLD):
    ############### Preprocessing
    ##Convert to high-saturation gray-scale.
    #print(image.shape)
    gray_image = cv2.cvtColor(image, config.GRAYING_MODE)
    ##Blur image to reduce "holes" left in the skull.
    blurred_image = cv2.GaussianBlur(gray_image, config.BLUR_AMOUNT, 0)

    ##Apply a threshold-based filter to acquire a contour_mask of the skull.
    _, threshold_mask = cv2.threshold(blurred_image,
            _threshold,
            config.MASK_VAL,
            cv2.THRESH_BINARY_INV)

    ##Fill the area outside of the skull with the same value as the contour_mask.
    cv2.floodFill(threshold_mask, None, config.MASK_FLOOD_SEED, 0)

    #This should only be called if the used image poorly fits the critera of strip_skull,
    #   is low quality, and/or has poorly definde edges.
    if poor_image_checks:
        offset = config.POOR_IMAGE_CHECK_SEED_OFFSET
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

    #print(f"Total: {total_pixels//1000}k\t\t\White: {white_pixels//10000}k\t\t Black: {black_pixels//10000}k")
    #print(f"White: {white_pixels:.2f}\t\t Black: {black_pixels:.2f}")

    if black_pixels > config.RETRY_THRESHOLD_RATIO and _threshold > config.RETRY_MIN_THRESHOLD:
        new_threshold = _threshold
        for i in range(0, len(config.THRESHOLD_DECREMENTS)):
            #print(i, THRESHOLD_DECREMENTS[i], THRESHOLD_DECREMENTS[i][0], THRESHOLD_DECREMENTS[i][1])

            if _threshold >= config.THRESHOLD_DECREMENTS[i][0]:
                new_threshold -= config.THRESHOLD_DECREMENTS[i][1]
                break
        #print(f"Failed to create mask, retrying... Old threshold: {_threshold}\t\tNew threshold: {new_threshold}")
        return strip_skull(image, poor_image_checks, debug, debug_name, new_threshold)
    print(f"Extracting skull with threshold set to: {_threshold}\t\
            [White: {white_pixels*100:.2f}%\t\tBlack: {black_pixels*100:.2f}%]")

    #print(image.shape)
    #print(contour_mask.shape)

    ##Fill the inside area with the inverted value, to ensure no hemorrage is lost to the threshold filter.
    contours, _ = cv2.findContours(threshold_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(threshold_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_mask = threshold_mask.copy()
    cv2.drawContours(contour_mask, contours, -1, 128, cv2.FILLED)

    ##Remove skull-adjaccent hemmorrage.
    ellipse_mask = contour_mask.copy()
    approx_mask = contour_mask.copy()
    convex_mask = contour_mask.copy()

    for i in range(0, len(contours)):
        cnt = contours[i]
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            if not math.isnan(ellipse[0][0]):
                cv2.ellipse(ellipse_mask, ellipse, (0, 0, 64), 0)

        eps = config.EPSILON_FOR_DP_APPROX * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        cv2.drawContours(approx_mask, [approx], 0, 64, 8)

        #isConvex = cv2.isContourConvex(cnt)
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
        figure = pyplot.figure(figsize=debug_config.FIG_SIZE)

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 1)
        pyplot.imshow(image)
        pyplot.title(f"Base ({debug_name})")

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 2)
        pyplot.imshow(gray_image)
        pyplot.title("Gray: cv2.COLOR_BGR2GRAY")

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 3)
        pyplot.imshow(ellipse_mask)
        pyplot.title("ellipse_mask")

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 4)
        pyplot.imshow(blurred_image)
        pyplot.title(f"Gray+Blurred: {config.BLUR_AMOUNT}")

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 5)
        pyplot.imshow(threshold_mask, cmap = "gray")
        pyplot.title(f"Threshold Mask:{_threshold}")

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 6)
        pyplot.imshow(approx_mask)
        pyplot.title("approx_mask")

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 7)
        pyplot.imshow(contour_mask)
        pyplot.title("Contour Mask")

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 8)
        pyplot.imshow(sharp_mask)
        pyplot.title("Sharpened Mask")

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 9)
        pyplot.imshow(convex_mask)
        pyplot.title("convex_mask")

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 10)
        pyplot.imshow(absolute_stripped)
        pyplot.title("Absolute Stripped")

        figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 11)
        pyplot.imshow(stripped)
        pyplot.title("Stripped + HiContrast")

        pyplot.show()

    return stripped

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
    pil_image = PIL.Image.fromarray(image)
    #Compute the 9 stats supported by PIL's Stat class.
    stats = PIL.ImageStat.Stat(pil_image)

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
            ############################## Features - First Order - Gray-Level ##############################

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
        
    
    
    
    
    
    
    