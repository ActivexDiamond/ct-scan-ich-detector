#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 04:53:23 2022

@author: activexdiamond
"""

############################## Dependencies ##############################
import timeit

import cv2

from matplotlib import pyplot

############################## Config ##############################
import debug_config

#Time the execution of custom-written feature extractors.
def time_fe_func(func, image, runs=100):
    print(f"Timing execution of{func}")
    dur = timeit.timeit(lambda: func(image), number=runs)
    print(f"Took: {dur / runs}s")


def image_plotter(images, labels, figureSize, count, superTitle=None, offset=0):
    ##Display the images as matplotlib plots, to get an idea of what we're working with.
    pyplot.figure(figsize=figureSize)
    pyplot.title("Final Images")
    for i in range(0, count):
        pyplot.subplot(330 + 1 + i)
        #rng = random.randrange(1, len(images))
        pyplot.imshow(images[i + offset], cmap=pyplot.get_cmap("gray"))
        pyplot.title(f"\nLabel: {labels[i + offset]}")
        
    if superTitle != None:
        pyplot.suptitle(superTitle)
    pyplot.show()
    
def plot_features(name, image, features):
    figure = pyplot.figure(figsize=debug_config.FIG_SIZE)

    figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 1)
    pyplot.imshow(image, cmap="gray")
    pyplot.title("Grayed")
    
    figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 2)
    pyplot.imshow(features["lbp"], cmap="gray")
    pyplot.title("LBP")

    figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 3)
    pyplot.plot(features["sfta"])
    pyplot.title("SFTA")
    
    #figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 4)
    #pyplot.imshow(features["hog_image"], cmap="gray")
    #pyplot.title("HOG Image")

    #figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 4)
    #d_display_hog_image = cv2.convertScaleAbs(features["hog_image"], alpha=3, beta=100)
    #pyplot.imshow(d_display_hog_image, cmap="gray")
    #pyplot.title("(Debug-Contrast - Viewing Only)")
    
    figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 4)
    pyplot.plot(features["hog"])
    pyplot.title("HOG")

    figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 5)
    line = 1
    for k, v in features["glcm"].items():
        pyplot.text(0.01, 0.15 * line, f"{k} = {v}", size=20)
        line += 1
    pyplot.title("GLCM")

    figure.add_subplot(debug_config.PLOT_ROWS, debug_config.PLOT_COLUMNS, 7)
    line = 1
    for k, v in features["fo"].items():
        pyplot.text(0.01, 0.1 * line, f"{k} = {v}", size=15)
        line += 1
    pyplot.title("1st Order Gray Stats")

    pyplot.show()
        


#detectable fail         6%
#undetectabl fail        0.5%
#?                       1
#total                   200

