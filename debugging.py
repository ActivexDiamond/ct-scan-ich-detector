#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 04:53:23 2022

@author: activexdiamond
"""

############################## Dependencies ##############################
from matplotlib import pyplot

############################## Constants ##############################


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
        


#detectable fail         6%
#undetectabl fail        0.5%
#?                       1
#total                   200

