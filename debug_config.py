#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 19:21:04 2022
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
import os

############################## Global Debugging ##############################
_USE_DUMMY_PATHS = False

############################## Preprocessing - Skull Stripping ##############################
##Debug run
CWD = os.getcwd()
IMAGE_DIR = CWD + "/input/head_ct/images/"
IMAGE_FILE = "000.png"

##Debug view
PLOT_COLUMNS = 3
PLOT_ROWS = 5
PLOT_SIZE = PLOT_COLUMNS * PLOT_ROWS
FIG_SIZE = (20, 20)

############################## Preprocessing - Output ##############################
PLOT_IMAGE_FIGURE_SIZE = (10, 10)
PLOTTED_IMAGE_COUNT = 9                #The number of raw-images to plot. int ; [0, 9]
PLOT_IMAGE_OFFSET = 0

PLOT_RATIO_BARS_H = 110

SKULL_STRIPS_TO_DISPLAY = 250

############################## Misc. ,##############################

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
#strip_skull(cv2.imread(os.getcwd() + "/input/head_ct/images/095.png"), True, True) #Hard time grabbing all brian.
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
