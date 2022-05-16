#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:27:07 2022
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
import os

import cv2

############################## Global Debugging ##############################
_USE_DUMMY_PATHS = False

############################## Skull Stripping ##############################
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

############################## Main Preprocessing Pipeline ##############################
##File paths.
if _USE_DUMMY_PATHS:
    IMAGE_RELATIVE_PATH = "/dummy_input/head_ct/images/*.png"
    LABEL_RELATIVE_PATH = "/dummy_input/labels.csv"
else:
    IMAGE_RELATIVE_PATH = "/input/head_ct/images/*.png"
    LABEL_RELATIVE_PATH = "/input/labels.csv"

##Preprocessing configs.
FINAL_GRAYING_MODE = cv2.COLOR_BGR2GRAY

IMAGE_W = 128
IMAGE_H = 128
IMAGE_SIZE = (IMAGE_W, IMAGE_H)

SKLEARN_SHUFFLE_SEED = 42
RANDOM_SEED = 1337
TEST_RATIO = 0.2                    #How much of the dataset to use for testing. float ; [0, 1]
VALIDATION_RATIO = 0.5              #How much of the trainning dataset to use for validation. float ; [0, 1]

############################## M.L. Model ##############################
INPUT_SHAPE = (IMAGE_W, IMAGE_H, 1)

EPOCHS = 200
BATCH_SIZE = 10

############################## Metadata ##############################
_METADATA = {
    "TITLE": "C.T. Scan I.C.H. Detector",
    "DESCRIPTION": "An M.L. model capable of detecting the precense of I.C.H. in 2-dimensional C.T. scans of the human brain.",
    "TYPE": "CLI",
    "VERSION": "dev-1.3.0",
    "LICENSE": "MIT",
    "AUTHOR": "Dulfiqar 'activexdiamond' H. Al-Safi"
}

def echo_metadata():
    print( "================---~~~ INFO ~~~---================")
    print( "=> Echoing project metadata!\t\t\t\t\t\t=")
    print(f"=> Title: {_METADATA['TITLE']}\t\t\t\t=")
    print(f"=> Type: {_METADATA['TYPE']}\t\t\t\t\t\t\t\t\t\t=")
    print(f"=> Version: {_METADATA['VERSION']}\t\t\t\t\t\t\t=")
    print(f"=> License: {_METADATA['LICENSE']}\t\t\t\t\t\t\t\t\t=")
    #print(f"=> Author: {_METADATA['AUTHOR']}\t=")
    print( "================---~~~ ~~~~ ~~~---================")
echo_metadata()

