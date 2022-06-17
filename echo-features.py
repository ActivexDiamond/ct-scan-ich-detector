#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 19:21:14 2022
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
##Path utils for image loading.
import os
from glob import glob
from pathlib import Path

##Maths
import numpy

## Data serialization and deserialization
import pandas
import deepdish

##Image manipulation.
import cv2

############################## Custom Modules ##############################
import preprocessing
import debugging 

############################## Config ##############################
import config
import debug_config

############################## Conversion Helper ##############################
def main():
    ############################## Import Data
    images = preprocessing.load_images(config.IMAGE_OUTPUT_PATH + "*.png", cv2.IMREAD_GRAYSCALE)
    #images = preprocessing.load_images(config.IMAGE_RELATIVE_PATH, cv2.IMREAD_GRAYSCALE)
    
    labels = preprocessing.load_labels(config.LABEL_RELATIVE_PATH)
    features = deepdish.io.load(os.getcwd() + config.FEATURE_OUTPUT_PATH)
    dataset_len = len(images)
    
    print(f"Data loaded. Got: {len(images)}images\t{len(labels)}labels")
    print(f"Image shape: {images[0].shape}")
    
    ############################## Flatten Images To Fit Into SVM
    for i, image in enumerate(images):
        image = cv2.resize(image, (128, 128))
        images[i] = image.flatten()
    string = ""
    for i, f in enumerate(features):
        string = f"===== Echoing features of image: <{i}> =====\n"
        for k, v in f.items():
            string += f"<{k}> = "
            if isinstance(v, dict):
                string += "A dictionary containing:\n"
                for _k, _v in v.items():
                    string += f"\t<{_k}> = <{_v}>\n"
            elif isinstance(v, int):
                string += f"<int><{v}>\n"
            else:
                string += f"<len:{v.size}><{v}>\n"
        string += f"\n===== ===== Image <{i}> done. ===== =====\n\n"
        print(string)
    
    
if __name__ == '__main__':
    main()