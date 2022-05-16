#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 20:26:31 2022

@author: activexdiamond
"""

VERSION = "dev-1.2.0"

print("=============== Info ===============")
print(f"Running C.T. Scan I.C.H. Detector Version: {VERSION}")
print("=============== ===============\n")

############################## Dependencies ##############################

############################## Custom Modules ##############################
from preprocessing import process_dataset

############################## Constants ##############################

############################## Execute Preprocessing Pipeline ##############################
processed_data = process_dataset()
(train_images, train_labels) = processed_data[0]
(test_images, test_labels) = processed_data[1]
(val_images, val_labels) = processed_data[2]
(images, d_raw_images, d_brain_images, d_gray_images) = processed_data[3]

dataset_len = len(images)

