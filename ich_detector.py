#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 06:13:49 2022
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

##Dataset Mangement
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.tree

##Visualization
from matplotlib import pyplot

############################## Custom Modules ##############################
import debugging 
import preprocessing

############################## Config ##############################
import config
import debug_config

############################## Conversion Helper ##############################
def compute_feature_vector(f):
    mat = numpy.zeros(1)
    for k, v in f.items():
        if k =="lbp":
            print("Ignoring LBP.")
            continue
        val = None
        if isinstance(v, dict):
            val = []
            for _k, _v in v.items():
                val.append(_v)
        elif isinstance(v, tuple):
            a, b = v
            val = [a, b]
        elif isinstance(v, list):
            val = v
        else:
            val = v
        mat = numpy.append(mat, val)
    mat = mat.flatten()
    return mat
            
    
    
    
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
    
    feature_matrix = []
    for i, feature in enumerate(features):
        feature_matrix.append(compute_feature_vector(feature))
    print(f"vec0={feature_matrix[0]}")    
    
    best_accuracy = 0
    best_accuracy_n_comps = 0
    #n_comps_pos = 48
    #for i in range(n_comps_pos, n_comps_pos + 1):
    for i in range(1, config.N_COMPONENTS_MAX):
        ############################## Feature Selection
        #print(f"Feature matrix: {len(feature_matrix[0])}, {len(labels)}")
        standardScaler = sklearn.preprocessing.StandardScaler()
        standard_matrix = standardScaler.fit_transform(feature_matrix)
        pca = sklearn.decomposition.PCA(n_components=i, random_state=config.SKLEARN_SHUFFLE_SEED)
        selected_features = pca.fit_transform(standard_matrix)
        #print(f"Feature matrix post selection: {len(selected_features[0])}, {len(labels)}")
        
        ############################## Splits
        ##Shuffle then split our dataset into train/test/validation.
        train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
                selected_features,
                labels,
                test_size=config.TEST_RATIO,
                random_state=config.SKLEARN_SHUFFLE_SEED)
        
        val_x, test_x, val_y, test_y = sklearn.model_selection.train_test_split(
                test_x,
                test_y,
                test_size=config.VALIDATION_RATIO,
                random_state=config.SKLEARN_SHUFFLE_SEED)
    
        ############################## Create Model
        model = sklearn.svm.SVC(kernel='linear',
                probability=True, 
                random_state=config.SKLEARN_SHUFFLE_SEED)
    
        ############################## Train Model
        model.fit(train_x, train_y)
        
        ############################## Test Model
        predictions = model.predict(test_x)
        accuracy = sklearn.metrics.accuracy_score(test_y, predictions)
        print(f"Tested N_COMPONENTS = {i} and got accuracy of %{accuracy * 100}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_n_comps = i
        
    ############################## Debug-Echo Splits
    ##Some debug echoing of the used splits.
    print("Using the following splits:")
    print("\tTRAIN\tTEST\t\tVALIDATE")
    string = "\t{}\t\t{}\t\t{}\t\t\t(images)".format(len(train_x), len(val_x), len(test_x))
    print(string)
    string = "\t{}%\t{}%\t{}%\t\t(of the dataset)".format(
            len(train_x) / dataset_len * 100,
            len(val_x) / dataset_len * 100,
            len(test_x) / dataset_len * 100)
    print(string)
    
    print(f"Best accuracy gotten was %{best_accuracy * 100} at N_COMPONENTS={best_accuracy_n_comps}")
    
    
if __name__ == '__main__':
    main()