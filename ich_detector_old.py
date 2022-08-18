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

##Maths
import numpy

##Timing
import time

## Data serialization and deserialization
import deepdish

##Dataset Mangement
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.tree

############################## Custom Modules ##############################
import debugging 
import preprocessing
import models

############################## Config ##############################
import config

############################## Conversion Helper ##############################
def compute_feature_vector(keys, f):
    mat = numpy.zeros(1)
    for k in keys:
        v = f[k]
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
            
def flag_decoder(flag, length, t):
    result = []
    for i in range(length):
        if flag & 2 ** i:
            result.append(t[i])
    return result
    
    
    
def main():
    numpy.random.seed(config.NUMPY_SEED)
    
    ############################## Import Data
    #images = preprocessing.load_images(config.IMAGE_OUTPUT_PATH + "*.png", cv2.IMREAD_GRAYSCALE)
    #images = preprocessing.load_images(config.IMAGE_RELATIVE_PATH, cv2.IMREAD_GRAYSCALE)
    
        
    labels = preprocessing.load_labels(config.LABEL_RELATIVE_PATH)
    features = deepdish.io.load(os.getcwd() + config.FEATURE_OUTPUT_PATH)
    #features = deepdish.io.load(os.getcwd() + config.RAW_FEATURE_OUTPUT_PATH)
    dataset_len = len(labels)
    
    #print(f"Data loaded. Got: {len(images)}images\t{len(labels)}labels")
    #print(f"Image shape: {images[0].shape}")
    
    ############################## Flatten Images To Fit Into models.
    #for i, image in enumerate(images):
    #    image = cv2.resize(image, (128, 128))
    #    images[i] = image.flatten()


        
    ############################## Test Features
    FEATURE_TEMPLATE = ["glcm", "fo", "sfta", "lbp", "hog"]
    #logger = debugging.Logger("specs/svm.txt")
    #logger = debugging.Logger("specs/rfc.txt")
    #logger = debugging.Logger("specs/dtc.txt")
    #logger = debugging.Logger("specs/dtc-test.txt")
    #logger = debugging.Logger("specs/raw-svm.txt")
    logger = debugging.Logger("specs/svm-seed44.txt")
    logger.log(f"Testing all possible configurations for the following features:\n{FEATURE_TEMPLATE}\n")
    test_dur = 0
    for i in range(1, (2 ** 5)):
        target_subset = flag_decoder(i, 5, FEATURE_TEMPLATE)
        logger.log(f"\n=====> Testing: {target_subset}")
        logger.log(f"===> Feature subset testing at step: {i}/{2 ** 5}")
        feature_matrix = []
        for f in features:
            feature_matrix.append(compute_feature_vector(target_subset, f))
        
        ############################## Splits
        max_n_comps = min(dataset_len, len(feature_matrix[0]) + 1)
        for n_comps in range(1, max_n_comps):
            ############################## Regress Dataset
            print("================= About to regress.")
            start_time = time.process_time()
            regressed_feature_matrix = models.regression_pca(n_comps, feature_matrix)
            #print("================= About to split.")
            train_x, test_x, train_y, test_y = preprocessing.split(regressed_feature_matrix, labels, config.TEST_RATIO)    
            regression_dur = time.process_time() - start_time
            ############################## Create & Train Model
            print("================= About to train.")
            start_time = time.process_time()
            model = models.model_svm(train_x, train_y)
            #model = models.model_rfc(train_x, train_y)
            #model = models.model_dtc(train_x, train_y)
            training_dur = time.process_time() - start_time
            ############################## Test Model
            print("================= About to predict.")
            start_time = time.process_time()
            predictions = model.predict(test_x)
            #accuracy = models.evaluate_nn(model, test_x, test_y)
            prediction_dur = time.process_time() - start_time
            total_dur = regression_dur + training_dur + prediction_dur
            test_dur += total_dur
            accuracy = sklearn.metrics.accuracy_score(test_y, predictions)
            
            logger.log(f"Got accuracy of <acc={accuracy * 100:.4f}%> " +
                    f"with <n_comps={n_comps}>.\t\tTook <total_dur={total_dur:.7f}s> in total." +
                    f" (<regression_dur={regression_dur:.7f}s>; " +
                    f"<training_dur={training_dur:.7f}s>; <prediction_dur={prediction_dur:.7f}s>).")
    logger.log(f"\n This entire spec took <spec_dur={test_dur / 60}m> to complete.")
    
    
    ############################## Test Model
   # predictions = model.predict(test_x)
   # accuracy = sklearn.metrics.accuracy_score(test_y, predictions)
    
    ############################## Debug-Echo Splits
    ##Some debug echoing of the used splits.
    #print("Using the following splits:")
    #print("\tTRAIN\tTEST\t\tVALIDATE")
    #string = "\t{}\t\t{}\t\t{}\t\t\t(images)".format(len(train_x), len(val_x), len(test_x))
    #print(string)
    #string = "\t{}%\t{}%\t{}%\t\t(of the dataset)".format(
    #        len(train_x) / dataset_len * 100,
    #        len(val_x) / dataset_len * 100,
    #        len(test_x) / dataset_len * 100)
    #print(string)
   # 
   # print(f"Best accuracy gotten was %{best_accuracy * 100} at N_COMPONENTS={best_accuracy_n_comps}")
    
    
if __name__ == '__main__':
    main()
    
    
#    ############################## Test
#    feature_matrix = []
#    for f in features:
#        feature_matrix.append(compute_feature_vector(["glcm", "fo", "lbp"], f))
#    for n_comps in range(19, 22):
#        print(f"n_comps={n_comps}")
#        ############################## Regress Dataset
#        start_time = time.process_time()
#        regressed_feature_matrix = models.regression_pca(n_comps, feature_matrix)
#        train_x, test_x, train_y, test_y = preprocessing.split(regressed_feature_matrix, labels)    
#        regression_dur = time.process_time() - start_time
#        ############################## Create & Train Model
#        start_time = time.process_time()
#        model = models.model_svm(train_x, train_y)
#        training_dur = time.process_time() - start_time
#        ############################## Test Model
#        start_time = time.process_time()
#        predictions = model.predict(test_x)
#        prediction_dur = time.process_time() - start_time
#        total_dur = regression_dur + training_dur + prediction_dur
#        accuracy = sklearn.metrics.accuracy_score(test_y, predictions)
#        
#        print(f"Got accuracy of <{accuracy * 100:.4f}%> " +
#                f"with <n_comps={n_comps}>.\t\tTook <{total_dur:.7f}s>." +
#                f" (regression_dur=<{regression_dur:.7f}s>; " +
#                f"training=<{training_dur:.7f}s>; prediction=<{prediction_dur:.7f}s>).")
