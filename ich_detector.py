#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""

############################## Dependencies ##############################
##Maths
import numpy

##Timing
import time
import datetime

##Image manipulation.
import cv2
import skimage

##Dataset Mangement
import sklearn
import sklearn.feature_extraction
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.tree

############################## Custom Modules ##############################
import debugging 
import preprocessing
import models
import sfta

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
    images = preprocessing.load_images(config.IMAGE_RELATIVE_PATH, cv2.IMREAD_GRAYSCALE)
    
    x = 3
    y = x
    MAX_PATCHES = 1000
    features = []
    for i, image in enumerate(images):
        if image.size != config.IMAGE_SIZE:
            image = cv2.resize(image, config.IMAGE_SIZE)
        patches = sklearn.feature_extraction.image.extract_patches_2d(image, (x, y), max_patches=MAX_PATCHES)
        if i == -1:
            cv2.imshow(f"{i} image", image)
            #cv2.waitKey(0)
            cv2.imshow(f"{i} patch[10]", patches[10])
            #cv2.waitKey(0)
            cv2.imshow(f"{i} patch[20]", patches[20])
            #cv2.waitKey(0)
            cv2.imshowl(f"{i} patch[30]", patches[30])
            #cv2.waitKey(0)
            cv2.imshow(f"{i} patch[40]", patches[40])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        f = {
            "patches_mean": [],
            #"patches_std": [],
            "patches_var": [],
        }
        for patch in patches:
            f["patches_mean"].append(numpy.mean(patch))
            #f["patches_std"].append(numpy.std(patch))
            f["patches_var"].append(numpy.var(patch))

        #f["lbp"] = preprocessing.compute_lbp(image)
        #f["hog"], hog_image = skimage.feature.hog(image, visualize=True)
        #f["sfta"] = sfta.compute_sfta(image, 5)
        #f["glcm"] = preprocessing.compute_glcm_stats(image)
        #f["fo"] = preprocessing.compute_fo_all(image)        
        
        #f["blob_log"] = skimage.feature.blob_log(image)
        #f["blob_dog"] = skimage.feature.blob_dog(image)
        f["corner_kitchen_rosenfeld"] = skimage.feature.corner_kitchen_rosenfeld(image)
        #f["corner_peaks"] = skimage.feature.corner_peaks(image)
        f["daisy"] = skimage.feature.daisy(image)
        f["draw_multiblock_lbp"] = skimage.feature.draw_multiblock_lbp(image, 0, 0, 3, 3)
        #f["orb"] = skimage.feature.orb(image)
        #f["sift"] = skimage.feature.sift(image)
        #f["Cascade"]
        
        features.append(f)
        
        #best: 87.5% with patch=3x3 and getting mean of patches
        #close: 85% with patch=2x2 and getting mean of patches
    
    labels = preprocessing.load_labels(config.LABEL_RELATIVE_PATH)
    #features = deepdish.io.load(os.getcwd() + config.FEATURE_OUTPUT_PATH)
    #features = deepdish.io.load(os.getcwd() + config.RAW_FEATURE_OUTPUT_PATH)
    dataset_len = len(labels)
    
    feature_template = []
    for k in features[0].keys(): feature_template.append(k)

    FLAG_LEN = len(feature_template)
    feature_matrix = []
    best = 0
    target_subset = ["patches_mean", "patches_var", "corner_kitchen_rosenfeld", "daisy", "draw_multiblock_lbp"]
    feature_matrix = []
    for f in features:
        feature_matrix.append(compute_feature_vector(target_subset, f))
    #print(f"feature_matrix length={len(feature_matrix)}")
    #for i in range(len(feature_matrix)): print(f"feature_matrix[{i}] length={len(feature_matrix[i])}")
    AVERAGES = 100
    total = 0
    for i in range(0, AVERAGES):
        config.SKLEARN_SHUFFLE_SEED += 1
        config.RANDOM_SEED += 1
        config.NUMPY_SEED += 1
        
        train_x, test_x, train_y, test_y = preprocessing.split(feature_matrix, labels, config.TEST_RATIO)
        model = models.model_rfc(train_x, train_y)            
        predictions = model.predict(test_x)
        accuracy = sklearn.metrics.accuracy_score(test_y, predictions)
        total += accuracy
        print(f"Got accuracy of <acc={accuracy * 100:.4f}%> ")
    average = total / AVERAGES
    print(f"Got average accuracy of <avg_acc={average* 100:.4f}%> ")
    
    ############################## Test Features
    #train_x, test_x, train_y, test_y = preprocessing.split(feature_matrix, labels, config.TEST_RATIO)
    #model = models.model_svm(train_x, train_y)             
    #model = models.model_dtc(train_x, train_y)            
    #model = models.model_rvm(tran_x, train_y)            
    
    ############################## Test Model
    #accuracy = models.evaluate_nn(model, test_x, test_y)

    
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
    
    
# Raw images:
#    svm = 72.5%
#    rfc = 75%
#    dtc = 62.5%
#
# Images with patches, lbp, hog, etc.. and a few other things:
#   svm = bad
#   rfc = 90%
#   dtc = 92.5 (sfta, glcm, fo)
#   rvm = 67.5
#
#   raw images + patches 3x3
#       svm = 62.5%
#       rfc = 72.5%
#       dtc = 57.5%
#       rvm = 55%
#
#   'patches_mean', 'patches_var', 'corner_kitchen_rosenfeld', 'daisy', 'draw_multiblock_lbp'
# patches=3x3, pathes_max = 1k
# rfc=92.5%
