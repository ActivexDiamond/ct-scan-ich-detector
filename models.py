#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 03:29:02 2022
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""
############################## Dependencies ##############################
##Math Libs
import numpy

##Machine Learning
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.tree
import sklearn.ensemble
import sklearn_rvm

import keras.preprocessing
import keras.models
import keras.layers

############################## Custom Modules ##############################

############################## Config ##############################
import config

############################## Classification Models ##############################
def model_svm(train_x, train_y):
    model = sklearn.svm.SVC(kernel='linear',
            cache_size=7000,
            probability=True, 
            random_state=config.SKLEARN_SHUFFLE_SEED)
    
    model.fit(train_x, train_y)
    return model 

def model_rfc(train_x, train_y):
    model = sklearn.ensemble.RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=config.SKLEARN_SHUFFLE_SEED,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0,
        max_samples=None)
    
    model.fit(train_x, train_y)
    return model

def model_dtc(train_x, train_y):
    model = sklearn.tree.DecisionTreeClassifier(criterion="gini",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=config.SKLEARN_SHUFFLE_SEED,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0)
    model.fit(train_x, train_y)
    return model

#Deprecated
def model_nn(train_x, train_y):
    #full_x = full_x[..., numpy.newaxis, numpy.newaxis]
    #print(f"full_x.shape={full_x.shape}")
    
    ##Split train into train/val
    #train_x, val_x, train_y, val_y = preprocessing.split(full_x, full_y, config.VALIDATION_RATIO)
    print(f"train_x.shape={train_x.shape}")
    #print(f"val_x.shape={val_x.shape}")
    #train_x = train_x[..., numpy.newaxis, numpy.newaxis]
    #val_x = val_x[..., numpy.newaxis, numpy.newaxis]
    train_y = numpy.array(train_y)
    train_y = train_y[..., numpy.newaxis]
    
    print(f"POST train_y.shape={train_y.shape}")    
    print(f"POST train_x.shape={train_x.shape}")
    #print(f"POST val_x.shape={val_x.shape}")
    #Create model.
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(160*4,
             input_shape=train_x.shape, 
            kernel_initializer="uniform",
            bias_initializer="uniform",
            activation="relu"))
    
    model.add(keras.layers.Dense(160*2,
            activation="relu",
            kernel_initializer="uniform"))
    
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation(config.OUTPUT_ACTIVATION_LAYER))
    
    #model.add(keras.layers.add(input_shape=train_x.shape[1:]))
    #model.add(keras.layers.Dense(numpy.prod(train_x.shape[1:]), activation=config.ACTIVATION_LAYER, input_shape=train_x.shape))
    #model.add(keras.layers.Dense(64, activation=config.ACTIVATION_LAYER, input_shape=train_x.shape))
    #model.add(keras.layers.Activation(config.ACTIVATION_LAYER))
    #model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(64))
    #model.add(keras.layers.Flatten())
    #model.add(keras.layers.Activation(config.ACTIVATION_LAYER))
    #model.add(keras.layers.Dense(64))
    #model.add(keras.layers.Activation(config.OUTPUT_ACTIVATION_LAYER))
    
    #Compile model.
    model.compile(loss=config.LOSS,
            optimizer=config.OPTIMIZER,
            metrics=["accuracy"],
            run_eagerly=config.D_RUN_EAGERLY)
    
    print(model.summary())
    
    model.fit(train_x,
            train_y, 
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            verbose=config.NN_VERBOSITY)
    
    
    #Prepare training data.
    #train_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=config.RESCALE,
    #        shear_range=config.RESCALE,
    #        zoom_range=config.ZOOM,
    #        rotation_range=config.ROTATION,
    #        width_shift_range=config.SHIFT_RANGE,
    #        height_shift_range=config.SHIFT_RANGE,
    #        horizontal_flip=config.HORIZONTAL_FLIP)
    
    #train_generator = train_data_generator.flow(train_x,
    #        train_y,
    #        batch_size=config.BATCH_SIZE)
    
    #Prepare validation data.
    #val_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    #val_generator = val_data_generator.flow(val_x,
    #        val_y,
    #        batch_size=config.BATCH_SIZE)
    
    #Begin training!
    #history = model.fit_generator(train_generator,
    #        steps_per_epoch=len(train_x) // config.BATCH_SIZE,
    #        epochs=config.EPOCHS,
    #        validation_data=val_generator,
    #        validation_steps=len(val_x // config.BATCH_SIZE))
    
    return model

def model_rvm(train_x, train_y):
    model = sklearn_rvm.EMRVC(kernel="rbf",
            gamma="auto")
    model.fit(train_x, train_y)
    return model
############################## Regression ##############################

def regression_pca(n_comps, feature_matrix):
    standardScaler = sklearn.preprocessing.StandardScaler()
    standard_matrix = standardScaler.fit_transform(feature_matrix)
    pca = sklearn.decomposition.PCA(n_components=n_comps, random_state=config.SKLEARN_SHUFFLE_SEED)
    regressed_features = pca.fit_transform(standard_matrix)
    return regressed_features

############################## Helpers ##############################
def evaluate_nn(model, test_x, test_y):
    predictions = model.predict(test_x)
    correct_predictions = 0
    for i, p in enumerate(predictions):
        if p < 0.5 and test_y[i] == 0:
            correct_predictions += 1
        elif p >= 0.5 and test_y[i] == 1:
            correct_predictions += 1
    return correct_predictions / len(test_y)