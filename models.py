#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 03:29:02 2022
@license: MIT

@author: Dulfiqar 'activexdiamond' H. Al-Safi
"""
############################## Dependencies ##############################
##Dataset Mangement
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.tree
import sklearn.ensemble

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

def model_nn():
    pass

############################## Regression ##############################

def regression_pca(n_comps, feature_matrix):
    standardScaler = sklearn.preprocessing.StandardScaler()
    standard_matrix = standardScaler.fit_transform(feature_matrix)
    pca = sklearn.decomposition.PCA(n_components=n_comps, random_state=config.SKLEARN_SHUFFLE_SEED)
    regressed_features = pca.fit_transform(standard_matrix)
    return regressed_features