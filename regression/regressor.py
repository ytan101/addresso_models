#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:52:51 2018

@author: balamurali_bt
"""

from __future__ import absolute_import, division, print_function
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# import tflearn
from sklearn.preprocessing import StandardScaler, RobustScaler
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr

import regressor_models

inpu = sio.loadmat("Train_text.mat")
mat_input = inpu["final_output"]

Actual_number_of_input = 88
number_of_features = mat_input.shape[1] - 1

# Regression data
X = mat_input[:, 0:Actual_number_of_input]
Y = mat_input[:, number_of_features]

scaler = StandardScaler()
# scaler = RobustScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)
print("Feature scaling")
#
PCA_enabled = 0


if PCA_enabled == 1:
    pca = PCA(n_components=20, whiten=True)
    pca.fit(X)
    X = pca.transform(X)

mlp_regre = 0
ensemble_learning = 1

user_input = 50

if ensemble_learning == 1:

    inpu_validation = sio.loadmat("Test_text.mat")
    mat_input_validation = inpu_validation["final_output"]

    X_validation = mat_input_validation[:, 0:Actual_number_of_input]
    X_validation = scaler.transform(X_validation)
    validation_identifier = mat_input_validation[:, number_of_features]

    print(validation_identifier)
    print("Ground Truth")

    # List of models
    regressor_models.linear_regressor(X, Y, X_validation, validation_identifier)
    regressor_models.random_forest_regressor(X, Y, X_validation, validation_identifier)
    regressor_models.mlp_regressor(X, Y, X_validation, validation_identifier)
    print(regressor_models.results_dict)
    print("emobase_large")
    # print(X_validation, validation_identifier)

    # classifier_models.mlp_classifier(X, Y, X_validation, validation_identifier)
    # classifier_models.clp_classifier(X, Y, X_validation, validation_identifier)

    # gnb = CategoricalNB()
    # clf0 = RandomForestClassifier(n_estimators=500, max_features = "auto", max_depth=50, min_samples_split=10, random_state=00, verbose = 2, warm_start = True)
    # clf0 = RandomForestClassifier(n_estimators=5000, random_state=00)

    # grid_bag_clf = GridSearchCV(bag_clf, bag_clf_parameters, scoring='balanced_accuracy')

    # grid_bag_clf.fit(X, Y)

    # eclf = VotingClassifier(estimators=[('MLP', reg),('AD', clf2),('GB', clp)], voting='soft')
    # eclf = VotingClassifier(estimators=[('AD', clf0),('GB', clf1)], voting='soft')
    # eclf2 = VotingClassifier(estimators=[('AD', clf0),('GB', clf1),('BC', clf2),('BC1', clf3)], voting='hard')
    # eclf2 = VotingClassifier(estimators=[('AD', clf0),('GB', clf3)], voting='hard')

    # bag_clf.fit(X, Y)

    # sig_reg = CalibratedClassifierCV(clf0, method="sigmoid", cv="prefit")

    # X_validation = pca.transform(X_validation)
    # X_validation = scaler.transform(X_validation)

    # sio.savemat("ypred_egemaps.mat", {"pred": ypred})

# def test_models

# error = mean_squared_error(test_identifier, ypred)
# print(error)


# print(reg.score(X_testig, test_identifier))
# print(accuracy_score(test_identifier, ypred))
# conf_matrix = confusion_matrix(test_identifier, ypred)
# print(confusion_matrix(test_identifier, ypred))
