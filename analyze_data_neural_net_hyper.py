#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 06:12:21 2017

@author: sahilnayyar
"""

## Initialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import make_scorer


## Load our test, train, dev objects
X_train = pickle.load(open( "X_train.p", "rb" ))
y_train = pickle.load(open( "y_train.p", "rb" ))
X_dev = pickle.load(open( "X_dev.p", "rb" ))
y_dev = pickle.load(open( "y_dev.p", "rb" ))

## Fitting hyperparameters

# For alpha, the best was found at 0.01
# Parameter grid
#alphas = 10.0 ** np.arange(-7, 7);
#param_grid = {'alpha': alphas}

# For hidden layer sizes, did two layers with combinations of sizes 1:21, and 
# really didnt seem to make a huge difference.
tuplemat = list(map((lambda y: list(map((lambda x: (x,y)), np.arange(1,21)))), np.arange(1,11)));
layer_sizes = [item for sublist in tuplemat for item in sublist]
param_grid = {'hidden_layer_sizes':layer_sizes}

X = pd.concat((X_train, X_dev));
y = pd.concat((y_train, y_dev));
nn = MLPClassifier(solver = 'lbfgs', alpha = 0.01, random_state=1);
test_fold = np.concatenate((-np.ones(len(X_train)), np.ones(len(X_dev))))
ps = PredefinedSplit(test_fold)
scorer = make_scorer(lambda y_true,y_pred: (fbeta_score(y_true,y_pred,0.5)));
clf = GridSearchCV(nn, param_grid, cv = ps, scoring = scorer, return_train_score=True);
clf.fit(np.asmatrix(X),np.ravel(y))
pickle.dump(clf,open("hyper_neural_network.p", "wb"))
clf = pickle.load(open("hyper_neural_network.p", "rb"))
train_acc = clf.cv_results_['split0_train_score']
dev_acc = clf.cv_results_['split0_test_score']
df = pd.DataFrame({'01 Layer Sizes': layer_sizes, '02 Training Accuracy':train_acc, '03 Dev Accuracy':dev_acc})

## Prints layer sizes vs. dev accuracy
with pd.option_context('display.max_rows', 200):
	print(df)