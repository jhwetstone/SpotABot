#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 04:17:42 2017

@author: sahilnayyar
"""
## Initialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model, ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

## Load our test, train, dev objects
X_train = pickle.load(open( "pickleFiles/X_train.p", "rb" ))
y_train = pickle.load(open( "pickleFiles/y_train.p", "rb" ))
X_dev = pickle.load(open( "pickleFiles/X_dev.p", "rb" ))
y_dev = pickle.load(open( "pickleFiles/y_dev.p", "rb" ))

fpointfive_scorer = make_scorer(fbeta_score, beta=0.5)

## Reshuffling stuff
X = pd.concat((X_train, X_dev));
y = pd.concat((y_train, y_dev));
test_fold = np.concatenate((-np.ones(len(X_train)), np.ones(len(X_dev))))
ps = PredefinedSplit(test_fold)

## Logistic Model
parameters = {'C':np.logspace(-5,5)}
logistic = linear_model.LogisticRegression()
clf_logistic = GridSearchCV(logistic, parameters, cv=ps, scoring=fpointfive_scorer)
clf_logistic.fit(np.asmatrix(X), np.ravel(y))
pickle.dump(clf_logistic, open("hyper_logistic_model.p","wb"))

parameters = {'learning_rate': np.linspace(0.005,0.25)}
gbm = ensemble.GradientBoostingClassifier()
clf_gbm = GridSearchCV(gbm, parameters, cv=ps, scoring=fpointfive_scorer)
clf_gbm.fit(np.asmatrix(X), np.ravel(y))
pickle.dump(clf_gbm, open("pickleFiles/hyper_gbm.p","wb"))

clf_logistic = pickle.load(open("pickleFiles/hyper_logistic_model.p","rb"))
C = list(map((lambda x: x['C']), clf_logistic.cv_results_['params'] ))
train_fscore = clf_logistic.cv_results_['split0_train_score']
dev_fscore = clf_logistic.cv_results_['split0_test_score']

df = pd.DataFrame({'0 Regularization Parameter': C, '1 Training F 0.5 Score': train_fscore, '2 Dev F 0.5 Score': dev_fscore})
print(df)

## GBM Model
plt.semilogx(C,train_fscore,C,dev_fscore)
plt.title('Regularizing the Logistic Model')
plt.xlabel('Regularization Parameter')
plt.ylabel('F Score')
plt.legend(['Training','Cross-Validation'])

clf_gbm = pickle.load(open("pickleFiles/hyper_gbm.p","rb"))
C = list(map((lambda x: x['learning_rate']), clf_gbm.cv_results_['params'] ))
train_fscore = clf_gbm.cv_results_['split0_train_score']
dev_fscore = clf_gbm.cv_results_['split0_test_score']

df = pd.DataFrame({'0 Regularization Parameter': C, '1 Training F 0.5 Score': train_fscore, '2 Dev F 0.5 Score': dev_fscore})
print(df)

plt.figure()
plt.semilogx(C,train_fscore,C,dev_fscore)
plt.title('Regularizing the GBM Model')
plt.xlabel('Regularization Parameter')
plt.ylabel('F Score')
plt.legend(['Training','Cross-Validation'])

## Neural network

# For alpha, the best was found at 0.01
# Parameter grid
# alphas = 10.0 ** np.arange(-7, 7);
# param_grid = {'alpha': alphas}

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
clf = GridSearchCV(nn, param_grid, cv = ps, scoring=fpointfive_scorer, return_train_score=True);
clf.fit(np.asmatrix(X),np.ravel(y))
pickle.dump(clf,open("pickleFiles/hyper_neural_network.p", "wb"))

clf = pickle.load(open("pickleFiles/hyper_neural_network.p", "rb"))
train_fscore = clf.cv_results_['split0_train_score']
dev_fscore = clf.cv_results_['split0_test_score']
df = pd.DataFrame({'01 Layer Sizes': layer_sizes, '02 Training F Score':train_fscore, '03 Dev F Score':dev_fscore})

## Prints layer sizes vs. dev accuracy
with pd.option_context('display.max_rows', 200):
	print(df)