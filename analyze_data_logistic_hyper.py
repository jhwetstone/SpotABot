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
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

## Load our test, train, dev objects
X_train = pickle.load(open( "X_train.p", "rb" ))
y_train = pickle.load(open( "y_train.p", "rb" ))
X_dev = pickle.load(open( "X_dev.p", "rb" ))
y_dev = pickle.load(open( "y_dev.p", "rb" ))

## Reshuffling stuff
X = pd.concat((X_train, X_dev));
y = pd.concat((y_train, y_dev));
test_fold = np.concatenate((-np.ones(len(X_train)), np.ones(len(X_dev))))
ps = PredefinedSplit(test_fold)

#parameters = {'C':np.logspace(-5,5)}
#
#svc = svm.SVC();
#clf_svc = GridSearchCV(svc, parameters, cv=ps)
#clf_svc.fit(np.asmatrix(X), np.ravel(y))
#
#logistic = svm.SVC();
#clf_logistic = GridSearchCV(logistic, parameters, cv=ps)
#clf_logistic.fit(np.asmatrix(X), np.ravel(y))

clf_logistic = pickle.load(open("hyper_logistic_model.p","rb"))
C = list(map((lambda x: x['C']), clf_logistic.cv_results_['params'] ))
train_acc = clf_logistic.cv_results_['split0_train_score']
dev_acc = clf_logistic.cv_results_['split0_test_score']

df = pd.DataFrame({'0 Regularization Parameter': C, '1 Training Accuracy': train_acc, '2 Dev Validation Accuracy': dev_acc})
print(df)

plt.semilogx(C,train_acc,C,dev_acc)
plt.title('Regularizing the Logistic Model')
plt.xlabel('Regularization Parameter')
plt.ylabel('Accuracy')
plt.legend(['Training','Cross-Validation'])