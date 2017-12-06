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
from sklearn import linear_model, svm, ensemble
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

## Load our test, train, dev objects
X_train = pickle.load(open( "X_train.p", "rb" ))
y_train = pickle.load(open( "y_train.p", "rb" ))
X_dev = pickle.load(open( "X_dev.p", "rb" ))
y_dev = pickle.load(open( "y_dev.p", "rb" ))

fpointfive_scorer = make_scorer(fbeta_score, beta=0.5)

## Reshuffling stuff
X = pd.concat((X_train, X_dev));
y = pd.concat((y_train, y_dev));
test_fold = np.concatenate((-np.ones(len(X_train)), np.ones(len(X_dev))))
ps = PredefinedSplit(test_fold)

#parameters = {'C':np.logspace(-5,5)}
#svc = svm.SVC();
#clf_svc = GridSearchCV(svc, parameters, cv=ps, scoring=fpointfive_scorer)
#clf_svc.fit(np.asmatrix(X), np.ravel(y))
#pickle.dump(clf_svc, open("hyper_gaussian_svm.p","wb"))
#
#logistic = linear_model.LogisticRegression()
#clf_logistic = GridSearchCV(logistic, parameters, cv=ps, scoring=fpointfive_scorer)
#clf_logistic.fit(np.asmatrix(X), np.ravel(y))
#pickle.dump(clf_logistic, open("hyper_logistic_model.p","wb"))
#
parameters = {'learning_rate': np.linspace(0.005,0.25)}
gbm = ensemble.GradientBoostingClassifier()
clf_gbm = GridSearchCV(gbm, parameters, cv=ps, scoring=fpointfive_scorer)
clf_gbm.fit(np.asmatrix(X), np.ravel(y))
pickle.dump(clf_gbm, open("hyper_gbm.p","wb"))

clf_logistic = pickle.load(open("hyper_logistic_model.p","rb"))
C = list(map((lambda x: x['C']), clf_logistic.cv_results_['params'] ))
train_fscore = clf_logistic.cv_results_['split0_train_score']
dev_fscore = clf_logistic.cv_results_['split0_test_score']

df = pd.DataFrame({'0 Regularization Parameter': C, '1 Training F 0.5 Score': train_fscore, '2 Dev F 0.5 Score': dev_fscore})
print(df)

plt.semilogx(C,train_fscore,C,dev_fscore)
plt.title('Regularizing the Logistic Model')
plt.xlabel('Regularization Parameter')
plt.ylabel('F Score')
plt.legend(['Training','Cross-Validation'])


clf_svm = pickle.load(open("hyper_gaussian_svm.p","rb"))
C = list(map((lambda x: x['C']), clf_svm.cv_results_['params'] ))
train_fscore = clf_svm.cv_results_['split0_train_score']
dev_fscore = clf_svm.cv_results_['split0_test_score']

df = pd.DataFrame({'0 Regularization Parameter': C, '1 Training F 0.5 Score': train_fscore, '2 Dev F 0.5 Score': dev_fscore})
print(df)

plt.figure()
plt.semilogx(C,train_fscore,C,dev_fscore)
plt.title('Regularizing the Gaussian SVM Model')
plt.xlabel('Regularization Parameter')
plt.ylabel('F Score')
plt.legend(['Training','Cross-Validation'])


clf_gbm = pickle.load(open("hyper_gbm.p","rb"))
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