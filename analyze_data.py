#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:01:45 2017
@author: sahilnayyar
"""

## Initialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model, svm
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

## Load our test, train, dev objects
X_train = pickle.load(open( "X_train.p", "rb" ))
X_test = pickle.load(open( "X_test.p", "rb" ))
X_dev = pickle.load(open( "X_dev.p", "rb" ))
y_train = pickle.load(open( "y_train.p", "rb" ))
y_test = pickle.load(open( "y_test.p", "rb" ))
y_dev = pickle.load(open( "y_dev.p", "rb" ))
y_train_dev = pickle.load(open( "y_train_dev.p", "rb" ))
X_train_dev= pickle.load(open( "X_train_dev.p", "rb" ))



## Creation of model objects
logistic = linear_model.LogisticRegression()
linear_svm = svm.LinearSVC();
gaussian_svm = svm.SVC();

## Application of fits (just two for now)
logistic.fit(np.asmatrix(X_train),np.ravel(y_train))
linear_svm.fit(np.asmatrix(X_train),np.ravel(y_train))
gaussian_svm.fit(np.asmatrix(X_train),np.ravel(y_train))

## Predicted classifications
y_train_logistic = logistic.predict(X_train);
y_train_linear_svm = linear_svm.predict(X_train);
y_train_gaussian_svm = gaussian_svm.predict(X_train);
y_train_dev_logistic = logistic.predict(X_train_dev);
y_train_dev_linear_svm = linear_svm.predict(X_train_dev);
y_train_dev_gaussian_svm = gaussian_svm.predict(X_train_dev);
y_dev_logistic = logistic.predict(X_dev);
y_dev_linear_svm = linear_svm.predict(X_dev);
y_dev_gaussian_svm = gaussian_svm.predict(X_dev);

## Precision score (true positives)
train_precision = [precision_score(y_train,y_train_logistic),
               precision_score(y_train,y_train_linear_svm),
               precision_score(y_train,y_train_gaussian_svm)]

train_dev_precision = [precision_score(y_train_dev,y_train_dev_logistic),
               precision_score(y_train_dev,y_train_dev_linear_svm),
               precision_score(y_train_dev,y_train_dev_gaussian_svm)]

dev_precision = [precision_score(y_dev,y_dev_logistic),
               precision_score(y_dev,y_dev_linear_svm),
               precision_score(y_dev,y_dev_gaussian_svm)]

## Training/dev accuracy
train_error = [logistic.score(X_train,y_train),
               linear_svm.score(X_train,y_train),
               gaussian_svm.score(X_train,y_train)]

train_dev_error = [logistic.score(X_train_dev,y_train_dev),
             linear_svm.score(X_train_dev,y_train_dev),
             gaussian_svm.score(X_train_dev,y_train_dev)]

dev_error = [logistic.score(X_dev,y_dev),
             linear_svm.score(X_dev,y_dev),
             gaussian_svm.score(X_dev,y_dev)]

#)

print( pd.DataFrame(data = [train_error, train_precision, train_dev_error, train_dev_precision, dev_error, dev_precision]
                    ,index = ['Training Accuracy','Training Precision', 'Train Dev Accuracy', 'Train Dev Precision', 'Dev Accuracy','Dev Precision']
                    ,columns = ['Logistic Regression', 'Linear SVM', 'Gaussian SVM'])
)    

## Save our final model (For "check_screenname.py")
pickle.dump(logistic,open("model.p","wb"))

### Print coefficients
coeffs = np.insert(logistic.coef_,0,logistic.intercept_)
df = pd.DataFrame(data = coeffs
                    ,index = np.insert(X_train.columns,0,'intercept')
                    ,columns = ['logistic coefficients'])
df = df.reindex(df['logistic coefficients'].abs().sort_values().index)
print(df)
