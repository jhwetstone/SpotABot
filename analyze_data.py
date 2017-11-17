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
import seaborn as sns
from sklearn import linear_model, decomposition, datasets
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

## Split data into train/dev/test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5)

## Creation of model objects
logistic = linear_model.LogisticRegression()
svm = SVC()
pca = decomposition.PCA() # coming soon!
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)]) # coming soon!

## Application of fits (just two for now)
logistic.fit(np.asmatrix(X_train),np.ravel(y_train))
svm.fit(np.asmatrix(X_train),np.ravel(y_train))

## Print errors
train_error = [logistic.score(X_train,y_train),svm.score(X_train,y_train)]
dev_error = [logistic.score(X_dev,y_dev),svm.score(X_dev,y_dev)]

print( pd.DataFrame(data = [train_error,dev_error]
                    ,index = ['Training Error','Validation Error']
                    ,columns = ['Logistic Regression', 'Support Vector Machine'])
)

### Print coefficients
#coeffs = np.insert(logistic.coef_,0,logistic.intercept_)
#print( pd.DataFrame(data = coeffs
#                    ,index = np.insert(X.columns,0,'intercept')
#                    ,columns = ['logistic coefficients'])
#)


## Some other really cool shit that isn't ready yet
def scatter_x1x2(categ1, categ2):

    X_pos = X_train.iloc[np.ravel(y_train == 1)]
    x1_pos = X_pos[categ1]
    x2_pos = X_pos[categ2]
    X_neg = X_train.iloc[np.ravel(y_train == 0)]
    x1_neg = X_neg[categ1]
    x2_neg = X_neg[categ2]
    
    x1max = max(
            np.concatenate(
                    (np.ravel(x1_pos)
                    ,np.ravel(x1_neg)
            )
        )   
    )
    
    x2max = max(
            np.concatenate(
                    (np.ravel(x2_pos)
                    ,np.ravel(x2_neg)
            )
        )   
    )
    
    ax = plt.axis([0,x1max,0,x2max])
    plt.scatter(x1_pos,x2_pos,marker='o',color='k',hold=True)
    plt.scatter(x1_neg,x2_neg,marker='o',color='r',facecolors='none',hold=True)
    
    plt.xlabel(categ1)
    plt.ylabel(categ2)
    
    return ax
    
#fig = plt.figure(figsize=(48,48));

#n = np.shape(X)[1];
#for i in range(0,n):
#    for j in range(0,n):
#        ax = fig.add_subplot(n,n,i*n+j+1);
##        ax.set_aspect(1);
#        scatter_x1x2(X.columns[i],X.columns[j])

# Getting rid of useless features (Unsupervised PCA analysis)
# http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#sphx-glr-auto-examples-plot-digits-pipe-py

#pca.fit(X_train)
#
#plt.figure(1, figsize=(4, 3))
#plt.clf()
#plt.axes([.2, .2, .7, .7])
#plt.plot(pca.explained_variance_, linewidth=2)
#plt.axis('tight')
#plt.xlabel('n_components')
#plt.ylabel('explained_variance_')
#
## Prediction
#n_components = [2]
#Cs = np.logspace(-4, 4, 3)
#
## Parameters of pipelines can be set using ‘__’ separated parameter names:
#estimator = GridSearchCV(pipe,
#                         dict(pca__n_components=n_components,
#                              logistic__C=Cs))
#estimator.fit(X_train, np.ravel(y_train))
#
#plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
#            linestyle=':', label='n_components chosen')
#plt.legend(prop=dict(size=12))
#plt.show()

Xpos = X_train.iloc[np.ravel(y_train == 1)];
Xneg = X_train.iloc[np.ravel(y_train == 0)];

bins = np.linspace(0,2,25)
fig1 = plt.figure(figsize = (6,6))
ax1 = plt.subplot(111)
plt.hist(Xpos['num_mentions_per_tweet'], bins, alpha=0.5, color = 'r', label='Bot', normed=1)
plt.hist(Xneg['num_mentions_per_tweet'], bins, alpha=0.5, color = 'k', label='Genuine User', normed=1)
plt.xlabel('Number of Mentions per Tweet')
plt.ylabel('Count (Normalized)')
x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect((x1-x0)/(y1-y0))
plt.legend()

fig2 = plt.figure(figsize = (6,6))
ax2 = plt.subplot(111)
plt.scatter(Xpos['retweet_count_per_tweet'],Xpos['favorite_count_per_tweet'],marker='o',color='r',hold=True, label='Bot')
plt.scatter(Xneg['retweet_count_per_tweet'],Xneg['favorite_count_per_tweet'],marker='o',color='k',hold=True, label='Genuine User')
plt.xlabel('Retweet Count Per Tweet')
plt.ylabel('Favorite Count Per Tweet')
x0,x1 = ax2.get_xlim()
y0,y1 = ax2.get_ylim()
ax2.set_aspect((x1-x0)/(y1-y0))
plt.legend()

