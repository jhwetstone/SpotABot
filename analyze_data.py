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
from sklearn import linear_model, svm, ensemble, neural_network
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score

## Load our test, train, dev objects
X_train = pickle.load(open( "X_train.p", "rb" ))
X_test = pickle.load(open( "X_test.p", "rb" ))
X_dev = pickle.load(open( "X_dev.p", "rb" ))
y_train = pickle.load(open( "y_train.p", "rb" ))
y_test = pickle.load(open( "y_test.p", "rb" ))
y_dev = pickle.load(open( "y_dev.p", "rb" ))
y_train_dev = pickle.load(open( "y_train_dev.p", "rb" ))
X_train_dev= pickle.load(open( "X_train_dev.p", "rb" ))


BETA = 0.5
datasets = {}
datasets['Train'] = {'Actuals': y_train, 'Design Matrix': X_train}
datasets['Train Dev'] = {'Actuals': y_train_dev, 'Design Matrix': X_train_dev}
datasets['Dev'] = {'Actuals': y_dev, 'Design Matrix': X_dev}
datasets['Test'] = {'Actuals': y_test, 'Design Matrix': X_test}

models = {}

## Creation of model objects
logistic = linear_model.LogisticRegression(C=2.02)
models['Logistic']={'model': logistic}
linear_svm = svm.LinearSVC()
models['Linear SVM']={'model': linear_svm}
gaussian_svm = svm.SVC(C=86.85);
models['Gaussian SVM'] = {'model': gaussian_svm}
gbm = ensemble.GradientBoostingClassifier(learning_rate=0.245,n_estimators=100)
models['GBM'] = {'model': gbm}
neural_net = neural_network.MLPClassifier(solver = 'lbfgs', alpha = 0.01, random_state=1, hidden_layer_sizes=(3,4))
models['Neural Net'] = {'model': neural_net}

## Application of fits 
for model_name, model_object in models.items():
    model = model_object['model']
    model.fit(np.asmatrix(X_train),np.ravel(y_train))

## Predicted classifications
for model_name, model_object in models.items():
    model = model_object['model']
    for dataset_name, dataset in datasets.items():
        model_object[dataset_name + ' Result'] = model.predict(dataset['Design Matrix'])
        model_object[dataset_name + ' Accuracy'] = model.score(dataset['Design Matrix'],dataset['Actuals'])
    
## Calculate scores
for model_name, model_object in models.items():
    model = model_object['model']
    for dataset_name, dataset in datasets.items():
        model_object[dataset_name + ' Precision'] = precision_score(dataset['Actuals'], model_object[dataset_name + ' Result'])
        model_object[dataset_name + ' Recall'] = recall_score(dataset['Actuals'], model_object[dataset_name + ' Result'])
        model_object[dataset_name + ' F Score'] = fbeta_score(dataset['Actuals'], model_object[dataset_name + ' Result'], BETA)

#)
reporting_fields = ['Accuracy','Precision','Recall','F Score']
data = []
index = []
for dataset_name, dataset_object in datasets.items():
    for rfield in reporting_fields:
        data.append([models[model_name][dataset_name + ' ' + rfield] for model_name in models])
        index.append(dataset_name + ' ' + rfield)

print( pd.DataFrame(data = data
                    ,index = index
                    ,columns = [model_name for model_name in models])
)    

best_model = [(model_name, model_object['model']) for model_name, model_object in models.items() if model_object['Dev F Score'] == np.max([model_object['Dev F Score'] for model_name, model_object in models.items()])][0]

print('Saving the ' + best_model[0] + ' model to model.p')
## Save our final model (For "check_screenname.py")
pickle.dump(best_model[1],open("model.p","wb"))

### Print coefficients
coeffs = np.insert(logistic.coef_,0,logistic.intercept_)
df = pd.DataFrame(data = coeffs
                    ,index = np.insert(X_train.columns,0,'intercept')
                    ,columns = ['logistic coefficients'])
df = df.reindex(df['logistic coefficients'].abs().sort_values().index)
print(df)


## Some other really cool shit that isn't ready yet
#def scatter_x1x2(categ1, categ2):
#
#    X_pos = X_train.iloc[np.ravel(y_train == 1)]
#    x1_pos = X_pos[categ1]
#    x2_pos = X_pos[categ2]
#    X_neg = X_train.iloc[np.ravel(y_train == 0)]
#    x1_neg = X_neg[categ1]
#    x2_neg = X_neg[categ2]
#    
#    x1max = max(
#            np.concatenate(
#                    (np.ravel(x1_pos)
#                    ,np.ravel(x1_neg)
#            )
#        )   
#    )
#    
#    x2max = max(
#            np.concatenate(
#                    (np.ravel(x2_pos)
#                    ,np.ravel(x2_neg)
#            )
#        )   
#    )
#    
#    ax = plt.axis([0,x1max,0,x2max])
#    plt.scatter(x1_pos,x2_pos,marker='o',color='k',hold=True)
#    plt.scatter(x1_neg,x2_neg,marker='o',color='r',facecolors='none',hold=True)
#    
#    plt.xlabel(categ1)
#    plt.ylabel(categ2)
#    
#    return ax
    
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

#Xpos = X_train.iloc[np.ravel(y_train == 1)];
#Xneg = X_train.iloc[np.ravel(y_train == 0)];
#
#bins = np.linspace(0,2,25)
#fig1 = plt.figure(figsize = (6,6))
#ax1 = plt.subplot(111)
#plt.hist(Xpos['num_mentions_per_tweet'], bins, alpha=0.5, color = 'r', label='Bot', normed=1)
#plt.hist(Xneg['num_mentions_per_tweet'], bins, alpha=0.5, color = 'k', label='Genuine User', normed=1)
#plt.xlabel('Number of Mentions per Tweet')
#plt.ylabel('Count (Normalized)')
#x0,x1 = ax1.get_xlim()
#y0,y1 = ax1.get_ylim()
#ax1.set_aspect((x1-x0)/(y1-y0))
#plt.legend()
#
#fig2 = plt.figure(figsize = (6,6))
#ax2 = plt.subplot(111)
#plt.scatter(Xpos['retweet_count_per_tweet'],Xpos['favorite_count_per_tweet'],marker='o',color='r',hold=True, label='Bot')
#plt.scatter(Xneg['retweet_count_per_tweet'],Xneg['favorite_count_per_tweet'],marker='o',color='k',hold=True, label='Genuine User')
#plt.xlabel('Retweet Count Per Tweet')
#plt.ylabel('Favorite Count Per Tweet')
#x0,x1 = ax2.get_xlim()
#y0,y1 = ax2.get_ylim()
#ax2.set_aspect((x1-x0)/(y1-y0))
#plt.legend()
#
