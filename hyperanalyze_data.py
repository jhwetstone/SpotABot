#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:15:23 2017

@author: sahilnayyar
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import fbeta_score

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:49:08 2017

@author: sahilnayyar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from random import shuffle
from sklearn import linear_model
from sklearn.metrics import fbeta_score

def repartitionExamples(X_t, y_t, X_d, y_d, perc_t, perc_d):
    m_t = len(X_t);
    m_d = len(X_d);
    
    i_t = int(perc_t*m_t);
    i_d = int(perc_d*m_d);

    order_t = [i for i in range(0, m_t)];
    shuffle(order_t);
    pt_t = [order_t[i] for i in range(0, i_t)];
    
    order_d = [i for i in range(0, m_d)]
    shuffle(order_d);
    pt_d1 = [order_d[i] for i in range(0, i_d)];
    pt_d2 = [order_d[i] for i in range(i_d+1, m_d)];
    
    X_t_new = pd.concat((X_t.iloc[pt_t], X_d.iloc[pt_d1])) 
    y_t_new = pd.concat((y_t.iloc[pt_t], y_d.iloc[pt_d1])) 
    X_d_new = X_d.iloc[pt_d2];
    y_d_new = y_d.iloc[pt_d2];
    
    return X_t_new, y_t_new, X_d_new, y_d_new;

def getModelsOfPartitions(X_t, y_t, X_d, y_d, ft_list, fd_list, modelConstructor, args):
    models = np.empty(np.shape(ft_grid), dtype=object);
    for i in range(0,len(ft_list)):
        for j in range(0,len(fd_list)):
            
            ft = ft_list[i];
            fd = fd_list[j];
        
            X_train, y_train, X_dev, y_dev = \
                repartitionExamples(X_t, y_t, X_d, y_d, ft, fd)
                        
            model = modelConstructor(**args);
            model.fit(np.asmatrix(X_train), np.ravel(y_train));
            models[i,j] = model;
            
    return models;

def mapFromModelGrid(f, grid):
    (m, n) = np.shape(grid);
    out = np.empty((m, n), dtype = float);
    for i in range(0,m):
        for j in range(0,m):
            out[i,j] = f(grid[i,j]);
    return out;

X_train = pickle.load(open( "X_train.p", "rb" ));
y_train = pickle.load(open( "y_train.p", "rb" ));
X_dev = pickle.load(open( "X_dev.p", "rb" ));
y_dev = pickle.load(open( "y_dev.p", "rb" ));

ft_list = np.arange(.1,1,.1);
fd_list = np.arange(0,.9,.1);
ft_grid, fd_grid = np.meshgrid(ft_list, fd_list)
models = getModelsOfPartitions(X_train, y_train, X_dev, y_dev, ft_list, fd_list, linear_model.LogisticRegression, dict(C=0.79));

BETA = 0.5;
fscoreFun = lambda m: fbeta_score(m.predict(X_dev), np.ravel(y_dev), BETA);
fscores = mapFromModelGrid(fscoreFun, models);

plt.axes().set_aspect('equal');
plt.pcolor(ft_grid, fd_grid, fscores);
plt.title('F-Score vs. Partitions of Datasets');
plt.xlabel('Fraction of Original Training Set');
plt.ylabel('Fraction of Original Dev Set');
plt.colorbar();

X_dev2 = pickle.load(open( "X_test.p", "rb" ));
y_dev2 = pickle.load(open( "y_test.p", "rb" ));

def hyper_fit(X, y, model_constructor, cv, fixed, var, filename = ""):
    
    model = model_constructor(**fixed);
    hyper = GridSearchCV(model, var, cv = cv, return_train_score=True);
    hyper.fit(X, y);
    
    if filename != "":
        pickle.dump(hyper, open("filename","wb"));
        
    return hyper;

def get_merged_data():
    
    X1 = np.asarray(pickle.load(open( "X_train.p", "rb" )));
    X2 = np.asarray(pickle.load(open( "X_dev.p", "rb" )));
    y1 = np.ravel(pickle.load(open( "y_train.p", "rb" )));
    y2 = np.ravel(pickle.load(open( "y_dev.p", "rb" )));

    X = np.concatenate((X1,X2));
    y = np.concatenate((y1,y2));
    groups = [1 for i in range(len(X1))] + [2 for i in range(0,len(X2))]
    
    return X, y, groups;

#(X, y), splitloc = get_merged_data();
#newsplitloc = int((splitloc + (len(X)-splitloc)/3));
#X_train = X[0: newsplitloc]
#y_train = y[0: newsplitloc]
#X_dev = X[newsplitloc+1: len(X)]
#y_dev = y[newsplitloc+1: len(X)]
#
#clf = hyper_fit( \
#        X, y, 
#        LogisticRegression, 
#        ShuffleSplit(), 
#        {"penalty":"l2"}, 
#        {"C":np.logspace(-2,2,5)}
#);
#
#print(fbeta_score(clf.predict(X_dev), y_dev, 0.5));