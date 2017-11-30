#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:31:15 2017

@author: sahilnayyar
"""

from os import listdir
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm


#=========#
# HELPERS #
#=========#
def getZafar(path):

    filenames = listdir(path);
    popularity_class_dict = {'1k':1 ,'100k':2,'1M':3,'10M':4};

    
#    print('Importing Zafar\'s processed classification datasets:')
    df = pd.DataFrame();
    for f in filenames:
        
#        print('\t{0}'.format(f));
        df_file = pd.read_csv(path+'/'+f, index_col='screen_name');
        
        # Add "is_bot" label, depending on substring within filename
        if f.find('bots') == 0:
            df_file['is_bot'] = 1;
        else:
            df_file['is_bot'] = 0;
            
        # Add "popularity class" feature, depending on substring within filename
        df_file['popularity_class'] = popularity_class_dict[f.split('.')[-2]];
        
        df = pd.concat([df, df_file]);
#    print('Complete!\n')
        
    # Sort by index alphabetically
    df = df.sort_index(); 
    
    # Move "is_bot" and "popularity_class" to the front of other columns
    df = df[list(df)[-2:] + list(df)[:-2]];
    
    # Remove duplicates (which have been classified as both a bot and a human)
    i_dupes = np.where(df.index.duplicated())[0] - 1;
    i_dupes = np.concatenate((i_dupes, i_dupes+1))
    i_dupes = np.sort(i_dupes);
    df.drop(i_dupes);
    
    X = df[df.columns.difference(['is_bot', 'source_identity'])];
    y = df['is_bot'];
    
    return X, y

def getTrains():
    
    # Load everything from the pickles, set training set to X_train, set training-dev set to X_test and X_dev
    X_train = pickle.load(open( "X_train.p", "rb" ));
    X_traindev1 = pickle.load(open( "X_test.p", "rb" ));
    X_traindev2 = pickle.load(open( "X_dev.p", "rb" ));
    X_traindev = pd.concat([X_traindev1, X_traindev2]);
    
    y_train = pickle.load(open( "y_train.p", "rb" ));
    y_traindev1 = pickle.load(open( "y_test.p", "rb" ));
    y_traindev2 = pickle.load(open( "y_dev.p", "rb" ));
    y_traindev = pd.concat([y_traindev1, y_traindev2]);
    
    ## TODO: adjust features to match those used by Zafar
    
    return X_train, X_traindev, y_train, y_traindev;

#======#
# MAIN #
#======#

# Notes to jesswetstone from sahilnayyar, 11/30/17

# So I tried using Zafar's data as a training set for both logistic and
# gaussian svm models, and the results were really shitty! (68% val error for logistic, 
# 56% val error for SVM with extreme overfitting).
path = 'classification_processed';
X_z, y_z = getZafar(path)
X_train, X_dev, y_train, y_dev = train_test_split(X_z, y_z, test_size = 0.2);

logistic = linear_model.LogisticRegression()
gaussian_svm = svm.SVC()
logistic.fit(np.asmatrix(X_train),np.ravel(y_train))
gaussian_svm.fit(np.asmatrix(X_train),np.ravel(y_train))
train_error = [logistic.score(X_train,y_train),gaussian_svm.score(X_train,y_train)]
dev_error = [logistic.score(X_dev,y_dev),gaussian_svm.score(X_dev,y_dev)]
print(pd.DataFrame(data = [train_error,dev_error]
    ,index = ['Training Error','Validation Error']
    ,columns = ['Logistic Regression', 'Support Vector Machine'])
)

# What I'd like to try next is, using the data that we prepared as a training
# (and traindev) set, test some models on an actual dev set derived from Zafar's
# data. I think this will be a valuable test of our methods so far, since the 
# cresci and Zafar datasets seem to be drawn from different distributions.

# The only caveat is that the features that we used are slightly different 
# than those used by Zafar. Given that we don't actually have access to 
# Zafar's raw, preprocessed data, we'll have to make a few adjustments to
# our selected features in order to use our prepared cresci data alongside
# Zafar's data. (Could you do this?)
    
# Repartition our prepared data from cresci into train and traindev sets (see above function)
X_train, X_traindev, y_train, y_traindev = getTrains();

# Get new dev and test sets from Zafar's data source
path = 'classification_processed';
X_z, y_z = getZafar(path)
X_dev, X_test, y_dev, y_test = train_test_split(X_z, y_z, test_size = 0.5);
