#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:54:48 2017

@author: jessicawetstone
"""

## Initialization
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import build_design_matrix 

def importDatasets():
    ## Import datasets
    genuine_account_folders = ['cresci-2017/genuine_accounts.csv', 'cresci-2015/TFP.csv', 'cresci-2015/E13.csv']
    bot_account_folders = [
                             'cresci-2017/social_spambots_1.csv'
                            , 'cresci-2017/social_spambots_2.csv'
                            , 'cresci-2017/social_spambots_3.csv'
                            , 'cresci-2017/traditional_spambots_1.csv']
                            #, 'cresci-2015/INT.csv'
                            #, 'cresci-2015/FSF.csv'
                            #, 'cresci-2015/TWT.csv']
    
    list_genuine_users = []
    list_genuine_tweets = []
    for folder in genuine_account_folders:
        df_users = pd.read_csv(folder+'/users.csv',index_col='id');
        df_users['source'] = folder; 
        list_genuine_users.append(df_users);
        df_tweets = pd.read_csv(folder+'/tweets.csv',index_col='id');
        df_tweets['source'] = folder; 
        list_genuine_tweets.append(df_tweets);
    genuine_tweets = pd.concat(list_genuine_tweets)
    genuine_users = pd.concat(list_genuine_users)
    
    list_bot_users = []
    list_bot_tweets = []
    for folder in bot_account_folders:
        df_users = pd.read_csv(folder+'/users.csv',index_col='id');
        df_users['source'] = folder;
        list_bot_users.append(df_users);    
        df_tweets = pd.read_csv(folder+'/tweets.csv',index_col='id');
        df_tweets['source'] = folder;
        list_bot_tweets.append(df_tweets);
    bot_tweets = pd.concat(list_bot_tweets)
    bot_users = pd.concat(list_bot_users)

    users = pd.concat([bot_users,genuine_users])
    tweets = pd.concat([bot_tweets,genuine_tweets])

    user_class = pd.concat([pd.DataFrame(np.ones(shape=(len(bot_users),1))),pd.DataFrame(np.zeros(shape=(len(genuine_users),1)))])
    user_class.rename(columns={0:'is_bot'},inplace=True);

    # Remove users that have no associated tweets
    user_class = user_class[users.index.isin(tweets.set_index('user_id').index)]
    users = users[users.index.isin(tweets.set_index('user_id').index)]

    ## Delete empty values
    del users['contributors_enabled'];
    del users['follow_request_sent'];
    del users['following'];
    del users['notifications'];
    del users['test_set_1'];
    del users['test_set_2'];
    
    del tweets['geo'];
    del tweets['contributors'];
    del tweets['favorited'];
    del tweets['retweeted'];
    
    ## Update timestamp values to be datetime
    tweets['timestamp_dt'] = pd.to_datetime(tweets['timestamp'],format='%Y-%m-%d %H:%M:%S')
    del tweets['timestamp']
    
    return users, user_class, tweets

def main():
    
    ## importData
    users, user_class, tweets = importDatasets()
    
    ## Build the design matrix
    X = build_design_matrix.buildDesignMatrix(users, tweets, 1)
    
    ## Split into Train, Train-Dev datasets
    X_train, X_train_dev, y_train, y_train_dev = train_test_split(X, user_class, test_size=0.2)
    
    pickle.dump(X_train, open( "X_train.p", "wb" ))
    pickle.dump(X_train_dev, open( "X_train_dev.p", "wb" ))
    pickle.dump(y_train, open( "y_train.p", "wb" ))
    pickle.dump(y_train_dev, open( "y_train_dev.p", "wb" ))

# To run - main()
main()