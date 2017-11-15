#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:54:48 2017

@author: jessicawetstone
"""

## Initialization
import numpy as np
import pandas as pd

## Import datasets
genuine_account_folders = ['cresci-2017/genuine_accounts.csv', 'cresci-2015/TFP.csv', 'cresci-2015/E13.csv']
bot_account_folders = [
                         'cresci-2017/social_spambots_1.csv'
                        , 'cresci-2017/social_spambots_2.csv'
                        , 'cresci-2017/social_spambots_3.csv'
                        , 'cresci-2017/traditional_spambots_1.csv'
                        , 'cresci-2015/INT.csv'
                        , 'cresci-2015/FSF.csv'
                        , 'cresci-2015/TWT.csv']

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

## Add friend-follower ratio as a feature
bot_users['friend_follower_ratio'] = bot_users['friends_count']/bot_users['followers_count']
## Remove divide-by-0 errors
bot_users.loc[~np.isfinite(bot_users['friend_follower_ratio']), 'friend_follower_ratio'] = np.nan
genuine_users['friend_follower_ratio'] = genuine_users['friends_count']/genuine_users['followers_count']
genuine_users.loc[~np.isfinite(genuine_users['friend_follower_ratio']), 'friend_follower_ratio'] = np.nan

## Add average-per-tweet information
genuine_averages = genuine_tweets[['num_hashtags','reply_count','retweet_count','favorite_count','num_mentions','num_urls','user_id']].groupby('user_id').mean()
bot_averages = bot_tweets[['num_hashtags','reply_count','retweet_count','favorite_count','num_mentions','num_urls','user_id']].groupby('user_id').mean()
bot_users = bot_users.join(bot_averages.add_suffix('_per_tweet'))
genuine_users = genuine_users.join(genuine_averages.add_suffix('_per_tweet'))


## Delete empty values
del bot_users['contributors_enabled'];
del bot_users['follow_request_sent'];
del bot_users['following'];
del bot_users['notifications'];
del bot_users['test_set_1'];
del bot_users['test_set_2'];

del genuine_users['contributors_enabled'];
del genuine_users['follow_request_sent'];
del genuine_users['following'];
del genuine_users['notifications'];
del genuine_users['test_set_1'];
del genuine_users['test_set_2'];