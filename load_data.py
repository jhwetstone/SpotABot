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
## Reply count is sometimes blank -- replace with 0's
genuine_averages['reply_count'] = genuine_averages['reply_count'].fillna(0)
bot_averages = bot_tweets[['num_hashtags','reply_count','retweet_count','favorite_count','num_mentions','num_urls','user_id']].groupby('user_id').mean()
bot_users = bot_users.join(bot_averages.add_suffix('_per_tweet'))
genuine_users = genuine_users.join(genuine_averages.add_suffix('_per_tweet'))

## Add number of unique places where the user has tweeted.  
# This will be 0 if no places have ever been tagged
genuine_users['unique_tweet_places'] = genuine_tweets.groupby('user_id').place.nunique()
bot_users['unique_tweet_places'] = bot_tweets.groupby('user_id').place.nunique()

## Update timestamp values to be datetime
bot_tweets['timestamp_dt'] = pd.to_datetime(bot_tweets['timestamp'],format='%Y-%m-%d %H:%M:%S')
genuine_tweets['timestamp_dt'] = pd.to_datetime(genuine_tweets['timestamp'],format='%Y-%m-%d %H:%M:%S')
del bot_tweets['timestamp']
del genuine_tweets['timestamp']

## Add: Variance in the user's number of tweets per hour.
genuine_tweets['date'] = genuine_tweets.timestamp_dt.dt.date
genuine_tweets['hour'] = genuine_tweets.timestamp_dt.dt.hour
variance_in_genuine_tweet_rate = genuine_tweets.groupby(['user_id','date','hour']).size().groupby('user_id').var()
variance_in_genuine_tweet_rate.rename('variance_in_tweet_rate',inplace=True)
genuine_users = genuine_users.join(variance_in_genuine_tweet_rate)
bot_tweets['date'] = bot_tweets.timestamp_dt.dt.date
bot_tweets['hour'] = bot_tweets.timestamp_dt.dt.hour
variance_in_bot_tweet_rate = bot_tweets.groupby(['user_id','date','hour']).size().groupby('user_id').var()
variance_in_bot_tweet_rate.rename('variance_in_tweet_rate',inplace=True)
bot_users = bot_users.join(variance_in_bot_tweet_rate)

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

## Delete empty values
del genuine_tweets['geo'];
del genuine_tweets['contributors'];
del genuine_tweets['favorited'];
del genuine_tweets['retweeted'];

del bot_tweets['geo'];
del bot_tweets['contributors'];
del bot_tweets['favorited'];
del bot_tweets['retweeted'];