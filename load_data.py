#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:54:48 2017

@author: jessicawetstone
"""

## Initialization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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

users = pd.concat([bot_users,genuine_users])
tweets = pd.concat([bot_tweets,genuine_tweets])

# Remove users that have no associated tweets
users = users[users.index.isin(tweets.set_index('user_id').index)]

user_class = pd.concat([pd.DataFrame(np.ones(shape=(len(bot_users),1))),pd.DataFrame(np.zeros(shape=(len(genuine_users),1)))])
user_class.rename(columns={0:'is_bot'},inplace=True);

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

## Add friend-follower ratio as a feature
users['friend_follower_ratio'] = users['friends_count']/users['followers_count']
## Remove divide-by-0 errors
users.loc[~np.isfinite(users['friend_follower_ratio']), 'friend_follower_ratio'] = np.nan

## Add average-per-tweet information
averages = tweets[['num_hashtags','reply_count','retweet_count','favorite_count','num_mentions','num_urls','user_id']].groupby('user_id').mean()
## Reply count is sometimes blank -- replace with 0's
averages = averages.fillna(0)
users = users.join(averages.add_suffix('_per_tweet'))

## Add number of unique places where the user has tweeted.  
# This will be 0 if no places have ever been tagged
users['unique_tweet_places'] = tweets.groupby('user_id').place.nunique()

## Update timestamp values to be datetime
tweets['timestamp_dt'] = pd.to_datetime(tweets['timestamp'],format='%Y-%m-%d %H:%M:%S')
del tweets['timestamp']

## Add: Variance in the user's number of tweets per hour.
tweets['date'] = tweets.timestamp_dt.dt.date
tweets['hour'] = tweets.timestamp_dt.dt.hour
variance_in_bot_tweet_rate = tweets.groupby(['user_id','date','hour']).size().groupby('user_id').var()
variance_in_bot_tweet_rate.rename('variance_in_tweet_rate',inplace=True)
users = users.join(variance_in_bot_tweet_rate)


## Feature selection code should go here (slimming down the users tables)
X = users[['favourites_count','followers_count','friends_count','verified','friend_follower_ratio','num_hashtags_per_tweet','reply_count_per_tweet','retweet_count_per_tweet','favorite_count_per_tweet','num_mentions_per_tweet','num_urls_per_tweet','unique_tweet_places','variance_in_tweet_rate']]
X['verified'] = X['verified'].fillna(0)
## Split data into train/dev/test 

## Find location rid of null values
iNotNull = pd.notnull(X).all(1).nonzero()[0]

X_train, X_test, y_train, y_test = train_test_split(X.iloc[iNotNull], user_class.iloc[iNotNull], test_size=0.4)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5)

logistic = LogisticRegression()
fitter = logistic.fit(X_train,y_train)
