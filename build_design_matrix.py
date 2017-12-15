#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:42:57 2017

@author: jessicawetstone
"""

import pickle
from sklearn.preprocessing import StandardScaler

def buildDesignMatrix(users, tweets, trainFlag=0):
    ## Add friend-follower ratio as a feature
    users.loc[:,'friend_follower_ratio'] = users['friends_count']/(users['followers_count'] + 1) ## To avoid divide by zero errors
    
    ## Add average-per-tweet information
    averages = tweets[['num_hashtags','retweet_count','favorite_count','num_mentions','num_urls','user_id']].groupby('user_id').mean()
    ## Reply count is sometimes blank -- replace with 0's
    averages = averages.fillna(0)
    users = users.join(averages.add_suffix('_per_tweet'))
    
    ## Add number of unique places where the user has tweeted.  
    # This will be 0 if no places have ever been tagged
    users.loc[:,'unique_tweet_places'] = tweets.groupby('user_id').place.nunique()

    ## Add: Variance in the user's number of tweets per hour.
    tweets.loc[:,'date'] = tweets.timestamp_dt.dt.date
    tweets.loc[:,'hour'] = tweets.timestamp_dt.dt.hour
    variance_in_tweet_rate = tweets.groupby(['user_id','date','hour']).size().groupby('user_id').var()
    variance_in_tweet_rate.rename('variance_in_tweet_rate',inplace=True)
    users = users.join(variance_in_tweet_rate)

    ## Feature selection code should go here (slimming down the users tables)
    X = users[['favourites_count','followers_count','friends_count','verified','friend_follower_ratio','num_hashtags_per_tweet','retweet_count_per_tweet','favorite_count_per_tweet','num_mentions_per_tweet','num_urls_per_tweet','unique_tweet_places','variance_in_tweet_rate']]
    
    values = {'verified': 0, 'variance_in_tweet_rate': 0}
    X = X.fillna(value=values)
    
    if trainFlag == 1:
        scaler = StandardScaler()
        scaler = scaler.fit(X)
        pickle.dump(scaler, open("pickleFiles/scaler.p","wb"))
        X.loc[:,:] = scaler.transform(X)
    else:
        scaler = pickle.load(open("pickleFiles/scaler.p","rb"))
        X.loc[:,:] = scaler.transform(X)
    
    return X