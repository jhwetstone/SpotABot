#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 21:32:23 2017

@author: jessicawetstone
"""
import os
import pandas as pd
import tweepy
import tweepy_utils
import pickle
import build_design_matrix
from sklearn.model_selection import train_test_split

def loadTestData():
    
    filename = 'data/varol-2017.dat'
    df_file = pd.read_csv(filename, sep='\t',header = None)
    
    df_file.rename(columns={0: 'id', 1: 'is_bot'},inplace=True)
    
    df_file.set_index('id',inplace=True)
    
    return df_file
    

def downloadDatasets(y_z, devFlag=1):
    
    ## Use dev flag to restrict to twenty users for testing purposes
    if devFlag:
        limit = 21
    else:
        limit = 100000
    cursor = 1
    
    ## FIRST: Download all user information for users who are still active on Twitter (accounts have not been deactivated or suspended)
    
    ## Download users
    user_columns = ['id','friends_count','favourites_count','geo_enabled','name','screen_name','statuses_count','verified','followers_count',
                    'default_profile','default_profile_image','description','has_extended_profile','profile_background_color','statuses_count','listed_count','url']
    
    test_user_list = [];
    # is_active column keeps track of the users who are still active on twitter
    # adding the user's id field to have a consistent index
    y = pd.DataFrame(y_z,columns=['is_bot','is_active'])
    api = tweepy_utils.connect_to_api()
    for user_id in y.index:
        if cursor > limit:
            break
        try:
            user_info = api.get_user(user_id=user_id)
            y.loc[user_id,'is_active'] = 1
            user_json = user_info._json
            test_user_list.append({col: user_json[col] for col in user_columns})
            cursor += 1
        ## Catch errors
        except tweepy.TweepError as e:
            if e.api_code == 50 or e.api_code == 63:
                y.loc[user_id,'is_active'] = 0
                continue
            else:
                print('Error: ' + str(e.api_code) + ' ' + e.reason )
                y.loc[user_id,'is_active'] = 0
                continue
    
    test_users = pd.DataFrame(test_user_list)
    test_users.set_index('id',inplace=True)
    print('Pickling test users')
    pickle.dump(test_users, open( "test_users.p", "wb" ))   
    
    ## NEXT: Download tweets corresponding to the active users
    test_tweet_list = []
    tweet_columns = ['id','favorite_count','retweet_count','created_at','text','lang','retweeted','source']
    tweet_entities = ['hashtags','urls','user_mentions']
    cursor = 1
    for user_id in test_users.index:
        if cursor > limit:
            break
        if cursor % 100 == 0:
            print('Pickling tweet list - size: ' + str(len(test_tweet_list)))
            pickle.dump(test_tweet_list, open( "test_tweet_list.p", "wb" ))
        try:
            for status_info in tweepy.Cursor(api.user_timeline, user_id=user_id, count=200).items(1000):
                status_json = status_info._json
                status_dict = {'user_id':user_id}
                status_dict.update({'num_'+en: len(status_json['entities'][en]) for en in tweet_entities})
                status_dict.update({col: status_json[col] for col in tweet_columns})
                status_place = status_json['place']
                if status_place is not None:
                    status_place = status_place['name']
                status_dict.update({'place':status_place})
                test_tweet_list.append(status_dict)
            cursor += 1
        except tweepy.TweepError as e:
            print('Error: ' + str(e.api_code) + ' ' + e.reason )
            continue
    
    test_tweets = pd.DataFrame(test_tweet_list)
    test_tweets.set_index('id',inplace=True)
    return test_users, test_tweets, y

def main():
    y_z = loadTestData()
    test_users, test_tweets, y = downloadDatasets(y_z,devFlag=0)
    print('Pickling test users')
    pickle.dump(test_users, open( "test_users.p", "wb" ))
    print('Pickling test y')
    pickle.dump(y, open("test_y.p","wb"))
    print('Pickling test tweets')
    pickle.dump(test_tweets, open("test_tweets.p","wb"))
    
    #test_users = pickle.load(open( "test_users.p", "rb" ))
    #y = pickle.load(open( "test_y.p", "rb" ))
    #test_tweets = pickle.load(open( "test_tweets.p", "rb" ))
    
    # Remove users that have no associated tweets
    test_users = test_users[test_users.index.isin(test_tweets.set_index('user_id').index)]
    test_tweets.rename(columns={'num_user_mentions': 'num_mentions'},inplace=True)
    test_tweets['timestamp_dt'] = pd.to_datetime(test_tweets['created_at'],infer_datetime_format=True)
    X_test = build_design_matrix.buildDesignMatrix(test_users,test_tweets)
    del y['is_active']
    y_test = y[y.index.isin(test_users.index)]
    
    X_test, X_dev , y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5)
    
    pickle.dump(X_test, open( "pickleFiles/X_test.p", "wb" ))
    pickle.dump(y_test, open( "pickleFiles/y_test.p", "wb" ))
    
    pickle.dump(X_dev, open( "pickleFiles/X_dev.p", "wb" ))
    pickle.dump(y_dev, open( "pickleFiles/y_dev.p", "wb" ))
    
main()