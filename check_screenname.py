#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:05:35 2017

@author: jessicawetstone
"""

import tweepy
import tweepy_utils
import pickle
import pandas as pd
import build_design_matrix


api = tweepy_utils.connect_to_api()
model = pickle.load(open("model.p", "rb"))

while True:
    screen_name=input("Enter the screen name to check: ")
    
    user_columns = ['id','friends_count','favourites_count','geo_enabled','name','screen_name','statuses_count','verified','followers_count']
    try:
        user_info = api.get_user(screen_name=screen_name)
        user_json = user_info._json
        test_user = pd.DataFrame([{col: user_json[col] for col in user_columns}])
        test_user.set_index('id',inplace=True)
        user_id = user_json['id']
        
        test_tweet_list = []
        tweet_columns = ['id','favorite_count','retweet_count','created_at','text']
        tweet_entities = ['hashtags','urls','user_mentions']
        tweets = api.user_timeline(user_id=user_id, count=200)
        for status_info in tweets:
            status_json = status_info._json
            status_dict = {'user_id':user_id}
            status_dict.update({'num_'+en: len(status_json['entities'][en]) for en in tweet_entities})
            status_dict.update({col: status_json[col] for col in tweet_columns})
            status_place = status_json['place']
            if status_place is not None:
                status_place = status_place['name']
            status_dict.update({'place':status_place})
            test_tweet_list.append(status_dict)
        if len(test_tweet_list) == 0:
            print("Sorry, this user doesn't have any tweets yet! Please try again.")
            continue
        test_tweets = pd.DataFrame(test_tweet_list)
        test_tweets.set_index('id',inplace=True)
        test_tweets.rename(columns={'num_user_mentions': 'num_mentions'},inplace=True)
        test_tweets['timestamp_dt'] = pd.to_datetime(test_tweets['created_at'],infer_datetime_format=True)
        X_test = build_design_matrix.buildDesignMatrix(test_user,test_tweets)
        if hasattr(model,'predict_proba'):
            scores = model.predict_proba(X_test)
            print("There is a %d%% chance that %s is a bot" % (scores[0][1] * 100, screen_name))
        elif model.predict(X_test) == 1:
            print("%s is a bot" % screen_name)
        else:
            print("%s is a human" % screen_name)
    ## Catch errors
    except tweepy.TweepError as e:
        if e.api_code == 50 or e.api_code == 63:
            print("Sorry, " + screen_name + " is not a valid screen name.  Please try again.")
            continue
        if e.reason == "Not authorized.":
            print("Sorry, we are not authorized to view " + screen_name + "'s tweets.  Please try a different screen name.")
            continue
        else:
            print('Error: ' + str(e.api_code) + ' ' + e.reason )
            print("Uh oh - we've experienced an error! Please try again.")
            continue

