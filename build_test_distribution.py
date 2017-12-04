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
import load_data
import time



def loadZafarTestData(path):

    filenames = os.listdir(path);
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
    df = df[~df.index.duplicated(keep='first')]
    
    X = df[df.columns.difference(['is_bot', 'source_identity'])];
    y = df['is_bot'];
    
    return X, y

def loadTestData(path):
    
    filename = 'varol-2017.dat'
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
    user_columns = ['id','friends_count','favourites_count','geo_enabled','name','screen_name','statuses_count','verified','followers_count']
    
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
    tweet_columns = ['id','favorite_count','retweet_count','created_at','text']
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
    path = 'classification_processed';
    y_z = loadTestData(path)
    test_users, test_tweets, y = downloadDatasets(y_z,devFlag=1)
    # Remove users that have no associated tweets
    test_users = test_users[test_users.index.isin(test_tweets.set_index('user_id').index)]
    test_tweets.rename(columns={'num_user_mentions': 'num_mentions'},inplace=True)
    test_tweets['timestamp_dt'] = pd.to_datetime(test_tweets['created_at'],infer_datetime_format=True)
    X_test = load_data.buildDesignMatrix(test_users,test_tweets)
    del y['is_active']
    y_test = y[y.index.isin(test_users.index)]
            
    pickle.dump(X_test, open( "X_test.p", "wb" ))
    pickle.dump(y_test, open( "y_test.p", "wb" ))
    
    ##FIXME: Still need to do the test/dev split

main()