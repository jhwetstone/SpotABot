# SpotABot
Code for CS 229 Project I Spot a Bot

## Getting Started: Downloading Data
1. Download cresci-2017.csv.zip
1. Unzip each of the files into the SpotABot directory, in a folder labeled 'cresci-2017'
1. Download cresci-2015.csv.zip
1. Unzip each of the files into the SpotABot directory, in a folder labeled 'cresci-2015'

## Prepare Datasets: load_data.py
1. Combines datasets, generate features, and saves into train/test/dev pickle files

## Models & Analysis: analyze_data.py
1. Loads train/test/dev datasets from pickle files and runs the following models:
1. * Logistic regression
1. * SVM with Gaussian Kernel

## Data Documentation
cresci-2015
* Bot: INT, FSF, TWT
* Not: TFP, E13

cresci-2017
* Bot: social_spambots_1, social_spambots_2, social_spambots_3, traditional_spambots_1, traditional_spambots_2, traditional_spambots_3, traditional_spambots_4
* Not: genuine_accounts