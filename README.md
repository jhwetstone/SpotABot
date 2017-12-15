# SpotABot
Code for CS 229 Project I Spot a Bot

## File index
* analyze_data.py: trains models and outputs results & scores, generates AUPRC curves
* analyze_data_logistic_hyper.py:
* analyze_data_neural_net_hyper.py:
* build_design_matrix.py: code for generating a design matrix from raw inputs (From csv files or twitter API)
* build_test_distribution.py:	downloads fresh tweets and user profile info from the Twitter API for annotated test distribution usernames
* check_screenname.py: Scores twitter handle input from user
* hyperanalyze_data.py
* load_data.py: code for loading and combining training datasets
* pca.py	file
* tweepy_utils.py: utility code for using the tweepy wrapper for the Twitter API

## Getting Started: Downloading Data
### Training Data
1. Download cresci-2017.csv.zip from https://botometer.iuni.iu.edu/bot-repository/datasets.html
1. Unzip each of the files into the SpotABot directory, in a folder labeled 'cresci-2017'
1. Download cresci-2015.csv.zip from https://botometer.iuni.iu.edu/bot-repository/datasets.html
1. Unzip each of the files into the SpotABot directory, in a folder labeled 'cresci-2015'

### Dev / Test Data
1. Download varol-2017 from https://botometer.iuni.iu.edu/bot-repository/datasets.html
1. Unzip into the SpotABot directory, in a file called 'varol-2017.dat'

## Prepare Datasets

### Training data: load_data.py
1. Combines datasets, generate features, and saves into train/test/dev pickle files

### Test data: build_test_distribution.py
For the test distribution, we use tweepy as a wrapper for the Twitter API.  
1. Install Tweepy by running `pip install tweepy==3.5.0`
1. In order to connect to the twitter API, you must get your own developer consumer key, consumer secret, access token, and access token secret from https://apps.twitter.com/
1. Once you have them, create a settings.py file in the project directory, which should look like this:
```
consumer_key = "PASTED CONSUMER KEY HERE"
consumer_secret = "PASTED CONSUMER SECRET HERE"
access_token = "PASTED ACCESS TOKEN HERE"
access_token_secret = "PASTED ACCESS TOKEN SECRET HERE"
```

## Models & Analysis: analyze_data.py
1. Loads train/test/dev datasets from pickle files and runs the following models:
    * Logistic regression
    * SVM with Gaussian Kernel

## Data Documentation
cresci-2015
* Bot: INT, FSF, TWT
* Not: TFP, E13

cresci-2017
* Bot: social_spambots_1, social_spambots_2, social_spambots_3, traditional_spambots_1, traditional_spambots_2, traditional_spambots_3, traditional_spambots_4
* Not: genuine_accounts

## References
Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2017, April). The paradigm-shift of social spambots: Evidence, theories, and tools for the arms race. In Proceedings of the 26th International Conference on World Wide Web Companion (pp. 963-972). ACM.

Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2015). Fame for sale: efficient detection of fake Twitter followers. Decision Support Systems, 80, 56-71.
