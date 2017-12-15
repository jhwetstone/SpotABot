# SpotABot
Code for CS 229 Project I Spot a Bot

## File index
* analyze_data.py: trains models and outputs results & scores, generates AUPRC curves
* build_design_matrix.py: code for generating a normalized design matrix from raw inputs (From either csv files or twitter API)
* check_screenname.py: Scores twitter handle input from user
* explore_distribution_mismatch.py: Test adding examples from dev dataset to training to explore distribution mismatch hypothesis
* pca.py: run pca on the test and train distributions
* prepare_test_distribution.py:	downloads fresh tweets and user profile info from the Twitter API for annotated test distribution usernames, saves test and dev pickle files
* prepare_training_distribution.py: loads training data from csvs, saves train and train-dev pickle files
* regularization_hyper_tuning.py: Tunes hyper parameters and regularization constants for models 
* tweepy_utils.py: utility code for using the tweepy wrapper for the Twitter API

## Getting Started: Downloading Data
1. Create a folder called "data" in the SpotABot directory

### Training Data
1. Download cresci-2017.csv.zip from https://botometer.iuni.iu.edu/bot-repository/datasets.html
1. Unzip each of the files into SpotABot/data, in a new folder labeled 'cresci-2017'
1. Download cresci-2015.csv.zip from https://botometer.iuni.iu.edu/bot-repository/datasets.html
1. Unzip each of the files into SpotABot/data, in a folder labeled 'cresci-2015'

### Dev / Test Data
1. Download varol-2017 from https://botometer.iuni.iu.edu/bot-repository/datasets.html
1. Unzip into SpotABot/data in a file called 'varol-2017.dat'

## Prepare Datasets

### Training data: prepare_training_distribution.py
1. Combines datasets, generate features, and saves into train/test/dev pickle files

### Test data: prepare_test_distribution.py
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

## Models & Analysis: 
1. regularization_hyper_tuning.py: for tuning hyperparameters
1. analyze_data.py: primary results module
1. explore_distribution_mismatch.py: test adding examples from dev dataset to training to explore distribution mismatch hypothesis
1. pca.py: run pca on the test and train distributions

## Application Prototype:
1. check_screenname.py: Enter Twitter username when prompted

## Shared Files
1. tweepy_utils.py
1. build_design_matrix.py

## References
Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2017, April). The paradigm-shift of social spambots: Evidence, theories, and tools for the arms race. In Proceedings of the 26th International Conference on World Wide Web Companion (pp. 963-972). ACM.

Cresci, S., Di Pietro, R., Petrocchi, M., Spognardi, A., & Tesconi, M. (2015). Fame for sale: efficient detection of fake Twitter followers. Decision Support Systems, 80, 56-71.

Varol, Onur, Emilio Ferrara, Clayton A. Davis, Filippo Menczer, and Alessandro Flammini. "Online Human-Bot Interactions: Detection, Estimation, and Characterization." ICWSM (2017)


