# SpotABot
Code for CS 229 Project I Spot a Bot

Working Doc: https://docs.google.com/document/d/1zYeQP1R2YI-tIu8YIyi-dPNhKB_qhv_8BlvgJGwR8Po/edit

## Getting Started: Downloading Data
1. Download cresci-2017.csv.zip from https://botometer.iuni.iu.edu/bot-repository/datasets.html
1. Unzip each of the files into the SpotABot directory, in a folder labeled 'cresci-2017'
1. Download cresci-2015.csv.zip from https://botometer.iuni.iu.edu/bot-repository/datasets.html
1. Unzip each of the files into the SpotABot directory, in a folder labeled 'cresci-2015'

## Prepare Datasets: load_data.py
1. Combines datasets, generate features, and saves into train/test/dev pickle files

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
