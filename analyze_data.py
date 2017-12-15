#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:01:45 2017

@author: sahilnayyar
"""

## Initialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model, ensemble, neural_network
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


## Load our test, train, dev objects
X_train = pickle.load(open( "pickleFiles/X_train.p", "rb" ))
X_test = pickle.load(open( "pickleFiles/X_test.p", "rb" ))
X_dev = pickle.load(open( "pickleFiles/X_dev.p", "rb" ))
y_train = pickle.load(open( "pickleFiles/y_train.p", "rb" ))
y_test = pickle.load(open( "pickleFiles/y_test.p", "rb" ))
y_dev = pickle.load(open( "pickleFiles/y_dev.p", "rb" ))
y_train_dev = pickle.load(open( "pickleFiles/y_train_dev.p", "rb" ))
X_train_dev= pickle.load(open( "pickleFiles/X_train_dev.p", "rb" ))


BETA = 0.5
datasets = {}
datasets['Train'] = {'Actuals': y_train, 'Design Matrix': X_train}
datasets['Train Dev'] = {'Actuals': y_train_dev, 'Design Matrix': X_train_dev}
datasets['Dev'] = {'Actuals': y_dev, 'Design Matrix': X_dev}
datasets['Test'] = {'Actuals': y_test, 'Design Matrix': X_test}

models = {}

## Creation of model objects
logistic = linear_model.LogisticRegression(C=2.02)
models['Logistic']={'model': logistic}
gbm = ensemble.GradientBoostingClassifier(learning_rate=0.245,n_estimators=100)
models['GBM'] = {'model': gbm}
neural_net = neural_network.MLPClassifier(solver = 'lbfgs', alpha = 0.01, random_state=1, hidden_layer_sizes=(3,4))
models['Neural Net'] = {'model': neural_net}

## Application of fits 
for model_name, model_object in models.items():
    model = model_object['model']
    model.fit(np.asmatrix(X_train),np.ravel(y_train))

## Predicted classifications
for model_name, model_object in models.items():
    model = model_object['model']
    for dataset_name, dataset in datasets.items():
        model_object[dataset_name + ' Result'] = model.predict(dataset['Design Matrix'])
        model_object[dataset_name + ' Accuracy'] = model.score(dataset['Design Matrix'],dataset['Actuals'])
    
## Calculate scores
for model_name, model_object in models.items():
    model = model_object['model']
    for dataset_name, dataset in datasets.items():
        model_object[dataset_name + ' Precision'] = precision_score(dataset['Actuals'], model_object[dataset_name + ' Result'])
        model_object[dataset_name + ' Recall'] = recall_score(dataset['Actuals'], model_object[dataset_name + ' Result'])
        model_object[dataset_name + ' F Score'] = fbeta_score(dataset['Actuals'], model_object[dataset_name + ' Result'], BETA)
#)

## Print results
reporting_fields = ['Accuracy','Precision','Recall','F Score']
data = []
index = []
for dataset_name, dataset_object in datasets.items():
    for rfield in reporting_fields:
        data.append([models[model_name][dataset_name + ' ' + rfield] for model_name in models])
        index.append(dataset_name + ' ' + rfield)
print( pd.DataFrame(data = data
                    ,index = index
                    ,columns = [model_name for model_name in models])
)    

## Generate PRC curves
plt.figure(figsize=(4.65,4.17))
for model_name, model_object in models.items():
    if model_name in ('Logistic','GBM','Neural Net'):
        actuals = datasets['Test']['Actuals']
        model = model_object['model']
        if hasattr(model,'decision_function'):
            predicted = model.decision_function(datasets['Test']['Design Matrix'])
        else:
            predicted = model.predict_proba(datasets['Test']['Design Matrix'])[:,1];
        
        average_precision = average_precision_score(actuals,predicted)
        precision, recall, _ = precision_recall_curve(actuals,predicted)
        plt.step(recall, precision, where='post', label= model_name + ': AUPRC={0:0.2f}'.format(
          average_precision)) 
        
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.xticks(fontsize=10);
plt.yticks(fontsize=10);
ax=plt.subplot(111)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels)
plt.savefig("precision-recall")

best_model = [(model_name, model_object['model']) for model_name, model_object in models.items() if model_object['Dev F Score'] == np.max([model_object['Dev F Score'] for model_name, model_object in models.items()])][0]

print('Saving the ' + best_model[0] + ' model to model.p')
## Save our final model (For "check_screenname.py")
pickle.dump(best_model[1],open("pickleFiles/model.p","wb"))

### Print logistic model coefficients
coeffs = np.insert(logistic.coef_,0,logistic.intercept_)
df = pd.DataFrame(data = coeffs
                    ,index = np.insert(X_train.columns,0,'intercept')
                    ,columns = ['logistic coefficients'])
df = df.reindex(df['logistic coefficients'].abs().sort_values().index)
print(df)