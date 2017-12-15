#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:20:35 2017

@author: sahilnayyar
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D;

X_train = pickle.load(open( "pickleFiles/X_train.p", "rb" ))
y_train = pickle.load(open( "pickleFiles/y_train.p", "rb" ))
y_train_dev = pickle.load(open( "pickleFiles/y_train_dev.p", "rb" ))
X_train_dev= pickle.load(open( "pickleFiles/X_train_dev.p", "rb" ))
X_test = pickle.load(open( "pickleFiles/X_test.p", "rb" ))
y_test = pickle.load(open( "pickleFiles/y_test.p", "rb" ))
X_dev = pickle.load(open( "pickleFiles/X_dev.p", "rb" ))
y_dev = pickle.load(open( "pickleFiles/y_dev.p", "rb" ))

X1 = np.concatenate((np.asarray(X_train), np.asarray(X_train_dev)))
y1 = np.concatenate((np.ravel(y_train), np.ravel(y_train_dev)))

X2 = np.concatenate((np.asarray(X_dev), np.asarray(X_test)))
y2 = np.concatenate((np.ravel(y_dev), np.ravel(y_test)))

U1,S1,V1 = np.linalg.svd(X1, full_matrices=False)
U2,S2,V2 = np.linalg.svd(X2, full_matrices=False)

fig = plt.figure(figsize=(10.5,5))
plt.plot(S1**2/sum(S1**2),'r')
plt.plot(S2**2/sum(S2**2),'b')
plt.legend(['Training dist.', 'Test dist.'])
plt.title('Dependence on principal components for both distributions');
plt.xlabel('Principal component no.')
plt.ylabel('Fraction of variance explained')

fig = plt.figure(figsize=(10.5,5))
ax = fig.add_subplot(121)
plt.plot(U1[y1==1,0],U1[y1==1,1],'.r')
plt.plot(U1[y1==0,0],U1[y1==0,1],'.k')
plt.title('Training distribution: Projection onto principal components')
plt.xlabel('u1')
plt.ylabel('u2')
plt.legend(['Bot','Gen. User']);

ax = fig.add_subplot(122, projection='3d')
ax.scatter(U1[y1==1,0],U1[y1==1,1],U1[y1==1,2],c='r')
ax.scatter(U1[y1==0,0],U1[y1==0,1],U1[y1==0,2],c='k')
ax.set_xlabel('u1')
ax.set_ylabel('u2')
ax.set_zlabel('u3')
plt.legend(['Bot','Gen. User']);

fig = plt.figure(figsize=(10.5,5))
ax = fig.add_subplot(121)
plt.plot(U2[y2==1,0],U2[y2==1,1],'b.')
plt.plot(U2[y2==0,0],U2[y2==0,1],'k.')
plt.title('Test distribution: Projection onto principal components')
plt.xlabel('u1')
plt.ylabel('u2')
plt.legend(['Bot','Gen. User']);

ax = fig.add_subplot(122, projection='3d')
ax.scatter(U2[y2==1,0],U2[y2==1,1],U2[y2==1,2],c='b')
ax.scatter(U2[y2==0,0],U2[y2==0,1],U2[y2==0,2],c='k')
ax.set_xlabel('u1')
ax.set_ylabel('u2')
ax.set_zlabel('u3')
plt.legend(['Bot','Gen. User']);
