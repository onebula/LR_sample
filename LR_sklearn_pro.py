#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import time
import glob

classifier = SGDClassifier(loss = 'log')

# 训练
filenames = sorted(glob.glob("./TrainData/train*"))
MaxIterNum = 100
count = 0
for c, filename in enumerate(filenames):
    TrainDF = pd.read_csv(filename, header = None, chunksize = 10)
    for Batch in TrainDF:
        count += 1
        print count
        
        y_train = np.array(Batch.iloc[:,0])
        X_train = np.array(Batch.iloc[:,1:])
        
        st1 = time.time()
        classifier.partial_fit(X_train,y_train, classes=np.array([0, 1, 2]))
        ed1 = time.time()
        st2 = time.time()
        predictions = classifier.predict(X_train)
        acc = accuracy_score(y_train,predictions)
        ed2 = time.time()
        print ed1-st1, ed2-st2, acc
        
        if count == MaxIterNum:
            break
    if count == MaxIterNum:
        break
        
# 测试
X_test = np.zeros([100, np.shape(X_train)[1]])
y_test = np.zeros(100)
TestSampleNum = 0

filenames = sorted(glob.glob("./TestData/test*"))
MaxIterNum = 100
count = 0
for c, filename in enumerate(filenames):
    TestDF = pd.read_csv(filename, header = None, chunksize = 10)
    for Batch in TestDF:
        count += 1
        print count
        
        y_test[TestSampleNum:TestSampleNum+np.shape(Batch)[0]] = np.array(Batch.iloc[:,0])
        X_test[TestSampleNum:TestSampleNum+np.shape(Batch)[0],:] = np.array(Batch.iloc[:,1:])
        TestSampleNum = TestSampleNum+np.shape(Batch)[0]
        
        if count == MaxIterNum:
            break
    if count == MaxIterNum:
        break

X_test = X_test[0:TestSampleNum,:]
y_test = y_test[0:TestSampleNum]

st1 = time.time()
predictions = classifier.predict(X_test)
acc = accuracy_score(y_test,predictions)
ed1 = time.time()
print ed1-st1, acc

A = confusion_matrix(y_test, predictions)
print A