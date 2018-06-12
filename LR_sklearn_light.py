#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time

from sklearn import datasets
Data = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(Data.data,Data.target,test_size=0.2, random_state=0)

# 训练
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 测试
st1 = time.time()
predictions = classifier.predict(X_test)
acc = accuracy_score(y_test,predictions)
ed1 = time.time()
print ed1-st1, acc

A = confusion_matrix(y_test, predictions)
print A

def softmax(x):
    e_x = np.exp(x - np.max(x)) # 防止exp()数值溢出
    return e_x / e_x.sum(axis=0)
pred = [np.argmax(softmax(np.dot(classifier.coef_, X_test[i,:]) + classifier.intercept_)) for i in range(len(X_test))] 
print np.sum(pred != predictions) # 检查是否存在差异