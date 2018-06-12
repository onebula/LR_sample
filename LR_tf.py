#!/usr/bin/python
# -*- coding: utf-8 -*-

ClassNum = 3
TrainBatchNum = 100
batch_size_train = 20
batch_size_test = 10
FeatureSize = 4

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

DataDir = './TrainData/train*'
pDataDir = './TestData/test*'

st = time.time()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 占用GPU50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
tf.reset_default_graph()
sess = tf.Session(config=config)

# 学习率
lr = 0.005
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = tf.placeholder(tf.int32, [])

def readMyFileFormat(fileNameQueue):
    reader = tf.TextLineReader()
    key, value = reader.read(fileNameQueue)
    record_defaults = [[0]] + [[0.0]] * 4
    user = tf.decode_csv(value, record_defaults=record_defaults)
    userlabel = user[0]
    userlabel01 = tf.cast(tf.one_hot(userlabel,ClassNum,1,0), tf.float32)
    userfeature = user[1:]
    return userlabel01, userfeature

def inputPipeLine_batch(fileNames, batchSize, numEpochs = None):
    fileNameQueue = tf.train.string_input_producer(fileNames, num_epochs = numEpochs, shuffle = False )
    example = readMyFileFormat(fileNameQueue)
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size_train
    YBatch, XBatch = tf.train.batch(
        example, batch_size = batchSize, 
        capacity = capacity)
    return YBatch, XBatch

filenames = tf.train.match_filenames_once(DataDir)
YBatch, XBatch = inputPipeLine_batch(filenames, batchSize = batch_size, numEpochs = 20)
pfilenames = tf.train.match_filenames_once(pDataDir)
pYBatch, pXBatch = inputPipeLine_batch(pfilenames, batchSize = batch_size, numEpochs = 1)

# LR
X_LR = tf.placeholder(tf.float32, [None, FeatureSize])
Y_LR = tf.placeholder(tf.float32, [None, ClassNum])
W_LR = tf.Variable(tf.truncated_normal([FeatureSize, ClassNum], stddev=0.1), dtype=tf.float32)
bias_LR = tf.Variable(tf.constant(0.1,shape=[ClassNum]), dtype=tf.float32)
Ypred_LR = tf.matmul(X_LR, W_LR) + bias_LR
Ypred_prob = tf.nn.softmax(Ypred_LR)
cost = -tf.reduce_mean(Y_LR*tf.log(Ypred_prob))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)


init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()  # local variables like epoch_num, batch_size
sess.run(init_op)
sess.run(local_init_op)

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# 训练
try:
    for i in range(TrainBatchNum):
        print i
        y, x = sess.run([YBatch, XBatch], feed_dict={batch_size: batch_size_train})
        flag, c = sess.run([optimizer, cost], feed_dict={X_LR: x, Y_LR: y})
        print c
except tf.errors.OutOfRangeError:
    print 'Done Train'
    
# 测试
Y = np.array([0, 0, 0])
Pred = np.array([0, 0, 0])
try:
    i = 0
    while True:
        print i
        i = i + 1
        y, x = sess.run([pYBatch, pXBatch], feed_dict={batch_size: batch_size_test})
        pred = sess.run(Ypred_prob, feed_dict={X_LR: x, Y_LR: y})
        Pred = np.vstack([Pred,pred])
        Y = np.vstack([Y,y])
except tf.errors.OutOfRangeError:
    print 'Done Test'

Y = Y[1:]
Pred = Pred[1:]
acc = accuracy_score(np.argmax(Y, axis = 1),np.argmax(Pred, axis = 1))
print acc
A = confusion_matrix(np.argmax(Y, axis = 1),np.argmax(Pred, axis = 1))
print A