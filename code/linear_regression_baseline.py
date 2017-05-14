import keras as K
import tensorflow as tf
import numpy as np
import pandas as pd
from code.read_data import inputBatch, inputData, inputNoShuffle
from keras.layers import Dense
from keras.models import Model, Sequential
from keras.objectives import mean_squared_error
from tensorlayer.utils import flatten_list
from keras.layers.normalization import BatchNormalization

epoch = 1000
batchSize = 32

trainx, trainy = inputBatch("../data/train.tfrecords", batchSize)

inputImg = tf.nn.l2_normalize(trainx, dim=1)  # 标准化
trainy = tf.log1p(trainy) # label log平滑

linearModel = Sequential()
linearModel.add(Dense(units=1, input_dim=612060)) # 线性回归
#linearModel = Model(inputs=inputImg, outputs=preds)
preds = linearModel(inputImg)

loss = tf.reduce_mean(tf.pow(trainy-preds, 2), axis=0)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

testBatchSize = 100
testNum = 20

testx = inputNoShuffle("../data/train.tfrecords", testBatchSize)
testx = tf.nn.l2_normalize(testx, dim=1)

predTest = linearModel(testx)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(epoch):
        print("%d epoch" %i)
        sess.run(train_step)
        lossArr = sess.run(loss)
        print("loss:", np.sqrt(lossArr))  # 评价指标是RMSE

    # fileNameQue = tf.train.string_input_producer(["../data/train.tfrecords"])
    # testImg = inputData(fileNameQue, trainFlag=False)

    predList = []
    for i in range(testNum):
        imgArr = sess.run(testx)
        print(imgArr.shape)
        preds = linearModel(testx)
        predsArr = sess.run(preds)
        print(predsArr)
        predsArr = np.expm1(predsArr) # exp数据还原
        predList.extend(predsArr)

    a = [j for i in predList for j in i] # 展平list

    res = pd.Series(a)
    res.to_csv("../result/result_linear_model.csv", index=False)

    coord.request_stop()
    coord.join(threads)





