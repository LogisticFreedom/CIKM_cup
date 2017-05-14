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

epoch = 4000
batchSize = 32

trainx, trainy = inputBatch("../data/train.tfrecords", batchSize)
#
# trainx = tf.cast(trainx, tf.float32)
# trainy = tf.cast(trainy, tf.float32)

#trainx = tf.reshape(trainx, [32, 15, 4, 101, 101]) # 恢复图像格式，batch*15个时间点*4个高度*长*宽

#lastImg = tf.slice(trainx, [0, 13, 3, 0, 0], [32, 1, 1, 101, 101]) # 切出最后一张雷达图
#lastImg = tf.reshape(lastImg, [32, 10201]) # 拉平

#lastImg = tf.slice(trainx, [0, 601858], [32, 10201])

inputImg = tf.nn.l2_normalize(trainx, dim=1)  # 标准化

# W = tf.Variable(tf.random_normal([612060, 1], stddev=0.01))
# b = tf.Variable(tf.random_normal([1], stddev=0.01))
# preds = tf.matmul(trainx, W)+b
# loss = tf.reduce_mean(tf.pow(preds-trainy, 2), 0)
linearModel = Sequential()
linearModel.add(Dense(units=1, input_dim=612060)) # 线性回归
#linearModel = Model(inputs=inputImg, outputs=preds)
preds = linearModel(inputImg)

loss = tf.reduce_mean(tf.pow(trainy-preds, 2), axis=0)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

testx = inputNoShuffle("../data/train.tfrecords", 20)
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

    testNum = 100
    predList = []
    for i in range(testNum):
        imgArr = sess.run(testx)
        print(imgArr.shape)
        preds = linearModel(testx)
        predsArr = sess.run(preds)
        print(predsArr)
        predList.extend(predsArr)

    a = [j for i in predList for j in i] # 展平list

    res = pd.Series(a)
    res.to_csv("../result/result_linear_model.csv", index=False)

    coord.request_stop()
    coord.join(threads)





