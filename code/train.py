import tensorflow as tf
from code.CNNModel import Conv2DModel, Conv3DModel, CNNRNNModel
from code.read_data import inputBatch, inputNoShuffle
from keras.objectives import mean_squared_error
import tensorlayer as tl
import numpy as np
import pandas as pd


epoch = 10
batchSize = 32

trainx, trainy = inputBatch("../data/train.tfrecords", batchSize)
trainy = tf.log1p(trainy) # log平滑


cnn = Conv3DModel()
network = cnn.inference(trainx, batchSize)

pred = network.outputs

loss = tf.reduce_mean(tf.pow(trainy-pred, 2), axis=0)

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# testx = inputNoShuffle("../data/test.tfrecords", 20)
#
# ans = cnn.inference(testx, 20)

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(epoch):
        print("%d epoch is running..." %i)
        trainxArr = sess.run(trainx)
        print(trainxArr.shape)
        sess.run(train_step)
        lossArr = sess.run(loss)
        print("loss", np.sqrt(lossArr))

    # testNum = 100
    #
    tl.files.save_npz(network.all_params, name='../model/cnn3d_model.npz')

    coord.request_stop()
    coord.join(threads)
