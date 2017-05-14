import tensorflow as tf
from code.CNNModel import Conv2DModel, Conv3DModel, CNNRNNModel
from code.read_data import inputBatch, inputNoShuffle
from keras.objectives import mean_squared_error
import tensorlayer as tl
import numpy as np
import pandas as pd

sess = tf.InteractiveSession()
epoch = 1
batchSize = 32

trainx, trainy = inputBatch("../data/train.tfrecords", batchSize)

cnn = Conv3DModel()
pred = cnn.inference(trainx, batchSize)

loss = tf.reduce_mean(tf.pow(trainy-pred, 2), axis=0)

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

testx = inputNoShuffle("../data/test.tfrecords", 20)



tl.utils.fit(sess, network, train_step, loss, trainx, trainy,
             batch_size=32, n_epoch=500, print_freq=5)

# with tf.Session() as sess:
#
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(epoch):
#         print("%d epoch is running..." %i)
#         trainxArr = sess.run(trainx)
#         print(trainxArr.shape)
#         #xArr = sess.run(network.outputs)
#         #print(xArr.shape)
#         sess.run(train_step)
#         lossArr = sess.run(loss)
#         print("loss", np.sqrt(lossArr))
#
#     testNum = 100
#     predList = []
#     for i in range(testNum):
#         print("%d epoch is running..." % i)
#         predsArr = sess.run(ans)
#         predList.extend(predsArr)
#
#         a = [j for i in predList for j in i]  # 展平list
#
#         res = pd.Series(a)
#         res.to_csv("../result/result_cnn3d_model.csv", index=False)
#
#
#     coord.request_stop()
#     coord.join(threads)
