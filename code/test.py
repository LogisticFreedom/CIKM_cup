import tensorlayer as tl
import tensorflow as tf
import pandas as pd
from code.read_data import inputNoShuffle
from code.CNNModel import Conv3DModel

batchSize = 200
testNum = 10

testx = inputNoShuffle("../data/test.tfrecords", batchSize)

cnn = Conv3DModel()
network = cnn.inference(testx, batchSize)

preds = network.outputs

with tf.Session() as sess:

    load_params = tl.files.load_npz(path='../model/', name='cnn3d_model.npz')
    tl.files.assign_params(sess, load_params, network)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    predList = []
    for i in range(testNum):
        print("%d epoch is running..." % i)
        testArr = sess.run(testx)
        print(testArr.shape)
        predArr = sess.run(preds)
        predList.extend(predArr)

    a = [j for i in predList for j in i]  # 展平list

    res = pd.Series(a)
    res.to_csv("../result/result_cnn3d_model.csv", index=False)

    coord.request_stop()
    coord.join(threads)
