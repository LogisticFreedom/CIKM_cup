import tensorflow as tf
from code.CNNModel import Conv2DModel, Conv3DModel, CNNRNNModel
from code.read_data import inputBatch
from keras.objectives import mean_squared_error

epoch = 1000
batchSize = 32

trainx, trainy = inputBatch("../data/train.tfrecords", batchSize)

cnn = Conv3DModel()

pred = cnn.inference(trainx, batchSize)

loss = tf.reduce_mean(mean_squared_error(trainy, pred))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(epoch):
        print("%d epoch is running..." %i)
        sess.run(train_step)
        lossArr = sess.run(loss)
        print("loss", lossArr)

    coord.request_stop()
    coord.join(threads)
