from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers import Input, Flatten, Dense, Activation, Dropout
from keras.models import Model
import tensorflow as tf
import tensorlayer as tl
from keras.models import load_model
import numpy as np
import os
import PIL
from PIL import Image
from keras import optimizers
import pandas as pd

# 模型基类
class BaseModel():

    def __init__(self, load = False):
        pass

    def buildModel(self): # 子类实现具体模型结构
        pass


class Conv2DModel():

    def inference(self, x):
        conv1 = tl.layers.Conv2d(x, filter_size=(3, 3), strides=(1,1))
        maxpool1 = tl.layers.PoolLayer(conv1, pool=tf.nn.max_pool)

        output = maxpool1
        return output


class Conv3DModel():

    def inference(self, x, batchSize, trainFlag = True):

        x = tf.nn.l2_normalize(x, dim=1)  # 标准化

        x = tf.reshape(x, [batchSize, 15, 4, 101, 101])
        x = tf.transpose(x, [0, 1, 3, 4, 2])

        inputx = tl.layers.InputLayer(x, name='input_layer')

        network = tl.layers.Conv3dLayer(inputx, act=tf.nn.relu, shape=[3, 3, 3, 4, 32], name="conv1")
        network = tl.layers.MaxPool3d(network, filter_size=[2, 2, 2], strides=[1, 1, 1], data_format="channels_last", name="maxpool1")

        network = tl.layers.Conv3dLayer(network, act=tf.nn.relu, shape=[3, 3, 3, 32, 64], name="conv2")
        network = tl.layers.MaxPool3d(network, filter_size=[2, 2, 2], strides=[ 1, 1, 1],
                                       data_format="channels_last", name="maxpool2")

        network = tl.layers.Conv3dLayer(network, act=tf.nn.relu, shape=[3, 3, 3, 64, 128], name="conv3")
        network = tl.layers.MaxPool3d(network, filter_size=[2, 2, 2], strides=[ 1, 1, 1],
                                       data_format="channels_last", name="maxpool3")

        # network = tl.layers.Conv3dLayer(network, act=tf.nn.relu, shape=[3, 3, 3, 128, 256], name="conv4")
        # network = tl.layers.MaxPool3d(network, filter_size=[2, 2, 2], strides=[1, 1, 1],
        #                                data_format="channels_last", name="maxpool4")

        network = tl.layers.FlattenLayer(network)
        network = tl.layers.DenseLayer(network, n_units=256, act=tf.nn.relu, name="fc1")
        network = tl.layers.DenseLayer(network, n_units=128, act=tf.nn.relu, name="fc2")
        network = tl.layers.DropoutLayer(network, keep=0.6, is_train=trainFlag, is_fix=True)
        output = tl.layers.DenseLayer(network, n_units=1, act=tf.nn.relu)

        output = output.outputs

        return output

class CNNRNNModel():

    def inference(self, x, batchSize):

        x = tf.nn.l2_normalize(x, dim=1)  # 标准化
        x = tf.reshape(x, [batchSize, 15, 4, 101, 101])
        x = tf.transpose(x, [0, 1, 3, 4, 2])

        tensorList = tf.split(x, num_or_size_splits=15, axis=1)

        conv2dModel = Conv2DModel()
        outTensorList = []
        for img in tensorList:
            img = tf.reshape(img, [batchSize, 101, 101, 4])
            out = conv2dModel.inference(img)
            outTensorList.append(out)

        dim = 512
        sequence = tf.stack(outTensorList, axis=0)
        sequence = tf.reshape(sequence, shape=[batchSize, 15, dim])
        sequence = tf.transpose(sequence, [0, 2, 1])

        network = tl.layers.RNNLayer(layer=sequence, cell_fn=tf.contrib.rnn.BasicLSTMCell,
                                  n_hidden=100, n_steps = 15, return_last = False, return_seq_2d = True, name = 'rnn_layer')

        pred = tl.layers.DenseLayer(network, n_units=1, act=tf.nn.relu)

        return pred














