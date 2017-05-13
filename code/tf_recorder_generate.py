import tensorflow as tf
import numpy as np
import pandas as pd

def praser(line):
#
    id = line[:, 0][0]
    label = line[:, 1][0]
    rawImg = np.array(list(map(np.int, line[:, 2][0].split(" "))))
    #imgSeq = rawImg.reshape((15, 4, 101, 101))
    imgSeq = rawImg
    return imgSeq, label

def create_TFRecorder(filePath, TFwriter, trainFlag = True):

    dataReader = pd.read_csv(filePath, delimiter=",", iterator=True) # 迭代读取

    loop = True
    while loop:
        try:
            line = dataReader.get_chunk(1)
            x, y = praser(line.values) # 解析每一条记录
            print(x.shape)
            print(y)
            # x = np.int64(x)
            # y = np.float32(y)

            if trainFlag:
                # 以浮点型存储label，以int64存储img

                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
                    "img": tf.train.Feature(int64_list=tf.train.Int64List(value=x))
                }))
                TFwriter.write(example.SerializeToString())
            else:
                example = tf.train.Example(features=tf.train.Features(feature={
                    "img": tf.train.Feature(int64_list=tf.train.Int64List(value=x))
                }))
                TFwriter.write(example.SerializeToString())
        except StopIteration:
            loop = False
            print("Iteration is stopped.")

if __name__ == "__main__":

    #filePath = "../data/CIKM2017_train/data_new/CIKM2017_train/data_sample.txt"
    # filePath = "../data/CIKM2017_train/data_new/CIKM2017_train/train.txt"
    # TFwriter_train = tf.python_io.TFRecordWriter("../data/train.tfrecords")
    #
    # create_TFRecorder(filePath, TFwriter=TFwriter_train)

    filePath = "../data/CIKM2017_testA/data_new/CIKM2017_testA/testA.txt"
    TFwriter_test = tf.python_io.TFRecordWriter("../data/test.tfrecords")
    create_TFRecorder(filePath, TFwriter=TFwriter_test)

    # f = generate_arrays_from_file(filePath, 8)
    # for x, y in f:
    #     print(x.shape)
    #     print(y.shape)
