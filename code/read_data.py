import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# def processLine(chunk, batchSize):
#
#     print(chunk.shape)
#     id = chunk[:, 0]
#     label = chunk[:, 1]
#     x = []
#     for i in range(batchSize):
#         x.append(chunk[i, 2].split(" "))
#     # rawImg = np.array(list(map(str.split(" "), list(chunk[:, 2]))))
#     # print(rawImg)
#     rawImg =np.array(x).astype(np.int)
#     print(rawImg)
#     imgSeq = rawImg.reshape((batchSize, 15, 4, 101, 101))
#     #print(imgSeq.shape)
#     return imgSeq, label



# def generate_arrays_from_file(filePath, batch_size):
#
#     dataReader = pd.read_csv(filePath, delimiter=",", iterator=True)
#
#     loop = True
#     while loop:
#         try:
#             chunk = dataReader.get_chunk(batch_size)
#             x, y = processLine(chunk.values, batch_size)
#
#             yield (x, y)
#
#         except StopIteration:
#             loop = False
#             print("Iteration is stopped.")



def inputData(fileNameQue, trainFlag = True):
    '''
    :param fileNameQue: 文件名队列
    :param size: 图片尺寸
    :param trainFlag: 读取的是否是训练集
    :return: 训练集返回图像和标签，测试集返回图像
    '''

    #fileNameQue = tf.train.string_input_producer(["./data/faceTF.tfrecords"])

    # 创建tfrecorder阅读器
    reader = tf.TFRecordReader()
    key, value = reader.read(fileNameQue)

    if trainFlag:

        # 这里以数据类型的固定长度解析数据，要和封装时的数据对应，这里的字长要指定
        features = tf.parse_single_example(value, features={ 'label': tf.FixedLenFeature(1, tf.float32),
                                               'img' : tf.FixedLenFeature(612060, tf.int64)})

        img = features["img"]
        img = tf.cast(img, dtype=tf.float32) # 转化图像数据格式
        label = tf.cast(features['label'], dtype=tf.float64) # 转换标签数据格式
        label = tf.cast(label, dtype=tf.float32)

        return img, label
    else:
        features = tf.parse_single_example(value, features={'label': tf.FixedLenFeature(1, tf.float32),
                                                            'img': tf.FixedLenFeature(612060, tf.int64)})
        img = features["img"]
        img = tf.cast(img, dtype=tf.float32)  # 转化图像数据格式
        return img

def inputBatch(filename, batchSize, dequeue=50):
    '''

    :param filename: 文件名
    :param batchSize: batch大小
    :param imgSize: 图像尺寸
    :param dequeue: 缓冲队列大小
    :return: 图像batch，标签batch
    '''
    fileNameQue = tf.train.string_input_producer([filename], shuffle=True)
    example, label = inputData(fileNameQue, trainFlag=True)
    min_after_dequeue = dequeue   # 样本池调整的大一些随机效果好
    capacity = min_after_dequeue + 3 * batchSize

    # 上一个函数生成的样本会在这里积蓄并打乱成batch输出
    exampleBatch, labelBatch = tf.train.shuffle_batch([example, label], batch_size=batchSize, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    return exampleBatch, labelBatch

def inputNoShuffle(filename, batchSize):
    '''

    :param filename: 文件名
    :param imgSize: 图像大小
    :return: 图像
    '''
    fileNameQue = tf.train.string_input_producer([filename], shuffle=False)
    #min_after_dequeue = dequeue
    #capacity = min_after_dequeue + 3 * batchSize
    example = inputData(fileNameQue, trainFlag=False)
    exampleBatch = tf.train.batch([example], batchSize, allow_smaller_final_batch=True)
    return exampleBatch




if __name__ == "__main__":

    with tf.Session() as sess:

        # 逐一产生test样本
        fileNameQue = tf.train.string_input_producer(["../data/train.tfrecords"])
        testimg = inputData(fileNameQue, trainFlag=False)

        # 产生训练样本
        imgBatch, labelBatch = inputBatch("../data/train.tfrecords", 32)

        # 初始化
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(10):
            imgArr = sess.run(imgBatch)
            labelArr = sess.run(labelBatch)
            testImgArr = sess.run(testimg)
            #print(imgArr.shape, labelArr)
            print(testimg.shape)
            #im = Image.fromarray(np.uint8(imgArr))
            # im.save("./data/img.jpg")

        coord.request_stop()
        coord.join(threads)

