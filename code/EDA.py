import pandas as pd
import keras as K
import  numpy as np
import matplotlib.pyplot as plt



# filePath = "../data/CIKM2017_testA/data_new/CIKM2017_testA/testA.txt"
# dataReader = pd.read_csv(filePath, delimiter=",", iterator=True)
# data = dataReader.get_chunk(1)
# print(data.values[:, 1])
#print(len(list((data.values[:, 2][0].split(" ")))))
#print(len(data.values[]))


# def cleanData(filename):
#     dataReader = pd.read_csv(filePath, delimiter=",", iterator=True)
#     loop = True
#     while loop:
#         try:
#             data = dataReader.get_chunk(1)
#             id = data.values[:, 0][0]
#             label = data.values[:, 1][0]
#             rawImage = np.array(list(map(np.int, list(data.values[:, 2][0].split(" ")))))
#             print(type(rawImage))
#             print(id, label, rawImage.shape)
#             imgSeq = rawImage.reshape((15, 4, 101, 101))
#             print(imgSeq.shape)
#         except StopIteration:
#             loop = False
#             print("Iteration is stopped.")
#
#
# if __name__ == "__main__":
#     filePath = "../data/CIKM2017_train/data_new/CIKM2017_train/data_sample.txt"
#     cleanData(filePath)


file = "../data/label_count.csv"
ansfile = "../result/result_linear_model.csv"
df = pd.read_csv(ansfile)
#df = np.log1p(df)
df.hist(bins=20)
plt.show()