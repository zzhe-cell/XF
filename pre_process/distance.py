import pandas as pd
from scipy.spatial import distance
import numpy as np
from preprocess import Preprocess

class Distance:
    def __init__(self, data):
        self.data = data
    def get_distance(self, x, y, weights=[1, 1, 1, 1]):
        """
        求data中两个样本的距离
        :param x: 第一个样本在data中的索引
        :param y: 第二个样本在data中的索引
        :return: float:两个样本的距离
        """
        wenben = self.data['文本向量']
        time = self.data['来访时间']
        mudi = pd.DataFrame()
        danwei = pd.DataFrame()
        str1 = '信访目的'
        str2 = '涉事单位'
        for column in list(self.data.columns):
            if str1 in column:
                mudi[column] = self.data[column]
            elif str2 in column:
                danwei[column] = self.data[column]
        dis1 = abs(time[x] - time[y])
        dis2 = distance.euclidean(np.array(mudi.iloc[x]), np.array(mudi.iloc[y]))
        dis3 = distance.euclidean(np.array(danwei.iloc[x]), np.array(danwei.iloc[y]))
        dis4 = distance.cosine(wenben[x], wenben[y])
        distances = [dis1, dis2, dis3, dis4]
        dis = sum(np.multiply(weights, distances))
        return dis

    def get_dis_matrix(self, weights=[1, 1, 1, 1]):
        n_samples = len(self.data)
        dis_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                dis_matrix[i][j] = self.get_distance(i, j, weights)
        for i in range(n_samples):
            for j in range(i):
                dis_matrix[i][j] = dis_matrix[j][i]
        return dis_matrix

    def get_wenben_distance(self, x, y):
        wenben = self.data['文本向量']
        dis = distance.cosine(wenben[x], wenben[y])
        return dis

# filepath = "../data/信访事项（1）.csv"
# data = pd.read_csv(filepath, encoding="GBK")
# p = Preprocess(data)
# data1 = p.preprocess()
# d = Distance(data1)
#
#
# for i in range(len(data1)):
#     dis = []
#     for j in range(len(data1)):
#         dis.append(d.get_wenben_distance(i, j))
#     print(dis)
#     dis_sort = np.argsort(dis)
#     slice = dis_sort[:10]
#     jinsi = data.iloc[slice, :]
#     jinsi.to_csv('../data/第{}条数据的10近邻_文本.csv'.format(i), encoding='GBK')


