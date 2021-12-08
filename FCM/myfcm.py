from pylab import *
from numpy import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris
import pandas as pd
import os
from collections import defaultdict

def readData(filename):
    df_full = pd.read_csv(filename)
    # 列出所有列
    columns = list(df_full.columns)
    # 特征值
    features = columns[:len(columns) - 1]
    df = df_full[features]
    return df


class myFcm(object):
    def __init__(self, k=3, MAX_ITER=100, n=2, m=2.00, num_attr=4):
        # 分类数
        self.k = k
        # 最大迭代数
        self.MAX_ITER = MAX_ITER
        # 样本数
        self.n = n
        # 模糊参数(加权指数)
        self.m = m

    # 初始化模糊矩阵U
    def initializeMembershipMatrix(self):
        '''
        步骤 1：用值在 0，1 间的随机数初始化隶属矩阵 U
        理解：每个数都有属于三类的概率
        '''
        membership_mat = list()
        for i in range(self.n):
            random_num_list = [random.random() for i in range(self.k)]
            summation = sum(random_num_list)
            temp_list = [x / summation for x in random_num_list]  # 首先归一化
            membership_mat.append(temp_list)
        # print(membership_mat)
        return membership_mat

    # 计算类中心点
    def calculateClusterCenter(self, membership_mat):
        '''
        步骤 2：用式（6.12）计算 c 个聚类中心 c
        return:返回三个维度的中心
        '''
        cluster_mem_val = zip(*membership_mat)
        cluster_centers = list()
        # 将n个数的三个聚类隶属度zip为三个列表
        cluster_mem_val_list = list(cluster_mem_val)
        # print(cluster_mem_val_list)
        # 分别对三个类别 的隶属度操作
        for j in range(k):
            # 分别拿出
            x = cluster_mem_val_list[j]
            # 做uij的m次方操作
            xraised = [e ** m for e in x]
            # 分母求和
            denominator = sum(xraised)
            temp_num = list()
            # 取出每一行的数据
            for i in range(n):
                # 去除某一行的数据
                data_point = list(df.iloc[i])
                # 6.12 分子——相乘
                prod = [xraised[i] * val for val in data_point]
                temp_num.append(prod)
            # 分子并不是一个数 而是一个向量，向量应该每个维度都进行求和再去和分母做商
            numerator = map(sum, zip(*temp_num))
            center = [z / denominator for z in numerator]  # 每一维都要计算。
            cluster_centers.append(center)
        # print(cluster_centers)
        return cluster_centers

    # 更新隶属度
    def updateMembershipValue(self, membership_mat, cluster_centers):
        #    p = float(2/(m-1))
        data = []
        for i in range(n):
            # 取出文件中的每一行数据
            x = list(df.iloc[i])
            data.append(x)
            # operator.sub 减法
            # 求每行数据与三个中心点的距离  ord默认是平方和开跟
            distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(self.k)]
            for j in range(self.k):
                # 6.13   求最新隶属度
                den = sum([math.pow(float(distances[j] / distances[c]), 2 / (self.m - 1)) for c in range(self.k)])
                membership_mat[i][j] = float(1 / den)
        return membership_mat, data

    # 得到聚类结果
    def getClusters(self, membership_mat):
        cluster_labels = list()
        for i in range(self.n):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
            cluster_labels.append(idx)
        return cluster_labels

    def fuzzyCMeansClustering(self):
        # 主程序
        # 初始化模糊矩阵
        membership_mat = self.initializeMembershipMatrix()
        # 迭代次数记录
        curr = 0
        while curr <= MAX_ITER:  # 最大迭代次数
            # 更新类中心点
            cluster_centers = self.calculateClusterCenter(membership_mat)
            # 更新隶属度
            membership_mat, data = self.updateMembershipValue(membership_mat, cluster_centers)
            # 得到聚类结果
            cluster_labels = self.getClusters(membership_mat)
            curr += 1
        return cluster_labels, cluster_centers, data, membership_mat, curr


if __name__ == '__main__':
    # 读取数据
    filename = "iris.csv"
    df = readData(filename)
    # 初始化一些参数
    # 分类数
    k = 3
    # 最大迭代数
    MAX_ITER = 100
    # 样本数
    n = len(df)  # the number of row
    # 模糊参数
    m = 2.00
    # 维度
    num_attr = len(df.columns) - 1
    myfcm = myFcm(k=k, MAX_ITER=MAX_ITER, n=n, m=m, num_attr=num_attr)
    labels, centers, data, membership, curr = myfcm.fuzzyCMeansClustering()
    value_cnt = {}
    for lable in labels:
        # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
        value_cnt[lable] = value_cnt.get(lable, 0) + 1
    print(value_cnt)
    print(curr, "次迭代")

    mainlist = defaultdict(list)
    for k, va in [(v, i) for i, v in enumerate(labels)]:
        mainlist[k].append(va)

    for i in mainlist:
        print("第"+str(i)+"类："+str(mainlist[i]))