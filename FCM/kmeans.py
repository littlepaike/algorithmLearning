import copy
import this

import numpy as np
import math
import random

'''
k-means
步骤一:初始化聚类中心C1，i=1,2....cn 。经典做法是从所有数据点中任取C个点。
步骤二:确定隶属矩阵U
步骤三:计算价值函数，如果它小于某个确定的阈值，或它相对上次价值函数的该变量小于某个阈值，则算法停止。
步骤四:修正聚类中心.
'''


class k_Means(object):
    def __init__(self, k=2):
        '''
        :param k:分组数
        '''
        self.k = k

    def distEclud(self, vectorA, vectorB):
        '''欧式距离
        :param vectorA:
        :param vectorB:
        :return:distance of vectorA and vectorB
        '''
        a = (vectorA[0, 0] - vectorB[0, 0]) ** 2
        b = (vectorA[0, 1] - vectorB[0, 1]) ** 2
        return math.sqrt(a + b)

    def testDistEclud(self, vectorA, vectorB):
        '''
        测试欧氏距离
        :return:
        '''
        vectorA = np.array([vectorA])
        vectorB = np.array([vectorB])
        return self.distEclud(vectorA, vectorB)

    def getData(self):
        '''
        :return:二维数组,模拟一些随机点,进行kmeans分类
        '''
        data = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [10, 10], [11, 10]])
        return data

    # 初始化质心
    def initCentroids(self, dataset):
        '''
        :param dataset:被无监督分类的数据集
        :param k:用户设定的分为几类
        :return:返回质心
        '''
        k = self.k
        numSamples, dim = dataset.shape
        # print(numSamples, dim)
        # 产生k行 dim列的零矩阵
        centroids = np.zeros((k, dim))
        # 定义重复列表
        repeat_list = []
        # 找出每个k的随机质点
        '''
        1、处理了重复的情况，通过一个repeate_list去解决重复问题导致两个质心是相同的 
        2、问题出现在array对比是元素对比，必须any或者all进行统一判断是否一样
        3、如果完全相同就跳过重新生成随机数。
        '''

        for i in range(k):
            while True:
                # flag每次都要初始化，如果有一次生成了两个相同的质心，flag变为0 要重新来一次，只有变回1  再去判断第二次生成的是否一样才可以跳出。
                flag = 1
                index = int(random.uniform(0, numSamples))  # 服从均匀分布0-数据量的随机数
                for repeatElement in repeat_list:
                    # print(repeatElement)
                    # print(dataset[index, :])
                    # print(all(repeatElement == dataset[index, :]))
                    if all(repeatElement == dataset[index, :]):
                        flag = 0
                        break
                if flag:
                    # print(dataset[index, :])
                    centroids[i, :] = dataset[index, :]
                    # print(centroids[i, :])
                    repeat_list.append(dataset[index, :])
                    break
                else:
                    continue
        return centroids

    def kMeans(self, dataset, centroids):
        '''
        :param dataset:初始数据集
        :param centroids:初始化的中心
        :return:列表，其中包含k簇（array）
        '''
        step = 0
        dataset_copy = copy.deepcopy(dataset)
        # print(dataset_copy, "数据集", type(dataset_copy))
        # print(centroids, "质心", type(centroids))

        # ##在深copy的数组中去掉两个质心。（弃用）
        # removeIndex = []
        # for i in range(len(centroids)):
        #     xList = np.where(dataset == centroids[i])[0].tolist()
        #     for i in range(len(xList)):
        #         if xList.count(xList[i]) > 1:
        #             removeIndex.append(xList[i])
        # removeIndex = list(set(removeIndex))
        # # print(removeIndex)
        # dataset_copy = np.delete(dataset_copy, removeIndex, axis=0)
        # # print(dataset_copy, "deepcopy")
        # 删除质心后的array距离
        # print(centroids[0])
        # print(dataset_copy[0])
        print("----------迭代开始-----------")
        # print(centroids, "一开始的质心")
        while True:
            km = [[] for i in range(self.k)]
            for i in range(len(dataset_copy)):
                # 计算到所有质心的距离
                _see = [self.testDistEclud(dataset_copy[i], centroids[j]) for j in range(self.k)]
                # 找出最近的簇
                min_index = _see.index(min(_see))
                # 把最近的并入对应簇
                km[min_index].append(i)
            # print(km)
            # print(dataset_copy[km[0][0]])
            # 更换质心
            step += 1
            k_new = []
            for i in range(self.k):
                # print("---------------")
                a = [dataset_copy[km[i][j]] for j in range(len(km[i]))]
                # 绝对不能把质心加入到列表
                # a.append(centroids[i])

                _centroids = sum(a) / len(a)
                # print(_centroids)
                k_new.append(_centroids)
            k_new = np.array(k_new)
            # print(k_new)
            # 更新质心
            print("newcentroids",k_new)
            # print(centroids, "centroids")
            if ~(k_new == centroids).all():
                centroids = k_new
                print("---------------->>>>>>>>>更换质心")
                # print(centroids)
            else:
                print("------------------------>>>>>>>迭代结束")
                print("迭代次数", step)
                # print(km)
                result_km = []
                for i in range(self.k):
                    a = [dataset_copy[km[i][j]] for j in range(len(km[i]))]
                    result_km.append(np.array(a))
                #[dataset_copy[km[i][j]] for i in range(self.k) for j in range(len(km[i]))] 有空试试二维列表生成
                return result_km


if __name__ == '__main__':
    # 定义分类数
    init_k = 2
    # 初始化对象
    k1 = k_Means()
    # 初始化分类数
    k1.k = init_k
    # 获取样例数据
    data = k1.getData()
    # 初始化质心
    centroids = k1.initCentroids(data)
    # print("初始化的多个质心为", centroids)
    # print(centroids.shape)

    # # 观察每个类别初始化的质心
    # for i in range(centroids.shape[0]):
    #     print("第" + str(i + 1) + "个类别的初始质心:", centroids[i])
    # print(k1.testDistEclud(centroids[0], centroids[1]))
    result = k1.kMeans(data, centroids)
    for item in range(len(result)):
        print("第"+str(item+1)+"簇")
        print(result[item])
