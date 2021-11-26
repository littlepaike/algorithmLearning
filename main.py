from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import operator
import random
from scipy.spatial.distance import pdist
from pylab import *

# 正常显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
INF = 9999999.0


def loadDataSet(fileName):
    '''
    加载数据集
    :param fileName:
    :return:
    '''
    # 初始化一个空列表
    dataSet = []
    # 读取文件
    fr = open(fileName)
    # 循环遍历文件所有行
    for line in fr.readlines():
        # 切割每一行的数据
        curLine = line.strip(',').split(',')
        fltLine = list(map(float, curLine))  # 映射所有的元素为 float（浮点数）类型
        dataSet.append(fltLine)
    return dataSet


def distEclud(vecA, vecB):
    """欧式距离
    输入：向量A, 向量B
    输出：两个向量的欧式距离
    """
    a = (vecA[0, 0] - vecB[0, 0]) ** 2
    b = (vecA[0, 1] - vecB[0, 1]) ** 2
    return sqrt(a + b)


def distManhattan(vecA, vecB):
    '''曼哈顿距离
    输入：向量A, 向量B
    输出：两个向量的曼哈顿距离
    '''
    return np.sum(np.abs(vecA - vecB))


def distCosine(vecA, vecB):
    '''余弦距离
    输入：向量A, 向量B
    输出：两个向量的余弦距离
    '''
    return pdist(np.vstack([vecA, vecB]), 'cosine')


def distChebyshev(vecA, vecB):
    '''切比雪夫距离
    输入：向量A, 向量B
    输出：两个向量的切比雪夫距离
    '''
    return np.max(np.abs(vecA - vecB))


def randCent(dataSet, k):
    """生成随机质心
    输入：数据集, 聚类个数
    输出：k个随机质心的矩阵
    """
    n = dataSet.shape[1]  # 每个数据的维度
    centroids = mat(zeros((k, n)))  # 生成k*n维数据
    for j in range(n):
        minJ = min(dataSet[:, j])  # 第j列最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)  # 第j列最大值与最小值的差值
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)  # 最小值加上差值的（0，1）之间的倍数
    return centroids


def w_update(dataSet, centroids, clusterAssment, distMeans=distEclud):
    """更新权重值
        输入：数据集, 质心
        输出：更新后的权值
    """
    for j in range(dataSet.shape[0]):
        dist_all = 0
        for cen in range(len(centroids)):
            dist = distMeans(dataSet[j, :], centroids[cen, :])
            dist_all += dist
        for cen in range(len(centroids)):
            dist_self = distMeans(dataSet[j, :], centroids[cen, :])
            clusterAssment[j, cen + 2] = dist_self / dist_all
    return clusterAssment


def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    """
    输入：数据集, 聚类个数, 距离计算函数, 生成随机质心函数
    输出：质心矩阵, 簇分配和距离矩阵
    """
    m = dataSet.shape[0]
    clusterAssment = mat(zeros((m, 2)))  # 初始化聚类矩阵
    centroids = createCent(dataSet, k)  # 生成随机质心
    clusterChanged = True  # 启动
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 寻找最近的质心
            minDist = INF
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, 0:2])  # 距离方法,计算该点与质心的距离
                if distJI < minDist:  # 得到与质心距离最少的距离下标
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 只要有一个点的簇发生变化，就继续进行
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist  # 存储该点的簇类，该点到簇心的距离
        for cent in range(k):  # 更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 计算均值
    return centroids, clusterAssment


def fuzzy_kmeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    """
    输入：数据集, 聚类个数, 距离计算函数, 生成随机质心函数
    输出：质心矩阵, 簇分配和距离矩阵
    """
    m = dataSet.shape[0]
    clusterAssment = mat(zeros((m, k + 2)))  # 初始化聚类矩阵
    centroids = createCent(dataSet, k)  # 生成随机质心
    clusterChanged = True  # 启动
    clusterAssment[:, 2:k + 2] = 1 / k
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 寻找最近的质心
            minDist = INF
            minIndex = -1
            for j in range(k):
                distJI = (clusterAssment[i, j + 2] ** 2) * distMeans(centroids[j, :],
                                                                     dataSet[i, 0:2])  # 距离方法,计算该点与质心的距离
                if distJI < minDist:  # 得到与质心距离最少的距离下标
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 只要有一个点的簇发生变化，就继续进行
                clusterChanged = True
            clusterAssment[i, 0:2] = minIndex, minDist  # 存储该点的簇类，该点到簇心的距离
            # 更新权值
            clusterAssment = w_update(dataSet[i, :], centroids, clusterAssment)
        c = mat(zeros((3, k)))
        for cent in range(dataSet.shape[0]):  # 更新质心的位置
            c[2, int(clusterAssment[cent, 0])] += clusterAssment[cent, int(clusterAssment[cent, 0]) + 2]
            c[0, int(clusterAssment[cent, 0])] += clusterAssment[cent, int(clusterAssment[cent, 0]) + 2] * dataSet[
                cent, 0]
            c[1, int(clusterAssment[cent, 0])] += clusterAssment[cent, int(clusterAssment[cent, 0]) + 2] * dataSet[
                cent, 1]
        for cent in range(k):
            cc0 = c[0, cent] / c[2, cent]
            cc1 = c[1, cent] / c[2, cent]
            centroids[cent, :] = cc0, cc1  # 计算均值
    return centroids, clusterAssment


def plotFeature(dataSet, centroids, clusterAssment):
    m = shape(centroids)[0]
    fig = plt.figure()
    scatterMarkers = ['v', '^']
    scatterColors = ['blue', 'green', 'black', 'purple', 'orange', 'black', 'yellow']
    ax = fig.add_subplot(111)
    for i in range(m):
        ptsInCurCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]  # 选择标记点
        colorSytle = scatterColors[i % len(scatterColors)]  # 选择颜色
        # flatten为返回一维数组
        ax.scatter(ptsInCurCluster[:, 0].flatten().A[0], ptsInCurCluster[:, 1].flatten().A[0], marker=markerStyle,
                   c=colorSytle, s=30)  # 添加每类数据点
    ax.scatter(centroids[:, 0].flatten().A[0], centroids[:, 1].flatten().A[0], marker='h', c='red', s=200)  # 添加质心的位置


def silhouette(dataset, clustAssing, K, distMeans=distEclud):
    '''计算每个点的轮廓系数
    :param dataset:所有数据集
    :param clustAssing:输入为每个簇的标签和其到簇心的距离
    :return:返回平均轮廓系数，和数组
    :k:簇数量
    '''
    mean_s = 0
    clustAssing_new = mat(zeros((dataset.shape[0], 2)))
    for i in range(dataset.shape[0]):
        a = 0;
        n1 = 0
        buffer = []
        for k in range(K):
            buffer.append([])
        for j in range(dataset.shape[0]):
            if i == j:
                continue
            elif clustAssing[i, 0] == clustAssing[j, 0]:  # 计算同簇内的距离
                n1 += 1
                distJI = distMeans(dataset[j, 0:2], dataset[i, 0:2])  # 距离方法,计算该点与质心的距离
                a += distJI
            elif clustAssing[i, 0] != clustAssing[j, 0]:  # 计算不同簇内的距离
                distJI = distMeans(dataset[j, 0:2], dataset[i, 0:2])
                buffer[int(clustAssing[j, 0])].append(distJI)
        b_min = INF
        for p in range(K):
            if p == clustAssing[i, 0]:
                continue
            else:
                b_buff = 0
                for q in range(len(buffer[p])):
                    b_buff += buffer[p][q]
                try:
                    b_buff = np.round(b_buff / len(buffer[p]), 5)
                except Exception as e:
                    print(e)
                if b_buff <= b_min:
                    b_min = b_buff
        aa = np.round(a / n1, 5)
        s = (b_min - aa) / max(aa, b_min)
        clustAssing_new[i, :] = clustAssing[i, 0], s
        mean_s += s
    mean_s = mean_s / dataset.shape[0]
    return mean_s, clustAssing


def SSE(K, clustAssing):
    value = []
    SSE_mean = 0
    for k in range(K):
        value.append([])
    for i in range(clustAssing.shape[0]):
        value[int(clustAssing[i, 0])].append(clustAssing[i, 1])
    for k in range(K):
        SSE_mean += np.mean(value[k])
    SSE_mean /= K
    return SSE_mean


def main():
    dataset = loadDataSet('Dataset.txt')
    dataSet = mat(dataset)
    resultCentroids, clustAssing = fuzzy_kmeans(dataSet, 5)
    #  resultCentroids, clustAssing = kMeans(dataSet, 5)
    plotFeature(dataSet, resultCentroids, clustAssing)
    mean_s, clustAssing_new = silhouette(dataSet, clustAssing, 5)
    mean_sse = SSE(5, clustAssing)
    print(mean_s, mean_sse)
    plt.show()


#   plt.xlabel('四种不同距离量度')  # 将坐标系x轴命名为x1
#   plt.ylabel('轮廓系数')  # 将坐标系y轴命名为y1
#   plt.plot([1,2,3,4],[0.557,0.579,0.497,0.725])
#   plt.show()
#    performence = [[], []]
#    for i in range(3, 11):
#        resultCentroids, clustAssing = fuzzy_kmeans(dataSet, i)
# #      print(i, resultCentroids)
#        mean_s, clustAssing_new = silhouette(dataSet, clustAssing, i)
#         # mean_s = SSE(i, clustAssing)
#        # print('簇K%-2d的SSE为 %.5f'%(i, mean_s))
#        print('簇K %-2d 的轮廓系数为 %.5f'%(i, mean_s))
#        performence[0].append(i)
#        performence[1].append(mean_s)
#    plt.plot(performence[0], performence[1])
#    plt.show()
if __name__ == '__main__':
    main()
