import numpy as np
import pandas as pd
from tqdm import tqdm
from data_proc import *

def classify(inX, dataSet, labels, k):
    """
    定义knn算法分类器函数
    :param inX: 测试数据
    :param dataSet: 训练数据
    :param labels: 分类类别
    :param k: k值
    :return: 所属分类
    """

    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # sqDiffMat = diffMat ** 2
    sqDiffMat = np.abs(diffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    # distances = sqDistances ** 0.5  # Euclidean distance
    distances = sqDistances
    sortedDistIndicies = distances.argsort()  # Sort and return index

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #default 0

    sortedClassCount = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
    return sortedClassCount[0][0]

def classify_two(inX, dataSet, labels, k):
    # Like the classify function, but is non-matrix implemented and has a slower computation rate
    m, n = dataSet.shape   # shape（m, n）m列n个特征
    # Calculate the Euclidean distance from the test data to each point
    distances = []
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += (inX[j] - dataSet[i][j]) ** 2
        distances.append(sum ** 0.5)

    sortDist = sorted(distances)

    # The category to which the k nearest values belong
    classCount = {}
    for i in range(k):
        voteLabel = labels[ distances.index(sortDist[i])]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 # 0:map default
    sortedClass = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
    return sortedClass[0][0]

def TrainSet():
    train_proc('train_data_all.json')
    dataSet = data2vector('train_proc.txt')
    labels = label2vector()
    return dataSet, labels

def TestSet(data_path:str):
    test_proc(data_path)
    dataSet = data2vector('test_proc.txt')
    return dataSet

def balance_method():
    '''
    Add the processing of unbalanced data. Details are shown in report.
    '''
    dataSet, labels = TrainSet()
    m, _ = dataSet.shape
    Store_data = dataSet[0:15]
    Store_label = labels[0:15]
    
    # For processing parts with unbalanced data volume, redundant data is removed and data volume is reduced
    for i in tqdm(range(m)):
        r = classify(dataSet[i], Store_data[:], Store_label[:], 5)
        if r != labels[i]:
            Store_data = np.concatenate((Store_data, dataSet[i].reshape(1,10)), axis=0)
            Store_label = np.append(Store_label, labels[i])
    
    np.save('Store_data.npy', Store_data)       # Save the supporting datset in the file 'Store_data.npy'
    np.save('Store_label.npy', Store_label)     # Save the supporting labels in the file 'Store_label.npy'