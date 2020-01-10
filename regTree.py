#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv,det
#载入数据
# def loadDataSet(fileName):
#     dataMat = []
#     fr = open(fileName)
#     for line in fr.readlines():
#         curLine = line.strip().split('\t')
#         print(type(curLine[0]))
#         # python3不适用：fltLine = map(float,curLine) 修改为：
#         fltLine = list(map(float, curLine))#将每行映射成浮点数，python3返回值改变，所以需要
#         dataMat.append(fltLine)
#     return dataMat


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName,"r")
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        dataMat.append(curLine)
    dataMat=np.array(dataMat)
    dataMat=dataMat.astype(np.float64)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1

#树回归
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])
def regErr(dataSet):#计算目标的平方误差（均方误差*总样本数）
    return np.var(dataSet[:,-1]) * dataSet.shape[0]#var是方差

#模型树
# def linearSolve(dataSet):
#     m,n=dataSet.shape
#     X=np.ones((m,n))#这里设置为m*n，已经把常数b给设置进去了，w1*x+w2*1=y
#     Y=np.ones((m,1))
#     X[:,0:n-1]=dataSet[:,0:n-1]
#     Y=dataSet[:,-1]
#     xTx=np.dot(X.transpose(),X)
#     if np.linalg.det(xTx)==0.0:
#         raise NameError('This matrix is singular')
#     ws=np.dot(  np.dot(np.linalg.inv(xTx),X.transpose())   ,Y)
#     return ws,X,Y
# def modelLeaf(dataSet):
#     ws,X,Y=linearSolve(dataSet)
#     return ws
# def modelErr(dataSet):
#     ws,X,Y=linearSolve(dataSet)
#     yHat=np.dot(X,ws)
#     return sum(np.power(Y-yHat,2)) #????? np.sum? sum?


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    #切分特征的参数阈值，用户初始设置好
    tolS = ops[0] #允许的误差下降值 !!!
    tolN = ops[1] #切分的最小样本数 !!!
    #若所有特征值都相同，停止切分
    #if len(set(dataSet[:,-1].T.tolist()[0])) == 1:#倒数第一列转化成list 不重复
    if len(set(dataSet[:, -1])) == 1:  # 倒数第一列即所有的标签值（y）转化成list 不重复
        return None,leafType(dataSet)  #如果剩余值只有一种，停止切分1。
        # 找不到好的切分特征，调用regLeaf直接生成叶结点
    m,n = dataSet.shape
    S = errType(dataSet)#最好的特征通过计算平均误差
    bestS = 10000000; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1): #遍历数据的每个属性特征
        for splitVal in set((dataSet[:, featIndex])):#遍历每个特征里不同的特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)#对每个特征进行二元分类
            if (mat0.shape[0] < tolN) | (mat1.shape[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:#更新为误差最小的特征
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果切分后误差效果下降不大，则取消切分，直接创建叶结点
    if (S - bestS) < tolS:
        return None,leafType(dataSet) #停止切分2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #判断切分后子集大小，小于最小允许样本数停止切分3
    if (mat0.shape[0] < tolN) | (mat1.shape[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue#返回特征编号和用于切分的特征值


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    #数据集默认NumPy Mat 其他可选参数【结点类型：回归树，误差计算函数，ops包含树构建所需的其他元组】
    feat,val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val #满足停止条件时返回叶结点值
    #切分后赋值
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #切分后的左右子树
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__=='dict') #判断为字典类型返回true
#返回树的平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


#树的后剪枝
def prune(tree, testData):#待剪枝的树和剪枝所需的测试数据，这个需要在测试集上进行，而不是训练集
    if shape(testData)[0] == 0: return getMean(tree)  # 确认数据集非空
    #假设发生过拟合，采用测试数据对树进行剪枝
    if (isTree(tree['right']) or isTree(tree['left'])): #左右子树非空
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    #剪枝后判断是否还是有子树
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #判断是否merge
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2)) #！！！这里tree['left']和tree['right']都是平均值，从叶子节点上来的
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        #如果合并后误差变小
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree



if __name__=="__main__":
    #树回归
    dataMatrix=loadDataSet("ex0.txt")
    print(createTree(dataMatrix))

    #模型树（即把叶子节点变成线性方程）
    # myMat2=loadDataSet('exp2.txt')
    # print(createTree(myMat2,modelLeaf,modelErr,(1,10)))

