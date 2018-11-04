
#encoding=utf-8
from numpy import *
import numpy as np
import pandas as pd
from math import log
import copy
import operator


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt


#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet,labels):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i


    return bestFeature



def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    return max(classCount)


#对离散变量划分数据集，取出该特征取值为value的所有样本
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def createTree(dataSet, labels, data_full, labels_full, data_test):
    classList = [example[-1] for example in dataSet]
    temp_labels = copy.deepcopy(labels)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        currentlabel = labels_full.index(labels[bestFeat])
    featValuesFull = [example[currentlabel] for example in data_full]
    uniqueValsFull = set(featValuesFull)
    del (labels[bestFeat])
    # 针对bestFeat的每个取值，划分出一个子树。
    for value in uniqueVals:
        subLabels = labels[:]
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            uniqueValsFull.remove(value)
        myTree[bestFeatLabel][value] = createTree(splitDataSet \
        (dataSet, bestFeat, value), subLabels, data_full, labels_full, splitDataSet(data_test,bestFeat,value))
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value] = majorityCnt(classList)
    if testing(myTree,data_test,temp_labels)<testingMajor(majorityCnt(classList),data_test):
        return myTree
    return majorityCnt(classList)


def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def testing(myTree,data_test,labels):
    error=0.0
    for i in range(len(data_test)):
        if classify(myTree,labels,data_test[i])!=data_test[i][-1]:
            error+=1
    print('myTree %d' %error)
    return float(error)

def testingMajor(major,data_test):
    error=0.0
    for i in range(len(data_test)):
        if major!=data_test[i][-1]:
            error+=1
    print('major %d' %error)
    return float(error)

data = [['青绿', '蜷缩', '浊响','清晰','是'],
               ['乌黑', '蜷缩', '沉闷','清晰','是'],
               ['乌黑', '蜷缩', '浊响','清晰','是'],
               ['青绿', '稍蜷', '浊响','清晰','是'],
               ['乌黑', '稍蜷', '浊响','稍糊','是'],
               ['乌黑', '稍蜷', '沉闷','稍糊','否'],
               ['青绿', '硬挺', '清脆','清晰','否'],
               ['浅白', '稍蜷', '沉闷','稍糊','否'],
               ['乌黑', '稍蜷', '浊响','清晰','否'],
               ['浅白', '蜷缩', '浊响','模糊','否'],
               ['青绿', '蜷缩', '沉闷','稍糊','否']]
data_test =  [['青绿', '蜷缩', '沉闷','清晰','是'],
               ['浅白', '蜷缩', '浊响','清晰','是'],
               ['乌黑', '稍蜷', '浊响','清晰','是'],
               ['浅白', '硬挺', '清脆','模糊','否'],
               ['浅白', '蜷缩', '浊响','模糊','否'],
               ['青绿', '稍蜷', '浊响','稍糊','否']]

labels = ['色泽','根蒂','敲声','纹理']
data_full = data[:]
labels_full = labels[:]
myTree = createTree(data, labels, data_full, labels_full, data_test)
print(myTree)

