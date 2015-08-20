#!/usr/bin/env python
# -*- coding:utf-8 -*-
#########################################################################
# File Name: kNN.py
# Author: Wang Biwen
# mail: wangbiwen88@126.com
# Created Time: 2015.07.30
#########################################################################

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

#分类函数
def classify(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0] #shape[0]表示矩阵的行数

	#以下就根据坐标系中求欧式距离的公式求距离
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet  #将inX扩展成特定行数的矩阵，并求与dataSet矩阵的差
	sqDiffMat = diffMat ** 2	#对矩阵每个元素求平方
	sqDistances = sqDiffMat.sum(axis = 1) #将矩阵的每一行求和，得到一个列表
	distances = sqDistances ** 0.5  #对列表中每个元素开方，得到距离列表

	#以下求出距离最小点的类别，并返回
	sortedDistIndicies = distances.argsort() #对距离排序得到排序完的下标列表
	classCount = dict()
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #统计每个点的类别
	sortedClassCount = sorted(classCount.iteritems(), key = lambda asd:asd[1], reverse = True) #排序得到最多点的类别及个数的tuple列表
	return sortedClassCount[0][0]

#将数据转化成矩阵的函数
def file2mat(inputFile):
	fin = open(inputFile, 'r')
	allLines = fin.readlines()
	numOfLines = len(allLines)
	resultMat = zeros((numOfLines, 3)) #创建numOfLines行3列的零矩阵
	labelList = []
	index = 0
	for line in allLines:
		line = line.strip()
		dataList = line.split('\t')
		resultMat[index, :] = dataList[0 : 3] #更新零矩阵的每一行
		labelList.append(int(dataList[-1])) #将对应的类别存入类别列表
		index += 1
	return resultMat, labelList

#数值归一化函数，自动将数据转化为0到1的区间
def autoNorm(dataSet):
	minMat = dataSet.min(0) #取每列的最小值
	maxMat = dataSet.max(0) #取每列的最大值
	rangesMat = maxMat - minMat #求得每列最大值和最小值之差，即取值范围
	m = dataSet.shape[0] #取得矩阵行数
	normMat = zeros(shape(dataSet)) #初始化成零矩阵
	normMat = dataSet - tile(minMat, (m, 1)) #求得矩阵每个元素与当前列最小值之差
	normMat = normMat / tile(rangesMat, (m, 1)) #然后除以当前列的范围，以转化为0到1之间的值
	return normMat, rangesMat, minMat

	



if __name__ == '__main__':
	group, labels = createDataSet()
	myclass = classify((0,0), group, labels, 3)
	print myclass

	mat, lst = file2mat('./data/datingTestSet2.txt')

	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#ax.scatter(mat[:, 2], mat[:, 0], 15.0 * array(lst), 15.0 * array(lst))
	#plt.show()
	
	normMat, rangesMat, minMat = autoNorm(mat)
	print normMat
	print rangesMat
	print minMat

