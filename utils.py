# encoding=utf-8
from PIL import Image
import torch
import numpy as np
import os
from config import opt


def hozFilp(img):
    """
    水平反转图像进行Data Argumentation
    """
    idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    imgFlip = img.index_select(3, idx)
    return imgFlip


def calScore(score, label):
    """[计算准确率]
    """
    score = score.data  # 对于Variable要做这个步骤
    label = label.data
    _, predict = torch.max(score, 1)  # 按行着最大值位置作为预测
    return torch.mean((predict == label).type(torch.FloatTensor))


def calAdj(queryF, testF):
    """根据test特征与query特征计算邻接矩阵
    Arguments:
        queryF {[type]} -- [description]
        testF {[type]} -- [description]
    """
    # 计算邻接矩阵
    if len(queryF.size()) == 1:
        queryNum = 1
    else:
        queryNum = queryF.size()[0]
    testNum = testF.size()[0]
    disMat = torch.zeros(queryNum, testNum)
    for ii in range(queryNum):
        # 查询图像特征
        queryVec = queryF[ii] if len(queryF.size()) != 1 else queryF
        disMat[ii] = torch.sqrt(torch.sum((testF - queryVec) ** 2, 1))
        print('第{0:d}个查询样本特征与测试集相似度计算完毕'.format(ii))
    return disMat


def getEva(dis, loc, isSingle=False, isSave=False):
    """获得评价参数CMC TOP6
    Arguments:
        loc--queryID
        disLocal--query图像对全部test数据集的相似度向量
    """
    testImgLab = [name for name in os.listdir(opt.testFolder)]  # 测试文件夹图像标签
    testImgLab.sort()  # 有17661个
    testImgCAM = np.array([int(name.split('_')[1][1])
                           for name in testImgLab])  # 视角
    testImgLab = np.array([int(name.split('_')[0])
                           for name in testImgLab])  # 标签
    queryImgLab = [name for name in os.listdir(opt.queryFolder)]  # 查询图像集合图像
    queryImgLab.sort()  # 有2228个
    queryImgCAM = np.array([int(name.split('_')[1][1])
                            for name in queryImgLab])  # 视角
    queryImgLab = np.array([int(name.split('_')[0])
                            for name in queryImgLab])  # 标签
    # 针对单个输入和多输入分别考虑
    if isSingle:
        _, sortLoc = torch.sort(dis[0])
    else:
        _, sortLoc = torch.sort(dis[loc])  # 获得第loc张查询图像对全测试集的相似度

    # 找到标签相同并且不在一个cam下的图像
    goodSam = list(set(np.where(testImgLab == queryImgLab[loc])[0]).intersection(
        set(np.where(testImgCAM != queryImgCAM[loc])[0])))
    # 找到标签相同但是在一个cam下图像
    junkSameCAM = list(set(np.where(testImgLab == queryImgLab[loc])[0]).intersection(
        set(np.where(testImgCAM == queryImgCAM[loc])[0])))
    # top 6
    if isSave:
        # 如果可以，保存下来
        queryImages = [os.path.join(opt.queryFolder, name)
                       for name in os.listdir(opt.queryFolder)]
        queryImages.sort()
        queryImg = Image.open(queryImages[loc])
        queryImg.save('queryRes/results/queryImg.png')  # 存储查询图像
    # 根据排序确定
    CMC, imgNameSort, mAP = calCMC(goodSam, junkSameCAM, sortLoc)
    if isSave and len(imgNameSort):
        # 要保存图像
        testImages = [os.path.join(opt.testFolder, name)
                      for name in os.listdir(opt.testFolder)]
        testImages.sort()
        for jj in range(len(imgNameSort)):
            topImg = Image.open(testImages[int(imgNameSort[jj])])  # 只找到top几
            topImg.save('queryRes/results/top{0:d}.png'.format(1 + jj))
    return torch.FloatTensor(CMC), mAP


def calCMC(goodSam, junkSameCAM, sortLoc):
    """
    计算CMC
    :return: CMC
    """
    CMC = np.zeros((1, len(sortLoc)))
    mAP = 0
    oldRecall = 0
    oldPrecision = 1
    intersS = 0
    isGood = 0
    numGood = len(goodSam)
    imgNameSort = np.zeros((1, len(sortLoc)))
    junkNum = 0  # 垃圾图像数目
    count = 0
    for ii in range(len(sortLoc)):
        flag = 0
        if len(np.where(np.asarray(goodSam) == sortLoc[ii])[0]):
            # 击中目标
            CMC[:, ii - junkNum:] = 1
            flag = 1
            isGood = isGood + 1
            imgNameSort[0, ii] = sortLoc[ii]  # 记录是哪张图像
        if len(np.where(np.asarray(junkSameCAM) == sortLoc[ii])[0]):
            # 同一摄像头，直接忽视
            junkNum = junkNum + 1
            continue
        if flag == 1:
            intersS = intersS + 1
        recall = intersS / numGood
        precision = intersS / (count + 1)
        mAP = mAP + (recall - oldRecall) * (0.5 * (oldPrecision + oldRecall))
        # 更新
        oldRecall = recall
        oldPrecision = precision
        count = count + 1
    finalSort = imgNameSort[imgNameSort != 0]  # 去除0
    finalSort = finalSort if np.shape(
        finalSort)[0] < opt.topN else finalSort[:opt.topN]
    return CMC, finalSort, mAP
