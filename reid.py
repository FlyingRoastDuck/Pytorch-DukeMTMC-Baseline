# -*-coding:utf-8-*-
import torch
import models
import torch.nn as nn
from config import opt
from torch.utils.data import DataLoader
from dataReader.reidReader import reidReader
from torch.autograd import Variable
import torchvision
import time
import pylab as pl
import numpy as np
import os
from PIL import Image
import signal


def train(**kwargs):
    """[训练函数]
    Arguments:
        **kwargs {[type]} -- [用户传入参数]
    """
    opt.parse(**kwargs)  # 参数更新
    global isTer
    isTer = False  # 设置全局变量方便中断时存储model参数
    # 设定模型
    model = eval('models.' + opt.model + '(' + str(opt.numClass) + ')')
    # if opt.isPreTrain:
    #     # 使用预训练方式进行训练
    #     preModel = eval('torchvision.models.' + opt.model + '(pretrained=True)')
    #     del preModel.fc
    #     # 修正以便进行权重初始化
    #     preDict = preModel.state_dict()  # 预训练权重
    #     modelD = model.state_dict()
    #     # 按照键值进行赋值
    #     newW = {k: v for k, v in preDict.items() if k in modelD.keys()}
    #     modelD.update(newW)
    #     model.load_state_dict(modelD)
    # 处理基本模型的路径
    if opt.modelPath:
        model.load(opt.modelPath)
    if opt.useGpu:
        model = model.cuda()
    # 获取数据
    trainSet = reidReader(srcFolder=opt.trainFolder)
    # 转化为dataloader
    trainLoader = DataLoader(trainSet, batch_size=opt.batchSize, shuffle=True, num_workers=opt.numWorker)
    if opt.trainRate != 1:
        cvSet = reidReader(srcFolder=opt.trainFolder, isCV=True)
        cvLoader = DataLoader(cvSet, batch_size=opt.batchSize, shuffle=True, num_workers=opt.numWorker)
    # 损失函数
    criterion = eval('nn.' + opt.lossFunc + '()')
    optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weightDecay)
    optimizer.zero_grad()
    # 开始读入样本训练
    log = []  # 记录损失
    trainAcc = []  # 记录训练精度
    cvAcc = []  # 记录交叉验证精度
    # 设置中断时转向保存函数
    signal.signal(signal.SIGINT, sigTerSave)
    for ii in range(opt.maxEpoch):
        for jj, (data, label) in enumerate(trainLoader):
            if not isTer:
                # 开始
                data = Variable(data)  # 没必要更新图像所以不设置requires_grad
                label = Variable(label)
                # 转到GPU
                if opt.useGpu:
                    data = data.cuda()
                    label = label.cuda()
                optimizer.zero_grad()
                score = model(data)  # 前向输出
                loss = criterion(score, label)
                loss.backward()
                optimizer.step()
                # 达到打印频率要打印
                if (jj + 1) % opt.printFreq == 0:
                    log.append(loss.cpu().data[0])  # 将损失
                    print('迭代次数:{0:d},损失函数值:{1:4.6f}'.format(ii, log[-1]))
            else:
                torch.save(model.state_dict(), 'temp.pth')
                print('完毕，中断')
                exit(-1)  # 中断
        if ii % opt.snapFreq == opt.snapFreq - 1:
            torch.save(model.state_dict(),
                       time.strftime('snapshots/' + opt.model + '%m_%H:%M:%S.pth'))  # 每训练完snapFreq轮就存一下
            print('暂存一次')
        # 训练完一轮就验证一次
        if opt.trainRate != 1:
            cvAcc.append(val(model, cvLoader))
            trainAcc.append(val(model, trainLoader))
            print('验证测试精度:{0:4.6f}%'.format(100 * cvAcc[-1]))
            print('在训练集上的精度:{0:4.6f}%'.format(100 * trainAcc[-1]))
        if (ii + 1) % opt.lrDecayRate == 0:
            # 降低学习率
            for param in optimizer.param_groups:
                if opt.minLR < param['lr']:
                    param['lr'] = param['lr'] * opt.lrDecay
            print('学习率下降至{0:4.6f}'.format(param['lr']))
    torch.save(model.state_dict(), 'snapshots/res18.pth')
    # 画图
    # pl.plot(log,'bo-',markersize=10)
    # pl.xlabel('iteration number')
    # pl.ylabel('loss value')
    # pl.savefig('loss.png')
    # print('训练损失随迭代次数的变化已存至loss.png')
    # pl.plot(np.arange(1,len(cvAcc)),100*cvAcc,'bo-',label='CV Accuracy',markersize=10)
    # pl.plot(np.arange(1,len(trainAcc)),100*trainAcc,'r*-',label='train Accuracy',markersize=10)
    # pl.xlabel('iteration number')
    # pl.ylabel('%')
    # pl.savefig('acc.png')
    # print('训练精度与交叉验证精度随迭代次数的变化已存至acc.png')


def test(**kwargs):
    """    
    [进行测试。提取图像特征]    
    Arguments:
        **kwargs {[type]} -- [参数]
    """
    opt.parse(**kwargs)
    # 加载训练数据
    testSet = reidReader(srcFolder=opt.testFolder, isTest=True)
    # 不能洗牌！！！！！
    testLoader = DataLoader(testSet, batch_size=opt.batchSize, shuffle=False, num_workers=opt.numWorker)
    # 先按照原始设计给新的模型导入参数
    model = eval('models.' + opt.model + '(' + str(opt.numClass) + ')')
    # model = torchvision.models.resnet18()
    # del model.fc
    # model.drop = nn.Dropout()
    # model.fc = nn.Linear(512, opt.numClass)
    # # 加载训练好的权重
    # model.load_state_dict(torch.load(opt.modelPath))
    # model.eval()
    # 转移到GPU
    if opt.useGpu:
        model = model.cuda()
    # 准备完毕，传入图像提取特征
    allF = np.asarray([])
    for ii, (data, label) in enumerate(testLoader):
        # 进行数据处理
        data = Variable(data)
        if opt.useGpu:
            data = data.cuda()
        calFeature = model(data, isTest=True)  # 获取特征
        # calFeature = model(data)
        if np.shape(allF)[0]:
            allF = np.vstack((allF, calFeature.data.cpu().numpy()))
        else:
            # 对于allF不存在的情况就直接复制
            allF = calFeature.data.cpu().numpy()
    allF = torch.FloatTensor(allF)
    torch.save(allF, 'snapshots/allF.pth')
    print('特征已保存至allF.pth')


def calScore(score, label):
    """[计算准确率]   
    """
    score = score.data  # 对于Variable要做这个步骤
    label = label.data
    _, predict = torch.max(score, 1)  # 按行着最大值位置作为预测
    return np.mean((predict == label).numpy()) if not opt.useGpu else np.mean((predict == label).cpu().numpy())


def val(model, loader):
    # 交叉验证
    acc = []
    criterion = eval('nn.' + opt.lossFunc)
    for ii, (data, label) in enumerate(loader):
        data = Variable(data)
        label = Variable(label)
        if opt.useGpu:
            data = data.cuda()
            label = label.cuda()
        # 进行验证
        score = model(data)
        score = calScore(score, label)
        acc.append(score)
    return torch.mean(torch.FloatTensor(acc))


def query(imgNum=None, **kwargs):
    """查询       
    Arguments:
        **kwargs {[type]} -- [description]
    """
    opt.parse(**kwargs)
    querySet = reidReader(opt.queryFolder, isQuery=True)
    # 不能洗牌
    queryLoader = DataLoader(querySet, opt.batchSize, shuffle=False, num_workers=opt.numWorker)
    # 加载模型
    model = eval('models.' + opt.model + '(' + str(opt.numClass) + ')')
    # model = torchvision.models.resnet18()
    # del model.fc
    # model.drop = nn.Dropout()
    # model.fc = nn.Linear(512, opt.numClass)
    # model.load_state_dict(torch.load(opt.modelPath))
    # model.eval()
    # 转移到GPU
    if opt.useGpu:
        model = model.cuda()
    queryF = np.array([])
    for ii, (data, label) in enumerate(queryLoader):
        # 导入查询集图像
        data = Variable(data)
        if opt.useGpu:
            data = data.cuda()
        # calFeature = model(data)
        calFeature = model(data, isTest=True)  # 获取特征
        if np.shape(queryF)[0]:
            queryF = np.vstack((queryF, calFeature.view(calFeature.size()[0], -1).data.cpu().numpy()))
        else:
            # 对于allF不存在的情况就直接复制
            queryF = calFeature.view(calFeature.size()[0], -1).data.cpu().numpy()
    queryF = torch.FloatTensor(queryF)
    torch.save(queryF, 'snapshots/queryF.pth')
    print('查询图像集合特征已保存至queryF.pth')
    # 使用欧式距离获得邻接矩阵,注意图像名字要排序
    allFiles = [os.path.join(opt.queryFolder, name) for name in os.listdir(opt.queryFolder)]
    allFiles.sort()
    # 只会计算某个样本
    testF = torch.load('snapshots/allF.pth')
    if imgNum is None:
        # 根据邻接矩阵计算CMC top6曲线
        disMat = calAdj(queryF, testF)
        curCMC = torch.zeros(disMat.size()[0], disMat.size()[1])  # 查询图数目*测试图像集合大小
        mAP = torch.zeros(disMat.size()[0], 1)
        for ii in range(disMat.size()[0]):
            # 对每一张图象分别查询
            curCMC[ii], mAP[ii] = getEva(disMat, ii)
            print('查询样本{0:d}比对完毕'.format(ii))
        print(torch.mean(curCMC, 0)[:opt.topN])
        print('mAP:{0:4.4f}'.format(torch.mean(mAP)))
    else:
        queryVec = queryF[imgNum]  # 对应查询图像特征
        disMat = calAdj(queryVec, testF)
        CMC, mAP = getEva(disMat, imgNum, isSingle=True, isSave=True)  # 找到带查询图像位置
        print(CMC[:, :opt.topN])
        print('mAP:{0:4.4f}'.format(mAP))


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


def sigTerSave(sigNum, frame):
    """
    使用ctrl+C时，将模型参数存储下来再退出，这是一个槽
    """
    global isTer
    isTer = True  # 全局变量设置为True
    print('保存模型参数至当前目录temp.pth中...')


def getEva(dis, loc, isSingle=False, isSave=False):
    """获得评价参数CMC TOP6
    Arguments:
        loc--queryID
        disLocal--query图像对全部test数据集的相似度向量
    """
    if isSingle:
        _, sortLoc = torch.sort(dis[0])
    else:
        _, sortLoc = torch.sort(dis[loc])  # 获得第loc张查询图像对全测试集的相似度
    testImgLab = [name for name in os.listdir(opt.testFolder)]  # 测试文件夹图像标签
    testImgLab.sort()  # 有17661个
    testImgCAM = np.array([int(name.split('_')[1][1]) for name in testImgLab])  # 视角
    testImgLab = np.array([int(name.split('_')[0]) for name in testImgLab])  # 标签
    queryImgLab = [name for name in os.listdir(opt.queryFolder)]  # 查询图像集合图像
    queryImgLab.sort()  # 有2228个
    queryImgCAM = np.array([int(name.split('_')[1][1]) for name in queryImgLab])  # 视角
    queryImgLab = np.array([int(name.split('_')[0]) for name in queryImgLab])  # 标签
    # top 6
    if isSave:
        # 如果可以，保存下来
        queryImages = [os.path.join(opt.queryFolder, name) for name in os.listdir(opt.queryFolder)]
        queryImages.sort()
        queryImg = Image.open(queryImages[loc])
        queryImg.save('queryRes/results/queryImg.png')  # 存储查询图像
    # 找到标签相同并且不在一个cam下的图像
    goodSam = list(set(np.where(testImgLab == queryImgLab[loc])[0]).intersection(
        set(np.where(testImgCAM != queryImgCAM[loc])[0])))
    # 找到标签相同但是在一个cam下图像
    junkSameCAM = list(set(np.where(testImgLab == queryImgLab[loc])[0]).intersection(
        set(np.where(testImgCAM == queryImgCAM[loc])[0])))
    # 根据排序确定
    CMC, imgNameSort, mAP = calCMC(goodSam, junkSameCAM, sortLoc)
    if isSave and len(imgNameSort):
        # 要保存图像
        testImages = [os.path.join(opt.testFolder, name) for name in os.listdir(opt.testFolder)]
        testImages.sort()
        for jj in range(len(imgNameSort)):
            topImg = Image.open(testImages[int(imgNameSort[0][jj])])  # 只找到top几
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
        if len(np.where(np.asarray(junkSameCAM) == sortLoc[ii])[0]):
            # 同一摄像头，直接忽视
            junkNum = junkNum + 1
            continue
        imgNameSort[:, ii] = sortLoc[ii]  # 记录是哪张图像
        if flag == 1:
            intersS = intersS + 1
        recall = intersS / numGood
        precision = intersS / (count + 1)
        mAP = mAP + (recall - oldRecall) * (0.5 * (oldPrecision + oldRecall))
        # 更新
        oldRecall = recall
        oldPrecision = precision
        count = count + 1
        if numGood == isGood:
            return CMC, imgNameSort[:opt.topN], mAP
    imgNameSort = imgNameSort[imgNameSort != 0]  # 去除0
    return CMC, imgNameSort[:opt.topN], mAP


# 转化pth为mat文件
def convert2Mat():
    import scipy.io as sci
    if not os.path.exists('snapshots/allF.pth') or not os.path.exists('snapshots/queryF.pth'):
        print('不存在测试集特征文件allF.pth或查询集特征文件queryF.pth')
    else:
        allF = torch.load('snapshots/allF.pth').numpy()
        queryF = torch.load('snapshots/queryF.pth').numpy()
        sci.savemat('snapshots/allF.mat', {'allF': allF})
        sci.savemat('snapshots/queryF.mat', {'queryF': queryF})
        print('转换完毕')


if __name__ == '__main__':
    import fire

    fire.Fire()
