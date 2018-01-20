from config import opt
from dataReader.conReader import conReader
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import models
import numpy as np
import os
import signal
import torch.nn as nn
from PIL import Image
from torch.optim import lr_scheduler


def train(**kwargs):
    """
    训练函数
    """
    opt.parse(**kwargs)
    global isTer
    isTer = False  # 设置全局变量方便中断时存储model参数
    trainData = conReader(opt.trainFolder)
    trainLoader = DataLoader(trainData, batch_size=opt.batchSize,
                             shuffle=True, num_workers=opt.numWorker)
    cvData = conReader(opt.trainFolder, isTrain=False,isCV=True)
    cvLoader = DataLoader(cvData, batch_size=opt.batchSize,
                          shuffle=True, num_workers=opt.numWorker)
    # 生成模型,使用预训练
    model = eval('models.' + opt.model + '(numClass=' + str(opt.numClass) + ')')
    model.train(True)  # 设置模型为训练模式（dropout等均生效）
    criterion = eval('nn.' + opt.lossFunc + '()')
    #初始化优化器,要使用不同的学习率进行优化
    acclerateParams = list(map(id, model.res.fc.parameters())) + list(map(id, model.classifierLayer.parameters()))
    baseParams = filter(lambda p: id(p) not in acclerateParams, model.parameters())
    # 观测多个变量
    optimizer = torch.optim.SGD([
        {'params': baseParams, 'lr': opt.lr},
        {'params': model.res.fc.parameters(), 'lr': 0.1},
        {'params': model.classifierLayer.parameters(), 'lr': 0.1}
    ], momentum=0.9, weight_decay=5e-4, nesterov=True)    
    timerOp = lr_scheduler.StepLR(optimizer,step_size=opt.lrDecayRate,gamma=opt.lrDecay) #每经过40轮迭代，学习率就变为原来的0.1
    lossVal = []  # 记录损失函数变化
    trainAcc = []  # 记录训练集精度变化
    cvAcc = []  # 记录交叉验证数据集精度变化
    if opt.useGpu:
        model = model.cuda()
    # 开始训练
    signal.signal(signal.SIGINT, sigTerSave)  # 设置监听器方便随时中断
    for ii in range(opt.maxEpoch):
        for phase in ['train', 'val']:
            if phase is 'val':              
                # 进入val模式
                model.train(False)
                cvAcc.append(val(model, cvLoader))
                print('验证测试精度:{0:4.6f}%'.format(100 * cvAcc[-1]))
            else:
                timerOp.step() #仅有达到40轮之后学习率才会下降
                model.train(True)
                for jj, (data, label) in enumerate(trainLoader):
                    if not isTer:
                        data = Variable(data)
                        label = Variable(label)
                        if opt.useGpu:
                            data = data.cuda()
                            label = label.cuda()
                        optimizer.zero_grad()
                        score = model(data)
                        loss = criterion(score, label)
                        lossVal.append(loss.cpu().data[0])  # 存储下来
                        loss.backward()
                        optimizer.step()  # 更新
                    else:
                        # 中断时要先存储模型参数再退出
                        model.save('temp.pth')
                        print('完毕，中断')
                        exit(-1)  # 中断
                    if jj % opt.printFreq == 0:
                        # 打印loss
                        print('迭代次数：{0:d},损失：{1:4.6f}'.format(ii, lossVal[-1]))
                    if ii % opt.snapFreq == opt.snapFreq - 1:
                        # 要保存
                        model.save()
    # 保存
    model.save('snapshots/' + opt.model + '.pth')
    # 保留数据方便作图
    np.savetxt("cvAcc.txt", cvAcc)
    np.savetxt("trainAcc.txt", trainAcc)
    np.savetxt("lossVal.txt", lossVal)
    print('完毕')


def test(**kwargs):
    #针对test（gallery）数据集进行
    opt.parse(**kwargs)
    # 进行测试，计算相似度
    model = eval('models.' + opt.model + '(' + str(opt.numClass) + ')')
    model.load(opt.modelPath)
    model.res.fc = nn.Sequential()
    model.classifierLayer = nn.Sequential() #注意这里不能直接删除，免得前传的时候找不到层，置空就好
    model = model.eval()  # 设置为测试模式，dropout等均失效    
    # 准备数据
    testData = conReader(opt.testFolder, isTrain=False)
    # 不能洗牌
    testLoader = DataLoader(
        testData, batch_size=opt.batchSize, num_workers=opt.numWorker)
    if opt.useGpu:
        model = model.cuda()
    features = torch.FloatTensor()  # 初始化一个floattensor用于存储特征
    for ii, (imgData, label) in enumerate(testLoader):
        n, _, _, _ = imgData.size()  # 获得图像数目
        doubleF = torch.FloatTensor(n, model.numFin).zero_()  # 存储反转前后提取特征的融合特征
        for jj in range(2):
            inImg = hozFilp(imgData)  # 第一次要反转
            if opt.useGpu:
                inImg = inImg.cuda()
            calF = model(Variable(inImg))  # 提取出2048维特征
            doubleF += calF.data.cpu() #融合即是相加        
        #feature归一化        
        normF = torch.norm(doubleF, p=2, dim=1, keepdim=True)
        doubleF = doubleF.div(normF.expand_as(doubleF))
        features = torch.cat((features, doubleF), 0)  # 将得到的特征按照垂直方向进行拼接
    torch.save(features, "snapshots/allF.pth")
    print("TEST所有特征已经保存")


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
    return torch.mean((predict==label).type(torch.FloatTensor))


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
        acc.append(calScore(score, label))
    return torch.mean(torch.FloatTensor(acc))


def query(imgNum=None, **kwargs):
    """查询
    Arguments:
        **kwargs {[type]} -- [description]
    """
    #针对query数据集进行处理的函数，主要仍然是提取特征
    opt.parse(**kwargs)
    querySet = conReader(opt.queryFolder, isTrain=False)
    # 不能洗牌
    queryLoader = DataLoader(
        querySet, batch_size=opt.batchSize, num_workers=opt.numWorker)
    # 加载模型
    # 进行测试，计算相似度
    model = eval('models.' + opt.model + '(' + str(opt.numClass) + ')')
    model.load(opt.modelPath)
    model.res.fc = nn.Sequential()
    model.classifierLayer = nn.Sequential() #注意这里不能直接删除，免得前传的时候找不到层，置空就好
    model = model.eval()  # 设置为测试模式，dropout等均失效    
    # 转移到GPU
    if opt.useGpu:
        model = model.cuda()

    queryF = torch.FloatTensor()  # 初始化一个floattensor用于存储特征
    for ii, (imgData, label) in enumerate(queryLoader):
        n, _, _, _ = imgData.size()  # 获得图像数目
        doubleF = torch.FloatTensor(n, model.numFin).zero_()  # 存储反转前后提取特征的融合特征
        for jj in range(2):
            inData = hozFilp(imgData)  # 第一次要反转
            if opt.useGpu:
                inData = inData.cuda()
            calF = model(Variable(inData))  # 提取出2048维特征
            doubleF += calF.data.cpu() #融合即是相加        
        #feature归一化        
        normF = torch.norm(doubleF, p=2, dim=1, keepdim=True)
        doubleF = doubleF.div(normF.expand_as(doubleF))
        queryF = torch.cat((queryF, doubleF), 0)  # 将得到的特征按照垂直方向进行拼接
    torch.save(queryF, 'snapshots/queryF.pth')
    print('查询图像集合特征已保存至queryF.pth')
    # 使用欧式距离获得邻接矩阵,注意图像名字要排
    allFiles = [os.path.join(opt.queryFolder, name)
                for name in os.listdir(opt.queryFolder)]
    allFiles.sort()
    # 只会计算某个样本
    testF = torch.load('snapshots/allF.pth')
    if imgNum is None:
        # 根据邻接矩阵计算CMC top6曲线
        disMat = calAdj(queryF, testF)
        curCMC = torch.zeros(disMat.size()[0], disMat.size()[
                             1])  # 查询图数�??*测试图像集合大小
        mAP = torch.zeros(disMat.size()[0], 1)
        for ii in range(disMat.size()[0]):
            # 对每一张图象分别查
            curCMC[ii], mAP[ii] = getEva(disMat, ii)
            print('查询样本{0:d}比对完毕'.format(ii))
        print(torch.mean(curCMC, 0)[:opt.topN])
        print('mAP:{0:4.4f}'.format(torch.mean(mAP)))
    else:
        queryVec = queryF[imgNum]  # 对应查询图像特征
        disMat = calAdj(queryVec, testF)
        CMC, mAP = getEva(disMat, imgNum, isSingle=True,
                          isSave=True)  # 找到带查询图像位�??
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


def sigTerSave(sigNum, frame):
    """
    使用ctrl+C时，将模型参数存储下来再退出，这是一个槽
    """
    global isTer
    isTer = True  # 全局变量设置为True
    print('保存模型参数至当前目录的temp.pth...')


if __name__ == '__main__':
    import fire

    fire.Fire()
