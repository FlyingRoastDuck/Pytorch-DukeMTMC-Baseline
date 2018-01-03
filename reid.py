from config import opt
from dataReader.reidReader import reidReader
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import models
import numpy as np
import os
import signal
import torch.nn as nn
from PIL import Image


def train():
    """
    训练函数
    """
    global isTer
    isTer = False  # 设置全局变量方便中断时存储model参数
    trainData = reidReader(opt.trainFolder)
    trainLoader = DataLoader(trainData, batch_size=opt.batchSize,
                             shuffle=True, num_workers=opt.numWorker)
    cvData = reidReader(opt.trainFolder, isCV=True)
    cvLoader = DataLoader(cvData, batch_size=opt.batchSize,
                          shuffle=True, num_workers=opt.numWorker)
    # 生成模型,使用预训练模�?
    model = eval('models.' + opt.model + '(numClass=' + str(opt.numClass) + ')')
    criterion = eval('nn.' + opt.lossFunc + '()')
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightDecay)
    optimizer.zero_grad()
    lossVal = []
    trainAcc = []
    cvAcc = []
    if opt.useGpu:
        model = model.cuda()
    # 开始训�?
    signal.signal(signal.SIGINT, sigTerSave)
    for ii in range(opt.maxEpoch):
        for jj, (data, label) in enumerate(trainLoader):
            if not isTer:
                data = Variable(data)
                label = Variable(label)
                if opt.useGpu:
                    data = data.cuda()
                    label = label.cuda()
                score = model(data)
                loss = criterion(score, label)
                lossVal.append(loss.cpu().data[0])  # 存储下来
                loss.backward()
                optimizer.step()  # 更新
            else:
                # 中断
                model.save('temp.pth')
                print('完毕，中�?')
                exit(-1)  # 中断
            if jj % opt.printFreq == 0:
                # 打印loss
                print('迭代次数：{0:d},损失：{1:4.6f}'.format(ii, lossVal[-1]))
            if ii % opt.snapFreq == opt.snapFreq - 1:
                # 要保存一�?
                model.save()
            if (ii + 1) % opt.lrDecayRate == 0:
                # 要降低学习率
                for param in optimizer.param_groups:
                    if opt.minLR < param['lr']:
                        param['lr'] *= opt.lrDecay
                        print('学习率下降至{0:4.6f}'.format(param['lr']))
        if opt.trainRate != 1:
            # 训完一轮测试一�?
            cvAcc.append(val(model, cvLoader))
            trainAcc.append(val(model, trainLoader))
            print('验证测试精度:{0:4.6f}%'.format(100 * cvAcc[-1]))
            print('在训练集上的精度:{0:4.6f}%'.format(100 * trainAcc[-1]))
    # 保存
    model.save('snapshots/' + opt.model + '.pth')
    # 作图
    np.savetxt("cvAcc.txt", cvAcc)
    np.savetxt("trainAcc.txt", trainAcc)
    np.savetxt("lossVal.txt", lossVal)


def test():
    # 进行测试，计算相似度
    model = eval('models.' + opt.model + '(' + str(opt.numClass) + ')')
    model.load_state_dict(opt.modelPath)
    # 准备数据�?
    testData = reidReader(opt.trainFolder, isTest=True)
    # 不能洗牌
    testLoader = DataLoader(testData, batch_size=opt.batchSize, num_workers=opt.numWorker)
    if opt.useGpu:
        model = model.cuda()
    features = np.array([])
    for ii, (data, label) in enumerate(testData):
        data = Variable(data)
        label = Variable(label)
        if opt.useGpu:
            data = data.cuda()
            label = label.cuda()
        calF = model(data, isTest=True)
        if np.shape(features[0]):
            np.vstack((features, calF.data.cpu().numpy()))
        else:
            features = calF.data.cpu().numpy()
    features = torch.FloatTensor(features)
    torch.save(features, "snapshots/allF.pth")
    print("所有特征已经保�?")


def calScore(score, label):
    """[计算准确率]
    """
    score = score.data  # 对于Variable要做这个步骤
    label = label.data
    _, predict = torch.max(score, 1)  # 按行着最大值位置作为预�?
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
        acc.append(calScore(score, label))
    return torch.mean(torch.FloatTensor(acc))


def query(imgNum=None):
    """查询
    Arguments:
        **kwargs {[type]} -- [description]
    """
    querySet = reidReader(opt.queryFolder, isQuery=True)
    # 不能洗牌
    queryLoader = DataLoader(querySet, batch_size=opt.batchSize, num_workers=opt.numWorker)
    # 加载模型
    model = eval('models.' + opt.model + '(' + str(opt.numClass) + ')')
    # 转移到GPU
    if opt.useGpu:
        model = model.cuda()
    queryF = np.array([])
    for ii, (data, label) in enumerate(queryLoader):
        # 导入查询集图�?
        data = Variable(data)
        if opt.useGpu:
            data = data.cuda()
        calFeature = model(data, isTest=True)  # 获取特征
        if np.shape(queryF)[0]:
            queryF = np.vstack((queryF, calFeature.view(calFeature.size()[0], -1).data.cpu().numpy()))
        else:
            # 对于allF不存在的情况就直接复�?
            queryF = calFeature.view(calFeature.size()[0], -1).data.cpu().numpy()
    queryF = torch.FloatTensor(queryF)
    torch.save(queryF, 'snapshots/queryF.pth')
    print('查询图像集合特征已保存至queryF.pth')
    # 使用欧式距离获得邻接矩阵,注意图像名字要排�?
    allFiles = [os.path.join(opt.queryFolder, name) for name in os.listdir(opt.queryFolder)]
    allFiles.sort()
    # 只会计算某个样本
    testF = torch.load('snapshots/allF.pth')
    if imgNum is None:
        # 根据邻接矩阵计算CMC top6曲线
        disMat = calAdj(queryF, testF)
        curCMC = torch.zeros(disMat.size()[0], disMat.size()[1])  # 查询图数�?*测试图像集合大小
        mAP = torch.zeros(disMat.size()[0], 1)
        for ii in range(disMat.size()[0]):
            # 对每一张图象分别查�?
            curCMC[ii], mAP[ii] = getEva(disMat, ii)
            print('查询样本{0:d}比对完毕'.format(ii))
        print(torch.mean(curCMC, 0)[:opt.topN])
        print('mAP:{0:4.4f}'.format(torch.mean(mAP)))
    else:
        queryVec = queryF[imgNum]  # 对应查询图像特征
        disMat = calAdj(queryVec, testF)
        CMC, mAP = getEva(disMat, imgNum, isSingle=True, isSave=True)  # 找到带查询图像位�?
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
        disLocal--query图像对全部test数据集的相似度向�?
    """
    testImgLab = [name for name in os.listdir(opt.testFolder)]  # 测试文件夹图像标�?
    testImgLab.sort()  # �?17661�?
    testImgCAM = np.array([int(name.split('_')[1][1]) for name in testImgLab])  # 视角
    testImgLab = np.array([int(name.split('_')[0]) for name in testImgLab])  # 标签
    queryImgLab = [name for name in os.listdir(opt.queryFolder)]  # 查询图像集合图像
    queryImgLab.sort()  # �?2228�?
    queryImgCAM = np.array([int(name.split('_')[1][1]) for name in queryImgLab])  # 视角
    queryImgLab = np.array([int(name.split('_')[0]) for name in queryImgLab])  # 标签
    # 针对单个输入和多输入分别考虑
    if isSingle:
        _, sortLoc = torch.sort(dis[0])
    else:
        _, sortLoc = torch.sort(dis[loc])  # 获得第loc张查询图像对全测试集的相似度

    # 找到标签相同并且不在一个cam下的图像
    goodSam = list(set(np.where(testImgLab == queryImgLab[loc])[0]).intersection(
        set(np.where(testImgCAM != queryImgCAM[loc])[0])))
    # 找到标签相同但是在一个cam下图�?
    junkSameCAM = list(set(np.where(testImgLab == queryImgLab[loc])[0]).intersection(
        set(np.where(testImgCAM == queryImgCAM[loc])[0])))
    # top 6
    if isSave:
        # 如果可以，保存下�?
        queryImages = [os.path.join(opt.queryFolder, name) for name in os.listdir(opt.queryFolder)]
        queryImages.sort()
        queryImg = Image.open(queryImages[loc])
        queryImg.save('queryRes/results/queryImg.png')  # 存储查询图像
    # 根据排序确定
    CMC, imgNameSort, mAP = calCMC(goodSam, junkSameCAM, sortLoc)
    if isSave and len(imgNameSort):
        # 要保存图�?
        testImages = [os.path.join(opt.testFolder, name) for name in os.listdir(opt.testFolder)]
        testImages.sort()
        for jj in range(len(imgNameSort)):
            topImg = Image.open(testImages[int(imgNameSort[0][jj])])  # 只找到top�?
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
            imgNameSort[0, ii] = sortLoc[ii]  # 记录是哪张图�?
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
        if numGood == isGood:
            return CMC, imgNameSort[:opt.topN], mAP
    imgNameSort = imgNameSort[imgNameSort != 0]  # 去除0
    return CMC, imgNameSort[:opt.topN], mAP


def sigTerSave(sigNum, frame):
    """
    使用ctrl+C时，将模型参数存储下来再退出，这是一个槽
    """
    global isTer
    isTer = True  # 全局变量设置为True
    print('保存模型参数至当前目录temp.pth�?...')


if __name__ == '__main__':
    import fire

    fire.Fire()
