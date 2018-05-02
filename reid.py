# encoding=utf-8
from dataReader.conReader import conReader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import models
import signal
import torch.nn as nn
from torch.optim import lr_scheduler
from utils import *
from tqdm import trange


def train(**kwargs):
    """
    训练函数
    """
    opt.parse(**kwargs)
    global isTer
    isTer = False  # 设置全局变量方便中断时存储model参数
    trainData = conReader(opt.trainFolder, fakeFolder=opt.fakeTrainFolder)
    trainNum = trainData.imgNum # 训练图像总数
    trainLoader = DataLoader(trainData, batch_size=opt.batchSize,
                             shuffle=True, num_workers=opt.numWorker)
    cvData = conReader(opt.trainFolder, isTrain=False, isCV=True, fakeFolder=opt.fakeTrainFolder)
    cvLoader = DataLoader(cvData, batch_size=opt.batchSize,
                          shuffle=True, num_workers=opt.numWorker)
    # 生成模型,使用预训练
    model = eval('models.' + opt.model + '(numClass=' + str(opt.numClass) + ')')
    model.train(True)  # 设置模型为训练模式（dropout等均生效）
    criterion = eval('nn.' + opt.lossFunc + '()')
    # 初始化优化器,要使用不同的学习率进行优化
    acclerateParams = list(map(id, model.res.fc.parameters())) + list(map(id, model.classifierLayer.parameters()))
    baseParams = filter(lambda p: id(p) not in acclerateParams, model.parameters())
    # 观测多个变量
    optimizer = torch.optim.SGD([
        {'params': baseParams, 'lr': opt.lr},
        {'params': model.res.fc.parameters(), 'lr': 0.1},
        {'params': model.classifierLayer.parameters(), 'lr': 0.1}
    ], momentum=0.9, weight_decay=5e-4, nesterov=True)
    timerOp = lr_scheduler.StepLR(optimizer, step_size=opt.lrDecayRate, gamma=opt.lrDecay)  # 每经过40轮迭代，学习率就变为原来的0.1
    lossVal = []  # 记录损失函数变化
    trainAcc = []  # 记录训练集精度变化
    cvAcc = []  # 记录交叉验证数据集精度变化
    if opt.useGpu:
        model = model.cuda()
    # 开始训练
    signal.signal(signal.SIGINT, sigTerSave)  # 设置监听器方便随时中断
    with trange(opt.maxEpoch, desc='Round') as epochBar:  # 控制外部迭代的进度条
        for ii in epochBar:
            for phase in ['train', 'val']:
                if phase is 'val':
                    # 进入val模式
                    model.train(False)
                    cvAcc.append(val(model, cvLoader))
                    epochBar.set_description('val acc after epoch{0:d}:{1:4.3f}%'.format(ii, 100 * cvAcc[-1]))
                else:
                    timerOp.step()  # 仅有达到40轮之后学习率才会下降
                    model.train(True)
                    roundNum = round(trainNum // opt.batchSize)
                    with trange(roundNum) as roundBar:
                        for jj, (data, label) in zip(roundBar, trainLoader):
                            roundBar.set_description('Round')
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
                            # if jj % opt.printFreq == 0:
                            #     # 打印loss
                            roundBar.set_description('current loss{}'.format(lossVal[-1]))
                            if ii % opt.snapFreq == opt.snapFreq - 1:
                                # 要保存
                                model.save()
    # 保存
    model.save('snapshots/' + opt.model + '.pth')
    # 保留数据方便作图
    np.savetxt("cvAcc.txt", cvAcc)
    np.savetxt("trainAcc.txt", trainAcc)
    np.savetxt("lossVal.txt", lossVal)
    print('done')


def test(**kwargs):
    # 针对test（gallery）数据集进行
    opt.parse(**kwargs)
    # 进行测试，计算相似度
    model = eval('models.' + opt.model + '(' + str(opt.numClass) + ')')
    model.load(opt.modelPath)
    model.res.fc = nn.Sequential()
    model.classifierLayer = nn.Sequential()  # 注意这里不能直接删除，免得前传的时候找不到层，置空就好
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
            doubleF += calF.data.cpu()  # 融合即是相加
        # feature归一化
        normF = torch.norm(doubleF, p=2, dim=1, keepdim=True)
        doubleF = doubleF.div(normF.expand_as(doubleF))
        features = torch.cat((features, doubleF), 0)  # 将得到的特征按照垂直方向进行拼接
    torch.save(features, "snapshots/allF.pth")
    print("TEST所有特征已经保存")


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
    # 针对query数据集进行处理的函数，主要仍然是提取特征
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
    model.classifierLayer = nn.Sequential()  # 注意这里不能直接删除，免得前传的时候找不到层，置空就好
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
            doubleF += calF.data.cpu()  # 融合即是相加
        # feature归一化
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
            1])  # 查询图数
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
