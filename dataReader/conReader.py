# -*-coding:utf-8-*-
from PIL import Image
import torchvision.transforms as T
from torch.utils import data
import os
import sys
from collections import defaultdict

sys.path.append('../')  # 为了导入opt
from config import opt


class conReader(data.Dataset):
    """
    准备数据集合，训练数据留1交叉验证，测试数据直接变
    """

    def __init__(self, srcFolder, transformation=None, isTrain=True, isCV=False, fakeFolder=None):
        """
        初始化数据集类，srcFolder代表数据文件夹，默认没有变换且是训练数据集
        """
        allFiles = defaultdict(list)
        files = [(name.split('_')[-3], os.path.join(srcFolder, name)) for name in os.listdir(srcFolder) if
                 name[-3:] == 'jpg']  # 读入所有jpg文件
        # for fake train
        if fakeFolder is not None:
            fakeFile = [(name.split('_')[-4], os.path.join(fakeFolder, name)) for name in os.listdir(fakeFolder) if
                        name[-3:] == 'jpg']
            # merge
            files = files + fakeFile
        [allFiles[k].append(v) for k, v in files]
        self.imgName = []  # 用于存储数据集对应的图像文件名
        if isTrain:
            # 训练数据集,留一个进行交叉验证,首先将同一类图像归类
            for ii in allFiles.keys():
                self.imgName += allFiles[ii][1:]  # 留下第一个用于CV
        elif isCV:
            # CV数据集
            for ii in allFiles.keys():
                self.imgName += [allFiles[ii][0]]  # 取第一个用于CV,注意这里要加上[]变成list对象
        else:
            # 查询或者test数据集
            for ii in allFiles.keys():
                self.imgName += allFiles[ii]  # 取全数据集
            self.imgName.sort()  # 注意对于测试数据一定要排好序好对号入座
        self.imgNum = len(self.imgName)  # 图像总数
        # 给出默认变换
        if transformation is None:
            # 给一个默认的变换
            if isTrain:
                # 训练数据集要使用特殊变换
                self.trans = T.Compose([
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                    T.Resize((288, 288)),
                    T.RandomCrop((256, 256)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.trans = T.Compose([
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                    T.Resize((256, 256)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        """
        重载此函数完成读入
        """
        fName = self.imgName[index]
        flag = fName.split('_')[0].split('/')[-1]  # 判断是真图还是假图
        if flag != 'fake':
            label = int(flag)
        else:
            label = int(fName.split('_')[1])
        srcImg = self.trans(Image.open(fName))
        return srcImg, label

    def __len__(self):
        """
        数据集长度
        """
        return len(self.imgName)


pass
