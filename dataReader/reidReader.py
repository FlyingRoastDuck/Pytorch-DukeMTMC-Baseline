# -*-coding:utf-8-*-
from PIL import Image
import torchvision.transforms as T
from torch.utils import data
import os
import sys

sys.path.append('../')  # 为了导入opt
from config import opt


class reidReader(data.Dataset):
    """
    继承torch.utils.data.Dataset实现一个数据读取器
    """
    def __init__(self, srcFolder, transformation=None, isCV=False, isQuery=False, isTest=False):
        """
        初始化参数，读入数据,isCV判断是不是交叉验证数据,默认训练集
        """
        # 读入所有文件
        allFiles = [os.path.join(srcFolder, name) for name in os.listdir(srcFolder)]
        fileNum = len(allFiles)
        if isCV:
            # 验证数据集
            self.imgName = allFiles[int(opt.trainRate * fileNum):]  # 后面20%拿来验证
        elif isQuery or isTest:
            # 查询数据集或者测试数据,此处应根据dir顺序重排序文件名
            allFiles.sort()  # 在查询与测试阶段注意将图像排序
            self.imgName = allFiles
        else:
            # 训练数据集
            if opt.trainRate != 1:
                self.imgName = allFiles[:int(opt.trainRate * fileNum)]
            else:
                self.imgName = allFiles
        if transformation is None:
            # 给一个默认的变换
            self.trans = T.Compose(
                [T.Scale((227, 227)),
                 T.ToTensor()])

    def __getitem__(self, index):
        """
        重载此函数完成读入
        """
        fName = self.imgName[index]
        label = int(fName.split('_')[0].split('/')[-1])
        srcImg = self.trans(Image.open(fName))
        return srcImg, label

    def __len__(self):
        """
        数据集长度
        """
        return len(self.imgName)


pass
