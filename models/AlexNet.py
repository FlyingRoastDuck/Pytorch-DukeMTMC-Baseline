# -*-coding:utf-8-*-
import torch.nn as nn
from .BasicNet import BasicNet
import numpy as np


class AlexNet(BasicNet):
    def __init__(self, outClass):
        """
        继承之前写的BasicNet，在此处完成参数初始化
        """
        super(AlexNet, self).__init__()
        """
        父类函数初始化之后，给模型名字赋值并进行前向传播结构设计
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation=1),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, dilation=(1, 1)),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation=(1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(4096, outClass)

    def forward(self, input, isTest=False):
        """
        前向传播
        """
        out = self.features(input)
        if not isTest:
            # 进入训练阶段
            out = out.view(out.size()[0], -1)
            out = self.classifier(out)
        else:
            # 测试阶段
            self.eval()  # 去除drop层
            out = out.view(out.size()[0], -1)
            for ii in range(6):
                # 到最后一层前一层终止
                out = self.classifier[ii](out)
        return out


pass
