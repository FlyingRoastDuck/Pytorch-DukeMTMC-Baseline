# -*- coding: utf-8 -*-
from .BasicNet import BasicNet
import torch.nn as nn
from .BasicBlock import BasicBlock
from torch.nn import functional as F


class resnet18(BasicNet):
    """
    resnet18，模型定义
    """

    def __init__(self, outClass):
        super(resnet18, self).__init__()
        # 初步处理
        self.preH = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 生成层
        self.layer1 = self.genLayer(64, 128, numBlocks=2)
        self.layer2 = self.genLayer(128, 256, numBlocks=2, stride=2)
        self.layer3 = self.genLayer(256, 512, numBlocks=2, stride=2)
        self.layer4 = self.genLayer(512, 512, numBlocks=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        # self.fc = nn.Linear(512, outClass)
        self.fc = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, outClass)
        )

    def forward(self, x, isTest=False):
        # 前传
        if isTest:
            self.eval()
            out = self.preH(x)
            out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
            out = self.avgpool(out),
            return out.view(out.size(0), -1)
        else:
            out = self.preH(x)
            out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
            out = self.avgpool(out),
            out = out.view(out.size(0), -1)
            return self.fc(out)

    def genLayer(self, inChannel, outChannel, numBlocks, stride=1):
        # 生成一个层
        # 侧边层，用于直传部分转换到与输出同样大小
        transf = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, 1, stride, bias=False),
            nn.BatchNorm2d(outChannel))
        layers = [BasicBlock(inChannel, outChannel, stride, isDownSam=transf)]
        for ii in range(1, numBlocks):
            layers.append(BasicBlock(outChannel, outChannel))
        return nn.Sequential(*layers)
