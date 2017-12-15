# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """
    设置resnet18基本模块
    """

    def __init__(self, inChannel, outChannel, stride=1, isDownSam=None):
        """
        基本构造函数
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannel)
        self.downsample = isDownSam  # 用于表示旁路（不经过前向传播的路）对应的变换

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        return F.relu(out)
