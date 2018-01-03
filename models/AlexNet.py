# -*-coding:utf-8-*-
import torch.nn as nn
from .BasicNet import BasicNet
import torchvision.models as models


class AlexNet(BasicNet):
    def __init__(self, outClass):
        """
        继承之前写的BasicNet，在此处完成参数初始化
        """
        super(AlexNet, self).__init__()
        """
        父类函数初始化之后，给模型名字赋值并进行前向传播结构设计
        """
        self.alex = models.alexnet(pretrained=True)
        preW = self.alex.state_dict()
        del self.alex.features
        self.alex.features = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, outClass)
        )
        # 载入已知权重
        curW = self.alex.state_dict()
        newW = {k: v for k, v in preW.items() if k in curW}
        curW.update(newW)
        self.alex.load_state_dict(newW)

    def forward(self, x, isTest=False):
        """
        前向传播
        """
        if isTest:
            self.alex.eval()
            out = self.alex.classifier(x)
            out = self.alex.features[0](out)
            out = self.alex.features[1](out)
            out = self.alex.features[2](out)
            out = self.alex.features[3](out)
            out = self.alex.features[4](out)
            out = self.alex.features[5](out)
            return out.view(out.size()[0], -1)
        else:
            return self.alex(x)


pass
