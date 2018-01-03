import torchvision.models as models
import torch.nn as nn
from .BasicNet import BasicNet


class resnet18(BasicNet):
    def __init__(self, numClass):
        super(resnet18, self).__init__()
        self.res = models.resnet18(pretrained=True)
        del self.res.fc
        self.res.fc = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, numClass)
        )

    def forward(self, x, isTest=False):
        # 前传播算法
        if isTest:
            self.eval()
            x = self.res.conv1(x)
            x = self.res.bn1(x)
            x = self.res.relu(x)
            x = self.res.maxpool(x)
            x = self.res.layer1(x)
            x = self.res.layer2(x)
            x = self.res.layer3(x)
            x = self.res.layer4(x)
            x = self.res.avgpool(x)
            # 只返回特征
            return x.view(x.size(0), -1)
        else:
            return self.res(x)
