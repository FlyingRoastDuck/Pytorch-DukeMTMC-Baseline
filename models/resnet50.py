import torchvision.models as models
import torch.nn as nn
from .BasicNet import BasicNet


class resnet50(BasicNet):
    def __init__(self, numClass):
        super(resnet50, self).__init__()
        self.res = models.resnet50(pretrained=True)
        self.res.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.numFin = self.res.fc.in_features  # 删除fc层之前，看看fc层映射的输入维度大小
        del self.res.fc
        self.res.fc = nn.Sequential(
            nn.Linear(self.numFin, self.numBottleNeck),
            nn.BatchNorm1d(self.numBottleNeck),
            nn.LeakyReLU(0.1),
            nn.Dropout()
        )
        self.res.fc.apply(self.weights_init_kaiming)  # 初始化权重
        # 将特征映射到对应类别
        self.classifierLayer = nn.Sequential(
            nn.Linear(self.numBottleNeck, numClass)
        )
        self.classifierLayer.apply(self.weights_init_classifier)  # 初始化权重

    def forward(self, x):
        # 前传播算法
        out = self.res.maxpool(self.res.relu(self.res.bn1(self.res.conv1(x))))
        out = self.res.avgpool(self.res.layer4(self.res.layer3(self.res.layer2(self.res.layer1(out)))))
        out = out.squeeze()
        return self.classifierLayer(self.res.fc(out))
