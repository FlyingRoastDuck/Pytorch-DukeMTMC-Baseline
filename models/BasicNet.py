import torch.nn as nn
import torch
import time


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.numBottleNeck = 512 #特征2048维，512是其后继层节点数
        self.modelName = str(type(self)).split('\'')[-2].split('.')[-1]

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            torch.save(self.state_dict(), time.strftime('snapshots/' + self.modelName + '%H:%M:%S.pth'))
        else:
            torch.save(self.state_dict(), path)

    def weights_init_kaiming(self,m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
            nn.init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            nn.init.normal(m.weight.data, 1.0, 0.02)
            nn.init.constant(m.bias.data, 0.0)

    def weights_init_classifier(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal(m.weight.data, std=0.001)
            nn.init.constant(m.bias.data, 0.0)
