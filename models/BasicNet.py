# -*-coding:utf-8-*-
import torch
import time
import torch.nn as nn


class BasicNet(nn.Module):
    """
    封装基本的保存加载操作
    """

    def __init__(self):
        super(BasicNet, self).__init__()
        self.modelName = str(type(self)).split('\'')[-2]  # 确定模型名字

    def save(self, fileName=None):
        """
        模型基本保存操作
        """
        if fileName is None:
            fileName = time.strftime('snapshots/' + self.modelName + '%m_%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), fileName)

    def load(self, path):
        """
        加载模型参数
        """
        self.load_state_dict(torch.load(path))


pass
