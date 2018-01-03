import torch.nn as nn
import torch
import time


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.modelName = str(type(self)).split('\'')[-2].split('.')[-1]

    def load(self, path):
        self.load_state_dict(path)

    def save(self, path=None):
        if path is None:
            torch.save(self.state_dict(), time.strftime('snapshots/' + self.modelName + '%H:%M:%S.pth'))
        else:
            torch.save(self.state_dict(), path)
