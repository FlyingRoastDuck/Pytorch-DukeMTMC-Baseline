# encoding=utf-8
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channel, disL):
        super(Discriminator, self).__init__()
        # 32 x 32
        self.layer1 = nn.Sequential(nn.Conv2d(channel, disL, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(disL),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16
        self.layer2 = nn.Sequential(nn.Conv2d(disL, disL * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(disL * 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8
        self.layer3 = nn.Sequential(nn.Conv2d(disL * 2, disL * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(disL * 4),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4
        self.layer4 = nn.Sequential(nn.Conv2d(disL * 4, 1, kernel_size=4, stride=1, padding=0),
                                    nn.Sigmoid())

    def forward(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))
