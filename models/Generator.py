# encoding=utf-8
import torch.nn as nn
from .BasicNet import BasicNet


class Generator(BasicNet):
    """
    GAN的生成器
    """

    def __init__(self, vecZ, channel, genL):
        super(Generator, self).__init__()
        # 开始构建网络结构
        # vecZ为输入噪声的大小，channel为要生成图像的通道数目，genL为基本的层数，将在隐层中作为基数进行拓展
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(vecZ, 4 * genL, kernel_size=4),
            nn.BatchNorm2d(4 * genL),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(4 * genL, 2 * genL, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * genL),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(2 * genL, genL, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(genL),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(genL, channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))
