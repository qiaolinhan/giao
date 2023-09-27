from multiprocessing.dummy import Pool
from turtle import forward
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d

class Convtest(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convtest, self).__init__()
        self.conv2d_test = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv2d_test(x)

class Poolingtest(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Poolingtest, self).__init__()
        self.pool_test = nn.MaxPool2d(in_channels, out_channels, 2)

    def forward(self, x):
        return self.pool_test(x)

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UBlock, self).__init__()
        self.tblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.tblock(x)

class Up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_conv, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.upconv(x)

if __name__ == '__main__':
    layer1 = DBlock(in_channels=1, out_channels=64)
    layer2 = UBlock(in_channels=128, out_channels=64)
    layer3 = Up_conv(in_channels=256, out_channels=1)

    convtest_layer = Convtest(in_channels = 1, out_channels = 64)
    pooling_layer = Poolingtest(in_channels = 64, out_channels=64)
    feature_in = torch.randn((5, 64, 512, 512))
    feature_out = pooling_layer(feature_in)
    print('feature out size:', feature_out.shape)
