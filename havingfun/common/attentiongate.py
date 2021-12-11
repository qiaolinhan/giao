import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from torch.nn.modules.batchnorm import BatchNorm2d

from doubleconv import Block as ResidualBlock

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size= 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.RReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_in = self.relu(g1 + x1)
        psi = self.psi(psi_in)

        return x *psi

if __name__ == '__main__':

    # batchsize = 1, channels = 128, inputsize = 255*255
    feature_g = torch.randn((1, 128, 255, 255))
    feature_x = torch.randn((1, 128, 255, 255))
    # model = Resnet34(img_channels=3, num_classes=3)
    model = Attention_block(F_g = 128, F_l = 128, F_int = 64)
    print(model.eval())
    preds = model(feature_g, feature_x)
    print('input shape:', feature_x.shape)
    print('preds shape:', preds.shape)

