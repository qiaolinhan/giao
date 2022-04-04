# 2022-04-03
# Re-modifying
# gate signal: in upsampling 
# attention signal: in downsampling, one more down.

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from depthwiseblock import DDepthwise, UDepthwise, Up_conv

import sys
sys.path.insert(1, 'havingfun/deving/tools')
from resizetensor import sizechange

class Attentiongate_block(nn.Module):
    def __init__(self, att_channels, gating_channels):
        super(Attentiongate_block, self).__init__()
        self.theta = nn.Sequential(
            nn.Conv2d(att_channels, gating_channels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(gating_channels),
            nn.Conv2d(gating_channels, gating_channels, 1, 2, 0), # to change H,W of feature map
            nn.BatchNorm2d(gating_channels),
        )

        self.phi = nn.Sequential(
            nn.ConvTranspose2d(gating_channels, gating_channels, kernel_size= 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(gating_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.ConvTranspose2d(gating_channels, att_channels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(att_channels),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2), # to change H, W of feature map
        )

    def forward(self, att, gate):
        att1 = self.theta(att)
        # print(f'att size before down: {att.size()}')
        # print(f'att size after down: {att1.size()}')

        gl = self.phi(gate)
        # print(f'gating size: {gl.size()}')
        # if gl.size() != att.size():
        #      gl = sizechange(gl, att1)
        # print(f'resized gating size: {gl.size()}')

        psi_in = self.relu(att1 + gl)
        # print(f'psi_in size: {psi_in.size()}')
        psi_out = self.psi(psi_in)
        if psi_out.size() != att.size():
            psi_out = sizechange(psi_out, att)
        # print(f'psi_out size: {psi_out.size()}')
        y = torch.mul(att, psi_out)
        # print(f'attention gate out size: {y.size()}')
        return y

if __name__ == '__main__':

    # batchsize = 2, channels = 128, inputsize = 255*255
    input = torch.randn((2, 128, 400, 400))
    Down00 = DDepthwise(in_channels=128, out_channels=256)
    down00 = Down00(input)
    # print(f'down00 size: {down00.size()}')
    Down01 = DDepthwise(in_channels = 256, out_channels=256)
    Pooling1 = nn.MaxPool2d(2, 2, 0) 
    down01 = Down01(down00)
    down02 = Pooling1(down01)
    print(f'down01 size: {down01.size()}')
    # print(f'down02 size: {down02.size()}')

    Down10 = DDepthwise(in_channels=256, out_channels=512)
    down10 = Down10(down02)
    # print(f'down10 size: {down10.size()}')
    Down11 = DDepthwise(in_channels=512, out_channels=512)
    down11 = Down11(down10)
    print(f'down11 size: {down11.size()}')

    att0 = down01
    print(f'att0 size: {att0.size()}')
    gate1 = down11
    print(f'gate1 size: {gate1.size()}')
    Attgate = Attentiongate_block(att_channels=256, gating_channels=512)
    attgate = Attgate(att0, gate1)
    print(f'attgate size: {attgate.size()}')

    _up10 = attgate
    print(f'_up10 size: {_up10.size()}')
    Up10 = UDepthwise(in_channels = 512, out_channels=256)
    _up11 = Up10(down11)
    print(f'_up11 size: {_up11.size()}')
    up_cat = torch.cat((_up10, _up11), 1)
    print(f'up_cat size: {up_cat.size()}')
    Up11 = Up_conv(in_channels=512, out_channels=256)
    up1 = Up11(up_cat)
    # print(f'up1 size: {up1.size()}')

