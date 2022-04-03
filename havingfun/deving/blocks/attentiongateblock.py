# 2022-04-03
# Re-modifying
# gate signal: in upsampling 
# attention signal: in downsampling, one more down.

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from depthwiseblock import DDepthwise, UDepthwise

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
            nn.BatchNorm2d(gating_channels)
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
        if gl.size() != att.size():
             gl = sizechange(gl, att1)
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
    feature_att = torch.randn((2, 64, 400, 400))
    feature_gate = torch.randn((2, 128, 200, 200))
    # model = Resnet34(img_channels=3, num_classes=3)
    attgate = Attentiongate_block(att_channels=64, gating_channels=128)
    # print(attgate.eval())
    feature_out = attgate(feature_att, feature_gate)
    print(f'att size: {feature_att.size()}')
    print('up-sampling part shape:', feature_out.shape)

