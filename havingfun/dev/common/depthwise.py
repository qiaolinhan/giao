# 1 kernel works for one channel
# deep wise 3x3: n_kernels = in_channels, then for every filter nn.Conv2d(1, out_channels, 3, 3, 1)
# num_parameter for each filter: 3x3x1xout_channels
# point wise 1x1: num_parameter for each filter: 1x1xin_channelsxout_channels

import torch
import torch.nn as nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forwarad(self, x):
        x = self.depthwise(x)
        x_out = self.pointwise(x)
        return x_out