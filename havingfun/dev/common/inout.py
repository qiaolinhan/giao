import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU


# input conv
class Conv0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv0, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        return self.conv0(x)

class Conv_f(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_f, self).__init__()
        # 1x1 Conv
        self.convf = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
    
    def forward(self, x):
        return self.convf(x)
