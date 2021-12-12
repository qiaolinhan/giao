import torch.nn as nn

class BeginConv(nn.Module):
    def __init__(self, input_channels, begin_channels):
        super(BeginConv, self).__init__()
        self.beginconv = nn.Sequential(
            nn.Conv2d(input_channels, begin_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(begin_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.beginconv(x)

class DoubleConv(nn.Module):
    def __init__(self, begin_channels, end_channels, identity_downsample=None):
        super(DoubleConv, self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(begin_channels, end_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(end_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(end_channels, end_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(end_channels),
        )
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.doubleconv(x)
        if identity is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = nn.ReLU(x)
        return self.doubleconv(x)



