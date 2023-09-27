# 2022-04-12
# this block is for building light u-net
# for preplacing res-blocks in resnet-based u-net
# using depth-wise and point-wise convs to replace original convs
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResDblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResDblock, self).__init__()

        # original
        # self.doubleconv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace = True),
        # )

        # squeezed
        self.ddepthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups = int(in_channels)),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups = int(out_channels)),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.BatchNorm2d(out_channels)
            )

        self.maxpool = nn.MaxPool2d(2, 2, 0)

    def forward(self, x):
        x_d = self.ddepthwise(x)
        x_r = self.shortcut(x)
        y = x_d + x_r
        y = F.relu(y)
        return y

class UnSqueezeblock(nn.Module):
    def __init__(self, in_channels, mid_channels_1, mid_channels, out_channels):
        super(UnSqueezeblock, self).__init__()

        self.expand_1 = nn.Conv2d(in_channels, mid_channels_1, 1, 1, 0)
        self.expand_3 = nn.Conv2d(in_channels, mid_channels - mid_channels_1, 3, 1, 1)
        self.squeeze = nn.Conv2d(mid_channels, out_channels, 1, 1, 0)
        self.outconv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.expand_1(x)
        x_3 = self.expand_3(x)
        x_mid = torch.cat([x_1, x_3], dim = 1)
        x_out = self.squeeze(x_mid)
        x_out = self.relu(x_out)
        x_out = self.outconv(x_out)
        x_out = self.relu(x_out)
        return x_out
    
if __name__ == "__main__":
        # batchsize = 1, channels = 64, inputsize = 255*255
    feature = torch.randn((1, 64, 255, 255))

    model_down = ResDblock(in_channels = 64, out_channels = 128)
    print(model_down.eval())
    preds = model_down(feature)
    print('downsapling input shape:', feature.shape)
    print('downsampling preds shape:', preds.shape)
    
    model = UnSqueezeblock(in_channels = 64, mid_channels = 55, mid_channels_1 = 15, out_channels = 32)
    print(model.eval())
    uppreds = model(feature)
    print('input shape:', feature.shape)
    print('uppreds shape:', uppreds.shape)