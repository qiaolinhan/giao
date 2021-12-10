import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, 'giao/havingfun/modules')

from inout import Conv0, Conv_f
from attentiongate import Attention_block
from doubleconv import Block, TBlock

class up_conv(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channel_in, channel_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.up(x)

class attunet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, scale_factor = 1):
        super(attunet, self).__init__()
        filters = np.array([64, 128, 256, 512])
        filters = filters // scale_factor

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # down-sampling
        self.Conv0 = Conv0(in_channels, filters[0])

        self.Conv1 = Block(filters[0], filters[1])
        self.Conv2 = Block(filters[1], filters[2])
        self.Conv3 = Block(filters[2], filters[3])

        # up_sampling
        self.Up3 = up_conv(filters[3], filters[2])
        self.Att3 = Attention_block(F_g = filters[2], F_l = filters[2], F_int = filters[1])
        self.up_conv3 = Block(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.Att2 = Attention_block(F_g = filters[1], F_l = filters[1], F_int = filters[0])
        self.up_conv2 = Block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Att1 = Attention_block(F_g = filters[0], F_l = filters[0], F_int = filters[0]//2)
        self.up_conv1 = Block(filters[1], filters[0])

        self.Conv_f = Conv_f(filters[0], out_channels)

    def forward(self, x):
        x0 = self.Conv0(x)

        x1 = self.Conv1(x0)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        up3 = self.Up3(x3)
        gate2 = self.Att3(g = up3, x = x2)
        up3 = torch.cat((up3, gate2), dim = 1)
        up3 = self.up_conv3(up3)

        up2 = self.Up2(up3)
        gate1 = self.Att2(g = up2, x = x1)
        up2 = torch.cat((up2, gate1), dim = 1)
        up2 = self.up_conv2(up2)

        up1 = self.Up1(up2)
        gate0 = self.Att1(g = up1, x = x0)
        up1 = torch.cat((up1, gate0), dim = 1)
        up0 = self.up_conv1(up1)

        out = self.Conv_f(up1)

        return out

if __name__ == '__main__':

    # batchsize = 1, channels = 3, inputsize = 255*255
    img = torch.randn((4, 3, 254, 254))
    # model = Resnet34(img_channels=3, num_classes=3)
    model = attunet(in_channels=3, out_channels = 1)
    print(model.eval())
    preds = model(img)
    print('input shape:', img.shape)
    print('preds shape:', preds.shape)






