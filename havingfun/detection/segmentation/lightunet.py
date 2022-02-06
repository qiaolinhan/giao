import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '/home/qiao/dev/giao/havingfun/deving/common')

from inout import Inputlayer, Outlayer
from attentiongate import Attention_block
from doubleconv import Block, TBlock, Up_conv


class LightUnet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, scale_factor = 1):
        super(LightUnet, self).__init__()

        filters = np.array([64, 128, 256, 512])
        filters = filters // scale_factor

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # down-sampling
        self.Conv0 = Inputlayer(in_channels, filters[0])

        self.Conv1 = Block(filters[0], filters[0])
        self.Conv2 = Block(filters[0], filters[1])
        self.Conv3 = Block(filters[1], filters[2])

        # neck part
        self.Neck = Block(filters[2], filters[3])

        # up_sampling
        # self.Up3 = Up_conv(filters[3], filters[2])
        self.Att3 = Attention_block(filters[2], filters[3])
        self.up_conv3 = TBlock(filters[3], filters[2])

        # self.Up2 = Up_conv(filters[2], filters[1])
        self.Att2 = Attention_block(filters[1], filters[2])
        self.up_conv2 = TBlock(filters[2], filters[1])

        # self.Up1 = Up_conv(filters[1], filters[0])
        self.Att1 = Attention_block(filters[0], filters[1])
        self.up_conv1 = TBlock(filters[1], filters[0])

        self.outlayer = Outlayer(filters[0], out_channels)

    def forward(self, x):
        x0 = self.Conv0(x)

        x1 = self.Conv1(x0)
        g1 = x1

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        g2 = x2

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        g3 = x3

        x_neck = self.Neck(x3)

        gate3 = self.Att3(g3, x_neck)
        # out = gate3
        up3 = self.up_conv3(gate3)

        gate2 = self.Att2(g2, up3)
        up2 = self.up_conv2(gate2)

        gate1 = self.Att1(g1, up2)
        up1 = self.up_conv1(gate1)

        out = self.outlayer(up2)

        return out

if __name__ == '__main__':
    # batchsize = 4, channels = 3, inputsize = 400*400
    img = torch.randn((4, 3, 400, 400))
    # model = Resnet34(img_channels=3, num_classes=3)
    model = LightUnet(in_channels=3, out_channels = 1)
    print(model.eval())
    preds = model(img)
    print('input shape:', img.shape)
    print('preds shape:', preds.shape)






