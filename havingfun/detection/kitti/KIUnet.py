# 2022-03-30
# Unet sematic segmentation for kitti dataset
# trying light weight structure and attention connection based on resnet model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import torchvision.transforms.functional as TF
# using the blocks in deving
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'havingfun/deving/blocks')
import torchvision.transforms as T
from inoutblock import Inputlayer, Outlayer
from attentiongateblock import Attention_block
from depthwiseblock import DDepthwise, UDepthwise, Up_conv
from doubleconvblock import DBlock # for bottleneck

# using tools in deving
sys.path.insert(1, 'havingfun/deving/tools')
from resizetensor import sizechange

class LightUnet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, filters = [64, 128, 256, 512], scale_factor = 2):
        super(LightUnet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()   
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # down-samplingS
        for filter in filters:
                self.downs.append(DDepthwise(in_channels, filter))
                in_channels = filter

        # up-sampling
        for filter in reversed(filters):
                self.ups.append(UDepthwise(filter * 2, filter))
                self.ups.append(Up_conv(filter * 2, filter))

        self.bottelneck = DBlock(filters[-1], filters[-1] * 2)
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

        # self.Conv0 = Inputlayer(in_channels, filters[0])

        # self.down1 = nn.Sequential(
        #         DDepthwise(filters[0], filters[0]),
        #         DDepthwise(filters[0], filters[0]),
        #         nn.MaxPool2d(kernel_size = 2, stride = 2),
        #         )
        # self.down2 = nn.Sequential(
        #         DDepthwise(filters[0], filters[1]),
        #         DDepthwise(filters[1], filters[1]),
        #         nn.MaxPool2d(kernel_size = 2, stride = 2),
        #         )
        # self.down3 = nn.Sequential(
        #         DDepthwise(filters[1], filters[2]),
        #         DDepthwise(filters[2], filters[2]),
        #         nn.MaxPool2d(kernel_size = 2, stride = 2),
        #         )
        # self.neck = nn.Sequential(
        #         DDepthwise(filters[2], filters[3]),
        #         DDepthwise(filters[3], filters[3])
        #         )
        # # self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # # up_sampling
        # self.Up3 = UDepthwise(filters[3], filters[2])
        # self.Att3 = Attention_block(filters[2], filters[3])
        # self.up_conv3 = Up_conv(filters[3], filters[2])

        # self.Up2 = UDepthwise(filters[2], filters[1])
        # self.Att2 = Attention_block(filters[1], filters[2])
        # self.up_conv2 = Up_conv(filters[2], filters[1])

        # self.Up1 = UDepthwise(filters[1], filters[0])
        # self.Att1 = Attention_block(filters[0], filters[1])
        # self.up_conv1 = Up_conv(filters[1], filters[0])

        # # self.up_conv0 = Up_conv(filters[0], out_channels)
        # self.outlayer = Outlayer(filters[0], out_channels)

    def forward(self, x):
        gating = []
        for down in self.downs:
            x = down(x)
            print('gating size', x.size())
            gating.append(x)
            x = self.pooling(x)

        x = self.bottelneck(x)
        attention = gating[::-1]

        print('up steps', len(self.ups))
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            attention = attention[idx // 2]
            print('attention size', attention.size())
            # resize the updepthwise output
            # if x.shape != gating.shape:
            #     x = TF.resize(x, size=gating.shape[2:])
        
            attention_gate = Attention_block(gating, attention)
            x = self.ups[idx + 1](attention_gate)

        return self.final_conv(x)

        # x0 = self.Conv0(x)
        # # print('input-c64 size :', x0.size())
        # x1 = self.down1(x0)
        # # print('c64-c64 size:', x1.size())
        # # x2 = self.Maxpool(x1)
        # x2 = self.down2(x1)
        # # print('c64-c128 size:', x2.size())
        # # x3 = self.Maxpool(x2)
        # x3 = self.down3(x2)
        # # x_neck = self.Maxpool(x3)
        # # print('c128-c256 size:', x3.size())
        # x_neck = self.neck(x3)
        # # print('c256-c512 neck size:', x_neck.size())
        # gate3 = self.Att3(x3, x_neck)
        # # print(gate3.size())     
        # _up3 = self.Up3(x_neck)
        # # print('_up3 size before resieze', _up3.size())
        # # print('gate3 size', gate3.size())
        # _up3 = sizechange(_up3, gate3)
        # # print('_up3 size after resize', _up3.size())
        
        # up3 = torch.cat((gate3, _up3), 1)
        # up3 = self.up_conv3(up3)

        # gate2 = self.Att2(x2, up3)
        # _up2 = self.Up2(up3)
        # _up2 = sizechange(_up2, gate2)
        # up2 = torch.cat((gate2, _up2), 1)
        # up2 = self.up_conv2(up2)

        # gate1 = self.Att1(x1, up2)
        # _up1 = self.Up1(up2)
        # _up1 = sizechange(_up1, gate1)
        # up1 = torch.cat((gate1, _up1), 1)
        # up1 = self.up_conv1(up1)
        # # up0 = self.up_conv0(up1)
        # out = self.outlayer(up1)
        # out = sizechange(out, x).squeeze(1)
        # return out


if __name__ == '__main__':
    # batchsize = 4, channels = 3, inputsize = 400*400
    img = torch.randn((4, 3, 400, 400))
    mask = torch.randn((4, 400, 400))
    # model = Resnet34(img_channels=3, num_classes=3)
    model = LightUnet(in_channels=3, out_channels = 1)
    print(model.eval())
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'=====>The depthwise seperable convolution uses {params} parameters.')
    preds = model(img)
#     if preds.shape != mask.shape:
#         # preds = TF.resize(preds, size=mask.shape[2:])
#         preds = sizechange(preds, mask)
    print('input shape:', img.size())
    print('preds shape:', preds.size())






