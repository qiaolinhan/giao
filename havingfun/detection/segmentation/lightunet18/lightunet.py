# 2022-03-14
# For U-net with structure of Resnet18, 31,036,481 params are needed.
# For this model, there are 2,764,403 parameters.
# this is a light U-net model based on the structure of resnet18 based Unet. The main purpose is tp decrease the model size
# so that it could be deplyed on the on-board computer of M300 for smoke and fire segmentation
import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import sys
import torchvision.transforms.functional as TF
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'havingfun/deving/blocks')
import torchvision.transforms as T
from inoutblock import Inputlayer, Outlayer
from attentiongateblock import Attentiongate_block
from depthwiseblock import DDepthwise, UDepthwise, Up_conv

sys.path.insert(1, 'havingfun/deving/tools')
from resizetensor import sizechange

class LightUnet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, scale_factor = 1):
        super(LightUnet, self).__init__()
        num = np.array([2, 2, 2, 2])
        filters = np.array([64, 128, 256, 512])
        filters = filters // scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        
        # down-sampling
        self.Conv0 = Inputlayer(in_channels, filters[0])

        self.down10 = DDepthwise(filters[0], filters[0])
        self.down11 = nn.Sequential(
                DDepthwise(filters[0], filters[0]),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                )

        self.down20 = DDepthwise(filters[0], filters[1])
        self.down21 = nn.Sequential(
                DDepthwise(filters[1], filters[1]),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                )

        self.down30 = DDepthwise(filters[1], filters[2])
        self.down31 = nn.Sequential(
                DDepthwise(filters[2], filters[2]),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                )

        self.neck0 = DDepthwise(filters[2], filters[3])
        self.neck1 = nn.Sequential(
                DDepthwise(filters[3], filters[3]),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                )
        # self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # up_sampling
        # self.Up3 = UDepthwise(filters[3], filters[2])
        self.Att3 = Attentiongate_block(filters[3], filters[3])
        self.up_conv3 = Up_conv(filters[3], filters[2])

        # self.Up2 = UDepthwise(filters[2], filters[1])
        self.Att2 = Attentiongate_block(filters[2], filters[2])
        self.up_conv2 = Up_conv(filters[2], filters[1])

        # self.Up1 = UDepthwise(filters[1], filters[0])
        self.Att1 = Attentiongate_block(filters[1], filters[1])
        self.up_conv1 = Up_conv(filters[1], filters[0])

        # self.up_conv0 = Up_conv(filters[0], out_channels)
        self.outlayer = Outlayer(filters[0], out_channels)

    def forward(self, x):
        x0 = self.Conv0(x)
        # print('input-c64 size :', x0.size())
        x1 = self.down1(x0)
        att1 = x1
        # print('c64-c64 size:', x1.size())
        # x2 = self.Maxpool(x1)
        x2 = self.down2(x1)
        att2 = x2
        # print('c64-c128 size:', x2.size())
        # x3 = self.Maxpool(x2)
        x3 = self.down3(x2)
        att3 = x3
        # x_neck = self.Maxpool(x3)
        # print('c128-c256 size:', x3.size())
        x_neck = self.neck(x3)
        # print('c256-c512 neck size:', x_neck.size())
        gate3 = x_neck
        # print(gate3.size())     
        _up30 = self.Att3(att3, gate3)
        # print(f'_up30 size: {_up30.size()}')
        _up31 = self.Up3(x_neck)
        _up31 = sizechange(_up31, _up30)
        _up3 = torch.cat((_up30, _up31), 1)
        # print('_up3 size before resieze', _up3.size())
        # print('gate3 size', gate3.size())
        up3 = self.up_conv3(_up3)
        # print('up3 size', up3.size())
        gate2 = up3
        _up20 = self.Att2(att2, gate2)
        _up21 = self.Up2(up3)
        _up21 = sizechange(_up21, _up20)
        _up2 = torch.cat((_up20, _up21), 1)
        up2 = self.up_conv2(_up2)

        gate1 = up2
        _up10 = self.Att1(att1, gate1)
        # print(f'_up10 size: {_up10.size()}')
        _up11 = self.Up1(up2)
        # print(f'_up11 size: {_up11.size()}')
        _up11 = sizechange(_up11, _up10)
        _up1 = torch.cat((_up10, _up11), 1)
        up1 = self.up_conv1(_up1)
        # print(f'up1 size: {up1.size()}')

        out = self.outlayer(up1)
        out = sizechange(out, x).squeeze(1)
        return out

if __name__ == '__main__':
    # batchsize = 4, channels = 3, inputsize = 400*400
    img = torch.randn((4, 3, 400, 400))
    mask = torch.randn((4, 1, 400, 400))
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
    print(31036481//2764403)






