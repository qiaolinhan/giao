# to build a resnet34-based classifier to recognize the image types which consist of: Normal, Smoke, Flame
import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Identity
from torch.nn.modules.pooling import MaxPool2d

# input conv
class Conv0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv0, self).__init__()
        self.conv0 = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):   
        return self.conv0(x)
# conv layer in residual blocks
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        super(Block,self).__init__()
        self.expansion = 1
        self.identity_downsample = identity_downsample
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        self.bn_1 = nn.BatchNorm2d(out_channels),
        self.relu = nn.ReLU(inplace=True),
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        self.bn_2 = nn.BatchNorm2d(out_channels),
    def forward(self, x):
        identity = x
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

# consist resnet blocks together (18 or 34)
class Resnet(nn.Module):
    def __init__(self, block, layers, img_channels, num_classes):
        super(Resnet, self).__init__()
        self.inputconv = Conv0(img_channels, 64)
        self.layer1 = self._make_layer(block, layers[0], out_channels = 64, stride = 1)
        self.layer2 = self._make_layer(block, layers[1], out_channels = 128, stride = 1)
        self.layer3 = self._make_layer(block, layers[2], out_channels = 256, stride = 1)
        self.layer4 = self._make_layer(block, layers[3], out_channels = 512, stride = 1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.inputconv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = [] 

        if stride == 1 and self.in_channels == out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 
                          kernel_size=3, stride = stride, 
                          bias=False),
                nn.BatchNorm2d(out_channels*2),
                )
        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride = stride)
            )
        self.in_channels = out_channels*1

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

def resnet34(img_channel=3, num_classes = 3):
    return Resnet(Block, [3, 4, 6, 3], img_channel, num_classes)
    

##########################
# test whether model is okay
x = torch.randn((1, 3, 255, 255)) # batchsize = 1, channels = 3, inputsize = 255*255
model = resnet34()
preds = model(x)
print('preds shape:', preds.shape)
print('input shape:', x.shape)
