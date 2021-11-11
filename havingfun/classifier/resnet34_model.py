import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU


# input conv
class Conv0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv0, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
    def forward(self, x):   
        return self.conv0(x)
# conv layer in residual blocks
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.res = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, 1,1,0, bias = False),
             nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.res(x)
        print('residual shape:', residual.shape)
        x = self.block(x)
        print('blocked shape:', x.shape)
        x += residual
        x = self.relu(x)
        return x

# # test: to match the blocked shapes
# x_block = torch.randn((1, 3, 255, 255))
# block = Block(in_channels = 3, out_channels = 64)
# print(block.eval())
# preds = block(x_block)
# print('preds shape:', preds.shape)
# print('input shape:', x_block.shape)


# consist resnet blocks together (18 or 34)
class Resnet34(nn.Module):
    def __init__(self, img_channels, num_classes):
        super(Resnet34, self).__init__()
        self.inputconv = Conv0(img_channels, out_channels = 64)

        structure = [3, 4, 6, 3]
        self.layer1 = nn.ModuleList(Block(in_channels=64, out_channels=64) for i in range(structure[0]))
        self.layer2 = nn.ModuleList(Block(in_channels=64, out_channels=128) for i in range(structure[1]))
        self.layer3 = nn.ModuleList(Block(in_channels=128, out_channels=256) for i in range(structure[2]))
        self.layer4 = nn.ModuleList(Block(in_channels=256, out_channels=512) for i in range(structure[3]))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        input = self.inputconv(x)

        residual1 = self.layer1(input)
        residual2 = self.layer2(residual1)
        residual3 = self.layer3(residual2)
        residual4 = self.layer4(residual3)

        res_out = self.avgpool(residual4)
        ap_out = res_out.reshape(res_out.shape[0], -1)
        output = self.fc(ap_out)
        return output

# ##########################
# test whether model is okay
img = torch.randn((1, 3, 255, 255)) # batchsize = 1, channels = 3, inputsize = 255*255
model = Resnet34(img_channels=3, num_classes=3)
print(model.eval())
preds = model(img)
print('preds shape:', preds.shape)
print('input shape:', img.shape)
