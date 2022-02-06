import torch
import torch.nn as nn
import numpy as np

class Attention_block(nn.Module):
    def __init__(self, layerp_channels, layera_channels):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(layerp_channels, layerp_channels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(layerp_channels),
        )

        self.W_x = nn.Sequential(
            nn.ConvTranspose2d(layera_channels, layerp_channels, kernel_size= 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(layerp_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(layerp_channels, layera_channels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(layera_channels),
            nn.Sigmoid(),
        )

        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_in = self.relu(g1 + x1)
        psi = self.psi(psi_in)
        y = torch.mul(x, psi)
        return y

if __name__ == '__main__':

    # batchsize = 1, channels = 128, inputsize = 255*255
    feature_g = torch.randn((1, 64, 255, 255))
    feature_x = torch.randn((1, 128, 255, 255))
    # model = Resnet34(img_channels=3, num_classes=3)
    gate = Attention_block(layerp_channels=64, layera_channels=128)
    print(gate.eval())
    feature_out = gate(feature_g, feature_x)
    print('input shape:', feature_g.shape)
    print('up-sampling part shape:', feature_out.shape)

