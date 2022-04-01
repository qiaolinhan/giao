import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn.modules.batchnorm import BatchNorm2d

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class TBlock(nn.Module):
    def __init__(self, in_channels, out_channels):  
        super(TBlock, self).__init__()
        self.tblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.tblock(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(Block(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(TBlock(feature*2, feature))

        self.bottleneck = Block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # down sampling part
        for down in self.downs:
            x = down(x)
            # print(x.shape)
            skip_connections.append(x)
            x = self.pool(x)

        # bottleneck part
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # up sampling part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # print(x.shape)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((1, 3, 255, 255))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    # print(x.shape)
    # print(preds.shape)
    assert preds.shape[2:] == x.shape[2:]

if __name__ == "__main__":
    test()
