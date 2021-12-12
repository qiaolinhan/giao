import torch
import torch.nn as nn
from commonconv import BeginConv, DoubleConv
import torchvision.transforms.functional as TF

# input_channels = 3
# begin_channels = 64

class unet_resnet18(nn.Module):
    def __init__(self, layers,
                blocks, 
                input_channels = 3, 
                begin_channels = 64, 
                end_channels = [64, 128, 256, 512], 
                out_channels = 1,
                 ):
        super(unet_resnet18, self).__init__()

        # inputconv
        self.conv0 = BeginConv(input_channels, begin_channels)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # down parts
        # self.downs = nn.ModuleList()
        # for feature in end_channels:
        #     self.downs.append(DoubleConv(begin_channels, feature))
        #     self.downs.append(DoubleConv(feature, feature))
        #     begin_channels = feature
        self.down1 = self._make_layer(blocks, layers[0], end_channels[0], stride = 1)
        self.down2 = self._make_layer(blocks, layers[1], end_channels[2], stride = 2)
        self.down3 = self._make_layer(blocks, layers[2], end_channels[2], stride = 1)
        self.down4 = self._make_layer(blocks, layers[3], end_channels[3], stride = 1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # bottleneck
        self.bottleneck = DoubleConv(end_channels[-1], end_channels[-1]*2)

        # up parts
        self.ups = nn.ModuleList()
        for feature in reversed(end_channels):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=3))
            self.ups.append(nn.Conv2d(feature, feature, kernel_size=3))

        # out
        self.outconv = nn.Conv2d(end_channels[0], out_channels, kernel_size=1, stride=1, padding=0)

    # def _make_layers():
    def _make_layer(self, blocks, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        # defined the layers empty
        layers = []
        
        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size = 1, stride = stride),
                                            nn.BatchNorm2d(out_channels*4))
        
        layers.append(blocks(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels

        for i in range(num_residual_blocks-1):
            layers.append(blocks(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        skip_connections = []
        # begin
        x = self.conv0(x)
        skip_connections.append(x)
        x = self.pool0(x)
        # down
        x = self.down1(x)
        skip_connections.append(x)
        x = self.pool(x)

        x = self.down2(x)
        skip_connections.append(x)
        x = self.pool(x)

        x = self.down3(x)
        skip_connections.append(x)
        x = self.pool(x)

        x = self.down4(x)
        skip_connections.append(x)
        x = self.pool(x)
        
        # bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections = skip_connections[::-1]

        # ups
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                # add padding, resizing, ...
                x = TF.resize(x, size = skip_connection.shape[2:]) 

            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)

        x = self.outconv(x)
        return x

def unetlight(input_channels = 3, out_channels = 1):
    # resnet18 --> [2, 2, 2, 2]; resnet34 --> [3, 4, 6, 8]
    return unet_resnet18(DoubleConv, [2, 2, 2, 2], input_channels, out_channels)

def test():
    x = torch.randn(1, 3, 255, 255)
    model = unetlight()
    y = model(x)
    print(y.shape)

test()







