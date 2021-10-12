import torch
import torch.nn as nn

# resnet blocks
class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if identity is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu2(x)
        return x

# common resnet structures
class resnet(nn.Module):
    def __init__(self, block, layers, img_channels, feature_channels):
        super(resnet, self).__init__()
        self.in_channels = 64
        # begin conv
        self.conv0 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU()
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels = 64, stride = 1)
        self.layer2 = self._make_layer(block, layers[1], out_channels = 128, stride = 2)
        self.layer3 = self._make_layer(block, layers[2], out_channels = 256, stride = 1)
        self.layer4 = self._make_layer(block, layers[3], out_channels = 512, stride = 1)
        self.fc = nn.Linear(512, 4, feature_channels)
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                                                nn.BatchNorm2d(out_channels))

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels

        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # begin
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        # resnet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def resnet18down(img_channels=3, feature_channels=512):
    return resnet(block, [2, 2, 2, 2], img_channels, feature_channels)

def test():
    net = resnet18down()
    x = torch.randn(2, 3, 255, 255)
    y = net(x)
    print(y.shape)

test()




