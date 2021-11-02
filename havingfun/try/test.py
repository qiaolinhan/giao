import torch.nn as nn
import torch

class trytry(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(trytry, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
    def forward(self, x):
        x = self.conv3(x)
        print("process,size", x.shape)
        return x

def test():
    x = torch.randn(1, 64, 255, 255)
    model = trytry(in_channels=64, out_channels=5)
    y = model(x)
    print(y.shape)

test()