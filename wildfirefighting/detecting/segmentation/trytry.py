import torch
import torch.nn as nn

class DDepthwise(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DDepthwise, self).__init__()
        self.ddepthwise = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.pool = nn.MaxPool2d(2,2, 1)
    

    def forward(self, x):
        y = self.ddepthwise(x)
        y = self.pool(y)
        return y


if __name__ == "__main__":
    in_tensor = torch.randn((4, 64, 395, 395))
    cal = DDepthwise(in_channels=64, out_channels=128)
    out_tensor = cal(in_tensor)
    print(out_tensor.size())


    x_r = torch.tensor([1, 2, 3])
    print(x_r.shape)
    x_c = torch.tensor([[1], [2], [3]])
    print(x_c.shape)