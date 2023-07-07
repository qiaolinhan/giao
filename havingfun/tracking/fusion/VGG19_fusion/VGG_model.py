import torch
import torch.nn as nn

# enumeration of VGG11, 13, 19 structures in a dictionary;
# where the "M" means "Max Pooling".
VGG_types = {
        "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512,
                  512, "M"],
        "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M",
                  512, 512, "M"],
        "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512,
                  512, "M", 512, 512, 512, "M"],
        "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512,
                  512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }
        
# a 'Global veration' is needed to note the structure version of VGG Net.
vggtype = "VGG19"
# a 'Class function' called VGGNet
class VGGNet(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[vggtype])

        # fully connected layers 
        self.fcs = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(4096, num_classes),
                )
        def forward(self, x):
            x = self.conv_layers(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fcs(x)
            return x

        def create_conv_layers(self, architecture):
            layers = []
            in_channels = self.in_channels

            for x in architecture:
                if type(x) == int:
                    out_channels = x
                    layers += [
                            nn.Conv2d(
                                in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = (3, 3),
                                stride = 1,
                                padding = 1,
                                ),
                            nn.BatchNorm2d(x),
                            nn.ReLU(),
                            ]
                    in_channels = x
                elif x == "M":
                    layers += [nn.MaxPool2d(kernel_size = (2, 2), stride =
                                            (2, 2))]
                    return nn.Sequential(*layers)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGNet(in_channels = 3, num_classes = 500).to(device)
    x = torch.randn(1, 3, 244, 244).to(device)
    print(model(x).shape)


