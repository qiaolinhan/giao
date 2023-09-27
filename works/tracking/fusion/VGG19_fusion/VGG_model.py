import torch
import torch.nn as nn
from .utils import load_state_dict_from_url

# The ports opened
__all__ = ["VGG", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16",
           "vgg16_bn", "vgg19", "vgg19_bn",
           ]

# int numbers represents the channel amounts
# M means maxpooling
# A-LRN represents local normalization reaction, there are 1x1 conv layers
# in C Net, which is spectial and not fitted by Nvidia, so did not put
# here.
cfgs = {
        "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512,
                  512, "M"],
        "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M",
                  512, 512, "M"],
        "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512,
                  512, "M", 512, 512, 512, "M"],
        "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512,
                  512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }
        
# the addresses where to download the pre-trained weightes
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }

# a 'Class function' called VGGNet
class VGGNet(nn.Module):
    def __init__(self, features, num_classes = 1000, init_weight = True):
        super(VGGNet, self).__init__()

        self.features = features
        # The adaptive average pooling, feature --> pooling 7*7
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))


        # fully connected layers as classifier
        self.fcs = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096), # 512 * 7 * 7 --> 4096
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(4096, 4096), # 4096 --> 4096
                nn.ReLU(),
                nn.Dropout(p = 0.5),
                nn.Linear(4096, num_classes), # 4096 --> 1000
                )
        
        self.conv_layers = self.create_conv_layers(VGG_types[vggtype])

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # feature extration
        x = self.features(x)
        # adaptive average pooling
        x = self.avgpool(x)
        # flatten feature map into vector
        x = torch.flatten(x, 1)
        # output from classifier (fully connected layers)
        x = self.fcs(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                        m.weight, model = 'fan_out', nonlinearity = 'relu')
                # initialize bias as 0 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # initialize the batch normalize weights as 1 
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # initialize the weights of fully connected layers
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)

def create_conv_layers(self, architecture):
    # return to the model list based on the configureation list
    layers = []
    in_channels = 3

    for x in architecture:
        # add the conv layers 
        if type(x) == int:
            # 3*3 conv layer
            conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace =
                                                              True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]

            in_channels = v
        elif x == "M":
            layers += [nn.MaxPool2d(kernel_size = (2, 2), stride =
                                    (2, 2))] 
            # or in short: kernel_size = 2, stride = 2
     return nn.Sequential(*layers)

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    # common network builder to estuplish a model and load pretrained
    # weights
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm = batch_norm), **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress =
                                              progress)
        model.load_state_dict(state_dict)
    return model

def vgg11(pretrained = False, progress = True, **kwargs):
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)

def vgg11_bn(pretrained = False, pregress = True, **kwargs):
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)

def vgg13(pretrained = False, progress = True, **kwargs):
    return _vgg("vgg11", "B", False, pretrained, progress, **kwargs)

def vgg13_bn(pretrained = False, pregress = True, **kwargs):
    return _vgg("vgg11_bn", "B", True, pretrained, progress, **kwargs)

def vgg16(pretrained = False, progress = True, **kwargs):
    return _vgg("vgg11", "D", False, pretrained, progress, **kwargs)

def vgg16_bn(pretrained = False, pregress = True, **kwargs):
    return _vgg("vgg11_bn", "D", True, pretrained, progress, **kwargs)

def vgg19(pretrained = False, progress = True, **kwargs):
    return _vgg("vgg11", "E", False, pretrained, progress, **kwargs)

def vgg19_bn(pretrained = False, pregress = True, **kwargs):
    return _vgg("vgg11_bn", "E", True, pretrained, progress, **kwargs)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGNet(in_channels = 3, num_classes = 500).to(device)
    x = torch.randn(1, 3, 244, 244).to(device)
    print(model(x).shape)


