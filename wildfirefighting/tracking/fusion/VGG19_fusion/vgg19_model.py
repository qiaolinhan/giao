import torch.nn as nn
import torch.optim as optim 
from torchvision import transforms, models 

import numpy as np 
import matplotlib.pyplot as plt  

vgg = models.vgg19(pretrained = True).features 
for param in vgg.parameters():
    param.requires_grad = False

if __name__ == "__main__":
    print(vgg)
