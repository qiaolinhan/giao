#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: tutorial_training_with_torch.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-07-29
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: This is a record of the tutorial of training with Pytorch 
#
# ------------------------------------------------------------------------------

# Data and Dataloader
'''
The `Dataset`  and `DataLoader` classes encapsulate the process of pulling your data from storage and exposing it to
your training loop in batches.

The `Dataset` is responsible for accessing and processing single instances of data.

The `Dataloader` pulls instances of data from the `Dataset`, collect them in batches and return them for consumption by
your training loop. The `Dataloader` works with all kinds of datasets, regardless  of the type of data they contain.

We use `torchvison.transform.Normalize()` to zero-center and normaize the distribution of the image title content, and
download both training and validation data splits.
'''

import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
        
# create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train = True, transform = transform, download = True)
validation_set = torchvision.datasets.FashionMNIST('./data', train = False, transform = transform, download = True)

# create dataloader for the datasets, shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size = 4, shuffle = True, num_workers = 2)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = 4, shuffle = False, num_workers = 2)

# class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))


# visualize the data as sanity check
import matplotlib.pyplot as plt
import numpy as np

# Helper function for inline image display
def matplotlib_imshow(img, one_channel = False):
    if one_channel:
        img = img.mean(dim = 0)
    img = img / 2 + 0.5 # unormalize
    npimg = img.numpy() # fogot the '()' first time
    if one_channel:
        plt.imshow(npimg, cmap = "Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(training_loader)
images, labels = dataiter.next()

# create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel = True)
print(' '.join(classes[labels[j]] for j in range(4)))
