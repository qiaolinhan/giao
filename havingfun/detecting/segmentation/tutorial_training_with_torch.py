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
    npimg = img.numpy() # don't forget the '()'
    if one_channel:
        plt.imshow(npimg, cmap = "Greys")
        # plt.show() # added this for avoiding the warning
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show() # added this for avoiding the warning

dataiter = iter(training_loader)
images, labels = dataiter.next()

# create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel = True)
print('\n ======> This is a anity check '.join(classes[labels[j]] for j in range(4)))

# ----------------------------------
'''
The Model
The Model used in this example is avariant of LeNet5
'''
import torch.nn as nn
import torch.nn.functional as F

# PyTorch models inherint from torch.nn.model
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__() # don't forget super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = GarmentClassifier()

# ----------------------------------
'''
Loss Function
Cross-Entropy loss is used. For demonstration purposes, we will create batches of dummy outputs and label values, run
them through the loss function, and examine the result.
'''
loss_fn = torch.nn.CrossEntropyLoss()
# NB: los functions expect data in batches, so we are creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
dummy_outputs = torch.rand(4, 10)
# Represents the correct class amon the 10 being tested
dummy_labels = torch.tensor([1, 5, 3, 7])

print('[INFO] dummy_outputs:',dummy_outputs)
print('[INFO] dummy_labels:', dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print('Total loss for this batch: {}'.format(loss.item()))

# ----------------------------------
'''
Optimizer
`Simple stochastic gradient descent with momentum` is used
It can be istructive to try some variations on this optimization scheme:
    * Learning rate determines the size of the steps the optimizer takes, it may impact the accuracy and convergence
    time.
    * Momentum nudges the optimizer in the direction of strongest gradient over multiple steps.
    * Different optimization algorithms, such as average SGD, Adgrad, or Adam have different performance.
'''
# OPtimizer specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9) # don't forget brankets `model.parameters()`

