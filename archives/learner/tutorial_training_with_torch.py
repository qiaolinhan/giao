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

# ---------------------------------
'''
The Training Loop
Below, we have function that performs one training  epoch, it enumerate data from the DataLoader, and on epoch pass of
the loop does the following:
    * Gets a batch of training data from DataLoader
    * Zeros the optimizer's gradients
    * Performs an inference-that is, gets predictions from the model for an input batch
    * Calculates the loss for that set of predictions vs. the label on the dataset
    * Calculates the backward gradients over the learning weights
    * Tells the optimizer to perform one learning step - that is, adjust the model's learning weights based on the
    observed gradients for this batch, according to the optimization algorithm we chose
    * It reports on the loss for every 1000 batches.
    * Finally, it reports the average per-batch loss for the last 1000 batches, for comparison with a validation run
'''
# Adding such package for solving the problem of 'too many open files'.
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# This problem is caused by the number of batches over 10000 'the change of i'

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    
    # Here, we use enumerate(train_loader) instead of iter(train_loader)
    # so that we can track the batch index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('======> batch {} loss {}'.format(int((i + 1)/1000), last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# -----------------------------
'''
Per-Epoch Activity
There are a couple of things we will want to do once per epoch:
    * Perform validation by checking our relative loss on a set of data that was not used for training, and report this
    * Save a copy of the model
Here, we will do our reporting in Tensorboard. This will require going to the comman line to start TensorBoard, and
opening it in another browser tab.
'''
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fasion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000

for epoch in range(EPOCHS):
    print('======> EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We do not need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i,vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('======> Loss train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
            {'Training' : avg_loss, 'Validation' : avg_vloss},
            epoch_number + 1)
    writer.flush()

    # track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

# -----------------------------
# To load a saved version of the model
saved_model = GarmentClassifier()
# saved_model.load_state_dict(torch.load(PATH))
