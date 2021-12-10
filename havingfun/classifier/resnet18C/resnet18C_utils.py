#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: resnet34_utils.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2021-12-04
#
#   @Email: 
#
#   @Description: 
#
#------------------------------------------------------------------------------


import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

# save the model
def save_model(epochs, model, optimizer, criterion):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, 'Param_resnet18C_process_1e5.pth')

def save_entire_model(epochs, model, optimizer, criterion):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, 'Param_resnet18C_1e5_e18.pth')

# save the figures of loss and accuracy
def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    # accuracy
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Acc_resnet18C_1e5_e18.png')

    # loss
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Loss_resnet18C_1e5_e18.png')

