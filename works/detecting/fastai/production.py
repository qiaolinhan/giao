import torch
from fastai import *
from fastai.vision import *
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

# independent variable: The thing we are using to make
# predictions from (The images)
# dependent variable: Our target (The categories)


# Create a datablock 
# TODO: What is the meaning of 'seed'?
imgs = DataBlock(
        # Provide a tuple where we specify what types we
        # want for the independent variables.
        blocks = (Imageblock, Categoryblock),
        get_items = get_iamge_files,
        splitter = RandomSplitter(valid_pct = 0.2, seed =
                                  42),
        get_y = parent_label,
        # The images need to be of the same size
        item_tfms = Resize(128)
        )

dls = imgs.dataloader(path)

#########
# Different ways to resize
# imgs = imgs.new(item_tfms = Resize(128,
# ResizeMethod.Squish))
# item_tfm = RandomResizeCrop(128, min_scale = 0.3)
# min_scale: determines how much of the image to select at minimum each
# time

# TODO: Resize into H20T image size
# imgs = imgs.new(item_tfms = Resize((*)), 
#                 batch_tfms = aug_transforms(mult = 2))


