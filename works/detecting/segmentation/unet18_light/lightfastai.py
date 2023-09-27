# 2022-04-07
# try to train the custom model using fastai

# The fastai Learner class combines 
# a model module with a data loader on a pytorch Dataset, 
# with the data part wrapper into the TabularDataBunch class. 
# So we need to prepare the DataBunch (step 1)
# then wrap our module and the DataBunch into a Learner object (step 2)

# NOTE: Before any work can be done a dataset needs to be converted into a DataBunch object.
# and in the case of the computer vision data - specifically into an ImageDataBunch subclass.

import fastai
import fastai.tabular
from fastai.vision.all import *
from torch.utils.data import Dataset, Subset
import timm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm

from lightunet import LightUnet
from lightdataCV import CVdataset, Atransform
# preparing the dataset for ImageDataBunch
path = Path('/home/qiao/dev/giao/datasets/')

path_img = path/'S_kaggle_wildfire/'
path_label = path/'S_kaggle_wildfire_label/'
codes = ['Smoke', 'Flame', 'Cloud', 'Background']
data = CVdataset(img_dir = path_img,mask_dir = path_label, transform = Atransform)
# split into train dataset and validation dataset
dataset_size = len(data)
print(f"Total number of images: {dataset_size}")
valid_split = 0.2
valid_size = int(valid_split*dataset_size)
indices = torch.randperm(len(data)).tolist()
train_data = Subset(data, indices[:-valid_size])
val_data = Subset(data, indices[-valid_size:])
print(f"Total training images: {len(train_data)}")
print(f"Total valid_images: {len(val_data)}")

dls = ImageDataLoaders(train_data, val_data)
model = LightUnet(in_channels=3, out_channels=1)
learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss())

learn.lr_find(start_lr = 1e-7, end_lr = 1e-2, num_it = 10)





 