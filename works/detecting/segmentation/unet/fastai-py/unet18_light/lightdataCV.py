#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: lightdataCV.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-07-27
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: this .py file is for loading dataset into training process through cv2
#
# ------------------------------------------------------------------------------

# for loading dataset in system folder, 'os' is needed
import os
import torch
from torch.utils.data import Dataset, Subset
# for checking whether correctly loaded the dataset
import matplotlib.pyplot as plt
# the main package for loading dataset into models
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
# loading dataset through cv2
import cv2
# for matching 
import glob
# for observing the process status
from tqdm import tqdm

# resize original images (H20T 960*770) into 480*385
Image_hight = 385
Image_weight = 480
# pre-processing the dataset (resize, augmentation, normailze)
atransform = A.Compose([
    A.Resize(Image_hight, Image_weight),
    A.HorizontalFlip(p = 0.2),
    A.RandomBrightnessContrast(p = 0.2),
    A.Normalize(
        mean = [0.0, 0.0, 0.0],
        std = [1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

# class: CVdataset
# parameters:
        # ---
        # img_dir, string: the image dataset path
        # mask_dir, string: the label dataset path
        # transform, None: empty transform
class CVdataset(Dataset):
    def __init__(self,  data_dir= 'data_dir', transform = None):
        self.data_dir = data_dir
        img_dir = data_dir + '/imgs'
        mask_dir = data_dir + '/labels'
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # read images in the image folder
        self.imgs = os.listdir(img_dir)
        # read labels in the label folder
        self.masks = os.listdir(mask_dir)
  
    def __len__(self):     
        return len(self.imgs)
        
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        mask_path = os.path.join(self.mask_dir, 'label_' + self.imgs[index])
        img_np = cv2.imread(img_path)
        # print(img_np.shape)

        # convert to original image channels, because cv2.imread may change it
        img_np = img_np[..., ::-1]
        # mask_np = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask_np = mask_np[..., ::-1]
        
        # there are multiple classes for segmentation, then no need 
        # mask_np[mask_np > 0.0] = 1.0
        # img_tensor = torch.tensor(img_np)
        # mask_tensor = torch.tensor(img_np)
        if self.transform:           
            augmentations = self.transform(image = img_np, mask = mask_np)
            img_tensor = augmentations['image']
            mask_tensor = augmentations['mask'].float()
        else: 
            print('======> Warning, image transformation missed')
            img_tensor = torch.tensor(img_np)
            mask_tensor = torch.tensor(mask_np)
        return img_tensor, mask_tensor

# build a dataloader based on the CVdataset
def build_loader(data_dir, batch_size):
    # load the dataset
    data_loaded = CVdataset(data_dir, transform = atransform) 
    data_tensor = data_loaded
    # split into train dataset and validation dataset
    dataset_size = len(data_tensor)
    print(f"======> Total number of images: {dataset_size}")
    valid_split = 0.2
    valid_size = int(valid_split*dataset_size)
    indices = torch.randperm(len(data_tensor)).tolist()
    train_data = Subset(data_tensor, indices[:-valid_size])
    valid_data = Subset(data_tensor, indices[-valid_size:])
    print(f"======> Total training images: {len(train_data)}")
    print(f"======> Total valid_images: {len(valid_data)}")
    
    # split into train_data and val_data
    train_dataloader = DataLoader(train_data,
            batch_size = batch_size,
            shuffle = True,)
    valid_dataloader = DataLoader(train_data,
            batch_size = batch_size,
            shuffle = True,)
    return train_dataloader, valid_dataloader 
if __name__ == '__main__':
    counter = 0
    batch_size = 1
    # ----------------------------------
    # image folder and label folder
    data_dir = ('datasets/KaggleWildfire20220729')
    # Using cv2 get image items from these two folders 
    # data = CVdataset(data_dir, transform = atransform)
    # The data augmentaion is True, based on the func of atransform
    # ----------------------------------
    # build dataloader 
    train_dataloader, valid_dataloader = build_loader(data_dir, batch_size)
    
    # ----------------------------------
    # prepare data for training with DataLoaders
    # data_loader = DataLoader(data, batch_size = batch_size, 
                          # num_workers = 0, 
                          # pin_memory = 1,
                          # shuffle = True)
   # ---------------------------------- 
    # ----------------------------------
    # check whether created dataloader correctly
    for j, data in tqdm(enumerate(train_dataloader), total = len(train_dataloader) / batch_size):
        counter += 1
        img, mask = data
    print('======> img_tensor size:', img.size())
    print('======> mask_tensor size:', mask.size())
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img.squeeze(0).permute(1, 2, 0))
    ax[0].axis('off')

    ax[1].imshow(mask.permute(1, 2, 0))
    ax[1].axis('off')

    plt.show()
