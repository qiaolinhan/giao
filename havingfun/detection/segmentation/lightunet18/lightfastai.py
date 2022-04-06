# 2022-04-05
# try traning the light unet model with fastai
import fastai
from fastai.vision.all import *
from lightCVloader import CVdataset, Atransform
from lightunet import LightUnet
from torch.utils.data import Dataset, Subset

Img_dir = ('datasets/S_kaggle_wildfire')
Mask_dir = ('datasets/S_kaggle_wildfire_label')
Data = CVdataset(img_dir=Img_dir, mask_dir = Mask_dir, transform = Atransform)
dataset_size = len(Data)
print(f"Total number of images: {dataset_size}")
valid_split = 0.2
valid_size = int(valid_split*dataset_size)
indices = torch.randperm(len(Data)).tolist()
train_data = Subset(Data, indices[:-valid_size])
val_data = Subset(Data, indices[-valid_size:])
print(f"Total training images: {len(train_data)}")
print(f"Total valid_images: {len(val_data)}")

dls = ImageDataLoaders(Data)

batch_size = 2
counter = 0
train_loader = DataLoader(train_data, batch_size = batch_size, 
                        num_workers = 0, 
                        pin_memory = 2,
                        shuffle = True)

learner = Learner(dls, LightUnet, loss_func = CrossEntropyLossFlat(), metrics=[accuracy])
learner.fit(5, lr = 1e-4, wd = 1e-2)