from pickle import FALSE, TRUE
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from lightunet import LightUnet
from lightutils import (
    save_model,
    load_model,
    check_accuracy,
    save_predictions_as_imgs,
    save_plots,
)
import argparse

# from albumentations.pytorch import ToTensorV2
import numpy as np
from lightdata import JinglingDataset, transform
from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import matplotlib.pyplot as plt
# Hyperparameters: batch size, number of workers, image size, train_val_split, model
Batch_size = 1
Num_workers = 2
Image_hight = 400
Image_weight = 400
Pin_memory = True
Valid_split = 0.2
Modeluse = LightUnet
# flexible hyper params: epochs, dataset, learning rate, load_model
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e',
    '--epochs',
    type = int,
    default = 18,
    help = 'Numbers of epochs to train the network'
)
parser.add_argument(
    '-t',
    '--troot',
    type = str,
    default = '/home/qiao/dev/giao/dataset/imgs/jinglingseg/images',
    help = 'Input the image dataset path'
)
parser.add_argument(
    '-m',
    '--mroot',
    type = str,
    default = '/home/qiao/dev/giao/dataset/imgs/jinglingseg/masks',
    help = 'Input the mask dataset path'
)
parser.add_argument(
    '-l',
    '--lr',
    type = np.float32,
    default = 1e-5,
    help = 'Learning rate for training'
)
parser.add_argument(
    '-load',
    '--load',
    default = None,
    help = 'loading the trained model for prediction'
)

args = vars(parser.parse_args())
Num_epochs = args['epochs']
Img_dir = args['troot']
Mask_dir = args['mroot']
Learning_rate = args['lr']
Load_model = args['load']

# the device used fir training
Device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the model
model = Modeluse(in_channels=3, out_channels=1)
model.to(device = Device)
# print the parameter numbers of the model
total_params = sum(p.numel() for p in model.parameters())
print(model.eval())
print('#############################################################')
print(f'There are {total_params:,} total parameters in the model.\n')
# optimizer used for training
optimizer = optim.Adam(model.parameters(), lr = Learning_rate)
# loss function for training
loss_fn = nn.CrossEntropyLoss()
# load dataset
data = JinglingDataset(img_dir = Img_dir,mask_dir = Mask_dir, transform = transform)
dataset_size = len(data)
print(f"Total number of images: {dataset_size}")
valid_split = 0.2
valid_size = int(valid_split*dataset_size)
indices = torch.randperm(len(data)).tolist()
train_data = Subset(data, indices[:-valid_size])
val_data = Subset(data, indices[-valid_size:])
print(f"Total training images: {len(train_data)}")
print(f"Total valid_images: {len(val_data)}")

print(f'\nComputation device: {Device}\n')

# for i in range(len(train_data)):
#     train_img, train_mask = train_data[i][0], train_data[i][1]
# for i in range(len(val_data)):
#     val_img, val_mask = val_data[i][0], val_data[i][1]

train_loader = DataLoader(train_data, batch_size = Batch_size, 
                          num_workers = Num_workers, 
                          pin_memory = Pin_memory,
                          shuffle = True)
val_loader = DataLoader(val_data, batch_size = Batch_size, 
                        num_workers = Num_workers, 
                        pin_memory = Pin_memory,
                        shuffle = True)

# # train_imgs, train_masks = next(iter(train_loader))
# # print(f"Images batch shape: {train_imgs.size()}")
# # print(f"Masks batch shape: {train_masks.size()}")
# # img = train_imgs[0].squeeze()
# # mask = train_masks[0]
# # plt.imshow(mask, cmap="gray")
# # plt.show()
# # print(f"mask: {mask}")
# codes = ['Target', 'Void']
# name2id = {v:k for k, v in enumerate(codes)}
# void_code = name2id['Void']

# def seg_acc(input, target):
#     target = target.squeeze(1)
#     mask = target != void_code
#     return (input.argmax(dim = 1)[mask]==target[mask]).float().mean()

# # def fit(dataloader, model, optimizer, loss_fn, scaler):
# print('====> Fitting process')

# train_running_loss = 0.0
# train_running_acc = 0.0
# counter = 0
# # _len_ = len(train_loader)
# for i, data in tqdm(enumerate(train_loader), total = len(train_data)):
#     # counter += 1
#     img, mask = data
#     img.to(device = Device)
#     mask.unsqueeze(1).to(device = Device)
#     # forward
#     with torch.cuda.amp.autocast():
#         preds = model(img)
#         # process = T.Resize(img.size()[2])
#         # preds = process(preds)
#         loss = loss_fn(preds, mask)
#         train_running_loss += loss.item()
#         train_running_acc += 0.0
#     # backward
#     optimizer.zero_garad()
#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()

#         # update tqdm loop
#         tqdm(enumerate(dataloader)).set_postfix(loss = loss.item()) 
#     epoch_loss = train_running_loss / counter
#     epoch_acc = 100. * seg_acc(preds, mask)
#     return epoch_loss, epoch_acc

# # def val_fn(val_loader, model, optimizer, loss_fn, scaler):
# #     val_running_loss = 0.0
# #     val_running_acc =0.0
# #     counter = 0
# #     with torch.no_grad():
# #         for i, val_data in tqdm(enumerate(val_loader), total = len(val_loader)):
# #             counter += 1
# #             img, mask = val_data
# #             img.to(device = Device)
# #             mask.unsqueeze(1).to(device = Device)
# #             preds = model(img)


# def main():
#     if Load_model is not None:
#         load_model(torch.load('Lightuent18S_1e5_e18.pth'), model)

#     train_loss, val_loss = [], []
#     train_acc, val_acc = [], []
#     # check_accuracy(val_loader, model, device = Device)
#     scaler = torch.cuda.amp.GradScaler()
#     for epoch in range(Num_epochs):
#         train_epoch_loss, train_epoch_acc = fit(train_loader, model,
#                                                      optimizer, loss_fn, scaler)
#         val_epoch_loss, val_epoch_acc = fit(val_loader, model,
#                                                optimizer, loss_fn, scaler)

#         train_loss.append(train_epoch_loss)
#         val_loss.append(val_epoch_loss)
#         train_acc.append(train_epoch_acc)
#         val_acc.append(val_epoch_acc)
# #         # save model
# #         checkpoint = {
# #             'sate_dict': model.state_dict(),
# #             'optimizer': optimizer.state_dict()
# #         }
# #         save_model(checkpoint)
# #         # check accuracy
# #         # check_accuracy(val_loader, model, device = Device)

# #         # print some examples to a folder
# #         save_predictions_as_imgs(val_loader, 
# #                                  model, 
# #                                  folder = 'saved_imgs/', 
# #                                  device = Device)
    
#         # plot loss and acc
#         save_plots(train_acc, val_acc, train_loss, val_loss)
# #         # save final model
# #         torch.save(model.state_dict(), 'saved_imgs/Lightuent18S_1e5_e18.pth')
# #         print('============> TEST PASS!!!')


# if __name__ == "__main__":
#     main()





