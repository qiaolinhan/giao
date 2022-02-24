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
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
# Hyperparameters: batch size, number of workers, image size, train_val_split, model
Batch_size = 2
Num_workers = 0
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

train_loader = DataLoader(train_data, batch_size = Batch_size, 
                          num_workers = Num_workers, 
                          pin_memory = Pin_memory,
                          shuffle = True)
val_loader = DataLoader(val_data, batch_size = Batch_size, 
                        num_workers = Num_workers, 
                        pin_memory = Pin_memory,
                        shuffle = True)

# segmentation codes
codes = ['Target', 'Void']
num_classes = 2
name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Void']

def seg_acc(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim = 1)[mask]==target[mask]).float().mean()

def modeltrans(input, target):
    preds = model(input)
    if preds.shape != target.shape:
        preds = TF.resize(preds, size=target.shape[2:])
    return preds

def fit(train_loader, modeltrans, optimizer, loss_fn, scaler):
    print('====> Fitting process')

    train_running_loss = 0.0
    train_running_acc = 0.0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total = len(train_data)):
        counter += 1
        img, mask = data
        img.to(device = Device)
        mask.unsqueeze(1).to(device = Device)
        # forward
        with torch.cuda.amp.autocast():
            preds = model(img)
            if preds.shape != mask.shape:
                preds = TF.resize(preds, size=mask.shape[2:])
                # preds = preds.float()
            # mask = mask.long()
            print(mask.type())
            print(preds.type())
            loss = loss_fn(preds, mask)
            train_running_loss += loss.item()
            train_running_acc += seg_acc(preds, mask)
        # backward
        optimizer.zero_garad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        tqdm(enumerate(train_loader)).set_postfix(loss = loss.item()) 
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * seg_acc(preds, mask)
    return epoch_loss, epoch_acc


def main():
    # if Load_model is not None:
    #     load_model(torch.load('Lightuent18S_1e5_e18.pth'), model)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    # check_accuracy(val_loader, model, device = Device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Num_epochs):
        train_epoch_loss, train_epoch_acc = fit(train_loader, model,
                                                     optimizer, loss_fn, scaler)
        val_epoch_loss, val_epoch_acc = fit(val_loader, model,
                                               optimizer, loss_fn, scaler)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_acc.append(val_epoch_acc)
#         # save model
#         checkpoint = {
#             'sate_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict()
#         }
#         save_model(checkpoint)
#         # check accuracy
#         # check_accuracy(val_loader, model, device = Device)

#         # print some examples to a folder
#         save_predictions_as_imgs(val_loader, 
#                                  model, 
#                                  folder = 'saved_imgs/', 
#                                  device = Device)
    
        # plot loss and acc
        save_plots(train_acc, val_acc, train_loss, val_loss)
#         # save final model
#         torch.save(model.state_dict(), 'saved_imgs/Lightuent18S_1e5_e18.pth')
#         print('============> TEST PASS!!!')


if __name__ == "__main__":
    main()





