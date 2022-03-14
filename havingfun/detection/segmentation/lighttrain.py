from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from lightunet import LightUnet
from lightutils import (
    save_model,
    save_entire_model,
    load_model,
    save_predictions_as_imgs,
    plot_img_and_mask,
    save_plots,
)
from lightevaluate import Segratio
import argparse

# from albumentations.pytorch import ToTensorV2
import numpy as np
from lightdata import JinglingDataset, transform
from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

# Hyperparameters: batch size, number of workers, image size, train_val_split, model
Batch_size = 1
Num_workers = 0
Image_hight = 400
Image_weight = 400
Pin_memory = True
Valid_split = 0.2
Modeluse = LightUnet
root = '/home/qiao/dev/giao/havingfun/detection/segmentation/saved_imgs/'
modelparam_path = root + 'Lightunet18SGD_1e4_e5.pth'
checkpoint = torch.load(modelparam_path)
# flexible hyper params: epochs, dataset, learning rate, load_model
parser = argparse.ArgumentParser()


parser.add_argument(
    '-e',
    '--epochs',
    type = int,
    default = 5,
    help = 'Numbers of epochs to train the network'
)

parser.add_argument(
    '-l',
    '--lr',
    type = np.float32,
    default = 1e-2,
    help = 'Learning rate for training'
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

# specifying whether to test the trained model
parser.add_argument(
    '-load',
    '--load',
    default = True,
    help = 'loading the trained model for prediction'
)
parser.add_argument(
    '-tar',
    '--tar_img',
    type = str,
    default = '/home/qiao/dev/giao/dataset/imgs/jinglingseg/images/img1.png',
    help = 'Load the target image to be detected'
)
tarmask_path = '/home/qiao/dev/giao/dataset/imgs/jinglingseg/masks/img1_mask.png'

# classes add codes
codes = ['Target', 'Void']
num_classes = 2
name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Void']
metric = Segratio(num_classes)

args = vars(parser.parse_args())
Num_epochs = args['epochs']
Img_dir = args['troot']
Mask_dir = args['mroot']
Learning_rate = args['lr']
Load_model = args['load']
Target_img = args['tar_img']

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
loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = nn.CrossEntropyLoss()

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

# resize tensor in up-sampling process                        
def sizechange(input_tensor, gate_tensor):
    sizechange = nn.UpsamplingBilinear2d(size = gate_tensor.shape[2:])
    out = sizechange(input_tensor)
    return out

# training process
def fit(train_loader, model, optimizer, loss_fn, scaler):
    print('====> Fitting process')

    train_running_loss = 0.0
    train_running_acc = 0.0
    train_running_mpa = 0.0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total = len(train_data) // Batch_size):
        counter += 1

        img, mask = data[i]
        img.to(device = Device)
        
        mask = mask.unsqueeze(1)
        mask = mask.float()
        mask.to(device = Device)

        # forward
        with torch.cuda.amp.autocast():
            preds = model(img)
            # print('preds size before resize', preds.size())
            # print('mask size', mask.size())

            # for now, the predictions are tensors
            # becaus of the U-net characteristic, the output is croped at edges
            # therefore, the tensor need to be resized
            if preds.shape != mask.shape:
                # preds = TF.resize(preds, size=mask.shape[2:])
                preds = sizechange(preds, mask)
                # print('preds size after resize', preds.size())

            loss = loss_fn(preds, mask)
            train_running_loss += loss.item()


            preds = preds.squeeze(1).permute(1, 2, 0)
            mask = mask.squeeze(1).permute(1, 2, 0)
            preds = (preds/255).detach().numpy().astype(np.uint8)
            mask = mask.detach().numpy().astype(np.uint8)

            # print('preds size:', preds.shape)
            # print('masks size:', mask.shape)

            hist = metric.addbatch(preds, mask)
            acc = metric.get_acc()
            train_running_acc += acc.item()

            mpa = metric.get_MPA()
            train_running_mpa += mpa.item()
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 

        tqdm(enumerate(train_loader)).set_postfix(loss = loss.item(), acc = acc.item(), MPA = mpa.item())

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * train_running_acc / counter
    epoch_mpa = 100. * train_running_mpa / counter
    return epoch_loss, epoch_acc, epoch_mpa

def valid(val_loader, model, loss_fn):
    print('====> Validation process')

    val_running_loss = 0.0
    val_running_acc = 0.0
    val_running_mpa = 0.0
    counter = 0
    for i, data in tqdm(enumerate(val_loader), total = len(val_data) // Batch_size):
        counter += 1

        img, mask = data
        img.to(device = Device)

        mask = mask.unsqueeze(1)
        mask = mask.float()
        mask.to(device = Device)

        # forwardSGD
        with torch.cuda.amp.autocast():
            preds = model(img)
            if preds.shape != mask.shape:
                # preds = TF.resize(preds, size = mask.shape[2:])
                preds = sizechange(preds, mask)

            val_loss = loss_fn(preds, mask)
            val_running_loss += val_loss.item()

            preds = preds.squeeze(1).permute(1, 2, 0)
            mask = mask.squeeze(1).permute(1, 2, 0)
            preds = (preds/255).detach().numpy().astype(np.uint8)
            mask = mask.detach().numpy().astype(np.uint8)

            # print('preds size:', preds.shape)
            # print('masks size:', mask.shape)

            hist = metric.addbatch(preds, mask)
            val_acc = metric.get_acc()
            val_running_acc += val_acc.item()
            val_mpa = metric.get_MPA()
            val_running_mpa += val_mpa.item()

        tqdm(enumerate(val_loader)).set_postfix(loss = val_loss.item(), acc = val_acc.item(), mpa = val_mpa.item())

    val_epoch_loss = val_running_loss / counter
    val_epoch_acc = 100. * val_running_acc / counter
    val_epoch_mpa = 100. * val_running_mpa / counter
    return val_epoch_loss, val_epoch_acc, val_epoch_mpa

def main():
    if Load_model is not None:
        img_path = Target_img
        img_im = Image.open(img_path).convert('RGB')
        mask_im =Image.open(tarmask_path).convert('L')
        trans2tensor = torchvision.transforms.ToTensor()
        img_tensor = trans2tensor(img_im).unsqueeze(0).to(device = Device)
        load_model(checkpoint, model)
        pred_tensor = 255 * model(img_tensor)

        # if pred_tensor.shape != img_tensor.shape:
        #     pred_tensor = TF.resize(pred_tensor, size = img_tensor.shape[2:])
        #     print(pred_tensor.size())

        pred_tensor = pred_tensor.squeeze(1)
        trans2img = torchvision.transforms.ToPILImage()
        pred_im = trans2img(pred_tensor).convert('L')
        # plt.imshow(pred_im)
        # plt.grid(False)
        # plt.show()
        plot_img_and_mask(img_im, pred_im, mask_im)

    else:
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []
        # check_accuracy(val_loader, model, device = Device)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(Num_epochs):
            train_epoch_loss, train_epoch_acc, _ = fit(train_loader, model,
                                                        optimizer, loss_fn, scaler)
            # tqdm(enumerate(train_loader)).set_postfix(loss = train_epoch_loss(), acc = train_epoch_loss())
            val_epoch_loss, val_epoch_acc, _ = valid(val_loader, model,loss_fn)
            # tqdm(enumerate(val_loader)).set_postfix(loss = val_epoch_loss.item(), acc = val_epoch_loss())
            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_acc.append(val_epoch_acc)
 
            # save entire model
            save_model(Num_epochs, model, optimizer, loss_fn)
            # check accuracy
            # check_accuracy(val_loader, model, device = Device)

            # print some examples to a folder
            # save_predictions_as_imgs(val_loader, 
            #                             model, 
            #                             folder = 'saved_imgs/', 
            #                             device = Device)

            # plot loss and acc
            save_plots(train_acc, val_acc, train_loss, val_loss)

            # # save final model
        save_entire_model(Num_epochs, model, optimizer, loss_fn)

    print('\n============> TEST PASS!!!\n')


if __name__ == "__main__":
    main()





