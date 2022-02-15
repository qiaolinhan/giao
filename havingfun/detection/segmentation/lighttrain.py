from pickle import FALSE
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
import albumentations as A
from albumentations.pytorch import ToTensorV2

from lightdata import JinglingDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Hyperparameters etc.


Batch_size = 4
Num_workers = 2
Image_hight = 400
Image_weight = 400
Pin_memory = True
Load_model = False
Valid_split = 0.2
Modeluse = LightUnet

# flexible hyper params
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
    type = float,
    default = 1e-5,
    help = 'Learning rate for training'
)

parser.add_argument(
    '-load',
    '--load',
    default = FALSE,
    help = 'Learning rate for training'
)

args = vars(parser.parse_args())
Num_epochs = args['epochs']
Img_dir = args['troot']
Mask_dir = args['mroot']
Learning_rate = args['lr']
Load_model = args['load']
Device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Computation device: {Device}\n')

# load the model
model = Modeluse(in_channels=3, out_channels=1)
model.to(device = Device)

total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
# total_trainable_params = sum(p.numel() for p in model.parameters()
#                                       if p.requires_grad)
# print(f'{total_trainable_params:} total parameters need training.')
optimizer = optim.Adam(model.parameters(), lr = Learning_rate)

loss_fn = nn.CrossEntropyLoss()


train_transform = A.Compose(
        [
            A.Resize(height = Image_hight, width = Image_weight),
            A.Rotate(limit = 35, p = 1.0),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.1),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
val_transform = A.Compose(
    [
        A.Resize(height = Image_hight, width = Image_weight),
        A.Normalize(
            mean = [0.0, 0.0, 0.0],
            std = [1.0, 1.0, 1.0],
            max_pixel_value = 255.0,
        ),
        ToTensorV2(), 
    ]
)
img, mask = JinglingDataset(img_dir = Img_dir,
                            mask_dir = Mask_dir,
                            )

train_img, val_img, train_mask, val_mask = train_test_split(img, mask, 
                                                                test_size=Valid_split,
                                                                random_state=42
    )
def getloaders(
    train_img, val_img, train_mask, val_mask, batch_size,
    train_transform, val_transform, pin_memory=True, 
):
    train_ds = ((train_img, train_mask), train_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size = Batch_size,
        num_workers=Num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_ds = ((val_img, val_mask), val_transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=Batch_size,
        num_workers=Num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader

train_loader, val_loader = getloaders(
    Img_dir,
    Mask_dir,
    Batch_size,
    )

def train_fn(train_loader, model, optimizer, loss_fn, scaler):
    print('====> Training process')
    train_running_loss = 0.0
    train_running_acc = 0.0
    counter = 0

    loop = tqdm(enumerate(train_loader), total = len(train_loader))
    for batch_idx, (img, mask) in loop:
        counter += 1
        data = img.to(device = Device)
        targets = mask.float().unsqueeze(1).to(devie = Device)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            train_running_loss += loss.item()
            
        # backward
        optimizer.zero_garad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss = loss.item()) 
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_acc / len(train_loader))
    return epoch_loss, epoch_acc

def main():
    if Load_model:
        load_model(torch.load('Lightuent18S_1e5_e18.pth'), model)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    check_accuracy(val_loader, model, device = Device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Num_epochs):
        train_epoch_loss, train_epoch_acc = train_fn(train_loader, model,
                                                     optimizer, loss_fn, scaler)
        val_epoch_loss, val_epoch_acc = train_fn(val_loader, model,
                                               optimizer, loss_fn, scaler)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_acc.append(val_epoch_acc)
        # save model
        checkpoint = {
            'sate_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_model(checkpoint)
        # check accuracy
        check_accuracy(val_loader, model, device = Device)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, 
                                 model, 
                                 folder = 'saved_imgs/', 
                                 device = Device)
    
        # plot loss and acc
        save_plots(train_acc, val_acc, train_loss, val_loss)
        # save final model
        torch.save(model.state_dict(), 'saved_imgs/Lightuent18S_1e5_e18.pth')
        print('============> TEST PASS!!!')


if __name__ == "__main__":
    main()





