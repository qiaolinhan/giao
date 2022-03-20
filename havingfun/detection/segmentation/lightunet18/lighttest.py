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
root = 'havingfun/detection/segmentation/saved_imgs'
modelparam_path = root + 'Lightunet18Adam_1e5_e30.pth'
checkpoint = torch.load(modelparam_path)

# flexible hyper params: epochs, dataset, learning rate, load_model
parser = argparse.ArgumentParser()

# specifying whether to test the trained model
parser.add_argument(
    '-tar',
    '--tar_img',
    type = str,
    default = 'datasets/S_kaggle_wildfire/000005.jpg',
    help = 'Load the target image to be detected'
)
tarmask_path = 'datasets/S_kaggle_wildfire_label/label_000005.jpg'

# classes add codes
codes = ['Fire', 'Smoke', 'Void']
num_classes = 3
name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Void']
metric = Segratio(num_classes)

args = vars(parser.parse_args())
Load_model = args['load']
Target_img = args['tar_img']

# the device used fir training
Device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the model
model = Modeluse(in_channels=3, out_channels=1)
model.to(device = Device)
# print the parameter numbers of the model
total_params = sum(p.numel() for p in model.parameters())
# print(model.eval())
print('#############################################################')
print(f'There are {total_params:,} total parameters in the model.\n')

# resize tensor in up-sampling process                        
def sizechange(input_tensor, gate_tensor):
    sizechange = nn.UpsamplingBilinear2d(size = gate_tensor.shape[2:])
    out = sizechange(input_tensor)
    return out

def main():    
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