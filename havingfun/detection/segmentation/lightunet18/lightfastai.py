import fastai
from fastai.vision.all import *
import timm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm

from lightunet import LightUnet

# creat model for cnn_larner


print(f'======> cuda: {torch.cuda.is_available()}')
defaults.use_cuda = False

path = Path('/home/qiao/dev/giao/datasets/')
path_img = path/'S_kaggle_wildfire/'
path_label = path/'S_kaggle_wildfire_label/'
# codes = ['Road', 'Person', 'Car', 'Bike', 'Pets', 'Light', 'Vegetation', 'Sky', 'Cloud', 'Bound', 'Sign', 'Pole']
codes = ['Smoke', 'Flame', 'Cloud', 'Background']

fnames = get_image_files(path_img)
lbl_names = get_image_files(path_label)

get_y_fn = lambda x: path_label/f'label_{x.name}'


dls = SegmentationDataLoaders.from_label_func(
    path_img,
    bs = 1,
    fnames = fnames,
    label_func = get_y_fn,
    codes = codes,
    item_tfms = [Resize((400, 400))],
    batch_tfms = [Normalize.from_stats(*imagenet_stats)],
)

name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Background']

def acc_smoke(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    # mask = target
    return (input.argmax(dim = 1)[mask]==target[mask]).float().mean()

metrics = acc_smoke
# pretrained = False
learn = create_cnn_model(dls, LightUnet, metrics = metrics)

device = 'cuda'
learn.model.to(device)
learn.lr_find(stop_div=False, num_it=10)
learn.lr_find.plot()

