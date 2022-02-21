import fastai
from fastai.vision import *
from fastai.vision.all import *
from lightunet import LightUnet
path = Path('dataset/imgs/jinglingseg1')
path_img = path/'images'
path_mask = path/'masks'

codes = ['Target', 'Void']

img_names = get_image_files(path_img)
mask_nams = get_image_files(path_mask)

get_y_fn = lambda x: path_mask/f'mask_{x.name}'

dls = SegmentationDataLoaders.from_label_func(
    path_img,
    bs = 1,
    fnames = img_names,
    label_func = get_y_fn,
    codes = codes,
    item_tfms = [Resize((400, 400))],
    batch_tfms = [Normalize.from_stats(*imagenet_stats)],    
)

name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Void']

def acc_smoke(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim = 1)[mask]==target[mask]).float().mean()

metrics = acc_smoke

learn = Learner(dls, LightUnet, metrics = metrics)
lr = 1e-5
learn.fit(13, slice(lr))
