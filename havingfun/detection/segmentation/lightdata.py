# data loading part
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class JinglingDataset(Dataset):
    def __init__(self,  img_dir, mask_dir, transform = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)

    def __len__(self):
        print(f'there are {len(self.imgs)} images')
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        img = np.array(Image.open(img_path).convert('RGB'))
        mask_path = os.path.join(self.mask_dir, self.imgs[index].replace('.png', '_mask.png'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype = np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transforms(img = img, mask = mask)
            img = augmentations['img']
            mask = augmentations['mask']

        return img, mask





