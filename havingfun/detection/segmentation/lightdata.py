# data loading part
from operator import index
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class JinglingDataset(Dataset):
    def __init__(self,  img_dir, mask_dir, transform = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):        
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        img = np.array(Image.open(img_path).convert('RGB'))
        img.sort()
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        mask = np.array(Image.open(mask_path).convert('L'), dtype = np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(img = img, mask = mask)
            img = augmentations['img']
            mask = augmentations['mask']

        return img, mask

if __name__ == '__main__':
    Img_dir = ('dataset/imgs/jinglingseg/images')
    Mask_dir = ('dataset/imgs/jinglingseg/masks')
    data = JinglingDataset(img_dir=Img_dir, mask_dir = Mask_dir)
    plt.imshow(data[0][1])
    plt.show()




