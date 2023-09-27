import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.png', '_mask.png'))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask > 0.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

if __name__ == '__main__':
    Img_dir = ('/home/qiao/dev/giao/dataset/imgs/jinglingseg/images')
    Mask_dir = ('/home/qiao/dev/giao/dataset/imgs/jinglingseg/masks')
    data = CarvanaDataset(Img_dir, Mask_dir)
    print(data[1])
