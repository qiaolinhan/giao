
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from lightunet import LightUnet
import requests
import albumentations as A
from albumentations.pytorch import ToTensorV2

img_path = 'datasets/imgs/jinglingseg/images/img6.png'
mask_path = 'datasets/imgs/jinglingseg/masks/img6_mask.png'
Device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_im = Image.open(img_path).convert('RGB')
img_np = np.array(img_im)
# img_tensor = transform(image = img_np)['image']
mask_im = Image.open(mask_path).convert('L')

trans2img = transforms.ToPILImage()
trans2tensor = transforms.ToTensor()
img_tensor = trans2tensor(img_im).unsqueeze(0)
# input_tensor = img_tensor.unsqueeze(0)
# rgb_img = np.float32(img_np)/255
# input_tensor = preprocess_image(rgb_img,
#                                 mean=[0, 0, 0],
#                                 std=[1, 1, 1])

modelparam_path = '/home/qiao/dev/giao/havingfun/detection/segmentation/saved_imgs/Lightunet18_1e4_e30.pth'
checkpoint = torch.load(modelparam_path)
img_tensor.to(device = Device)
model = LightUnet(in_channels=3, out_channels=1).to(device=Device)

def load_model(checkpoint, model):
    print('====> Loading checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])

load_model(checkpoint, model)
# print(model.eval())


if torch.cuda.is_available():
    model = model.cuda()
    img_tensor = img_tensor.cuda()

preds = model(img_tensor)

if preds.shape != img_tensor.shape:
    preds = TF.resize(preds, size = img_tensor.shape[2:])
output = preds.squeeze(0)
output = torch.sigmoid(output)
print(output.size())
# normalized_masks = F.softmax(output, dim=1).cpu()

# sem_classes = ['void', 'smoke']
# sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

# smoke_category = sem_class_to_idx['smoke']
# smoke_mask = normalized_masks[0, :, :].argmax(axis = 0).detach().cpu().numpy()
# # print(smoke_mask)
# smoke_mask_uint8 = 255 * np.uint8(smoke_mask == smoke_category)
# smoke_mask_float = np.float32(smoke_mask == smoke_category)


img_in = trans2img(img_tensor.squeeze(0))
img_out = trans2img(output).convert('L')
mask_tar = mask_im

plt.figure()

f, axarr = plt.subplots(1, 2)

axarr[0].imshow(img_im)
axarr[1].imshow(img_out)
# axarr[2].imshow(mask_tar)

for axarr in axarr[:]:
    axarr.get_xaxis().set_visible(False)
    axarr.get_yaxis().set_visible(False)
plt.show()


# both_images = np.hstack((input_tensor.squeeze(0), np.repeat(smoke_mask_uint8[:, :, None], 3, axis=-1)))
# Image.fromarray(both_images)
# Image.show()



# class SemanticSegmentationTarget:
#     def __init__(self, category, mask):
#         self.category = category
#         self.mask = torch.from_numpy(mask)
#         if torch.cuda.is_available():
#             self.mask = self.mask.cuda()
        
#     def __call__(self, model_output):
#         return (model_output[self.category, :, : ] * self.mask).sum()

    
# target_layers = [model.neck]
# targets = [SemanticSegmentationTarget(smoke_category, smoke_mask_float)]
# # with GradCAM(model=model,target_layers=target_layers,
# #              use_cuda=torch.cuda.is_available()) as cam:
# with GradCAM(model=model,target_layers=target_layers) as cam:
#     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
# print(grayscale_cam)
# camed_image = show_cam_on_image(img_im, grayscale_cam, use_rgb=True)
# plt.imshow(camed_image)
# plt.show()
# # # Image.fromarray(cam_image)
