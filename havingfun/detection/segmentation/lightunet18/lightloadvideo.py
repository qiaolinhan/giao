# 2022-03-28
# This model is used to test the video loading into network and save the output video
# Thanks for shun's example

import cv2
import torch
from lightunet import LightUnet
from lightutils import (
    load_model,
    # save_predictions_as_imgs,
    plot_img_and_mask,
)
import argparse

import torchvision

from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
# import os

Device = 'cuda' if torch.cuda.is_available() else 'cpu'
Modeluse = LightUnet
root = 'havingfun/detection/segmentation/saved_imgs/'
modelparam_path = root + 'Lightunet18_MSE_Adam_1e4_e30.pth'
checkpoint = torch.load(modelparam_path, map_location=torch.device(Device))
# load the model
model = Modeluse(in_channels=3, out_channels=1)
model.to(device = Device)
load_model(checkpoint, model)
# print the parameter numbers of the model
total_params = sum(p.numel() for p in model.parameters())
print(f'======> There are {total_params:,} total parameters of this model.\n')
# print(model.eval())

# flexible hyper params: dataset for testing
parser = argparse.ArgumentParser()
parser.add_argument(
    '-tar',
    '--tar_video',
    type = str,
    default = 'datasets/DJI_0023.MOV',
    help = 'Load the target video to be detected'
)
args = vars(parser.parse_args())
INPUT_VIDEO_PATH = args['tar_video']
TOSAVE_VIDEO_SIZE = (400, 400)

capture = cv2.VideoCapture(INPUT_VIDEO_PATH)
all_frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'======> There are {all_frame_num:} total frames.')
video_saver = cv2.VideoWriter("segresult.avi", cv2.VideoWriter_fourcc(*'DIVX'), 5,
                              TOSAVE_VIDEO_SIZE)


if __name__ == "__main__":
    for img_cv2 in range(0, all_frame_num, 10):
        ret, frame = capture.read() 
        resized = cv2.resize(frame, TOSAVE_VIDEO_SIZE)
        # print("read the frame %d\n", img_cv2)
        img_im = Image.fromarray(resized)
        trans2tensor = torchvision.transforms.ToTensor()
        img_tensor = trans2tensor(img_im).unsqueeze(0).to(device = Device)    
        pred_tensor = model(img_tensor).squeeze(1)

        pred_tensor = pred_tensor.squeeze(1)
        trans2img = torchvision.transforms.ToPILImage()
        pred_im = trans2img(pred_tensor)
        pred_cv2 = np.asarray(pred_im)[:,::-1].copy()
        cv2.imshow('img', pred_cv2)
        cv2.waitKey(1)

        video_saver.write(resized)
        # video_saver.write(pred_cv2)
        print("write the frame %d\n", img_cv2)

    video_saver.release()
    capture.release()
    print('======> Test Pass!')
# capture = cv2.VideoCapture(INPUT_VIDEO_PATH)
# all_frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


# video_saver = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 5,
#                               TOSAVE_VIDEO_SIZE)
# if __name__ == "__main__":

    

#     for i in range(all_frame_num):
#         ret, frame = capture.read()
#         print("read the frame %d\n", i)
#         resized = cv2.resize(frame, TOSAVE_VIDEO_SIZE)
#         video_saver.write(resized)

#     video_saver.release()
#     capture.release()
#     print("Quit!")

# Batch_size = 1
# Num_workers = 0
# Image_hight = 400
# Image_weight = 400
# Pin_memory = True
# Valid_split = 0.2








# def main():
    
#     video_path = INPUT_VIDEO_PATH

#     img_cv2 = cv2.imread(video_path, cv2.COLOR_BGR2RGB)
#     img_im = Image.fromarray(img_cv2)


#     trans2img = torchvision.transforms.ToPILImage()
#     pred_im = trans2img(pred_tensor).convert('L')
#     plt.imshow(pred_im)
#     plt.grid(False)
#     plt.show()

#     plot_img_and_mask(img_im, pred_im, mask_im)


# if __name__ == '__main__':
#     main()
#     print('======> Test Pass!')
