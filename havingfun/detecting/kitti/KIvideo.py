# 2022-03-30
# To test the kitti images loading into network 
# functions: 
# 1. Resize the images and output the video
# 2. Save the processed images as a video
# 3. * concatenate the videos to compare
# Thanks for shun's example

import cv2
import torch
from KIUnet import LightUnet
from KIutils import (
    load_model,
)
import argparse

import torchvision

from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
# import os

Device = 'cuda' if torch.cuda.is_available() else 'cpu'
Modeluse = LightUnet
root = 'havingfun/detection/segmentation/saved_imgs'
modelparam_path = root + 'Lightunet18_CE_Adam_5.96e6_e10.pth'
checkpoint = torch.load(modelparam_path, map_location=torch.device(Device))
# load the model
model = Modeluse(in_channels=3, out_channels=1)
model.to(device = Device)
load_model(checkpoint, model)
# print the parameter numbers of the model
total_params = sum(p.numel() for p in model.parameters())
print(f'======> There are {total_params:,} total parameters of this model.\n')
# print(model.eval())


# cv2 read the video cut images in a folder
kiimgs_cv2 = [cv2.imread(img_o) 
            for img_o in glob.glob('datasets/kittiseg/testing/image_2/*.png')]
print(f'======> There are total {kiimgs_cv2:} images.')

# flexible hyper params: 
parser = argparse.ArgumentParser()
# dataset for testing
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
# video_saver0 = cv2.VideoWriter("resizeresult.avi", cv2.VideoWriter_fourcc(*'DIVX'), 5,
#                               TOSAVE_VIDEO_SIZE)
# video_saver1 = cv2.VideoWriter("segresult.avi", cv2.VideoWriter_fourcc(*'DIVX'), 5,
#                               TOSAVE_VIDEO_SIZE)
video_saver2 = cv2.VideoWriter("concatresult.avi", cv2.VideoWriter_fourcc(*'DIVX'), 5,
                              TOSAVE_VIDEO_SIZE)

if __name__ == "__main__":
    for img_cv2 in range(0, all_frame_num, 10):
        ret, frame = capture.read()
        # resize the original video frames
        resized = cv2.resize(frame, TOSAVE_VIDEO_SIZE)
        # video_saver0.write(resized)
        print("resized the frame %d\n", img_cv2)

        # process the resized video frame by our model
        img_im = Image.fromarray(resized)
        trans2tensor = torchvision.transforms.ToTensor()
        img_tensor = trans2tensor(img_im).unsqueeze(0).to(device = Device)    
        pred_tensor = model(img_tensor)

        trans2img = torchvision.transforms.ToPILImage()
        pred_im = trans2img(pred_tensor)
        pred_cv2 = np.asarray(pred_im)[:,::-1].copy()
        print("processed the frame %d\n", img_cv2)
        # cv2.imshow('img', pred_cv2)
        # cv2.waitKey(1)
        # video_saver1.write(pred_cv2)

        # concatenate the resized video and processed video frames
        video_op = cv2.vconcat([resized, pred_cv2])
        video_saver2.write(video_op)
        print('concatenated the frame %d\n', img_cv2)

    # video_saver0.release()
    # video_saver1.release()
    video_saver2.release()
    capture.release()
    print('======> Test Pass!')