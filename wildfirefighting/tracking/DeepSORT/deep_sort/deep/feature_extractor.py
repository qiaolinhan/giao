#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: feature_extractor.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-10-27
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description:
#   to extract the features in bounding box, to acquire an embedding with fixed dimension to represent
#   bounding box.
#   The training is based on traditional ReID.
#   the input is an image typed list when using this Extractor class.
#   
# ------------------------------------------------------------------------------
import torch
import torchvision.transforms as transforms
import cv2
from .model import Net

class Extractor(object):
    def __init__(self, model_path, use_cuda = True):
        self.net = Net(reid = True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location = lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
                # RGB image values (0, 255) --> to tensor --> normalization to (0, 1) --> (x - mean)/std to (-1, 1)
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

    def _preprocess(self, im_crops):
        '''
        TODO: 
        1. to float with scale from 0 to 1
        2. resize to (64, 128) as Market1501 dataset did
        3. concatenate to a numpy array
        4. normalize
        '''
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim = 0).float()
        return im_match

    # call is a special usage, its function is similar as reloading () computing in class
    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

if __name__ == "__main__":
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)


