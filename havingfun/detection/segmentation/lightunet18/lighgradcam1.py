# use the grad-cam to make layers output more expalinale
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import numpy as np
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image

from lightunet import LightUnet
import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import matplotlib.pyplot as plt

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_apth = 'datasets/S_kaggle_wildfire/000002.jpg'

input_cv2 = cv2.imread(img_apth, cv2.COLOR_BGR2RGB)
input_im = Image.fromarray(input_cv2)
transform = T.ToTensor()
input_tensor = transform(input_im)
input_tensor = input_tensor.to(device = Device)
input_tensor = input_tensor.unsqueeze(0)
print(input_tensor.size())

model = LightUnet().to(device = Device)
target_layers = [model.neck[-1]]

codes = ['Void', 'Smoke', 'Fire',]
# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(codes), 3))

def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [codes[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    
    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices

def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

masks = predict(input_tensor, model, Device, 0.9)

# cam = EigenCAM(model,
#                target_layers, 
#                use_cuda=torch.cuda.is_available(),)

# grayscale_cam = cam(input_tensor, targets=targets)
# # Take the first image in the batch:
# grayscale_cam = grayscale_cam[0, :]
# cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
# Image.fromarray(cam_image)
# plt.show()