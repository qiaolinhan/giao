# to build a resnet34-based classifier to recognize the image types which consist of: Normal, Smoke, Flame
import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
print(model.eval())