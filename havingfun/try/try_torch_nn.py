from  PIL import Image  
import torch
import math
import numpy as np
import torchviz

img = Image.open("datas/wildfireeg001.jpg").convert("RGB")
img.resize((255, 255))
x = np.array(img)
print(img)
print(x.shape)

# conver tnp.array into torch.tensor
x, = map(torch.tensor,(x,))
print(x.shape)
# weights and bias
weights = torch.randn(3, 1200)/math.sqrt(784)
weights.requires_grad_()
biases = torch.zeros(10, requires_grad=True)

# activation
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(2)

def model(xb):
    return log_softmax(xb @ weights + biases)

# batch size
bs = 1
# a mini-batch from x
xb = x[0:bs]

preds = model(xb)
print(preds[0])
print(preds.shape)

torchviz.make_dot(preds, params = {"W": weights, "b": biases})

