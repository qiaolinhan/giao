# Commonly Used Evaluation Methods for Image Segmentation
2022-07-05

This md is mainly for recording some commonly used evaluation schemes for image
segmentation.
## 1. citing from [Ekin Tiu](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)
Includes:  
* Pixel Accuracy
* Intersection-Over-Union (Jaccard Index)
* Dice Coefficient (F1 Score)

### 1.1. Pixel Accuracy
__Define:__ It is the present of pixels in your image that are classified correctly.  

:x:__However, it is in no way the mest metric__ :fearful:  
__Disadvantage: class imbalance__
  * When our class are extremely imbalanced , it means that a class or some classes dominate the image, while some of
    other classes make up only a small portion of the image.  

### 1.2. Intersection-Over-Union (IOU, Jaccard Index)
__Define:__ It is the area of overlap between the predicted segmentation and the ground truth.  

<img src="https://miro.medium.com/max/300/0*kraYHnYpoJOhaMzq.png" alt="Visualized IOU calculation"/>

The mean IOU of image is calculated by taking the IOU of each class and average them.  

__What is overlap and Union in our context?__  
* Overlap: move the predicted segmentation directly above the ground truth to see how many foreground pixels and
  background pixels are classified.  
* Union: (predicted foreground + ground truth) - overlap pixels  

Keras implementation
```python
from keras import backend as K

# y_true, y_pred: [m, r, c, n]
# ------
# m: number of images
# r: number of rows
# c: number of colums
# n: number of calsses 
def iou_coef(y_true, y_pred, smooth = 1):
  intersection = K.sum(K.abs(y_true * y_pred), axis = [1, 2, 3])
  union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis = 0)
  return iou
```

