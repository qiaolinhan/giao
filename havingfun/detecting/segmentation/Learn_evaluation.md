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

:x:__However, it is in no way the best metric__ :fearful:  
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

### 1.3. Dice Coefficient (F1 Score)
__Define:__ the dice coefficient is 2* the area of overlap divided by the total number of pixels in both images.  

<img src="https://miro.medium.com/max/429/1*yUd5ckecHjWZf6hGrdlwzA.png" alt="2xOverlap/Total number of pixels">

Dice = (Ships + Background)/2

(The dice coefficient is very similar to IOU)

Implementation of Dice Score
```python
def dice_coef(y_true, y_pred, smooth = 1):
  intersection = K.sum(y_true * y_pred, axis = 1)
  union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, axis = [1, 2, 3])
  # the difference is there is a '2*' in dice cofficient
  dice = K.mean((2. * intersection + smooth) / (union + smooth), axis = 0) 
  return dice
```

## Cited from [Kiprono Elijah Koech](https://towardsdatascience.com/on-object-detection-metrics-with-worked-example-216f173ed31e#:~:text=At%20a%20low-level%2C%20evaluating%20performance%20of%20an%20object,%28FP%29%20%E2%80%94%20Incorrect%20detection%20made%20by%20the%20detector.)
The most popular metrics:  
* Average precision (AP)
* mean Average precision (mAP)

At a low level, evaluating performance of an object detector down to determining if detection is correct or net.  

| Definition          | Explain                                                        |
| ----                | ----                                                           |
| True Positive (TP)  | Correct detection made by the model.                           |
| False Positve (FP)  | Incorrect detection made by the detector.                      |
| False Negative (FN) | A ground-truth misser (not detected) by the object detector.   |
| True Negative (TN)  | This is background region correctly not detected by the model. |

### Precision and Recall
* Precision: The degree of exactness of the model in identifying only relevant objects. It is the ration of TPs over all
  detections made by the model.
* Recall: It measures the ability of the model to detect all ground truths -- proposition of TPs among all ground
  truths.

$$
P = \frac{TP}{TP + FP} = \frac{TP}{ all detections }\\
R = \frac{TP}{TP + FN} = \frac{TP}{ all ground-truth}\\
$$
Higher precision and higher recall means better model. A perfect model has zero FNs and zero FPs.  

### Precision x Recall Curve (PR Curve)
<img src="https://miro.medium.com/max/312/1*WL8PnVSPE_0Pem9bnSeseQ.png">

Confidence score also rely on threshold.  
Raising confidence score threshold means that more objects will be missed by the model.  
(More FNs and therefore low recall and high precision)  
Lower confidence score will mean that hte model gets more FPs (low precision adn high recall).  
The recision-recall (PR) curve is a plot of precision adn re call at varying values of confidence. For a good model,
precision and recall stays high even when confidence sore is varied.

### Average Precision
* $AP@\alpha$: Area under the Precision-Recall Curve (AUC-PR)
* __11-point interpolation method__
* __All-point interpolation method__

```python
import numpy as np
from sklearn.metrics import average_precision_score
y_true = np.array([0, 0, 1, 1])
y_score = np.array([0.1, 0.4, 0.35, 0.8])
average_precision_score(y_true, y_scores)
```
#### $AP@\alpha$ 
It is evaluated at $\alpha$ IoU threshold. Formally, it is defined as follows
$$
AP@\alpha = \int_{0}^{1} p(r) dr
$$
(AP50 and AP 75 mean AP calculated at IoU = 0.5 and IoU = 0.75)  
A high area under PR curve means high recall and high precision. PR curve is a zig-zag like plot.

#### 11-point interpolation method
A 11-point AP is a plot of interpolated precision scores for a model results at 11 equally spaced standard recall level,
nemely, 0.0, 0.1, 0.2, ..., 1.0. It is defined as:
$$
AP@\alpha_{11} = \frac{1}{11}\sum_{r \in R} p_{interp}(r)
$$
where $R = {0.0, 0.1, 0.2, ..., 1.0}$ and interpolated precision at recall value, r. It is the highest precision for any
recall value $r' \leq r$.  

#### All-point interpolation method
$$
AP@\alpha = \sum_i (r_{i + 1} - r_i)p_{interp}(r_{i + 1})
$$

### Mean Average Precision (mAP)
__Remark (AP and the number of classes):__ AP is calculated individually for each class. This means that there are as
many AP values as the number of calsses (loosely). These AP values are averaged to obtain the metric: mean Average
Precision (mAP). Precisely, mean Average Precision (mAP) is the average of AP values over all classes.
$$
mAP@\alpha = {1\over n} \sum_{i = 1}^{n} AP_i \text{for n classes}
$$
<br>
__Remark (AP and IoU):__ AP is calculated at a given IoU threshold $\alpha$.
