# Commonly Used Evaluation Methods for Image Segmentation
2022-07-05
This md is mainly for recording some commonly used evaluation schemes for image
segmentation.
## citing from [Ekin Tiu][https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2]
Includes:  
* Pixel Accuracy
* Intersection-Over-Union (Jaccard Index)
* Dice Coefficient (F1 Score)

### 1. Pixel Accuracy
__Define:__ It is the present of pixels in your image that are classified correctly.  

__However, it is in no way the mest metric__  
__Disadvantage: class imbalance__
  * When our class are extremely imbalanced , it means that a class or some classes dominate the image, while some of
    other classes make up only a small portion of the image.  

### 2. Intersection-Over-Union (IOU, Jaccard Index)
__Define:__ It is the area of overlap between the predicted segmentation and the ground truth.  





