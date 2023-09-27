# Model structure: Resnet18  
Total number of images: 818  
Total training images: 655  
Total valid_images: 163  
Computation device: cuda  

11,536,195 total parameters.  
11,536,195 training parameters.

## 2021-12-03
When transform the brightness with 0.5, the traning is unacceptable, the accuracy for 'smoke' is around $10\%$.    
Therefore the brightness is changed with 0.2. Then the learning is fine to separately reach $70\%$, where:  
|Epochs| Acc Fire | Acc Normal | Acc Smoke| Total Train Acc| Total Val Acc|  
|------| ---------|------------|----------|-----------------|---------------|   
|17| 96.42857142857143| 88.63636363636364| 71.42857142857143| 80.458| 84.663|  
|18| 91.07142857142857| 79.54545454545455| 80.95238095238095| 79.084| 84.049|  
|19| 71.42857142857143| 88.63636363636364| 73.01587301587301| 79.389| 76.687|  
|49| 96.42857142857143| 84.09090909090911| 36.50793650793651| 83.664| 69.939|  
|50| 80.35714285714286| 93.18181818181819| 22.22222222222222| 86.565| 61.350|  

epochs = 20 is almost enough,. At e19, the loss of validation is climbing and unstable. And the detection of smoke is getting worse.  

## 2021-12-04
It is considered to use lr = 1e-5 to slow the learning. The accuracy of $80\%$ reached fast (at e6).
|Epochs| Acc Fire| Acc Normal| Acc Smoke| Train Loss| Val Loss| Acc Train| Acc Loss| 
|------|---------|-----------|----------|-----------|---------|----------|---------|  
|1| 87.03703703703704| 82.14285714285714| 62.264150943396224| 0.843| 0.592| 61.679| 77.301|  
|2| 88.88888888888889| 73.21428571428571| 90.56603773584905| 0.724| 0.687| 70.534| 84.049|
|3| 79.62962962962963| 82.14285714285714| 71.69811320754717| 0.750| 0.550| 69.313| 77.914|  
|4| 85.18518518518519| 76.78571428571429| 83.01886792452831| 0.702| 0.551| 71.756| 81.595|  
|5| 85.18518518518519| 71.42857142857143| 86.79245283018868| 0.656| 0.492| 74.046| 80.982|
|6| 92.5925925925926| 89.28571428571429| 50.943396226415096| 0.595| 0.648| 76.794| 77.914|

It is hard to learn the detection of smoke, more epochs, result worse. 
It is considered that smaller learning rate makes the curve more stable. Therefore lr = 1e-6 is going to be teested. With e16, there is not an obvious trend of convergence of smoke detection.  
Back to lr = 1e-4, epochs = 17 ,augmentation with brightness = 0.1. Or lr = 1e-5, epochs = 3, brightness = 0.1 

|Learning Rate| Epochs| Acc Fire| Acc Normal| Acc Smoke| Train Loss| Val Loss| Acc Train| Acc Val|
|-------------|-------|---------|-----------|----------|-----------|---------|----------|---------|
|1e-5| 18| 98.21428571428571| 81.48148148148148| 67.9245283018868| 0.446| 0.574| 82.748| 82.822|
|1e-5| 3| 84.31372549019608| 78.46153846153847| 65.95744680851064| 0.665| 0.687| 71.450| 76.687|
|1e-4| 18| 83.6734693877551| 76.78571428571429| 74.13793103448276| 0.443| 0.859| 83.206| 77.914|
|1e-4| 4| 80.76923076923077| 79.62962962962963| 84.21052631578948| 0.713| 0.755| 71.298| 81.595|

It could be seen that: when lr = 1e-5, epochs = 18, the Acc Smoke is just 67.9, but Acc Fire is impressive 98.2. Its loss performed best!  
Best Acc Smoke is lr = 1e-4, epochs = 4, where Acc Smoke is 84.2, but its loss is not that excellent.


