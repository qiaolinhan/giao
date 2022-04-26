__Gradient-weighted class activation mapping (Grad-CAM)__, 
which allows explanations for a specific class of image classification CNNs.  

* Sacrifice a degree of interpretability in pipeline modules inorder to achieve greater performance through greater abstraction (more layers) and tighter intergration (end-to-end training).  

__Class Activation Mapping (CAM)__:  
Produce a localization map from image classification CNNs where <u>global-average-pooled</u> convolutional feature maps are fed directly into a softmax.

In Pytorch, it is able to use __forward hooks__ to visualize activations of NNs.  
__Forward Hooks__: functions called after do forward pass.

