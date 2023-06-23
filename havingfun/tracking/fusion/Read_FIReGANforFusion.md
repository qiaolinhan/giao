

## DL-based image fusion
* Reduced the complexity compared to methods on the multi-scale transform
  and representation learning domains.
## Data Fusion
Deep learning has an inbuilt automatic stage feature process that learns rich hierarchical representations (i.e.
features).
Ptchy Operation
1. Apply a set of weights -- a filter -- to extract **local features**
2. Use **multiple filters** to extract different features  
3. **Specially share** parameters of each filter

## FIRe-GAN
------------------------------
decompose paramount properties regard latter trivial thorough endeavour  
------------------------------
Function:  
* The fusion of thermal and visible information into a single image can potentially increase the robustness and accuracy
  of wildfire detection models. 

Characteristic:
* There is a growing interest in Deep Learning (DL)-based image fusion, **due to** their reduced complexity;
* DL-based image fusion have not been evaluated in the domain of the imagery

Dataset:  
1. Corsican Dataset, visible-infrared image pairs [12]. Contains 640 pairs of visible and near-infrared (NIR) fire images.
2. RGB-NIR dataset [18], which contains 477 non-fire visible-NIR image pairs.

Selected state-of-the-art (SOTA):  

| Code        | Author      | Model                                 | Characteristic                                                                                           |
| ---         | ---         | ---                                   | ---                                                                                                      |
| VGG19Fusion | Li et al.   | Pre-trained VGG19 DCNN                | The authors only selected layers of the network, no further training on new datasets is need.           |
| GAN1Fusion  | Ma et al.   | Generative Adversarial Networks (GANs) | Advantage: end-to-end trainable, which significantly reduces its implementation and training complexity. |
| GAN2Fusion  | Zhao et al. | GAN-based approach                    | Being able to generate approximate infrared images from visible ones.                                    |

Limitation:  
It is relevant to note that many of the existing visible-infrared fusion methods output greyscale fused image, which
means that the color information of the visible image is lost.

For FIRe-GAN:
* Allows for the processing of higher resolution image and the generation of color fused images as outputs;  
* The latter is relevant due to color being one of the most used features in visible-image-based fire detection;  

Main contributions:
1. Carry out a thorough analysis and comparison of existing DL-fusion methods for conventional imagery;  
2. Provide a quantitative demonstration of the feasibility of applying DL-based fusion methods for infrared imagery from
   wildfires;
3. IR and fused image generator that has been tested both in conventional and fire imagery.

### VGG19Fusion  
1. Decompose the original image into base parts and detail content;
2. Fuse these base parts through weight-averaging;
3. For the fusion of detail parts, the authors employ a DL framework in which they first use selected layers of the
   pre-trained VGG19 model to extract deep features;
4. Use a multi-layer fusion strategy to extract weight maps;
5. Use the deep features and weight maps to reconstruct the fused detail content;
6. Construct the final output image by combining the fused detail and base contents.

### GAN1Fusion  
* The first to propose a GAN model for image fusion tasks.  
* The architecture is an end-to-end model that generates fused images automatically from the source images without the
  need of defining fusion rules.  
The generator --> to produce a fused image with thermal information from IR image and gradients (qiao: Weights?) from
the visible one.  
The discriminator --> forces the generated image to contain more  details from the visible image. It is named as
FusionGAN model with samples from the Corsican Fire Database.  

### GAN2Fusion  
GAN to fuse unmatched infrared and visible images.  
Visible image --> generator (G1)  --> Synthetic infrared image;  
Visible image, Synthetic infrared image --> generator (G2) --> fused image;  
Visile image, fused image --> discriminator(D1) --> to close to source visible image, containing more textural details;  
The source infrared image, generated infrared image, fused image --> discriminator(D2) -->.  
It could be called as an UnmatchGAN  

| Model                   | Advantage                                                                    | Disadvantage                                                                                                                                                                                                                                                  |
| ---                     | ---                                                                          | ---                                                                                                                                                                                                                                                           |
| VGG19                   | 1. Only need some layersto perform feature extraction                        | 1. Not an end-to-end method. <br> 2. The required intermediate steps increase its implementation complexity.                                                                                                                                                  |
| GAN1Fusion (FusionGAN)  | 1. An end-to-end model, significantly reducing its implementation complexity | 1. Need to be trained on visible-infrared image pairs <br> 2. In consequence, its performance depends on the quality of the training process <br> 3. GANs haveadditional challenge of training stability (qiao: What is the training stability, see ref [16]) |
| GAN2Fusion (UnmatchGAN) | 1. end-to-end procedure                                                      | 1. Training stability <br> 2. Need good training dataset|

For UnmatchGAN:  
* Additional capability of learning to generate approximate infrared images based on source visible ones.  
* The fusion process requires perfectly aligned source images.  
* For the particular context of fire images, this could prove a significant advantage for the research community given
  the burden of obtaining perfectly matched visible-infrared fire images on realistic operative scenarios.

### Metrics
* They are the most common in the image fusion area.

#### Information entropy (EN)
It reflects the average amount of information in an image.  
Definition: $EN = - \sum_{l = 1}^{L - 1}p_l\log_2{p_l}$
$L$: the gray level of the image; $p_l$: the proportion of gray-valued pixels $i$ in the total number of pixels.  
Larger $EN$, more information is in the fused image.   

#### Correlation coefficient (CC) 
It measures the degree of linear correlation between the fused image and either the visible or infrared image.  
Definition: $CC(X, Y) = \frac{Cov(X, Y)}{\sqrt{Var(x) Var(y)}}$
$Cov(X, Y)$: The covariance between the fused image and the reference images; $Var(X)$, $Var(Y)$: variance of the two
images.  
Larger $CC$, higher the correlation between the fused and the reference images.

#### Peak signal-to-noise ratio(PSNR)
The PSNR assumes that the difference between the fused image and reference image is noise.  
Definition: 
$PSNR = 10\log_{10}(\frac{MAX^2}{MSE})$  
$MAX$: the maximum value of the image color; $MSE$: Mean squared error.  
An accepted benchmark for this metric is 30dB. $PSNR$ lower than this threshold means that the fused image presents
significant deterioration.  

#### Structural similarity index measure (SSIM)
A method for measuring the similarity between two images. It is based on the degradation of structural information.
Definition:
$SSIM(X, Y) = (\frac{2u_x u_y + c_1}{u_x^2 + u_y^2 + c_1})^{\alpha} * (\frac{2\sigma_x \sigma_y + c_2}{\sigma_x^2 + \sigma_y^2 + c_2})^{\alpha} * (\frac{\sigma_{xy} + c_3}{\sigma_x \sigma_y + c_3})^{\gamma}$  
$x$, $y$ are the reference and fused images; $u_x$, $u_y$, $\sigma_x^2$, $\sigma_y^2$, $\sigma_{xy}^2$ represent the
mean value, variance, and covariance of image $x$ and $y$, $c_1$, $c_2$, $c_3$ are small numbers that help to avoid
division by zero, $\alpha$, $\beta$, $\gamma$ atr used to adjust the proportions.  
The Range of values for $SSIM$ goes from 0 to 1, with 1 being the best possible one.

### Comparison
UNet improved G1 of UnmatchedGAN --> FIRe-GAN  
* For consistency and to make the comparison fair with these methods, pre-trained the proposed FIRe-GAN model with the
  RGB_NIR dataset.
* Test on the Corsican Fire Detection Database.
