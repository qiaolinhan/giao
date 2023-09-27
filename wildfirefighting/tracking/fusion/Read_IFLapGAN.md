Generative adversarial network (GAN), powerful tool for infrared and visible image fusion.  
* Extracting methods extract the features incompletely:
  * Miss some texture
  * Lack the stability of training

The Generator:  
* shallow features extraction module 
* Laplacian pyramid module 
  * a pyramid-style encoder-decoder architecture, which progressively extracts the multi-scale features; 
  * attention module is equipped in the decoder to decode salient features.
* reconstruciton module 

Two Discriminators:
Two discriminators are adopted to discriminate the fused image and two different modalities respectively.

Develop another side supervised loss based on the side pre-trained fusion network, reduces the bias the mixed
distributions of infrared/visible images and fused image, improves the stability of GAN training.

The loss function:
* Adversarial Loss of generator and discriminator 
* Content loss 
  * used to promote the spatial consistency between fused image and source images.   
  * Inspired by the loss function described in GANMcC
  * adopt the pixel-aware and gradient-aware consistency to define the content loss.
* Side supervised loss

Dataset: TNO dataset 39 pairs of infrared/visible images
Ratios improved:  
$Q_{NMI}, Q_{M}, Q_{Yang}, Q^{AB/F}$, MI, VIF, FMI


Infrared: robust to various disturbances such as low light, smog and disguises, but lack of textures.

##### Classical traditional image fusion methods  
* Multi-scale decomposition (MSD)  
  * MSD decompositions 
  * sub-band fusion 
  * inverse MSD
  * (Laplacian pyramid, wavelet, and nonsubsampled contourlet)
###### Drawbacks of MSD  
* handcrafted with limited features representation capacity 

##### Sparse representation (SR) 
* Classical SR 
* Joint SR 
* Group SR 
* Convolutional SR 
The learned dictionary can provide more accurate representation than MSD.  
Nevertheless, it is still a problem to choose a rule for fusing the sparse coefficients of infrared and visible images,
since the non-zero elements are random.  

##### Deep neural networks (DNNs) for solving the image fusion problem 
* Primal work for multi-focus image fusion (MFIF) that employs a DNN to predict the decision maps.  
  * challenges: unaviliable ground truth for supervised learning;
  * for solving the challenge:
    * DNNs-based features extraction: then fuse using well-designed rule (auto-encoder with dense block, nest connection
      network, image decomposition network, auto-encoder with squeeze and excitation (SE) block, residual network,
      neural architecture search)
    * End-to-end architectures with unsupervised learning or features transfering.

