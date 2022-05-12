## [Light U-net for wildfire segmentation](https://github.com/qiaolinhan/giao/tree/master/havingfun/detecting/segmentation/lightunet18)
### Some Packages to Install
First, get a Conda environment. (Not nessesary, just for showing and recording the steps)    
`conda create -n 'name'`  
Install the relying packages  
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`  
`conda install numpy`  
`conda install matplotlib`  
`pip install opencv-python`  
`pip install -U albumentations`  
`pip install torch-lr-finder`
`conda install tqdm`

### Steps to Run
1. Change the dataset folder in [data](https://github.com/qiaolinhan/giao/blob/master/havingfun/detecting/segmentation/lightunet18/lightdataCV.py) to load in the prepared images and masks to check.
2. Check whether the [Network strucure](https://github.com/qiaolinhan/giao/blob/master/havingfun/detecting/segmentation/lightunet18/lightunet.py) is okay enough, this model is based on Resnet18 Encoder-Decoder, add layers for more complex segmentation missions.
3. Find appropriate learning rate based on Leslie method with [lrfinder](https://github.com/qiaolinhan/giao/blob/master/havingfun/detecting/segmentation/lightunet18/lightlrfind.py), there is an [appliable package](https://pypi.org/project/torch-lr-finder/?msclkid=c492365aaf6c11ec9d78518a9cef19b9) could simply work it.
4. Ready to train the model, traing on [train](https://github.com/qiaolinhan/giao/blob/master/havingfun/detecting/segmentation/lightunet18/lighttrain.py)
5. Test on images with [imgtest](https://github.com/qiaolinhan/giao/blob/master/havingfun/detecting/segmentation/lightunet18/lighttestimg.py) and videos with [videotest](https://github.com/qiaolinhan/giao/blob/master/havingfun/detecting/segmentation/lightunet18/lighttestvideo.py). The opencv is recommonded to load the video and image, but ther is convert part happen in the `.py` files.

### Files and Packages Explain
#### Cross Entropy Loss
commonly, use __Sum of the Squared Residuals__ to determine how well the Neural network fits the data. 
$$SSR = \sum_{i = 1}^{n = 3}{(Observed_i - Predicted_i)^2}$$

The `.py` files are relying on the blocks and tools in [deving](https://github.com/qiaolinhan/giao/tree/master/havingfun/deving)
* `util.py`: Some functions are stored, such as: saving model at every iteration; save entire model; saving predicted result; saving accuracy's and loss's plots; plot image, pred, mask in a figure ...  


