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
1. Change the dataset folder in [data](havingfun/detection/segmentation/lightunet18/lightdata.py) to load in the prepared images and masks to check.
2. Check whether the [Network strucure](havingfun/detection/segmentation/lightunet18/lightunet.py) is okay enough, this model is based on Resnet18 Encoder-Decoder, add layers for more complex segmentation missions.
3. Find appropriate learning rate based on Leslie method with [lrfinder](havingfun/detection/segmentation/lightunet18/lightlrfind.py), there is an [appliable package](https://pypi.org/project/torch-lr-finder/?msclkid=c492365aaf6c11ec9d78518a9cef19b9) could simply work it.
4. Ready to train the model, traing on [train](havingfun/detection/segmentation/lightunet18/lighttrain.py)
5. Test on images with [imgtest](havingfun/detection/segmentation/lightunet18/lighttest.py) and videos with [videotest](havingfun/detection/segmentation/lightunet18/lightloadvideo.py). The opencv is recommonded to load the video and image, but ther is convert part happen in the `.py` files.

### Files Explain
* `util.py`: Some functions are stored, such as saving accuracy's and loss's plots.

