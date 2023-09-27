# 2022-04-08
# some functions for training results analysis and saving
import sklearn
import sklearn.metrics as metrics
import numpy as np
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

# -----------------------------
# load the dataset path
import os
root = os.path.dirname(os.path.join(
    'havingfun/detection/segmentation/saved_imgs/'
    ))
# -----------------------------

# -----------------------------
# the model name
modelname = 'Lightunet18'
# learning rate
lr = '2.22e3'
# traning epochs
epochs = '20'

# strings: the name of files to save, which includes:
#   process_model_param
#   entire_model_param
#   loss statisctics
#   pixel level accuracy statistics
#   (other metrics statistics) 
#   prediction example image
process_model_param = 'process_' + modelname + '_' + lr + '_' + epochs + '.pth'
model_param = modelname + '_' + lr + '_' + epochs + '.pth'
loss_imgs = 'Loss_'+ modelname + '_' + lr + '_' + epochs +'.png'
acc_imgs = 'Acc_' + modelname + '_' + lr + '_' + epochs +'.png'
show_imgs = 'Show_' + modelname + '_' + lr + '_' + epochs +'.png'
# -----------------------------

# -----------------------------
# evaluation raios from sklearn
# pixel-level accuracy
def pixelaccuracy(y_true, y_pred):
    pixelaccuracy = metrics.accuracy_score(y_true, y_pred, normalize = False)
    return pixelaccuracy

# ROC-AUC-score: area under the receiver oprating characteristic curve from prediction scores.
def rocaucscore(y_true, y_pred):
    # for the parameter of multi class:
    # orv: one-vs-rest
    # ovo: one-vs-one
    rocaucscore = metrics.roc_auc_score(y_true, y_pred, average = 'macro', multi_class = 'ovr')
    return rocaucscore

# AP, average-precision-score
# summarize a precision-recall curve as the weighted mean of precision achieved at each threshold
def apscore(y_true, y_pred):
    apscore = metrics.average_precision_score(y_true, y_pred, average = 'macro')
    return apscore

# f1-score: balanced f1-score
# F1 = 2 * (precision * recall) / (precision + recall)
def f1score(y_true, y_pred):
    f1score = metrics.f1_score(y_true, y_pred, average = 'macro')
    return f1score
# -----------------------------

# -----------------------------
# saving results functions
# save the model
# save the model during the training process, just save as stoping it during observation
def save_processing_model(epochs, model, optimizer, criterion):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, os.path.join(root,process_model_param))

# save the entire model, finish all training steps
def save_entire_model(epochs, model, optimizer, criterion):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, os.path.join(root, model_param))

# load the model for testing
def load_model(checkpoint, model):
    print('======> Loading checkpoint')
    model.load_state_dict(checkpoint['model_state_dict'])
# -----------------------------

# -----------------------------
# compute accuracy
# segmentation codes
codes = ['Target', 'Void']
num_classes = 2
name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Void']
       
def save_predictions_as_imgs(dataloader, model, folder = root, device = 'cuda'):
    print('===========> saving prediction')
    for idx, (x, y) in enumerate(dataloader):
        x = x.to(device = device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, 
            os.path.join(root, 'seg_result.png'),
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f'{folder}{idx}.png')

    model.train()

# -----------------------------
# save the accuracy and loss figures
def save_training_plots(train_acc, val_acc, train_loss, val_loss):
    print(f'====> Saving processing ratios')
    plt.figure(figsize = (10, 7))
    plt.plot(
        train_acc, color = 'green', linestyle = '-', label = 'Train Accuracy'
    )
    plt.plot(
        val_acc, color = 'purple', linestyle = '-', label = 'Validation Accuracy'
    )
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Segmentation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(root, acc_imgs))

    plt.figure(figsize = (10, 7))
    plt.plot(
        train_loss, color = 'orange', linestyle = '-', label = 'Training Loss'
    )
    plt.plot(
        val_loss, color = 'blue', linestyle = '-', label = 'Validation Loss'
    )
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Segmentation Loss')
    plt.legend()
    
    plt.savefig(os.path.join(root, loss_imgs))

def plot_img_and_mask(img, pred, mask):
    print('=====> Saving prediction result')
    fig, ax = plt.subplots(3, 1)
    # plt.grid = False 
    # plt.xticks([]), plt.yticks([])

    fig = plt.figure()
    fig.set_size_inches(50,20)
    ax1 = fig.add_subplot(131)
    ax1.grid(False)
    ax1.set_title('Input Image')
    ax1.imshow(img)

    ax2 = fig.add_subplot(132)
    ax2.grid(False)
    ax2.set_title('Output Prediction')
    ax2.imshow(pred)
     
    ax3 = fig.add_subplot(133)
    ax3.grid(False)
    ax3.set_title('Target Mask')
    ax3.imshow(mask)

    plt.savefig(os.path.join(root, show_imgs))
# -----------------------------
# test whether the functions in training work
# if __name__ == '__main__':
    # save_model()
    # load_model()
    # save_model()
    # check_accuracy()
    # save_predictions_as_imgs()
    # save_plots()
    
