import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import os
root = os.path.join('giao/havingfun/detection/segmentation/saved_imgs')

def save_model(epochs, model, optimizer, loss_fn):
    print('====> Saving model')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_fn,
    },
    root/'Lightuent18S_1e5_e18.pth')

def load_model(checkpoint, model):
    print('====> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])

def check_accuracy(loader, model, device = 'cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds * y).sum())/(
                (preds + y).sum() + 1e-8
            )

    print(f'Got {num_correct}/{num_pixels} with acc: {num_correct/num_correct * 100:.2f}')
    print(f'Got dice score of: {dice_score/len(loader)}')
    model.train()

def save_predictions_as_imgs(loader, model, folder = 'saved_imgs', device = 'cuda'):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device = device)
        with torch.no_gard():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f'{folder}/pred_{idx}.png'
        )
        torchvision.utils.save_image(y.unsqueeze(1), f'{folder}{idx}.png')

    model.train()

def save_plots(train_acc, val_acc, train_loss, val_loss):
    print(f'====> Saving processing results')
    plt.figure(figsize = (10, 7))
    plt.plot(
        train_acc, color = 'green', linestyle = '-', label = 'Train accuracy'
    )
    plt.plot(
        val_acc, color = 'blue', linestyle = '-', label = 'Validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Segmentation Accuracy')
    plt.legend()
    plt.savefig(root/'Acc_Lightunet18S_1e5_e18.png')

    plt.figure(figsize = (10, 7))
    plt.plot(
        train_loss, color = 'orange', linestyle = '-', label = 'Train loss'
    )
    plt.plot(
        val_loss, color = 'red', linestyle = '-', label = 'Validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Segmentation Loss')
    plt.legend()
    plt.savefig(root/'Loss_Lightunet18_1e5_e18.png')

def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes +1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Ouput mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.savefig(root/'seg_result.png')

if __name__ == '__main__':
    # save_model()
    load_model()
    save_model()
    check_accuracy()
    save_predictions_as_imgs()
    save_plots()
    