#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: train.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2022-10-28
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: 
#
# ------------------------------------------------------------------------------

import torch.backends.cudnn as cudnn
import torchvision

from model import Net

parser = argparse.ArgumentParser(description = "Train on Market1501")
parser.add_argument("--data-dir", default = r"./yolov5deepsort/Market-1501-v15.09.15/pytorch", typpe = str)
parser.add_argument("--no-cuda", action = "store_true")
parser.add_argument("--gpu-id", default = 0, type = int)
parser.add_argument("--lr", default = 0.0001, type = float)
parser.add_argument("--iterval", '-i', default = 20, type = int)
parser.add_argument("--resume", '-r', action = 'store_true')
args = parser.parse_args()

device = "cuda{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
root = args.data_dir
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding = 4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0,485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform = transform_train),
        batch_size = 64, shuffle = True
        )
testloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(test_dir, transform = transform_test),
        )
num_classes = max(len(trainloader.dataset.dataset.classes), len(testloader.dataset.classes))
print("num_classes = %s" %num_classes)

######
# definition of the net
start_epoch = 0
net = Net(num_classes = num_classes)
if arg.resume:
    assert.os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print(r"Loading from checkpoint/ckp.t7")
    checkpoint = torch.load("./checkpoint/ckp.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

######
# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameter(), args.lr, momentum = 0.9, weight_decay = 5e-4)
best_acc = 0.

def train(epoch):
    print("\nEpoch: %d"%(epoch + 1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0.
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (input, labels) in enumerate(trainloader):
        # forward
        inputs,labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim = 1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        if (idx + 1) % interval == 0:
            end = time.time()

    return train_loss/len(trainloader), 1. - correct/total

def test(epoch):
    global best acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            input, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim = 1)[1].eq(labels).sum().item()
            total += labels.size(0)

        print("testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:5f} Correct: {}/{} Acc:{:.3f}%".format(
            100.*(idx + 1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
            ))

    # saving checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc =acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
                'net_dict':net.state_dict(),
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')
    return test_loss/len(testloader), 1. - correct/total

######
# plot figure
x_epoch = []
record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
fig = plt.figure()
ax0 = fig.add_subplot(121, title = "loss")
ax1 = fig.add_subplot(122, title = "toplerr")
def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label = 'train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label = 'val')
    ax1.plot(x_epoch, record['train-err'], 'bo-', label = 'train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label = 'val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")

######
# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

def main():
    total_epoches = 40
    for epoch in range(start_epoch, start_epoch + total_epoches):
        train_loss, tain_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch + 1) % (total_epoches // 2) == 0:
            lr_decay()


if __name__ == "__main__":
    main()

