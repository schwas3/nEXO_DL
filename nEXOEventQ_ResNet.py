#!/usr/bin/env python
# coding: utf-8

#Dataset code copied from https://github.com/utkuozbulak/pytorch-custom-dataset-examples
#model code copied from https://github.com/DeepLearnPhysics/pytorch-uresnet

import pandas as pd
import numpy as np
from numpy import load
import scipy.misc
from scipy.special import exp10
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data.sampler import SubsetRandomSampler
import os
import shutil

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import argparse
import resnet_regression
import traceback
import matplotlib.pyplot as plt
import pickle

#device = 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
list_of_gpus = range(torch.cuda.device_count())
device_ids = range(torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = str(list_of_gpus).strip('[]').replace(' ', '')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epochs = 200

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = lr/exp10(epoch/20)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class nEXODatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
	    # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        npimg = load(single_image_name, allow_pickle=True).astype(np.float32)[:224, 30:254,:]
        # Transform image to tensor.
        img_as_tensor = self.to_tensor(npimg).type(torch.FloatTensor)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

# Training
def train(trainloader, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_acc = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        targets = targets.view(-1, 1)
        for m in range(outputs.size(0)):
            if np.absolute(outputs[m].item() -  targets[m].item())/targets[m].item() < 0.02:
                train_acc += 1
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f'
            % (train_loss/(batch_idx+1), train_acc/total) )
    return train_loss/len(trainloader)

def test(testloader, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    test_acc = 0
    total = 0
    score = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            targets = targets.view(-1, 1)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            total += targets.size(0)
            for m in range(outputs.size(0)):
                score.append([outputs[m].item(), targets[m].item()])
                if np.absolute(outputs[m].item() -  targets[m].item())/targets[m].item() < 0.02:
                    test_acc += 1
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f '
                % (test_loss/(batch_idx+1), test_acc/total))

    acc = test_acc/total
    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_sens' ):
            os.mkdir('checkpoint_sens' )
        torch.save(state, './checkpoint_sens/ckpt_%d.t7' % epoch)
        torch.save(state, './checkpoint_sens/ckpt.t7' )
        best_acc = acc
    return test_loss/len(testloader), score

def TagEvent(event):
    net.eval()
    npimg = load(event , allow_pickle=True).astype(np.float32)
    # Transform image to tensor
    img_as_tensor = self.to_tensor(npimg).type(torch.FloatTensor)
    img_as_tensor = torch.unsqueeze(img_as_tensor, 0)
    with torch.no_grad():
        img_as_tensor = img_as_tensor.to(device)
        output = net(img_as_tensor)
        return output[0].item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch nEXO background rejection')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--tag', '-t', action='store_true', default=False, help='tag event with trained network')
    parser.add_argument('--channels', '-p', type=int, default=2, help='Input Channels')
    parser.add_argument('--start', '-s', type=int, default=0, help='start epoch')
    parser.add_argument('--csv', '-c', type=str, default='image2dcharge_sens.csv', help='csv files of training sample')

    args = parser.parse_args()
    transformations = transforms.Compose([transforms.ToTensor()])
    # Data
    print('==> Preparing data..')
    nEXODataset = nEXODatasetFromImages(args.csv)
    # Creating data indices for training and validation splits:
    dataset_size = len(nEXODataset)
    indices = list(range(dataset_size))
    validation_split = .15
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True
    random_seed= 42
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=400, sampler=train_sampler, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=400, sampler=validation_sampler, num_workers=4)

    lr = 1.0e-3
    momentum = 0.9
    weight_decay = 1.0e-3
    batchsize = 50
    batchsize_valid = 500
    start_epoch = 0
    epochs      = 100

    print('==> Building model..')
    net = resnet_regression.resnet18(pretrained=False, input_channels=args.channels)
    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()
    # We use SGD
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        cudnn.benchmark = True

    net = net.to(device)
    if args.resume and os.path.exists('./checkpoint_sens/ckpt.t7'):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint_sens'), 'Error: no checkpoint directory found!'
        if device == 'cuda':
            checkpoint = torch.load('./checkpoint_sens/ckpt.t7' )
        else:
            checkpoint = torch.load('./checkpoint_sens/ckpt.t7', map_location=torch.device('cpu') )
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
    if args.tag:
        # Use trained network to tag event.
        print('==> Load the network from checkpoint..')
        assert os.path.isdir('checkpoint_sens'), 'Error: no checkpoint directory found!'
        if device == 'cuda':
            checkpoint = torch.load('./checkpoint_sens/ckpt.t7' )
        else:
            checkpoint = torch.load('./checkpoint_sens/ckpt.t7', map_location=torch.device('cpu') )
        net.load_state_dict(checkpoint['net'])

    x = np.linspace(start_epoch,start_epoch + 100,1)
    # numpy arrays for loss and accuracy
    y_train_loss = np.zeros(100)
    y_valid_loss = np.zeros(100)
    test_score = []
    for epoch in range(start_epoch, start_epoch + 4):
        # set the learning rate
        adjust_learning_rate(optimizer, epoch, lr)
        iterout = "Epoch [%d]: "%(epoch)
        for param_group in optimizer.param_groups:
            iterout += "lr=%.3e"%(param_group['lr'])
            print(iterout)
            try:
                train_ave_los = train(train_loader, epoch)
            except Exception as e:
                print("Error in training routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            print("Epoch [%d] train aveloss=%.3f "%(epoch,train_ave_loss))
            y_train_loss[epoch] = train_ave_loss

            # evaluate on validationset
            try:
                valid_loss, score = test(validation_loader, epoch)
            except Exception as e:
                print("Error in validation routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            print("Test[%d]:Result* \tLoss %.3f"%(epoch, valid_loss))
            test_score.append(score)
            y_valid_loss[epoch] = valid_loss
        np.save('test_score_%d.npy' % (start_epoch + 1), test_score)
    print(y_train_loss, y_valid_loss)
