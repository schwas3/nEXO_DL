#!/usr/bin/env python
# coding: utf-8

#Dataset code copied from https://github.com/utkuozbulak/pytorch-custom-dataset-examples

import time
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data.sampler import SubsetRandomSampler

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import argparse

device = 'cuda' #if torch.cuda.is_available() else 'cpu'
best_acc = 0 # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
NUM_EPOCHS = 5
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cropandflip(npimg2d):
    transformimg = np.zeros( (240, 1024, 3), dtype=np.float32)
    for i in range(2):
        transformimg[:,:,i] = npimg2d[:,:,i]
    return transformimg

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=3, stride=2),
             nn.Conv2d(64, 192, kernel_size=5, padding=2),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=3, stride=2),
             nn.Conv2d(192, 384, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(384, 256, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(256, 256, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=3, stride=2),
             )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.regression = nn.Sequential(
             nn.Dropout(0.5),
             nn.Linear(256 * 6 * 6, 4096),
             nn.ReLU(inplace=True),
             nn.Dropout(0.5),
             nn.Linear(4096, 1024),
             nn.ReLU(inplace=True),
             )

        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.regression(x)
        logits = self.fc(x)
        return logits

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
        npimg = np.load(single_image_name, allow_pickle=True).astype(np.float32)
        transformed = cropandflip(npimg)
        # Transform image to tensor
        img_as_tensor = self.to_tensor(transformed)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return img_as_tensor, single_image_label

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch nEXO background rejection')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    transformations = transforms.Compose([transforms.ToTensor()])
    # Data
    print('==> Preparing data..')
    nEXODataset = nEXODatasetFromImages('image2dcharge_eventq.csv')

    # Creating data indices for training and validation splits:
    dataset_size = len(nEXODataset)
    indices = list(range(dataset_size))
    validation_split = .2
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
    train_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=200, sampler=train_sampler, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=200, sampler=validation_sampler, num_workers=4)

    lr = 1.0e-3
    momentum = 0.9
    weight_decay = 1.0e-3
    batchsize = 50
    batchsize_valid = 500
    start_epoch = 0

    print('==> Building model..')
    net = AlexNet()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net = net.to(device)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint_regression'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint_regression/ckpt.t7' )
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        net.train()
        total_loss = 0
        total_acc = 0
        total = 0
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device).view(-1, 1)

            # FORWARD AND BACK PROP
            outputs = net(features)
            if batch_idx == 0:
                print(outputs, targets)
            loss = criterion(outputs, targets)
            total_loss += loss
            for m in range(outputs.size(0)):
                if np.absolute(outputs[m].item() -  targets[m].item())/targets[m].item() < 0.02:
                    total_acc += 1
            optimizer.zero_grad()
            loss.backward()
            # UPDATE MODEL PARAMETERS
            optimizer.step()
            total += targets.size(0)
            # LOGGING
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | Acc: %.3f '
                    % (epoch+1, NUM_EPOCHS, batch_idx,
                      len(train_loader), total_loss/(batch_idx+1), total_acc*1.0/total))
            print(s)

        net.eval()
        with torch.set_grad_enabled(False):  # save memory during inference
            total_loss = 0
            total_acc = 0
            total = 0
            test_targets = []
            test_predicts = []
            for batch_idx, (features, targets) in enumerate(validation_loader):
                features = features.to(device)
                targets = targets.to(device).view(-1, 1)
                outputs = net(features)
                if batch_idx == 0:
                    print(outputs, targets)
                loss = criterion(outputs, targets)
                total_loss += loss
                for m in range(outputs.size(0)):
                    if np.absolute(outputs[m].item() -  targets[m].item())/targets[m].item() < 0.02:
                        total_acc += 1
                test_targets.append(targets)
                test_predicts.append(outputs)
                total += targets.size(0)
                s = ('Test Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | Acc: %.3f '
                    % (epoch+1, NUM_EPOCHS, batch_idx,
                      len(validation_loader), total_loss/(batch_idx+1), total_acc*1.0/total))
                print(s)

            np.save('eventq_predicts_%d.npy' % epoch, np.array(test_predicts) )
            np.save('eventq_targets_%d.npy' % epoch, np.array(test_targets) )

            print(s)
            # Save checkpoint.
            acc = total_acc*1.0/total
            if acc > best_acc:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint_regression' ):
                    os.mkdir('checkpoint_regression' )
                torch.save(state, './checkpoint_regression/ckpt_%d.t7' % epoch)
                torch.save(state, './checkpoint_regression/ckpt.t7' )
                best_acc = acc
    s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
    print(s)
