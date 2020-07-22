#!/usr/bin/env python
# coding: utf-8

#Dataset code copied from https://github.com/utkuozbulak/pytorch-custom-dataset-examples

import time
import pandas as pd
import numpy as np

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
best_acc = 10000  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
NUM_CLASSES = 1000
NUM_EPOCHS = 100
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cropandflip(npimg2d):
    transformimg = np.zeros( (96, 1024, 3), dtype=np.float32)
    for i in range(3):
        transformimg[24:72,:,i] = npimg2d[:,:,i]
    return transformimg

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
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
        self.classifier = nn.Sequential(
             nn.Dropout(0.5),
             nn.Linear(256 * 6 * 6, 4096),
             nn.ReLU(inplace=True),
             nn.Dropout(0.5),
             nn.Linear(4096, 4096),
             nn.ReLU(inplace=True),
             )

        self.fc = nn.Linear(4096, (self.num_classes-1)*2)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        logits = self.fc(x)
        logits = logits.view(-1, (self.num_classes-1), 2)
        probas = F.softmax(logits, dim=2)[:, :, 1]
        return logits, probas

def cost_fn(logits, levels):
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*levels
            + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels)), dim=1))
    return torch.mean(val)

def compute_mae_and_mse(net, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    for i, (features, targets, levels) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = net(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        print(predicted_labels, targets)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets)**2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse


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
        levels = [1]*single_image_label + [0]*(NUM_CLASSES - 1 - single_image_label)
        levels = torch.tensor(levels, dtype=torch.float32)

        return img_as_tensor, single_image_label, levels

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
    nEXODataset = nEXODatasetFromImages('image2dcharge_sens.csv')

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
    net = AlexNet(NUM_CLASSES)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net = net.to(device)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint_regression/ckpt.t7' )
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        net.train()
        for batch_idx, (features, targets, levels) in enumerate(train_loader):
            features = features.to(device)
            targets = targets
            targets = targets.to(device)
            levels = levels.to(device)

            # FORWARD AND BACK PROP
            logits, probas = net(features)
            cost = cost_fn(logits, levels)
            optimizer.zero_grad()
            cost.backward()
            # UPDATE MODEL PARAMETERS
            optimizer.step()
            # LOGGING
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                    % (epoch+1, NUM_EPOCHS, batch_idx,
                      len(train_loader), cost))
            print(s)

    s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    net.eval()
    with torch.set_grad_enabled(False):  # save memory during inference
        train_mae, train_mse = compute_mae_and_mse(net, train_loader,
                                                  device=device)
        test_mae, test_mse = compute_mae_and_mse(net, validation_loader,
                                                  device=device)

        s = 'MAE/RMSE: | Train: %.2f/%.2f | Test: %.2f/%.2f' % (
            train_mae, torch.sqrt(train_mse), test_mae, torch.sqrt(test_mse))
        print(s)

    s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
    print(s)
