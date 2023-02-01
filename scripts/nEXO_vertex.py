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
import sys
sys.path.append("..")

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
from networks.resnet_regression import resnet18
from utils.data_loaders import VertexDataset
from utils.data_loaders import DatasetFromSparseMatrix
import traceback
#import matplotlib.pyplot as plt
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
    # lr = lr/exp10(epoch/10)
    lr = lr/np.exp(epoch/10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cropandflip(npimg2d):
    imgpad  = np.zeros( (200,255), dtype=np.float32 )
    transformimg = np.zeros( (200,255, 3), dtype=np.float32)
    flip1 = np.random.rand()
    flip2 = np.random.rand()
    for i in range(3):
        imgpad[:,:] = npimg2d[ :,:,i]
        #if flip1>0.5:
        #    imgpad = np.flip( imgpad, 0 )
        #if flip2>0.5:
        #    imgpad = np.flip( imgpad, 1 )
        transformimg[:,:,i] = imgpad[:,:]
    return transformimg

# Training
def train(trainloader, epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    #correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        #_, predicted = outputs
        #total += targets.size(0)
        #correct += predicted.eq(targets).sum().item()

        #print(targets[:10,:], outputs[:10,:])
        print(batch_idx, '/', len(trainloader), 'Loss: %.3f ' % (train_loss/(batch_idx+1.)))
        #print(batch_idx, '/', len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/len(trainloader) #, 100.*correct/total

def test(testloader, epoch, saveall=False):
    
    global best_acc
    net.eval()
    test_loss = 0
    #correct = 0
    total = 0
    score = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            print(batch_idx, '/', len(testloader), 'Loss: %.3f ' % (test_loss/(batch_idx+1.)))
            #_, predicted = outputs.max(1)
            #total += targets.size(0)
            #correct += predicted.eq(targets).sum().item()
            #softmax = nn.Softmax(dim=0)
            #for m in range(outputs.size(0)):
            #    score.append([softmax(outputs[m])[1].item(), targets[m].item()])
                # score.append([outputs[m][1].item(), targets[m].item()])
            #print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    is_better = False
    #acc = 100.*correct/total
    
    # If we want to save all training records
    if saveall:
        print ('Saving...')
        state = {
            'net': net.state_dict(),
            #'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_sens'):
            os.mkdir('checkpoint_sens' )
        if not os.path.isdir('training_outputs'):
            os.mkdir('training_outputs')
        torch.save(state, './checkpoint_sens/ckpt_%d.t7' % epoch)
        torch.save(state, './checkpoint_sens/ckpt.t7' )
        best_acc = acc
    # Otherwise only save the best one
    #elif acc > best_acc:
    #    is_better = True
    #    print ('Saving...')
    #    state = {
    #        'net': net.state_dict(),
    #        'acc': acc,
    #        'epoch': epoch,
    #    }
    #    if not os.path.isdir('checkpoint_sens'):
    #        os.mkdir('checkpoint_sens' )
    #    if not os.path.isdir('training_outputs'):
    #        os.mkdir('training_outputs')
    #    torch.save(state, './checkpoint_sens/ckpt_%d.t7' % epoch)
    #    torch.save(state, './checkpoint_sens/ckpt.t7' )
    #    best_acc = acc
        
    return test_loss/len(testloader) #, 100.*correct/total, score, is_better

def TagEvent(event):
    
    global best_acc
    net.eval()
    npimg = load(event , allow_pickle=True).astype(np.float32)
    # npimg = (npimg - npimg.min())/(npimg.max() - npimg.min())
    # Transform image to tensor
    img_as_tensor = self.to_tensor(npimg).type(torch.FloatTensor)
    img_as_tensor = torch.unsqueeze(img_as_tensor, 0)
    softmax = nn.Softmax()
    
    with torch.no_grad():
        img_as_tensor = img_as_tensor.to(device)
        output = net(img_as_tensor)
        # print(output.shape)
        # print(softmax(output)[0][1].item())
        return softmax(output[0])[1].item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch nEXO background rejection')
    parser.add_argument('--lr', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--save_all', action='store_true', default=False, help='save all training records')
    parser.add_argument('--tag', '-t', action='store_true', default=False, help='tag event with trained network')
    parser.add_argument('--channels', '-n', type=int, default=2, help='Input Channels')
    parser.add_argument('--start', '-s', type=int, default=0, help='start epoch')
    parser.add_argument('--csv', '-c', type=str, default='test.csv', help='csv files of training sample')
    parser.add_argument('--h5file', '-d', type=str, default='test.h5', help='h5 files of training sample')

    args = parser.parse_args()
    transformations = transforms.Compose([transforms.ToTensor()])
    
    # Data
    print('==> Preparing data..')
    nEXODataset = VertexDataset(args.h5file, args.csv, n_channels=args.channels)
    # Creating data indices for training and validation splits:
    dataset_size = len(nEXODataset)
    indices = list(range(dataset_size))
    validation_split = .2
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True
    random_seed= 40
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    batch_size = 256
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(nEXODataset, batch_size=batch_size, sampler=validation_sampler, num_workers=0)

    lr = args.lr
    momentum = 0.9
    # Weight_decay = 1.0e-3
    weight_decay = 5.0e-3
    batchsize_valid = 500
    start_epoch = 0
    epochs      = 12

    print('==> Building model..')
    # net = preact_resnet.PreActResNet18(num_channels=args.channels)
    net = resnet18(input_channels=args.channels)
    # Define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()
    # We use SGD
    # optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), 
                                  weight_decay=1e-4, eps=1e-08, amsgrad=False)

    net = net.to(device)
    
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
        
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
        import glob
        files = glob.glob('tl208/*')
        tags_bb0n = []
        tagresult = open('quicktest.txt', 'w')
        for exfile in files:
            extag = TagEvent(exfile)
            tagresult.write("%s %f\n" % (exfile, extag))
            tags_bb0n.append(extag)

        # plt.hist(np.array(tags_bb0n), bins = np.linspace(0, 1, 100))
        # plt.savefig('gamma_tag.pdf')

    # Numpy arrays for loss and accuracy, if resume from check point then read the previous results
    if args.resume and os.path.exists('./training_outputs/loss_acc.npy'):
        arrays_resumed = np.load('./training_outputs/loss_acc.npy', allow_pickle=True)
        y_train_loss = arrays_resumed[0]
        y_train_acc  = arrays_resumed[1]
        y_valid_loss = arrays_resumed[2]
        y_valid_acc  = arrays_resumed[3]
        test_score   = arrays_resumed[4].tolist()
    else:
        y_train_loss = np.array([])
        y_train_acc  = np.array([])
        y_valid_loss = np.array([])
        y_valid_acc  = np.array([])
        test_score   = []
    
    for epoch in range(start_epoch, start_epoch + epochs):
        # Set the learning rate
        adjust_learning_rate(optimizer, epoch, lr)
        iterout = "\nEpoch [%d]: "%(epoch)
        
        for param_group in optimizer.param_groups:
            iterout += "lr=%.3e"%(param_group['lr'])
            print(iterout)
            try:
                train_ave_loss = train(train_loader, epoch)
            except Exception as e:
                print("Error in training routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
            #print("Train[%d]: Result* Loss %.3f\t Accuracy: %.3f"%(epoch, train_ave_loss, train_ave_acc))
            y_train_loss = np.append(y_train_loss, train_ave_loss)
            #y_train_acc = np.append(y_train_acc, train_ave_acc)

            # Evaluate on validationset
            try:
                valid_loss = test(validation_loader, epoch, args.save_all)
            except Exception as e:
                print("Error in validation routine!")
                print(e.message)
                print(e.__class__.__name__)
                traceback.print_exc(e)
                break
                
            #print("Test[%d]: Result* Loss %.3f\t Precision: %.3f"%(epoch, valid_loss, prec1))
            
            #test_score.append(score)
            y_valid_loss = np.append(y_valid_loss, valid_loss)
            #y_valid_acc = np.append(y_valid_acc, prec1)
            
            # If we want to save all training records
            #if args.save_all:
            np.save('./training_outputs/loss_acc.npy', np.array([y_train_loss, y_train_acc, y_valid_loss, y_valid_acc, test_score], dtype=object))
            # Otherwise only save the best one
            #elif is_better:
            #    # print (np.array(test_score).shape)
            #    np.save('./training_outputs/loss_acc.npy', np.array([y_train_loss, y_train_acc, y_valid_loss, y_valid_acc, test_score], dtype=object))
        
    # print(y_train_loss, y_train_acc, y_valid_loss, y_valid_acc)
    # np.save('test_score_%d.npy' % (start_epoch + 1), test_score)