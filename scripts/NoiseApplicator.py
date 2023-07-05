# %%
#!/usr/bin/env python
# coding: utf-8

#Dataset code copied from https://github.com/utkuozbulak/pytorch-custom-dataset-examples
#model code copied from https://github.com/DeepLearnPhysics/pytorch-uresnet

import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import os

import torch,h5py
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

import argparse
#from networks.preact_resnet import PreActResNet18
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from networks.resnet_example import resnet18
from utils.data_loaders import DenseDataset
from utils.data_loaders import DatasetFromSparseMatrix
import traceback
import yaml 

def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param

#%%
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'
# list_of_gpus = range(torch.cuda.device_count())
# device_ids = range(torch.cuda.device_count())
# os.environ["CUDA_VISIBLE_DEVICES"] = str(list_of_gpus).strip('[]').replace(' ', '')

# best_acc = 0  # best test accuracy
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# epochs = 200

# def adjust_learning_rate(optimizer, epoch, lr):
#     """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
#     lr = lr/np.exp(epoch/10)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# # Training
# def train(trainloader, epoch):
#     # print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         # print(inputs.shape)
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#         print(batch_idx, '/', len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
#     return train_loss/len(trainloader), 100.*correct/total

# def test(testloader, epoch, saveall=False):
    
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     score = []
    
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#             softmax = nn.Softmax(dim=0)
#             for m in range(outputs.size(0)):
#                 score.append([softmax(outputs[m])[1].item(), targets[m].item()])
#                 # score.append([outputs[m][1].item(), targets[m].item()])
#             print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#     # Save checkpoint.
#     is_better = False
#     acc = 100.*correct/total
    
#     # If we want to save all training records
#     if saveall:
#         print ('Saving...')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint_sens'):
#             os.mkdir('checkpoint_sens' )
#         if not os.path.isdir('training_outputs'):
#             os.mkdir('training_outputs')
#         torch.save(state, './checkpoint_sens/ckpt_%d.t7' % epoch)
#         torch.save(state, './checkpoint_sens/ckpt.t7' )
#         best_acc = acc
#     # Otherwise only save the best one
#     elif acc > best_acc:
#         print ('Saving...')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint_sens'):
#             os.mkdir('checkpoint_sens' )
#         if not os.path.isdir('training_outputs'):
#             os.mkdir('training_outputs')
#         torch.save(state, './checkpoint_sens/ckpt_%d.t7' % epoch)
#         torch.save(state, './checkpoint_sens/ckpt.t7' )
#         best_acc = acc
        
#     return test_loss/len(testloader), 100.*correct/total, score

#%%
# parser = argparse.ArgumentParser(description='PyTorch nEXO background rejection')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--config', '-f', type=str, default="baseline.yml", help="specify yaml config")

# args = parser.parse_args()

resume = False
config = '/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/config/noise_test.yml'

# parameters
config = yaml_load(config)
data_name = config['data']['name']
h5_path = config['data']['h5name']
fcsv = config['data']['csv']
input_shape = [int(i) for i in config['data']['input_shape']]
lr = config['fit']['compile']['initial_lr']
batch_size = config['fit']['batch_size']
epochs = config['fit']['epochs']

#%%
# h5file = h5py.File(h5_path, 'r')
# h5file2=[h5file]


# #%%

# # Data
# nEXODataset1 = NoisedDatasetFromSparseMatrix(h5_path, fcsv, n_channels=input_shape[2])
# print(len(nEXODataset1))
# print(nEXODataset1[129753][1])
# nEXODataset1.mislabeled_indices = [129753]
# print(nEXODataset1[129753][1])


# # print(nEXODataset1[129752][1])
# # nEXODataset2 = NoisedDatasetFromSparseMatrix(h5_path, fcsv, n_channels=input_shape[2],seed=1,noise_amplitude=1)
# # nEXODataset2 = NoisedDatasetFromSparseMatrix(h5_path, fcsv, n_channels=input_shape[2],seed=1,noise_amplitude=0)
# # print(1)

#%%
# noise plots (subplots only 1/25 waves shown)
from utils.data_loaders import NoisedDatasetFromSparseMatrix,DatasetFromSparseMatrix
fig=plt.figure(figsize=(50,5))
nEXODataset = NoisedDatasetFromSparseMatrix(h5_path, fcsv, n_channels=input_shape[2],noise_amplitude=0,restore_quiet=True)
sq1 = np.random.SeedSequence()
sq2 = np.random.SeedSequence()
seed1 = sq1.entropy
seed2 = sq2.entropy
dataset_size = len(nEXODataset)
print('seed1:', seed1,'seed2:', seed2)
if 0: # determines whether to reseed noise each time (should NOT bc that would look bad probably?)
    nEXODataset.seed_list1 = sq1.generate_state(dataset_size)
# DO FIX THE RENOISE SEED
nEXODataset.seed_list2 = sq2.generate_state(dataset_size)
j = 0
for noise_amplitude in [0,.5,1,2,4]:
    plt.subplot(1,5,j+1)
    nEXODataset.seed_list1 = sq1.generate_state(dataset_size)
    nEXODataset.noise_amplitude = noise_amplitude
    n1 = nEXODataset[0]
    n1 = n1[0]
    n1 = np.array(n1)
    n1 = n1[0][::25]
    k = 0
    for n in n1:
        plt.plot(np.arange(255),n+k*400)#,label=noise_amplitude)
        k += 1
    j+=1
    plt.yticks([])
    plt.xlabel('time (not in linear units)',fontsize=20)
    plt.ylabel('Waveform amplitude',fontsize=20)
    plt.title('Excess Noise = %g'%noise_amplitude,fontsize=24)
# plt.legend()
plt.show()
#%%
# plots of all waveforms (x and y)
from utils.data_loaders import NoisedDatasetFromSparseMatrix,DatasetFromSparseMatrix
fig=plt.figure(figsize=(20,15))
nEXODataset = NoisedDatasetFromSparseMatrix(h5_path, fcsv, n_channels=input_shape[2],noise_amplitude=0,restore_quiet=True)
sq1 = np.random.SeedSequence()
sq2 = np.random.SeedSequence()
seed1 = sq1.entropy
seed2 = sq2.entropy
dataset_size = len(nEXODataset)
print('seed1:', seed1,'seed2:', seed2)
if 0: # determines whether to reseed noise each time (should NOT bc that would look bad probably?)
    nEXODataset.seed_list1 = sq1.generate_state(dataset_size)
# DO FIX THE RENOISE SEED
nEXODataset.seed_list2 = sq2.generate_state(dataset_size)
j = 0
for noise_amplitude in [0]:#,.5,1,2,4]:
    # plt.subplot(1,5,j+1)
    nEXODataset.seed_list1 = sq1.generate_state(dataset_size)
    nEXODataset.noise_amplitude = noise_amplitude
    n1 = nEXODataset[200001]
    print(n1[1])
    n1 = n1[0]
    n1 = np.array(n1)
    # n1 = n1[0][::]
    k = 0
    plt.subplot(121)
    plt.title('X-channels',fontsize=24)
    for n2 in n1:
        for n in n2:
            plt.plot(np.arange(255),n+k*400)#,label=noise_amplitude)
            k += 1
        # plt.yticks([])
        plt.yticks([])
        plt.xlabel('time (not in linear units)',fontsize=20)
        plt.ylabel('Waveform amplitude',fontsize=20)
        plt.subplot(122)
        plt.title('Y-channels',fontsize=24)
    j+=1
    # plt.title('Excess Noise = %g'%noise_amplitude,fontsize=24)
# plt.legend()
plt.suptitle('Gamma Event',fontsize=28)
plt.show()

#%%


nEXODataset1 = DatasetFromSparseMatrix(h5_path, fcsv, n_channels=input_shape[2])
nEXODataset2 = NoisedDatasetFromSparseMatrix(h5_path, fcsv, n_channels=input_shape[2],noise_amplitude=1,restore_quiet=False)
nEXODataset3 = NoisedDatasetFromSparseMatrix(h5_path, fcsv, n_channels=input_shape[2],noise_amplitude=1,restore_quiet=True)
random_seed = 40
sq1 = np.random.SeedSequence()
sq2 = np.random.SeedSequence()
seed1 = sq1.entropy
seed2 = sq2.entropy
print('random_seed:',random_seed,'seed1:', seed1,'seed2:', seed2)
dataset_size = len(nEXODataset2)
# if not args.reseed_noise:
nEXODataset2.seed_list1 = sq1.generate_state(dataset_size)
nEXODataset3.seed_list1 = sq1.generate_state(dataset_size)
# if args.restore_quiet and not args.reseed_quiet:
nEXODataset2.seed_list2 = sq2.generate_state(dataset_size)
nEXODataset3.seed_list2 = sq2.generate_state(dataset_size)
# nEXODataset2 = NoisedDatasetFromSparseMatrix(h5_path, fcsv, n_channels=input_shape[2],seed=1,noise_amplitude=1)
# nEXODataset3 = NoisedDatasetFromSparseMatrix(h5_path, fcsv, n_channels=input_shape[2],seed=2,noise_amplitude=1)
n1 = nEXODataset1[0]
# print(n1[0].shape)
n1=n1[0]
n1=np.array(n1)
# print(np.all(np.any(n1,(1,0))==(np.any(n1,(0,1)))))
n2 = nEXODataset2[0][0]
n3 = nEXODataset3[0][0]
# print(n1.shape)
# # -print(n1)
# n10,n11 = n1
# n10,n11=np.array(n10),np.array(n11)
# print(n10.shape)
n2=np.array(n2)
n3=np.array(n3)
# data=np.array(np.any(n1,2))
# -print(data.shape)
fig=plt.figure(figsize=(50,15))
# plt.subplot(121)
# [plt.plot(np.arange(255),n1[0,i]+10*i)for i in range(z0,z1,1)]
# plt.subplot(122)
# [plt.plot(np.arange(255),n1[1,i]+10*i)for i in range(z0,z1,1)]
# print(n1[data].shape)
# n1[data]+=np.random.normal(0,100,255)
# plt.subplot(121)
# active_channels = np.any(n1,2,keepdims=1)
# active_times = np.any(n1,(0,1),keepdims=1)
# print(n1[0][121])
# print(n1[np.any(n1,2)][:,np.any(n1,(0,1))].shape)
n1x=n1[1][97:]
# n2=n2[0]
# n1x = n1x[np.any(np.abs(n1x),1)]
# n2 = n2[np.any(np.abs(n2),1)]
# n1=n1[1]
# n2=n2[1]
# plt.plot(np.arange(255),n2)
plt.plot(np.arange(255),n1x[1])
# n2=n2[np.abs(np.arange(255)-130)<55]
n1x=n1x[1][np.abs(np.arange(255)-170)<55]
# n1x=n1x[1][np.abs(np.arange(255)-60)<55]
# n2=n2[np.abs(np.arange(255)-60)<55]
print('1 np.mean:',np.mean(n1x))
print('1 np.std:',np.std(n1x))
# print('2 np.mean:',np.mean(n2))
# print('2 np.std:',np.std(n2))
plt.plot(np.arange(109)+116,n1x)
# plt.plot(np.arange(109)+76,n2)
# plt.plot(np.arange(109)+6,n1x)
# plt.plot(np.arange(109)+6,n2)
plt.show()
#%%
fig=plt.figure(figsize=(50,15))
x=7500
y=100
z0=0
z1=200
z=1
plt.subplot(121)
[plt.plot(np.arange(255),n1[0,i]+y*i)for i in range(z0,z1,z)if np.any(n1[0,i])]
# plt.subplot(121)
# [plt.plot(np.arange(255),n2[0,i]+y*i+x)for i in range(z0,z1,z)if np.any(n2[0,i])]
# plt.ylim(3400,4500)
# [plt.plot(np.arange(255),n3[0,i]+y*i+x*2)for i in range(z0,z1,z)if np.any(n3[0,i])]
# plt.subplot(122)
# [plt.plot(np.arange(255),n1[1,i]+y*i)for i in range(z0,z1,z)if np.any(n1[1,i])]
# plt.subplot(122)
# [plt.plot(np.arange(255),n2[1,i]+y*i+x)for i in range(z0,z1,z)if np.any(n2[1,i])]
# plt.ylim(2500,6000)
# [plt.plot(np.arange(255),n3[1,i]+y*i+x*2)for i in range(z0,z1,z)if np.any(n3[1,i])]
plt.show()
# %%
