#!/usr/bin/env python
"""
This is a main script to perform training or prediction of network on provided data.
To be called as main_sparseconvnet.py -conf conf_filename -a train/predict
"""
import os
import torch
from argparse     import ArgumentParser
from argparse     import Namespace

import numpy  as np
import pandas as pd
import tables as tb
import yaml 

from networks.sparseresnet import ResNet

from utils.train_utils      import train_net
from utils.train_utils      import predict_gen
import time
start = time.time()

def is_valid_action(parser, arg):
    if not arg in ['train', 'predict']:
        parser.error("The action %s is not allowed!" % arg)
    else:
        return arg

def is_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    return arg


def get_params(confname):
    full_file_name = os.path.expandvars(confname)
    parameters = {}

    builtins = __builtins__.__dict__.copy()

    with open(full_file_name, 'r') as config_file:
        exec(config_file.read(), {'__builtins__':builtins}, parameters)
    return Namespace(**parameters)


if __name__ == '__main__':
    parser = ArgumentParser(description="parameters for models")
    parser.add_argument("-conf", dest = "confname", required=True,
                        help = "input file with parameters", metavar="FILE",
                        type = lambda x: is_file(parser, x))
    parser.add_argument("-a", dest = "action" , required = True,
                        help = "action to do for NN",
                        type = lambda x : is_valid_action(parser, x))
    args     = parser.parse_args()
    confname = args.confname
    action   = args.action
    parameters = get_params(confname)

    net = ResNet(parameters.spatial_size,
                     parameters.init_conv_nplanes,
                     parameters.init_conv_kernel,
                     parameters.kernel_sizes,
                     parameters.stride_sizes,
                     parameters.basic_num,
                     momentum = parameters.momentum,
                     nlinear = parameters.nlinear)
    net = net.cuda()

    print('net constructed', time.time() - start)
    
    loss = np.inf
    if parameters.saved_weights:
        dct_weights = torch.load(parameters.saved_weights)['state_dict']
        loss = torch.load(parameters.saved_weights)['loss']
        net.load_state_dict(dct_weights, strict=False)
        print('weights loaded')

        if parameters.freeze_weights:
            for name, param in net.named_parameters():
                if name in dct_weights:
                    param.requires_grad = False


    if action == 'train':
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                     lr = parameters.lr,
                                     betas = parameters.betas,
                                     eps = parameters.eps,
                                     weight_decay = parameters.weight_decay)
	if parameters.saved_weights:
		saved_optimizer = torch.load(parameters.saved_weights)['optimizer']
		optimizer.load_state_dict(saved_optimizer)

        train_net(nepoch = parameters.nepoch,
                  train_data_path = parameters.train_file,
                  valid_data_path = parameters.valid_file,
                  train_batch_size = parameters.train_batch,
                  valid_batch_size = parameters.valid_batch,
                  net = net,
                  datafile = parameters.datafile,
                  criterion = criterion,
                  optimizer = optimizer,
                  checkpoint_dir = parameters.checkpoint_dir,
                  num_workers = parameters.num_workers,
		  start_loss = loss,
                  nevents_train = parameters.nevents_train,
                  nevents_valid = parameters.nevents_valid,
                  )

    if action == 'predict':
        gen = predict_gen(data_path = parameters.predict_file,
                          datafile = parameters.datafile,
                          net = net,
                          batch_size = parameters.predict_batch,
                          nevents = parameters.nevents_predict)
        coorname = ['xbin', 'ybin', 'zbin']
        output_name = parameters.out_file

        tname = 'EventPred'
        with tb.open_file(output_name, 'w') as h5out:
            for dct in gen:
                if 'coords' in dct:
                    coords = dct.pop('coords')
                    #unpack coords and add them to dictionary
                    dct.update({coorname[i]:coords[:, i] for i in range(3)})
                predictions = dct.pop('predictions')
                #unpack predictions and add them back to dictionary
                dct.update({f'class_{i}':predictions[:, i] for i in range(predictions.shape[1])})

                #create pandas dataframe and save to output file
                df = pd.DataFrame(dct)
                df.to_hdf(h5out) 
