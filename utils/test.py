import numpy as np
import torch
import sys
import sparseconvnet as scn
from data_loaders import DataGen, collatefn
from torch.utils.tensorboard import SummaryWriter
import time

if __name__ == '__main__':
    train_gen = DataGen('/scratch/zpli/nexo_train.csv', '/scratch/zpli/nexo.h5')
    start = time.time()
    loader_train = torch.utils.data.DataLoader(train_gen,
                                               batch_size = 100,
                                               shuffle = False,
                                               num_workers = 1,
                                               collate_fn = collatefn,
                                               drop_last = True,
                                               pin_memory = False)
    for batchid, (coord, ener, label, event) in enumerate(loader_train):
        print(len(event), time.time() - start)
