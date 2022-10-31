import numpy  as np
import pandas as pd
import torch
import h5py

class DataGen(torch.utils.data.Dataset):
    def __init__(self, filename, datafile, nevents=None, augmentation = False):
        """ This class yields events from pregenerated MC file.
        Parameters:
            filename : str; filename to read
            table_name : str; name of the table to read
                         currently available BinClassHits and SegClassHits
        """
        self.csv_info = pd.read_csv(filename, skiprows=1 , header=None)
        self.groupname = np.asarray(self.csv_info.iloc[:,0])
        self.datainfo = np.asarray(self.csv_info.iloc[:,1])
        self.h5file = h5py.File(datafile, 'r')
        self.augmentation = augmentation

    def __getitem__(self, idx):
        dset_entry = self.h5file[self.groupname[idx]][self.datainfo[idx]]
        eventtype = dset_entry.attrs[u'tag']
        label = 1 
        if eventtype == 'gamma':
            label = 0 
        return (dset_entry[:, 0]).astype(int) + 120, (dset_entry[:, 1]).astype(int) + 120, (dset_entry[:, 2]).astype(int) + 20, dset_entry[:, 3]/10000, np.unique([label]), idx

    def __len__(self):
        return len(self.datainfo)
    def __del__(self):
        if self.h5file is not None:
            self.h5file.close()
        if self.csv_info is not None:
            del self.csv_info

def collatefn(batch):
    coords = []
    energs = []
    labels = []
    events = np.zeros(len(batch))
    for bid, data in enumerate(batch):
        x, y, z, E, lab, event = data
        batchid = np.ones_like(x)*bid
        coords.append(np.concatenate([x[:, None], y[:, None], z[:, None], batchid[:, None]], axis=1))
        energs.append(E)
        labels.append(lab)
        events[bid] = event

    coords = torch.tensor(np.concatenate(coords, axis=0), dtype = torch.short)
    energs = torch.tensor(np.concatenate(energs, axis=0), dtype = torch.float).unsqueeze(-1)
    labels = torch.tensor(np.concatenate(labels, axis=0), dtype = torch.long)

    return  coords, energs, labels, events
