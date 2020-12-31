#!/usr/bin/env python
'''Reads NeXus HDF5 files using h5py and prints the contents'''

import h5py    # HDF5 support
import glob
import time
import csv
def h5merger(filedir, csvfile, outfile):
    filelist = glob.glob('%s/*.h5' % filedir)
    csvfile = open(csvfile, 'w')
    fieldnames = ['groupname', 'dsetname']
    writer = csv.DictWriter(csvfile, fieldnames)
    with h5py.File(outfile, 'w') as fid:
        nexodata = fid.create_group(u'nexo_data' )
        for i in range(len(filelist)):
            fileName = filelist[i]
            print(fileName, time.time() - t1)
            f = h5py.File(fileName,  "r")
            f.copy(f['bb0n.tar'], nexodata, name='nexo_data_%d' % i)
            dset = f['bb0n.tar']
            for item in dset.keys():
                writer.writerow({'groupname':'nexo_data_%d' % i, 'dsetname':item})
            f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset builder.')
    parser.add_argument('--filedir', '-f', type=str, help='directory of h5 files.')
    parser.add_argument('--outfile', '-o', type=str, help='output h5 file.')
    parser.add_argument('--csvfile', '-c', type=str, help='csv file of dataset info.')
    args = parser.parse_args()
    (args.filedir, args.csvfile, args.outfile)
