import numpy as np
import tarfile

def SaveH5(infile, outfile):
    import h5py    # HDF5 support
    import six
    import os, time
    print("Write a HDF5 file")
    fileName = outfile
    timestamp = u'%s' % time.ctime()

    # create the HDF5 file
    f = h5py.File(fileName, "w")
    # point to the default data to be plotted
    f.attrs[u'default']          = u'entry'
    # give the HDF5 root some more attributes
    f.attrs[u'file_name']        = fileName
    f.attrs[u'file_time']        = timestamp
    f.attrs[u'creator']          = u'tarToh5.py'
    f.attrs[u'HDF5_Version']     = six.u(h5py.version.hdf5_version)
    f.attrs[u'h5py_version']     = six.u(h5py.version.version)
    exodata = f.create_group(os.path.basename(infile))
    tar=tarfile.TarFile(infile, 'r')
    for item in tar.getmembers():
        f = tar.extractfile(item)
        event2dimg = np.load(item.name)
        eventtype = -1
        if 'bb0n' in item.name:
            eventtype = 1
        elif 'gamma' in item.name:
            eventtype = 0
        if eventtype not in [0, 1]:
            continue
        dset = exodata.create_dataset(item.name, data=event2dimg, dtype='float16')
        dset.attrs[u'tag'] = eventtype
    f.close()

if __name__ == '__main__':
    SaveH5('bb0n.tar', 'test.h5')
