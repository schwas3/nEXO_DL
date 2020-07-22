#!/usr/bin/env python
# coding: utf-8
import csv
import os, glob
with open('image2dcharge_sens.csv', 'w') as csvfile:
    fieldnames = ['filename', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    filelist = glob.glob('/home/zl423/scratch60/gamma_10/*npy')
    filelist += glob.glob('/home/zl423/scratch60/gamma_11/*npy')
    filelist += glob.glob('/home/zl423/scratch60/gamma_12/*npy')
    filelist += glob.glob('/home/zl423/scratch60/gamma_13/*npy')
    filelist += glob.glob('/home/zl423/scratch60/gamma_14/*npy')
    filelist += glob.glob('/home/zl423/scratch60/gamma_15/*npy')
    filelist += glob.glob('/home/zl423/scratch60/gamma_16/*npy')
    filelist += glob.glob('/home/zl423/scratch60/gamma_17/*npy')
    filelist += glob.glob('/home/zl423/scratch60/gamma_18/*npy')
    filelist += glob.glob('/home/zl423/scratch60/gamma_19/*npy')
    nsig = 0
    nbkg = 0
    for img in filelist:
        if 'bb0n' in img:
            continue
            nsig += 1
            writer.writerow({'filename': img, 'label': '1'})
        else:
            #if nbkg > 700000:
            #    continue
            nbkg += 1
            writer.writerow({'filename': img, 'label': '0'})
    filelist1 = glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_33/*npy')
    filelist1 += glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_330/*npy')
    filelist1 += glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_331/*npy')
    filelist1 += glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_332/*npy')
    filelist1 += glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_333/*npy')
    filelist1 += glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_334/*npy')
    filelist1 += glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_335/*npy')
    filelist1 += glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_336/*npy')
    filelist1 += glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_337/*npy')
    filelist1 += glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_338/*npy')
    filelist1 += glob.glob('/home/zl423/project/dlphysics/nEXO_DL_norm/bb0n_339/*npy')
    for img in filelist1:
        nsig += 1
        if nsig > 160000:
            break
        writer.writerow({'filename': img, 'label': '1'})
