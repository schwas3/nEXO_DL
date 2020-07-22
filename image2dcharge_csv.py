#!/usr/bin/env python
# coding: utf-8
import csv
import os, glob
with open('image2dcharge_sens.csv', 'w') as csvfile:
    fieldnames = ['filename', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    filelist = glob.glob('/home/zl423/project/dlphysics/nEXO_energy/channelq_npy/*npy')
    maxlabel = 0
    for img in filelist:
        label = int(float(img[:-4].split('_')[-1])/135)
        if label > maxlabel:
            maxlabel = label
        writer.writerow({'filename': img, 'label': label})
    print(maxlabel)
