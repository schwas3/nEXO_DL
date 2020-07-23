#!/usr/bin/env python
import ROOT
import os
import numpy as np
import argparse

def read3dimage(rootfile):
    tfile = ROOT.TFile(rootfile, 'READ')
    ttree = tfile.Get('SSigTree')

    pitch = 6
    maxlen = 0 #1540
    iteration = 0
    remove = 0
    for i in range(ttree.GetEntries()):
        #image_2dcharge = np.zeros((200,255,3),dtype=np.float16)
        ttree.GetEntry(i)
        iteration += 1
        fxwf = ttree.fxwf
        xposition = ttree.fxposition
        xtile = ttree.fxqtile
        xcharge = ttree.xq
        fywf = ttree.fywf
        yposition = ttree.fyposition
        ycharge = ttree.yq
        ytile = ttree.fyqtile
        for j in range(len(xcharge)):
            posj = xposition[j]
            tilej = xtile[j]
            image_2dcharge = np.zeros((48,1024,3),dtype=np.float16)
            if xcharge[j] == 0:
                continue
            for k in range(len(xcharge)):
                posk = xposition[k]
                tilek = xtile[k]
                if abs(posk - posj) < 100 and abs(tilek - tilej) < 100:
                    H = int((posk - posj)/6) + 16*int((tilek - tilej)/96) + 24
                    if H >= 47:
                        continue
                    samplet = 0
                    for m in range(1024):
                        if m < 512:
                            samplet = m
                        else:
                            samplet = 512 + (m-512)*2
                        if samplet >= 1024 or samplet >= len(fxwf[k]):
                            break
                        image_2dcharge[H, m, 0] += fxwf[k][samplet]
            for k in range(len(ycharge)):
                posk = yposition[k]
                tilek = ytile[k]
                if abs(posk - tilej) < 100 and abs(tilek - posj) < 100:
                    H = int((posk - tilej)/6) + 16*int((tilek - posj)/96) + 24
                    if H >= 47:
                        continue
                    samplet = 0
                    for m in range(1024):
                        if m < 512:
                            samplet = m
                        else:
                            samplet = 512 + (m-512)*2
                        if samplet >= 1024 or samplet >= len(fywf[k]):
                            break
                        image_2dcharge[H, m, 1] += fywf[k][samplet]
            np.save('./channelq_npy/%s_channelQ_%d_x%d_%f.npy' % (rootfile, i, j, xcharge[j]) , image_2dcharge)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Root file to process.')
    parser.add_argument('--rootfile', type=str)
    args = parser.parse_args()
    read3dimage(args.rootfile)
