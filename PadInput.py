import pandas as pd
import uproot as up
import numpy as np
import argparse
import os

def SaveNumpy(rootfile, npdir):
    print(rootfile, npdir)
    if not os.path.exists(npdir):
        os.mkdir(npdir)
    input_TFile = up.open(rootfile)
    elecEvent_TTree = input_TFile['Event/Elec/ElecEvent']
    elec_columns_to_get = [ 'fElecChannels.fChannelCharge',\
                        'fElecChannels.fWFAmplitude',\
                        'fElecChannels.fNoiseWF',\
                        'fElecChannels.fXPosition',\
                        'fElecChannels.fYPosition',\
                        'fElecChannels.fChannelNoiseTag']
    dfelec = elecEvent_TTree.arrays( elec_columns_to_get, outputtype=pd.DataFrame )

    for i in range(len(dfelec)):
        row = dfelec.iloc[i]
        xpos = []
        ypos = []
        indices = []
        npimg = np.zeros((224, 224, 12), dtype=np.float16)
        for j in range(len(row['fElecChannels.fChannelCharge'])):
            if row['fElecChannels.fChannelCharge'][j] and row['fElecChannels.fChannelNoiseTag'][j] ==0:
                xpos.append(row['fElecChannels.fXPosition'][j])
                ypos.append(row['fElecChannels.fYPosition'][j])
                indices.append(j)
        xmax = 0
        xmin = 0
        ymax = 0
        ymin = 0
        if len(xpos) > 0:
            xmax = max(xpos)
            xmin = min(xpos)
        if len(ypos) > 0:
            ymax = max(ypos)
            ymin = min(ypos)

        if xmax - xmin > 11*12 or ymax - ymin > 11*12:
            print('skip large size event')
            continue
        for x, y, index in zip(xpos, ypos, indices):
            ch = int((y - ymin)/12)
            #print(x, y, xmin, ymin)
            for tslice in range(10):
                H = int((x-xmin)/12) + 20*tslice
                for W in range(224):
                    samplet = W + 224*tslice
                    if samplet >= len(row['fElecChannels.fWFAmplitude'][index]):
                        continue
                    else:
                        npimg[H, W, ch] = row['fElecChannels.fWFAmplitude'][index][-1-samplet] + row['fElecChannels.fNoiseWF'][index][-1-samplet]
        np.save('%s/%s_%d.npy' % (npdir, os.path.basename(rootfile), i), npimg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Building')
    parser.add_argument('--eventtype', default='bb0n', type=str, help='Event Type')
    parser.add_argument('--num', default=1, type=int, help='File Number')
    args = parser.parse_args()
    SaveNumpy('/home/zl423/scratch60/pad/rootfile/%s%d.root' % (args.eventtype, args.num), '/home/zl423/scratch60/pad/%s%d' % (args.eventtype, args.num))
