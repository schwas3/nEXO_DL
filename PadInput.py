import matplotlib.pyplot as plt
import pandas as pd
import uproot as up
import numpy as np

input_TFile = up.open('bb0n_seed1.root') #../Baseline2019_bb0n_FullLXe_seed100.nEXOevents.root')
elecEvent_TTree = input_TFile['Event/Elec/ElecEvent']
# We need to ignore the fBits column, which causes errors (and isn't useful anyway.)
# See the issue for more info: https://github.com/scikit-hep/uproot/issues/475
#elec_columns_to_get = [x for x in events.allkeys() \
#                  if events[x].interpretation is not None and x != b"fBits"]
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
    npimg = np.zeros((8, 224, 224), dtype=np.float16)
    for j in range(len(row['fElecChannels.fChannelCharge'])):
        if row['fElecChannels.fChannelCharge'][j] and row['fElecChannels.fChannelNoiseTag'][j] ==0:
            xpos.append(row['fElecChannels.fXPosition'][j])
            ypos.append(row['fElecChannels.fYPosition'][j])
            indices.append(j)
    if max(xpos) - min(xpos) > 15*12 or max(ypos) - min(ypos) > 15*12:
        print('skip large size event')
        break
    xmin = min(xpos)
    ymin = min(ypos)
    for x, y, index in zip(xpos, ypos, indices):
        H = int((x - xmin)/12 + (y - ymin)/12*15)
        for ch in range(8):
            for W in range(224):
                samplet = W + 224*ch
                if samplet >= len(row['fElecChannels.fWFAmplitude'][index]):
                    continue
                else:
                    npimg[ch, H, W] = row['fElecChannels.fWFAmplitude'][index][samplet]
    np.save('bb0n_%d.npy' % i, npimg)

