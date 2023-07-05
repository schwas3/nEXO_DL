#%%
import matplotlib.pyplot as plt
import numpy as np
import os

# class trainingCase:
#     def __init__(self,short_cmd='0 -q',prefix='misc_1_'):
#         self.bestEpoch = -1
#         self.short_cmd = short_cmd
#         self.prefix = prefix
#         self.filename = '/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/scripts/output/%s/%s_loss_acc.npy'%(short_name,short_name)
#         data = prepareData(self.filename)
#         self.gammaData = data[data[:,1]==0][:,1]
#         self.betaData = data[data[:,1]==1][:,1]

#     def getBestEpoch


def getShortname(short_cmd='0 -q',prefix = 'misc_1_'):
    shortname = '%s%s_Q%i_%s-Q_%s%s'%(prefix,short_cmd.split()[0],'-q'in short_cmd,['fixed','reseeded']['-Q'in short_cmd],['fixed','reseeded']['-N'in short_cmd],''if '-m'not in short_cmd else'_%s'%(float(short_cmd.split()[-1])))
    return shortname

def getFilename(short_name):
    return '/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/scripts/output/%s/%s_loss_acc.npy'%(short_name,short_name)

def prepareData(filename):
    data = np.load(filename,allow_pickle=True)[-1]
    return data

def getEpoch(short_cmd='0 -q',epoch=0):
    return np.array(prepareData(getFilename(getShortname(short_cmd)))[epoch])

def getGammasAndBetas(short_cmd='0 -q',epoch=0):
    data = getEpoch(short_cmd,epoch)
    gammaData = data[data[:,1]==0][:,0]
    betaData = data[data[:,1]==1][:,0]
    return gammaData,betaData

def getGammaBetaHistograms(short_cmd='0 -q',epoch=0,percentilesOrThresholds=1,nDivisions=100):
    '''Return percentiles (0) or thresholds (1) of nDivisions equally spaced from 0 to 1'''
    gammaData, betaData = getGammasAndBetas(short_cmd,epoch)
    # gammaData = np.zeros(gammaData.size)
    # betaData = np.ones(betaData.size)
    # gammaData = np.random.random(gammaData.size)
    # betaData = np.random.random(betaData.size)
    # np.arange(nDivisions)/nDivisions
    if percentilesOrThresholds:
        gammaData = np.histogram(gammaData,nDivisions,(0,1))
        betaData = np.histogram(betaData,nDivisions,(0,1))
    else:
        print("THIS DOESNT WORK YET SORRY")
        # np.sort blah blah blah
    return gammaData, betaData

def getGammaClear85ThresholdRate(short_cmd='0 -q',nEpochs = 20):
    '''Returns the clearance rate of gammas for a threshold allowing 85% of betas and the epoch it occurs'''
    bestEpoch, bestThresh, bestClearance = 0,0,1
    for epoch in range(nEpochs):
        gammaData,betaData = getGammasAndBetas(short_cmd,epoch)
        fifteenthPercentileIndex = round(len(betaData)*0.15)
        threshold = np.sort(betaData)[fifteenthPercentileIndex]
        clearance = np.sum(gammaData >= threshold) / len(gammaData)
        if clearance <= bestClearance:
            bestEpoch = epoch
            bestThresh = threshold
            bestClearance = clearance
    return bestEpoch, bestThresh, bestClearance



#%%
fig=plt.figure(figsize=(30,3))
j=0
betas = 129752
gammas = 112961
for n in [0,1,2,4,10]:
    relabeledGammas = (gammas + betas) * n / 100
    trainingBetas_p = betas * 0.8
    trainingGammas_c = (gammas - relabeledGammas) * 0.8
    trainingGammas_p = relabeledGammas
    validationGammas = gammas - trainingGammas_c - trainingGammas_p
    validationBetas = betas - trainingBetas_p
    n = trainingGammas_p / trainingBetas_p * 100
    plt.subplot(1,5,j+1)
    j+=1
    plt.bar(['C','P','V'],[trainingGammas_c,trainingGammas_p,validationGammas],bottom=[0,trainingBetas_p,validationBetas])
    plt.bar(['P','V'],[trainingBetas_p,validationBetas])
    plt.title('Physics: %g%% $\\gamma$\'s'%(round(100*trainingGammas_p/(trainingGammas_p+trainingBetas_p),1)),fontsize=28)
    plt.xticks(fontsize=24)
    print(n,trainingBetas_p,trainingGammas_c,trainingGammas_p,validationBetas,validationGammas)
    plt.ylim(0,130000)
plt.show()
#%%

short_cmd = '0 -q -m 0'
epoch, thresh, clearance = getGammaClear85ThresholdRate(short_cmd)
gammaData,betaData = getGammaBetaHistograms(short_cmd,epoch,nDivisions=100)
gammaData, xData = gammaData
betaData, _ = betaData
xData = xData[:-1]
# gammaData = gammaData / np.sum(gammaData)
#%%
fig,ax=plt.subplots(figsize=(15,12))
plt.bar(xData,gammaData,width=xData[1]-xData[0],align='edge',label='Gammas',alpha=0.5)
# xData = xData[:-1]
# betaData = betaData / np.sum(betaData)
plt.bar(xData,betaData,width=xData[1]-xData[0],align='edge',label='Betas',bottom=0*gammaData,alpha=0.5)
plt.xlim(0*xData[gammaData>0][0],1)
# plt.plot(xData,gammaData/(gammaData+betaData))
# plt.plot(xData,betaData/(gammaData+betaData))
# plt.title(short_cmd)
plt.vlines(thresh,0,np.max([gammaData,betaData]),linestyles='dashed')
plt.text(thresh+.01,np.max([gammaData,betaData])/2,'85%% $\\beta$\'s $\\rightarrow$\n%g%% $\\gamma$\'s $\\rightarrow$   '%round(clearance*100,1),ha='center',fontsize=32)
plt.xlabel('DNN Confidence Parameter',fontsize=32)
plt.ylabel('Counts',fontsize=32)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.show()

#%%
fig,ax=plt.subplots(figsize=(15,12))
# plt.plot(gammaData/np.sum(gammaData),betaData/np.sum(betaData))
plt.plot(np.cumsum(gammaData[::-1]/np.sum(gammaData)),np.cumsum(betaData[::-1]/np.sum(betaData)))
plt.scatter(np.cumsum(gammaData[::-1]/np.sum(gammaData))[(xData-thresh)**2 == np.min((xData-thresh)**2)],np.cumsum(betaData[::-1]/np.sum(betaData))[(xData-thresh)**2 == np.min((xData-thresh)**2)],s=100)
plt.vlines(np.cumsum(gammaData[::-1]/np.sum(gammaData))[(xData-thresh)**2 == np.min((xData-thresh)**2)],0,np.cumsum(betaData[::-1]/np.sum(betaData))[(xData-thresh)**2 == np.min((xData-thresh)**2)],linestyles='dashed')
plt.hlines(np.cumsum(betaData[::-1]/np.sum(betaData))[(xData-thresh)**2 == np.min((xData-thresh)**2)],0,np.cumsum(gammaData[::-1]/np.sum(gammaData))[(xData-thresh)**2 == np.min((xData-thresh)**2)],linestyles='dashed')
plt.xlabel('$\\gamma$ acceptance rate',fontsize=32)
plt.ylabel('$\\beta$ acceptance rate',fontsize=32)
plt.text(.1+np.cumsum(gammaData[::-1]/np.sum(gammaData))[(xData-thresh)**2 == np.min((xData-thresh)**2)],-.05+np.cumsum(betaData[::-1]/np.sum(betaData))[(xData-thresh)**2 == np.min((xData-thresh)**2)],'85%% $\\beta$\'s\n%g%% $\\gamma$\'s    '%round(clearance*100,1),fontsize=32,ha='center',va='center')
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.ylim(0,1)
plt.xlim(0,1)
plt.show()

#%%

# arr = np.load('/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/training_outputs/loss_acc.npy',allow_pickle=True)
# arr = np.load('/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/scripts/training_outputs/loss_acc.npy',allow_pickle=True)
# arr = np.load('/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/scripts/output/noise_test_3_0/noise_test_3_0_loss_acc.npy',allow_pickle=True)


# # # %%
# # prefix = 'noise_test_6_'
# # noise_amplitude = 99
# # arr = np.load('/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/scripts/output/%s%g/%s%g_loss_acc.npy'%(prefix,noise_amplitude,prefix,noise_amplitude),allow_pickle=True)
# # data = arr[-1]
# # fig, ax=plt.subplots(figsize=(10,10))
# # j = 0
# # axin = ax.inset_axes([0.25,0.08,0.55,0.55])
# # axin.set_xticks([0.1,0.2,0.3])
# # axin.set_yticks([0.7,0.8,0.9])
# # axin.set_xlim(0.1,0.3)
# # axin.set_ylim(0.7,0.9)
# # ax.indicate_inset_zoom(axin)
# # for data1 in data[::]:
# #     data1=np.array(data1)
# #     # data2=np.array(data2)
# #     # print(np.min(data1),np.max(data1))
# #     xData = np.arange(1001)/1000
# #     yData0 = []
# #     yData1 = []
# #     for i in xData:
# #         yData0 += [np.sum((data1[:,0]>=i) * (data1[:,1]==0))]
# #         yData1 += [np.sum((data1[:,0]>=i) * (data1[:,1]==1))]
# #     yData0 = np.array(yData0)/np.sum(data1[:,1]==0)
# #     yData1 = np.array(yData1)/np.sum(data1[:,1]==1)
# #     axin.text(yData0[yData1>0.7+j*0.005][-1],yData1[yData1>0.7+j*0.005][-1],s='%g'%j)
# #     # plt.plot(xData,yData0)
# #     # plt.plot(xData,yData1)
# #     plt.plot(yData0,yData1,lw=2,alpha=1,label='Epoch%g'%j)
# #     axin.plot(yData0,yData1,lw=2,alpha=1)
# #     plt.xlabel('Ratio of gamma events above threshold')
# #     plt.ylabel('Ratio of electron events above threshold')
# #     j+=1
# #     # for y0,y1,s in zip(yData0[::10],yData1[::10],xData[::10]):
# #     #     plt.text(y0,y1,'%g'%s)
# # axin.set_xticklabels([])
# # axin.set_yticklabels([])
# # plt.legend()
# # plt.title('%s, Noise_Amp = %g [times RMS of ~25 units]'%(prefix,noise_amplitude))
# # plt.grid()
# # axin.grid()
# # plt.gca().set_aspect('equal')
# # plt.show()
# # %%
# prefixes = ['noise_test_6_']*7
# # prefixes = ['noise_test_5_'],'noise_test_6_','noise_test_6_']
# noise_amplitudes = [0,1,4,9,19,49,99]
# # bestEpochs = [10,10,10]
# # arr1 = [np.load('/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/scripts/output/%s%g/%s%g_loss_acc.npy'%(prefix,noise_amplitude,prefix,noise_amplitude),allow_pickle=True)for prefix,noise_amplitude in zip(prefixes,noise_amplitudes)]
# fig, ax=plt.subplots(figsize=(50,15))
# for prefix,noise_amplitude in zip(prefixes,noise_amplitudes):
#     ax=plt.subplot(1,len(noise_amplitudes),noise_amplitudes.index(noise_amplitude)+1)
#     arr = np.load('/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/scripts/output/%s%g/%s%g_loss_acc.npy'%(prefix,noise_amplitude,prefix,noise_amplitude),allow_pickle=True)
#     data = arr[-1]
#     j = 0
#     axin = ax.inset_axes([0.25,0.08,0.55,0.55])
#     axin.set_xticks([0.0,0.1,0.2,0.3,0.4])
#     axin.set_yticks([0.6,0.7,0.8,0.9,1.0])
#     axin.set_xlim(0,0.4)
#     axin.set_ylim(0.6,1)
#     ax.indicate_inset_zoom(axin)
#     for data1 in data:
#         # if j != bestEpoch: j+=1;continue
#         data1=np.array(data1)
#         # data2=np.array(data2)
#         # print(np.min(data1),np.max(data1))
#         xData = np.arange(201)/200
#         yData0 = []
#         yData1 = []
#         for i in xData:
#             yData0 += [np.sum((data1[:,0]>=i) * (data1[:,1]==0))]
#             yData1 += [np.sum((data1[:,0]>=i) * (data1[:,1]==1))]
#         yData0 = np.array(yData0)/np.sum(data1[:,1]==0)
#         yData1 = np.array(yData1)/np.sum(data1[:,1]==1)
#         # axin.text(yData0[yData1>0.7+j*0.005][-1],yData1[yData1>0.7+j*0.005][-1],s='%g'%j)
#         axin.text(yData0[(yData0>0.1+j*0.0075)*(yData1>0.7+j*0.0075)][-1],yData1[(yData0>0.1+j*0.0075)*(yData1>0.7+j*0.0075)][-1],s='%g'%j)
#         # plt.plot(xData,yData0)
#         # plt.plot(xData,yData1)
#         plt.plot(yData0,yData1,lw=2,alpha=1,label='Epoch%g'%(j))
#         # plt.plot(yData0,yData1,lw=2,alpha=1,label='%s%g_Epoch%g'%(prefix,noise_amplitude,j))
#         axin.plot(yData0,yData1,lw=2,alpha=1)
#         plt.xlabel('Ratio of gamma events above threshold')
#         plt.ylabel('Ratio of electron events above threshold')
#         j+=1
#         # for y0,y1,s in zip(yData0[::10],yData1[::10],xData[::10]):
#         #     plt.text(y0,y1,'%g'%s)
#     plt.title('%g times RMS Noise (~25 units)'%noise_amplitude)
#     axin.set_xticklabels([])
#     axin.set_yticklabels([])
#     plt.legend(fontsize=8)
#     plt.grid()
#     axin.grid()
#     # plt.title('%s, Noise_Amp = %g [times RMS of ~25 units]'%(prefix,noise_amplitude))
#     plt.gca().set_aspect('equal')
# plt.show()
# #%%
# prefixes = ['noise_test_6_']*7
# noise_amplitudes = [0,1,4,9,19,49,99]
# fig, ax=plt.subplots(figsize=(50,15))
# for prefix,noise_amplitude in zip(prefixes,noise_amplitudes):
#     ax = plt.subplot(2,len(noise_amplitudes),noise_amplitudes.index(noise_amplitude)+1)
#     arr = np.load('/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/scripts/output/%s%g/%s%g_loss_acc.npy'%(prefix,noise_amplitude,prefix,noise_amplitude),allow_pickle=True)
#     y_train_loss, y_train_acc, y_valid_loss, y_valid_acc = arr[:-1]
#     plt.plot(np.arange(len(y_train_acc)),y_train_loss,label='Training')
#     plt.plot(np.arange(len(y_train_acc)),y_valid_loss,label='Validation')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.xticks(np.arange(10)*2)
#     plt.ylim(0,2)
#     plt.legend()
#     plt.xlim(0,19)
#     plt.grid()
#     plt.title('%g times RMS Noise (~25 units)'%noise_amplitude)
#     ax = plt.subplot(2,len(noise_amplitudes),noise_amplitudes.index(noise_amplitude)+1+len(noise_amplitudes))
#     plt.plot(np.arange(len(y_train_acc)),y_train_acc,label='Training')
#     plt.xticks(np.arange(10)*2)
#     plt.plot(np.arange(len(y_train_acc)),y_valid_acc,label='Validation')
#     plt.ylabel('Accuracy (Percent)')
#     plt.ylim(70,100)
#     plt.xlim(0,19)
#     plt.legend()
#     plt.grid()
#     plt.xlabel('Epoch')
# plt.show()
    
# # %%
# np.random.seed(1)
# mask1 = np.random.normal(0,25,1000)
# mask1 *= 5
# np.random.seed(1)
# mask2 = np.random.normal(0,5*25,1000)
# # print(mask1)
# # print(mask2)
# print(np.any(mask1-mask2))
# # %%
# sq1 = np.random.SeedSequence()
# print(sq1)
# seedList = sq1.generate_state(100)
# print(*seedList)
#%%
prefix = 'misc_1_'
short_cmds='''0
1
5
10
25
50
100
0 -N
1 -N
5 -N
10 -N
25 -N
50 -N
100 -N
0 -q
1 -q
5 -q
10 -q
25 -q
50 -q
100 -q
0 -q -N
1 -q -N
5 -q -N
10 -q -N
25 -q -N
50 -q -N
100 -q -N
0 -q -Q
1 -q -Q
5 -q -Q
10 -q -Q
25 -q -Q
50 -q -Q
100 -q -Q
0 -q -Q -N
1 -q -Q -N
5 -q -Q -N
10 -q -Q -N
25 -q -Q -N
50 -q -Q -N
100 -q -Q -N'''.split('\n')
if 1:
    short_cmds='''0 -q -m 0
    0.1 -q
    0.25 -q
    0.5 -q
    1 -q
    2 -q
    3 -q
    4 -q
    5 -q'''.splitlines()
#     0 -q -N
#     0.1 -q -N
#     0.25 -q -N
#     0.5 -q -N
#     1 -q -N
#     2 -q -N
#     3 -q -N
#     4 -q -N
#     5 -q -N
# '''.splitlines()
# 10 -q
# 25 -q
# 50 -q
# 100 -q
# 10 -q -N
# 25 -q -N
# 50 -q -N
# 100 -q -N'''.splitlines()
relabel = 1
if relabel:
    short_cmds='''0 -m 0
    0 -m 1
    0 -m 2
    0 -m 3
    0 -m 4
    0 -m 5
    0 -m 7.5
    0 -m 10'''.splitlines()
# short_names = ['%s%s_Q%i_%s-Q_%s'%(prefix,i.split()[0],'-q'in i,['fixed','reseeded']['-Q'in i],['fixed','reseeded']['-N'in i])for i in ]
if relabel not in [0,1]:fig,ax = plt.subplots(figsize=(15,12))
# fig,ax = plt.subplots(figsize=(50,12))
invert_axes = 0
labels,exp0,exp1=[],np.zeros((len(short_cmds),20)),[]
for short_cmd in short_cmds:
    index = short_cmds.index(short_cmd)
    short_name = '%s%s_Q%i_%s-Q_%s%s'%(prefix,short_cmd.split()[0],'-q'in short_cmd,['fixed','reseeded']['-Q'in short_cmd],['fixed','reseeded']['-N'in short_cmd],''if '-m'not in short_cmd else'_%s'%(float(short_cmd.split()[-1])))
    print(short_name)
    filename = '/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/scripts/output/%s/%s_loss_acc.npy'%(short_name,short_name)
    if not os.path.exists(filename):continue
    amp = float(short_cmd.split()[0])
    noised_quiet = '-q' in short_cmd
    reseed_nosie = '-N' in short_cmd
    reseed_quiet = '-Q' in short_cmd
    if relabel and '-m'in short_cmd:translated_meaning='(0,0) - %g%% of gammas -> electrons'%(float(short_cmd.split()[-1]))
    else:translated_meaning = '(%g,%g)_%s+(0,%g)_%s'%(amp,amp*noised_quiet,['000','012'][reseed_nosie],1*noised_quiet,['000','012'][reseed_quiet])
    # if reseed_quiet:ax = plt.subplot(4,7,22)
    # ax = plt.subplot(6,7,index+1)
    if relabel not in [0,1]:plt.title(translated_meaning)
    if 0: # evenly spaced thresholds
        data = np.load(filename,allow_pickle=True)[-1]
        j = 0
        if invert_axes:
            axin = ax.inset_axes([0.53,0.07,0.45,0.45])
            axin.set_xticks(np.arange(5)/4)
            axin.set_yticks(np.arange(5)/4)
            axin.set_xticklabels(['0','','0.5','','1'])
            axin.set_yticklabels(['0','','0.5','','1'])
            axin.set_xlim(0,1)
            axin.set_ylim(0,1)
            plt.ylim(0.6,1)
            plt.xlim(0,0.4)
            axin.set_title('Entire ROC')
        else:
            axin = ax.inset_axes([0.25,0.08,0.5,0.5])
            axin.set_xticks(np.arange(5)*.4/4)
            axin.set_yticks(np.arange(5)*.4/4+.6)
            axin.set_ylim(0.6,1)
            axin.set_xlim(0,0.4)
            ax.indicate_inset_zoom(axin)
            plt.ylim(0,1)
            plt.xlim(0,1)
            axin.set_xticklabels(['0','','','','0.4'])
            axin.set_yticklabels(['0.6','','','','1.0'])
        for data1 in data[10:11]:
            # if j != bestEpoch: j+=1;continue
            data1=np.array(data1)
            # data2=np.array(data2)
            # print(np.min(data1),np.max(data1))
            xData = np.arange(101)/100
            yData0 = []
            yData1 = []
            for i in xData:
                yData0 += [np.sum((data1[:,0]>=i) * (data1[:,1]==0))]
                yData1 += [np.sum((data1[:,0]>=i) * (data1[:,1]==1))]
            yData0 = np.array(yData0)/np.sum(data1[:,1]==0)
            yData1 = np.array(yData1)/np.sum(data1[:,1]==1)
            if invert_axes:
                thresh = 0.05 + j * 0.0075
                # plt.text(yData0[yData1>=thresh][-1],yData1[yData1>=thresh][-1],s='%g'%j,fontsize=6)
                plt.text(yData0[(yData1>=thresh+.6)*(yData0>=thresh)][-1],yData1[(yData1>=thresh+.6)*(yData0>=thresh)][-1],s='%g'%j,fontsize=6)
                # plt.text(yData0[(yData0>j*0.0075)*(yData1>0.6+j*0.0075)][-1],yData1[(yData0>0+j*0.0075)*(yData1>0.6+j*0.0075)][-1],s='%g'%j)
            else:
                thresh = 0.05 + j * 0.0075
                # plt.text(yData0[yData1>=thresh][-1],yData1[yData1>=thresh][-1],s='%g'%j,fontsize=6)
                axin.text(yData0[(yData1>=thresh+.6)*(yData0>=thresh)][-1],yData1[(yData1>=thresh+.6)*(yData0>=thresh)][-1],s='%g'%j,fontsize=6)
                # # plt.text(yData0[(yData0>j*0.0075)*(yData1>0.6+j*0.0075)][-1],yData1[(yData0>0+j*0.0075)*(yData1>0.6+j*0.0075)][-1],s='%g'%j)
                # axin.text(yData0[yData1>0.7+j*0.01][-1],yData1[yData1>0.7+j*0.01][-1],s='%g'%j,fontsize=6)
            # plt.plot(xData,yData0)
            # plt.plot(xData,yData1)
            epochColor=plt.plot(yData0,yData1,lw=1,alpha=1,label='Epoch%g'%(j))[0].get_color()
            # plt.plot((yData0[1:][::-1]+yData0[:-1][::-1])/2,-np.cumsum((yData0[1:][::-1]-yData0[:-1][::-1])*(yData1[1:][::-1]+yData1[:-1][::-1])/2),c=epochCOLOR)
            # plt.plot(yData0,yData1,lw=2,alpha=1,label='%s%g_Epoch%g'%(prefix,noise_amplitude,j))
            axin.plot(yData0,yData1,lw=1,alpha=1)
            plt.xlabel('Ratio of gamma events above threshold')
            plt.ylabel('Ratio of electron events above threshold')
            j+=1
            # for y0,y1,s in zip(yData0[::10],yData1[::10],xData[::10]):
            #     plt.text(y0,y1,'%g'%s)
        # axin.set_xticklabels([])
        # axin.set_yticklabels([])
        if invert_axes:
            ax.legend(fontsize=6,loc='upper left')
        else:
            ax.legend(fontsize=6,loc='lower right')
        ax.grid()
        ax.set_aspect('equal')
        axin.set_aspect('equal')
        axin.grid()
        # plt.title('%s, Noise_Amp = %g [times RMS of ~25 units]'%(prefix,noise_amplitude))
        # ax = plt.subplot(1,8,index+1)
    elif 0: # evenly spaced ratios of gammas/betas above threshold (i.e. 0% of betas, 1% of betas,... AND 0% of gammas, 1% of gammas,... sorted to form straight curve of 2x points as intervals)
        data = np.load(filename,allow_pickle=True)[-1]
        j = 0
        if invert_axes:
            axin = ax.inset_axes([0.53,0.07,0.45,0.45])
            axin.set_xticks(np.arange(5)/4)
            axin.set_yticks(np.arange(5)/4)
            axin.set_xticklabels(['0','','0.5','','1'])
            axin.set_yticklabels(['0','','0.5','','1'])
            axin.set_xlim(0,1)
            axin.set_ylim(0,1)
            plt.ylim(0.6,1)
            plt.xlim(0,0.4)
            axin.set_title('Entire ROC')
        else:
            axin = ax.inset_axes([0.25,0.08,0.5,0.5])
            axin.set_xticks(np.arange(5)*.4/4)
            axin.set_yticks(np.arange(5)*.4/4+.6)
            axin.set_ylim(0.6,1)
            axin.set_xlim(0,0.4)
            ax.indicate_inset_zoom(axin)
            plt.ylim(0,1)
            plt.xlim(0,1)
            axin.set_xticklabels(['0','','','','0.4'])
            axin.set_yticklabels(['0.6','','','','1.0'])
        for data1 in data[10:11]:
            # if j != bestEpoch: j+=1;continue
            data1=np.array(data1)
            # data2=np.array(data2)
            # print(np.min(data1),np.max(data1))
            xData = np.concatenate((np.sort(data1[data1[:,1]==0][:,0])[np.int0(np.floor((np.sum(data1[:,1]==0)*np.arange(100)/100)))],np.sort(data1[data1[:,1]==1][:,0])[np.int0(np.floor((np.sum(data1[:,1]==1)*np.arange(100)/100)))]))
            xData = np.sort(xData)
            yData0 = []
            yData1 = []
            for i in xData:
                yData0 += [np.sum((data1[:,0]>=i) * (data1[:,1]==0))]
                yData1 += [np.sum((data1[:,0]>=i) * (data1[:,1]==1))]
            yData0 = np.array(yData0)/np.sum(data1[:,1]==0)
            yData1 = np.array(yData1)/np.sum(data1[:,1]==1)
            if invert_axes:
                thresh = 0.05 + j * 0.0075
                # plt.text(yData0[yData1>=thresh][-1],yData1[yData1>=thresh][-1],s='%g'%j,fontsize=6)
                plt.text(yData0[(yData1>=thresh+.6)*(yData0>=thresh)][-1],yData1[(yData1>=thresh+.6)*(yData0>=thresh)][-1],s='%g'%j,fontsize=6)
                # plt.text(yData0[(yData0>j*0.0075)*(yData1>0.6+j*0.0075)][-1],yData1[(yData0>0+j*0.0075)*(yData1>0.6+j*0.0075)][-1],s='%g'%j)
            else:
                thresh = 0.05 + j * 0.0075
                # plt.text(yData0[yData1>=thresh][-1],yData1[yData1>=thresh][-1],s='%g'%j,fontsize=6)
                axin.text(yData0[(yData1>=thresh+.6)*(yData0>=thresh)][-1],yData1[(yData1>=thresh+.6)*(yData0>=thresh)][-1],s='%g'%j,fontsize=6)
                # # plt.text(yData0[(yData0>j*0.0075)*(yData1>0.6+j*0.0075)][-1],yData1[(yData0>0+j*0.0075)*(yData1>0.6+j*0.0075)][-1],s='%g'%j)
                # axin.text(yData0[yData1>0.7+j*0.01][-1],yData1[yData1>0.7+j*0.01][-1],s='%g'%j,fontsize=6)
            # plt.plot(xData,yData0)
            # plt.plot(xData,yData1)
            epochColor=plt.plot(yData0,yData1,lw=1,alpha=1,label='Epoch%g'%(j))[0].get_color()
            # plt.plot((yData0[1:][::-1]+yData0[:-1][::-1])/2,-np.cumsum((yData0[1:][::-1]-yData0[:-1][::-1])*(yData1[1:][::-1]+yData1[:-1][::-1])/2),c=epochCOLOR)
            # plt.plot(yData0,yData1,lw=2,alpha=1,label='%s%g_Epoch%g'%(prefix,noise_amplitude,j))
            axin.plot(yData0,yData1,lw=1,alpha=1)
            plt.xlabel('Ratio of gamma events above threshold')
            plt.ylabel('Ratio of electron events above threshold')
            j+=1
            # for y0,y1,s in zip(yData0[::10],yData1[::10],xData[::10]):
            #     plt.text(y0,y1,'%g'%s)
        # axin.set_xticklabels([])
        # axin.set_yticklabels([])
        if invert_axes:
            ax.legend(fontsize=6,loc='upper left')
        else:
            ax.legend(fontsize=6,loc='lower right')
        ax.grid()
        ax.set_aspect('equal')
        axin.set_aspect('equal')
        axin.grid()
        # plt.title('%s, Noise_Amp = %g [times RMS of ~25 units]'%(prefix,noise_amplitude))
        # ax = plt.subplot(1,8,index+1)
    elif 0: # unused?
        data=np.load(filename,allow_pickle=True)[-1]
        j=0
        auc = []
        axin = ax.inset_axes([0.35,0.08,0.45,0.45])
        axin.set_xticks([0.0,0.1,0.2,0.3,0.4])
        axin.set_yticks([0.6,0.7,0.8,0.9,1.0])
        axin.set_xlim(0,0.4)
        axin.set_ylim(0.6,1)
        ax.indicate_inset_zoom(axin)
        for data1 in data:
            # if j != bestEpoch: j+=1;continue
            data1=np.array(data1)
            # data2=np.array(data2)
            # print(np.min(data1),np.max(data1))
            xData = np.arange(101)/100
            yData0 = []
            yData1 = []
            for i in xData:
                yData0 += [np.sum((data1[:,0]>=i) * (data1[:,1]==0))]
                yData1 += [np.sum((data1[:,0]>=i) * (data1[:,1]==1))]
            yData0 = np.array(yData0)/np.sum(data1[:,1]==0)
            yData1 = np.array(yData1)/np.sum(data1[:,1]==1)
            # axin.text(yData0[yData1>0.7+j*0.005][-1],yData1[yData1>0.7+j*0.005][-1],s='%g'%j)
            axin.text(yData0[(yData0>j*0.0075)*(yData1>0.6+j*0.0075)][-1],yData1[(yData0>0+j*0.0075)*(yData1>0.6+j*0.0075)][-1],s='%g'%j)
            # axin.text(yData0[(yData0>j*0.0075)*(yData1>0.6+j*0.0075)][-1],yData1[(yData0>0+j*0.0075)*(yData1>0.6+j*0.0075)][-1],s='%g'%j)
            # plt.plot(xData,yData0)
            # plt.plot(xData,yData1)
            axin.plot(yData0,yData1,lw=1,alpha=1)
            plt.plot(yData0,yData1,lw=1,alpha=1,label='Epoch%g'%(j))
            auc += [-np.sum((yData0[1:]-yData0[:-1])*(yData1[1:]+yData1[:-1])/2)]
            # plt.plot(yData0,yData1,lw=2,alpha=1,label='%s%g_Epoch%g'%(prefix,noise_amplitude,j))
            # axin.plot(yData0,yData1,lw=2,alpha=1)
            plt.xlabel('Ratio of gamma events above threshold')
            plt.ylabel('Ratio of electron events above threshold')
            j+=1
            # for y0,y1,s in zip(yData0[::10],yData1[::10],xData[::10]):
            #     plt.text(y0,y1,'%g'%s)
        # axin.set_xticklabels([])
        # axin.set_yticklabels([])
        axin.set_xticklabels([])
        axin.set_yticklabels([])
        plt.legend(fontsize=6)
        plt.grid()
        plt.gca().set_aspect('equal')
        axin.grid()
        plt.xlim(0,1)
        plt.ylim(0,1)
        ax = plt.subplot(2,9,10)
        plt.plot(np.arange(len(auc)),auc,label=translated_meaning,lw=1)
        plt.grid()
        plt.legend(fontsize=6)
        # plt.ylim(0,1)
        # plt.xlim(0,1)
        plt.title('AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        # plt.gca().set_aspect('equal')
        # axin.grid()
        # plt.title('%s, Noise_Amp = %g [times RMS of ~25 units]'%(prefix,noise_amplitude))
    elif relabel == 1: # Relabeling
        data = np.load(filename,allow_pickle=True)[-1]
        j = 0
        labels += [float(short_name.split('_')[-1])]
        # exp0+=[[]] 
        for data1 in data:
            # if j != bestEpoch: j+=1;continue
            data1=np.array(data1)
            # data2=np.array(data2)
            # print(np.min(data1),np.max(data1))
            # xData = np.arange(11)/10
            # yData0 = []
            # yData1 = []
            # for i,k in zip(xData,xData[1:]):
            #     if 1:
            #         yData0 += [np.sum((data1[:,0]>=i) * (data1[:,0]<k) * (data1[:,1]==0))]
            #         yData1 += [np.sum((data1[:,0]>=i) * (data1[:,0]<k) * (data1[:,1]==1))]
            #     else:
            #         yData0 += [np.sum((data1[:,0]>=i) * (data1[:,1]==0))]
            #         yData1 += [np.sum((data1[:,0]>=i) * (data1[:,1]==1))]
            # yData0 = np.array(yData0)/np.sum(data1[:,1]==0)
            # yData1 = np.array(yData1)/np.sum(data1[:,1]==1)
            # axin.text(yData0[yData1>0.7+j*0.005][-1],yData1[yData1>0.7+j*0.005][-1],s='%g'%j)
            # axin.text(yData0[(yData0>0.1+j*0.0075)*(yData1>0.7+j*0.0075)][-1],yData1[(yData0>0.1+j*0.0075)*(yData1>0.7+j*0.0075)][-1],s='%g'%j)
            # plt.plot(xData,yData0)
            # plt.plot(xData,yData1)
            # plt.plot((xData[1:]+xData[:-1])/2,yData1/yData0,lw=2,label='Ratio_Epoch%g'%(j))
            #plt.plot((xData[1:]+xData[:-1])/2,yData1,'--',lw=2,label='Betas - '+short_name.split('_')[-1]+'%% relabled gammas')#label='E%g'%(j))
            #plt.plot((xData[1:]+xData[:-1])/2,yData0,'--',lw=2,label='Gamma - '+short_name.split('_')[-1]+'%% relabled gammas')#label='G%g'%(j))
            # print(np.sum((xData[1:]+xData[:-1])/2*yData0))
            # print(np.sum((xData[1:]+xData[:-1])/2*yData1))
            # exp0 += [(np.mean(data1[:,0][data1[:,1]==0]))]
            # exp1 += [(np.mean(data1[:,0][data1[:,1]==1]))]
            # print(len(data1))
            exp0[index][j]=np.sum(data1[:,0][data1[:,1]==0]>np.sort(data1[:,0][data1[:,1]==1])[round(0.15*np.sum(data1[:,1]==1))])/np.sum(data1[:,1]==0)*100
            # exp0 += [(np.mean(data1[:,0][data1[:,1]==0]))]
            # exp1 += [(np.mean(data1[:,0][data1[:,1]==1]))]
            # plt.plot((xData[1:]+xData[:-1])/2,yData1,lw=2,label='E%g'%(j))
            # plt.plot((xData[1:]+xData[:-1])/2,yData0,lw=2,label='G%g'%(j))
            # plt.plot(yData0,yData1,lw=2,alpha=1,label='Epoch%g'%(j))
            # plt.plot(yData0,yData1,lw=2,alpha=1,label='%s%g_Epoch%g'%(prefix,noise_amplitude,j))
            # axin.plot(yData0,yData1,lw=2,alpha=1)
            j+=1
            # for y0,y1,s in zip(yData0[::10],yData1[::10],xData[::10]):
            #     plt.text(y0,y1,'%g'%s)
        # plt.xlabel('DNN Value (0.1 width Bins - 10 bins)')
        # plt.ylabel('Percent of Events w/ DNN Value +/- 0.05')
        # plt.ylabel('$\\left<\\beta-Likeness\\right>$')
        # plt.ylabel('Ratio of total electron tagged over ratio of total gamma tagged')
        # plt.xlim(0,1)
        # plt.gca().set_aspect('equal')
        # axin.grid()
        # plt.title('%s, Noise_Amp = %g [times RMS of ~25 units]'%(prefix,noise_amplitude))
    elif relabel == 0: # Noise
        data = np.load(filename,allow_pickle=True)[-1] # Currently just picks out the 10th epoch, need to optimize
        j = 0
        labels += [float(short_name.split('_')[2])]
        # exp0 += [[]]
        for data1 in data:
            # if j != bestEpoch: j+=1;continue
            data1=np.array(data1)
            # data2=np.array(data2)
            # print(np.min(data1),np.max(data1))
            # xData = np.arange(11)/10
            # yData0 = []
            # yData1 = []
            # for i,k in zip(xData,xData[1:]):
            #     if 1:
            #         yData0 += [np.sum((data1[:,0]>=i) * (data1[:,0]<k) * (data1[:,1]==0))]
            #         yData1 += [np.sum((data1[:,0]>=i) * (data1[:,0]<k) * (data1[:,1]==1))]
            #     else:
            #         yData0 += [np.sum((data1[:,0]>=i) * (data1[:,1]==0))]
            #         yData1 += [np.sum((data1[:,0]>=i) * (data1[:,1]==1))]
            # yData0 = np.array(yData0)/np.sum(data1[:,1]==0)
            # yData1 = np.array(yData1)/np.sum(data1[:,1]==1)
            # axin.text(yData0[yData1>0.7+j*0.005][-1],yData1[yData1>0.7+j*0.005][-1],s='%g'%j)
            # axin.text(yData0[(yData0>0.1+j*0.0075)*(yData1>0.7+j*0.0075)][-1],yData1[(yData0>0.1+j*0.0075)*(yData1>0.7+j*0.0075)][-1],s='%g'%j)
            # plt.plot(xData,yData0)
            # plt.plot(xData,yData1)
            # plt.plot((xData[1:]+xData[:-1])/2,yData1/yData0,lw=2,label='Ratio_Epoch%g'%(j))
            #plt.plot((xData[1:]+xData[:-1])/2,yData1,'--',lw=2,label='Betas - '+short_name.split('_')[-1]+'%% relabled gammas')#label='E%g'%(j))
            #plt.plot((xData[1:]+xData[:-1])/2,yData0,'--',lw=2,label='Gamma - '+short_name.split('_')[-1]+'%% relabled gammas')#label='G%g'%(j))
            # print(np.sum((xData[1:]+xData[:-1])/2*yData0))
            # print(np.sum((xData[1:]+xData[:-1])/2*yData1))
            # exp0 += [(np.mean(data1[:,0][data1[:,1]==0]))]
            # exp1 += [(np.mean(data1[:,0][data1[:,1]==1]))]
            exp0[index][j]=np.sum(data1[:,0][data1[:,1]==0]>=np.sort(data1[:,0][data1[:,1]==1])[round(0.15*np.sum(data1[:,1]==1))])/np.sum(data1[:,1]==0)*100
            # exp0[index][0][j]=np.sum(data1[:,0][data1[:,1]==0]>=np.sort(data1[:,0][data1[:,1]==1])[round(0.15*np.sum(data1[:,1]==1))])/np.sum(data1[:,1]==0)*100
            # exp0[index][1][j]=np.sum(data1[:,0][data1[:,1]==0]>=np.sort(data1[:,0][data1[:,1]==1])[round(0.20*np.sum(data1[:,1]==1))])/np.sum(data1[:,1]==0)*100
            # exp0[index][2][j]=np.sum(data1[:,0][data1[:,1]==0]>=np.sort(data1[:,0][data1[:,1]==1])[round(0.25*np.sum(data1[:,1]==1))])/np.sum(data1[:,1]==0)*100
            # print(np.sum(data1[:,0][data1[:,1]==0]>sorted(data1[:,0][data1[:,1]==1])[round(0.15*np.sum(data1[:,1]==1))])/np.sum(data1[:,1]==0))
            # print(sorted(data1[:,0][data1[:,1]==1])[round(0.15*np.sum(data1[:,1]==1))])
            # exp0 += [(np.mean(data1[:,0][data1[:,1]==0]))]
            # exp1 += [(np.mean(data1[:,0][data1[:,1]==1]))]
            # plt.plot((xData[1:]+xData[:-1])/2,yData1,lw=2,label='E%g'%(j))
            # plt.plot((xData[1:]+xData[:-1])/2,yData0,lw=2,label='G%g'%(j))
            # plt.plot(yData0,yData1,lw=2,alpha=1,label='Epoch%g'%(j))
            # plt.plot(yData0,yData1,lw=2,alpha=1,label='%s%g_Epoch%g'%(prefix,noise_amplitude,j))
            # axin.plot(yData0,yData1,lw=2,alpha=1)
            j+=1
            # for y0,y1,s in zip(yData0[::10],yData1[::10],xData[::10]):
            #     plt.text(y0,y1,'%g'%s)
        # plt.xlabel('DNN Value (0.1 width Bins - 10 bins)')
        # plt.ylabel('Percent of Events w/ DNN Value +/- 0.05')
        # plt.ylabel('$\\left<\\beta-Likeness\\right>$')
        # plt.ylabel('Ratio of total electron tagged over ratio of total gamma tagged')
        # plt.xlim(0,1)
        # plt.legend(fontsize=12)
        # plt.gca().set_aspect('equal')
        # axin.grid()
        # plt.title('%s, Noise_Amp = %g [times RMS of ~25 units]'%(prefix,noise_amplitude))
    else: # training vs validation - LOSS AND/OR ACCURACY
        arr = np.load(filename,allow_pickle=True)
        y_train_loss, y_train_acc, y_valid_loss, y_valid_acc = arr[:-1]
        if 0:
            plt.plot(np.arange(len(y_train_acc)),y_train_loss,label='Training_Loss')
            plt.plot(np.arange(len(y_train_acc)),y_valid_loss,label='Validation_Loss')
            plt.ylabel('Loss')
            # plt.xlabel('Epoch')
            plt.ylim(0,2)
            # plt.xticks(np.arange(10)*2)
            # plt.legend()
            # plt.xlim(0,19)
            # plt.grid()
        else:
            # plt.plot(np.arange(len(y_train_acc)),y_train_acc,label='Training_Acc')
            # plt.legend()
            # plt.grid()
            # plt.xlim(0,19)
            # plt.xticks(np.arange(10)*2)
            # plt.xlabel('Epoch')
            plt.plot(np.arange(len(y_train_acc)),y_valid_acc,label='Validation_Acc_'+short_name.split('_')[-1])
            plt.ylabel('Accuracy (Percent)')
            plt.ylim(0,100)
        plt.legend()
        plt.grid()
        plt.xlim(0,19)
        plt.xticks(np.arange(10)*2)
        plt.xlabel('Epoch')
labels=np.array(labels)
exp0=np.array(exp0)
#%%
exp1=np.min(exp0,-1)
exp0=list(exp0)
exp2=[list(i).index(j) for i,j in zip(exp0,exp1)]
exp2 = [[i-1,i,i+1]for i in exp2]
print(exp2)
# exp0 = np.min(exp0,-1)
#%%
fig,ax = plt.subplots(figsize=(30,24))
if relabel==1:
    betas = 129752
    gammas = 112961
    relabeledGammas = (gammas + betas) * labels / 100
    trainingBetas_p = betas * 0.8
    trainingGammas_c = (gammas - relabeledGammas) * 0.8
    trainingGammas_p = relabeledGammas
    labels2 = trainingGammas_p / (trainingGammas_p + trainingBetas_p) * 100
    # labels = labels/(100+labels)
    # plt.scatter(labels2,1.35e28*(exp0/exp0[0])**(-1/3),label='Relabeled',s=400)
    plt.xlabel('Physics Gamma Composition (%)',fontsize=36)
    plt.ylabel('Accepted Gammas (%)',fontsize=40)
    # plt.ylabel('Sensitivity',fontsize=40)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    if 1:
        exp3 = []
        for k,i,j in zip(exp0,exp2,labels):
            exp3 += [*k[i]]
        plt.scatter(labels2,exp3[::3],s=400,label='Previous Epoch')#,label='Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
        plt.scatter(labels2,exp3[1::3],s=400,label='Best Epoch')#,label='Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
        plt.scatter(labels2,exp3[2::3],s=400,label='Next Epoch')#,label='Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
        plt.legend(fontsize=36)
        plt.ylim(0,20)
    else:
        plt.scatter(labels2,exp0,s=400)#,label='Relabeled')

    # plt.ylabel('Predicted Sensitivity (yrs) - ceteris paribus')
    plt.grid(True)
    print(exp0)
    # plt.ylim(1e28,1.36e28)
    plt.title('')
    # plt.xlim(0,20)
elif relabel==0:
    plt.xlabel('Excess Noise (25 units in waveform amplitude)',fontsize=40)
    plt.ylabel('Accepted Gammas (%)',fontsize=40)
    if 0:
        labels2 = []
        exp3 = []
        for k,i,j in zip(exp0,exp2,labels):
            labels2 += 3*[j]
            exp3 += [*k[i]]
        plt.scatter(labels2[::3],exp3[::3],s=400,label='Previous Epoch')#,label='Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
        plt.scatter(labels2[1::3],exp3[1::3],s=400,label='Best Epoch')#,label='Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
        plt.scatter(labels2[2::3],exp3[2::3],s=400,label='Next Epoch')#,label='Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
    if 0: # 3 different beta acceptance rates
        plt.scatter(labels,exp0[:,0],s=400,label='85% Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
        plt.scatter(labels,exp0[:,1],s=400,label='80% Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
        plt.scatter(labels,exp0[:,2],s=400,label='75% Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
    else:
        plt.scatter(labels,exp0,s=400)#,label='Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
        plt.ylabel('Accepted Gammas (%)',fontsize=40)
    if 1: # sensitivity circles / labels
        for label,val in zip(labels[[0,3,4,5,7]],exp0[[0,3,4,5,7]]):
            plt.text(label-.095,val-[1,1][label in labels[[3,5]]]*3.9,'%.3g yrs'%(1.35e28*(val/exp0[0])**(-1/3)),ha='center',va='bottom',fontsize=36,rotation=75)
            plt.scatter(label,val,edgecolors='r',facecolors='none',s=1000,lw=4)
        # plt.scatter(labels,1.35*10**28*(exp0/exp0[0])**(-1/3),label='Renoised plus excess noise (excess noise fixed between epochs)')#,marker='+',alpha=0.8,zorder=0)
        # plt.ylabel('Predicted Sensitivity (yrs) - ceteris paribus')
        # consider not scaling by 1.35E28 and instead just scaling to 1 and saying sensitiivty relative to unperturbed performance
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.grid(True)
    plt.ylim(0,20)
    # plt.ylim(0,20/exp0[0])#,100)
    plt.title('')
else:
    pass
    # plt.scatter(labels[len(exp0)//2+1:],exp0[len(exp0)//2+1:],label='Renoised plus excess noise (excess noise changes between epochs)',marker='x',alpha=0.8,zorder=-1)
# plt.scatter(labels,exp1,label='Betas')
# plt.legend(fontsize=36)
plt.show()
# %%
a=np.random.normal(0,25,(20,2,200,255))
# # %%
# a=np.random.normal(0,25,(2,15,255))
# # %%
# a=np.random.normal(0,25,(20,102000))
# # %%

# %%
