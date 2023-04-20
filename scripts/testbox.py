#%% 
import matplotlib.pyplot as plt
import numpy as np
import os
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
# #%%
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
5 -q -Q
10 -q -Q
25 -q -Q
50 -q -Q
100 -q -Q
0 -q -Q -N
5 -q -Q -N
10 -q -Q -N
25 -q -Q -N
50 -q -Q -N
100 -q -Q -N'''.split('\n')
short_names = ['%s%s_Q%i_%s-Q_%s'%(prefix,i.split()[0],'-q'in i,['fixed','reseeded']['-Q'in i],['fixed','reseeded']['-N'in i])for i in a]
fig,ax = plt.subplots(figsize=(50,40))
for short_cmd in short_cmds:
    index = short_cmds.index(short_cmd)
    short_name = '%s%s_Q%i_%s-Q_%s'%(prefix,short_cmd.split()[0],'-q'in short_cmd,['fixed','reseeded']['-Q'in short_cmd],['fixed','reseeded']['-N'in short_cmd])
    filename = '/p/lustre2/nexouser/scotswas/DNN_study/nEXO_DL/scripts/output/%s/%s_loss_acc.npy'%(short_name,short_name)
    if not os.path.exists(filename):continue
    amp = float(short_cmd.split()[0])
    noised_quiet = '-q' in short_cmd
    reseed_nosie = '-N' in short_cmd
    reseed_quiet = '-Q' in short_cmd
    translated_meaning = '(%g,%g)_%s+(0,%g)_%s'%(amp,amp*noised_quiet,['000','012'][reseed_nosie],1*noised_quiet,['000','012'][reseed_quiet])
    # if reseed_quiet:ax = plt.subplot(4,7,22)
    ax = plt.subplot(6,7,index+1)
    if 1:
        data = np.load(filename,allow_pickle=True)[-1]
        # data = arr[-1]
        j = 0
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
            xData = np.arange(51)/50
            yData0 = []
            yData1 = []
            for i in xData:
                yData0 += [np.sum((data1[:,0]>=i) * (data1[:,1]==0))]
                yData1 += [np.sum((data1[:,0]>=i) * (data1[:,1]==1))]
            yData0 = np.array(yData0)/np.sum(data1[:,1]==0)
            yData1 = np.array(yData1)/np.sum(data1[:,1]==1)
            # axin.text(yData0[yData1>0.7+j*0.005][-1],yData1[yData1>0.7+j*0.005][-1],s='%g'%j)
            axin.text(yData0[(yData0>0.1+j*0.0075)*(yData1>0.7+j*0.0075)][-1],yData1[(yData0>0.1+j*0.0075)*(yData1>0.7+j*0.0075)][-1],s='%g'%j)
            # plt.plot(xData,yData0)
            # plt.plot(xData,yData1)
            plt.plot(yData0,yData1,lw=2,alpha=1,label='Epoch%g'%(j))
            # plt.plot(yData0,yData1,lw=2,alpha=1,label='%s%g_Epoch%g'%(prefix,noise_amplitude,j))
            axin.plot(yData0,yData1,lw=2,alpha=1)
            plt.xlabel('Ratio of gamma events above threshold')
            plt.ylabel('Ratio of electron events above threshold')
            j+=1
            # for y0,y1,s in zip(yData0[::10],yData1[::10],xData[::10]):
            #     plt.text(y0,y1,'%g'%s)
        axin.set_xticklabels([])
        axin.set_yticklabels([])
        plt.legend(fontsize=6)
        plt.grid()
        plt.gca().set_aspect('equal')
        axin.grid()
        # plt.title('%s, Noise_Amp = %g [times RMS of ~25 units]'%(prefix,noise_amplitude))
    else:
        arr = np.load(filename,allow_pickle=True)
        y_train_loss, y_train_acc, y_valid_loss, y_valid_acc = arr[:-1]
        if 1:
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
            plt.plot(np.arange(len(y_train_acc)),y_train_acc,label='Training_Acc')
            # plt.legend()
            # plt.grid()
            # plt.xlim(0,19)
            # plt.xticks(np.arange(10)*2)
            # plt.xlabel('Epoch')
            plt.plot(np.arange(len(y_train_acc)),y_valid_acc,label='Validation_Acc')
            plt.ylabel('Accuracy (Percent)')
            plt.ylim(0,100)
        plt.legend()
        plt.grid()
        plt.xlim(0,19)
        plt.xticks(np.arange(10)*2)
        plt.xlabel('Epoch')
    plt.title(translated_meaning)

plt.show()
# %%
