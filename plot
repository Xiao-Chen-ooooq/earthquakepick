import json
with open(r'C:\Users\87582\EQTransformer-master\新建文件夹\traceNmae_dic.json','r') as f:
    r=json.load(f)
#print(r)
s={}
for k in r:
    if k[0]=='2':
#         print(len(r[k]))
        if len(r[k])>=20:
            s.update({k:r[k]})
# print(s)
import os
d={}
f=open(r'C:\Users\87582\EQTransformer-master\新建文件夹\Y2000.phs')
t=[]
for q in f.readlines():
    if len(q)<50:
        continue
    elif len(q)>50 and len(q)<70:
        t.append(q)
    elif len(q)>70:
        k=q.split()[0]
        d.update({k:t})
        t=[]
import matplotlib.pyplot as plt
import os
import obspy
from obspy.core import UTCDateTime
from tensorflow.keras.models import load_model
count=0
modek=load_model(r'D:\gstreamer\weights-0-38-0.00588.hdf5')
modes=load_model(r'D:\gstreamer\weights-0-92-0.00731.hdf5')
import numpy as np
from scipy import signal
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
def _normalizes( data,size, mode = 'max'):
    'Normalize waveforms in each batch'
    data_new=np.zeros((size,5984,3))
    for i in range(size):
        data_new[i,:,:]=data[i,:,:]/np.max(np.max(abs(data[i,:,:])))           
    return data_new

for i in s:
    r=i
    if 1:
        print('ok')
        k=(len(s[r])-1)*5
        plt.figure(figsize=(20,10))
        l=0
        preds_p=[]
        preds_s=[]
        for i in s[r]:
            
            j0=obspy.read('E:'+'/0/'+i.split('*')[1]+'/'+i.split('*')[1]+'_E_'+i.split('*')[0][0:3]+'.sac')
            j1=obspy.read('E:'+'/0/'+i.split('*')[1]+'/'+i.split('*')[1]+'_N_'+i.split('*')[0][0:3]+'.sac')
            j2=obspy.read('E:'+'/0/'+i.split('*')[1]+'/'+i.split('*')[1]+'_Z_'+i.split('*')[0][0:3]+'.sac')
            j0.decimate(factor=5, strict_length=False)
            j1.decimate(factor=5, strict_length=False)
            j2.decimate(factor=5, strict_length=False)
            j0.filter(type='bandpass', freqmin = 0.1, freqmax =10)
            j1.filter(type='bandpass', freqmin = 0.1, freqmax =10)
            j2.filter(type='bandpass', freqmin = 0.1, freqmax =10)
            sa=100
            #r=UTCDateTime(i.split('*')[2])
            if k==(len(s[r])-1)*5:
                a=UTCDateTime(i.split('*')[2])
    #             print
    #             print(d[r][l])
        #     print(r)
        #     print(type(r))
        #     print(r-j[0].stats.starttime)
            font0 = {'family': 'serif',
            'color': 'white',
            'stretch': 'condensed',
            'weight': 'normal',
            'size': 12,
            } 
            if d[r][l]!='0':
                if d[r][l][42:47]=='60.00':
                    ts=UTCDateTime(int(d[r][l][18:22]),int(d[r][l][22:24]),int(d[r][l][24:26]),int(d[r][l][26:28]),int(d[r][l][28:30]),59.95)
                else:
                    ts=UTCDateTime(int(d[r][l][18:22]),int(d[r][l][22:24]),int(d[r][l][24:26]),int(d[r][l][26:28]),int(d[r][l][28:30]),float(d[r][l][42:47]))
    #             print(ts)
            else:
                ts=0
            if d[r][l+1]!='0':
                if d[r][l+1][30:35]=='60.00':
                    tp=UTCDateTime(int(d[r][l+1][18:22]),int(d[r][l+1][22:24]),int(d[r][l+1][24:26]),int(d[r][l+1][26:28]),int(d[r][l+1][28:30]),59.95)
                else:
                    tp=UTCDateTime(int(d[r][l+1][18:22]),int(d[r][l+1][22:24]),int(d[r][l+1][24:26]),int(d[r][l+1][26:28]),int(d[r][l+1][28:30]),float(d[r][l+1][30:35]))
    #             print(tp)
            else:
                tp=0
            data0=j0[0].data[int((a-j0[0].stats.starttime)*sa)-7*sa:int((a-j0[0].stats.starttime)*sa)-7*sa+5984]
            data1=j1[0].data[int((a-j1[0].stats.starttime)*sa)-7*sa:int((a-j1[0].stats.starttime)*sa)-7*sa+5984]
            data2=j2[0].data[int((a-j2[0].stats.starttime)*sa)-7*sa:int((a-j2[0].stats.starttime)*sa)-7*sa+5984]
            data=np.concatenate([data0.reshape(1,5984,1),data1.reshape(1,5984,1),data2.reshape(1,5984,1)],axis=-1)
            data=_normalizes(data,1)
    #         if len(data)==0:
    #             continue
#             data0=data0/max(j0[0].data[int((a-j0[0].stats.starttime)*sa)-20*sa:int((a-j0[0].stats.starttime)*sa)+100*sa])
            predp=modek.predict(data).reshape(5984)
#             preds_p.append(predp)
            preds=modes.predict(data).reshape(5984)

            font0 = {'family': 'serif',
                        'color': 'white',
                        'stretch': 'condensed',
                        'weight': 'normal',
                        'size': 23,
                        } 
            fig = plt.figure(figsize=(20,20),constrained_layout=False)
            widths = [6, 1]
            heights = [1, 1, 1, 1, 1, 1, 1.8]
            spec5 = fig.add_gridspec(ncols=2, nrows=7, width_ratios=widths,
                                  height_ratios=heights, left=0.1, right=0.9, hspace=0.1)
            #             preds_s.append(preds)
            ax = fig.add_subplot(spec5[0, 0])         
            plt.plot(data[0,:, 0], 'k')
            plt.xlim(0, 5984)
            x = np.arange(5984)

            ax.set_xticks([])
            ax.tick_params(axis='y',labelsize=17)
            plt.ylim(-1.2,1.2)
            plt.rcParams["figure.figsize"] = (10, 10)
            legend_properties = {'weight':'bold'} 
            ymin, ymax = ax.get_ylim()
            pt=int((tp-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
            plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
            st=int((ts-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
            plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
            ax = fig.add_subplot(spec5[0, 1])                 
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['E', 'Picked P', 'Picked S'], fancybox=True, shadow=True,loc='center left', bbox_to_anchor=(-0.47, 0.5))
            plt.axis('off')
            plt.rcParams["figure.figsize"] = (10, 10)
            ax = fig.add_subplot(spec5[1, 0])         
            f, t, Pxx = signal.stft(data[0,:, 0], fs=100, nperseg=80)
            Pxx = np.abs(Pxx)                       
            plt.pcolormesh(t, f, Pxx, alpha=None, cmap='hot', shading='flat', antialiased=True)
            plt.ylim(0, 40)
            plt.text(1, 1, 'STFT', fontdict=font0)
            plt.ylabel('Hz', fontsize=23)
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='y',labelsize=17)
            ax.set_xticks([])

            ax = fig.add_subplot(spec5[2, 0])         
            plt.plot(data[0,:, 1], 'k')
            plt.xlim(0, 5984)
            x = np.arange(5984)

            ax.set_xticks([])
            plt.rcParams["figure.figsize"] = (10, 10)
            legend_properties = {'weight':'bold'} 
            ymin, ymax = ax.get_ylim()
            pt=int((tp-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
            plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
            st=int((ts-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
            plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='y',labelsize=17)
            ax = fig.add_subplot(spec5[2, 1])                 
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines,['N', 'Picked P', 'Picked S'], fancybox=True, shadow=True,loc='center left', bbox_to_anchor=(-0.47, 0.5))
            plt.axis('off')
            plt.rcParams["legend.fontsize"] =25
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='y',labelsize=17)
            ax = fig.add_subplot(spec5[3, 0]) 
            f, t, Pxx = signal.stft(data[0,:, 1], fs=100, nperseg=80)
            Pxx = np.abs(Pxx)                       
            plt.pcolormesh(t, f, Pxx, alpha=None, cmap='hot', shading='flat', antialiased=True)
            plt.ylim(0, 40)
            plt.text(1, 1, 'STFT', fontdict=font0)
            plt.ylabel('Hz', fontsize=23)
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='y',labelsize=17)
            ax.set_xticks([])
            ax = fig.add_subplot(spec5[4, 0])         
            plt.plot(data[0,:, 2], 'k')
            plt.xlim(0, 5984)
            x = np.arange(5984)
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='y',labelsize=17)
            ax.set_xticks([])
            plt.rcParams["figure.figsize"] = (10, 10)
            legend_properties = {'weight':'bold'} 
            ymin, ymax = ax.get_ylim()
            pt=int((tp-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
            plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
            st=int((ts-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
            plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
            ax = fig.add_subplot(spec5[4, 1])                 
            custom_lines = [Line2D([0], [0], color='k', lw=0),
                            Line2D([0], [0], color='c', lw=2),
                            Line2D([0], [0], color='m', lw=2)]
            plt.legend(custom_lines, ['Z', 'Picked P', 'Picked S'], fancybox=True, shadow=True,loc='center left', bbox_to_anchor=(-0.47, 0.5))
            plt.axis('off')
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='y',labelsize=17)
            ax = fig.add_subplot(spec5[5, 0]) 
            f, t, Pxx = signal.stft(data[0,:, 2], fs=100, nperseg=80)
            Pxx = np.abs(Pxx)                       
            plt.pcolormesh(t, f, Pxx, alpha=None, cmap='hot', shading='flat', antialiased=True)
            plt.ylim(0, 40)
            plt.text(1, 1, 'STFT', fontdict=font0)
            plt.ylabel('Hz', fontsize=23)
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis='y',labelsize=17)
            ax.set_xticks([])
            ax = fig.add_subplot(spec5[6, 0])
            x = np.array(range(5984))+10

            plt.plot(x, predp[:], '--', color='b', alpha = 0.5, linewidth=2, label='P_arrival')
            plt.plot(x, preds[:], '--', color='r', alpha = 0.5, linewidth=2, label='S_arrival')
            plt.tight_layout()       
            plt.ylim((-0.1, 1.1)) 
            plt.xlim(0, 5984)
            plt.ylabel('Probability', fontsize=23) 
            plt.xlabel('Time(s)', fontsize=23) 
            plt.xticks([0,1000,2000,3000,4000,5000],[0,10,20,30,40,50])
            plt.yticks(np.arange(0, 1.1, step=0.2))
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=7))
            ax.tick_params(axis='both',labelsize=17)
            axes = plt.gca()
            axes.yaxis.grid(color='lightgray')
            ax = fig.add_subplot(spec5[6, 1])  
            custom_lines = [
                            Line2D([0], [0], linestyle='--', color='b', lw=2),
                            Line2D([0], [0], linestyle='--', color='r', lw=2)]
            plt.legend(custom_lines, ['P_arrival', 'S_arrival'], fancybox=True, shadow=True)
            plt.axis('off')

            plt.xlim(0, 5984)
            fig.tight_layout()
#             fig.savefig(r'C:\Users\87582\Documents\0\0.png')
            break
        break
  fig = plt.figure(figsize=(20,20),constrained_layout=False)
widths = [6, 1]
heights = [1, 1, 1, 1.6]
spec5 = fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths,
                      height_ratios=heights, left=0.1, right=0.9, hspace=0.1)
#             preds_s.append(preds)
ax = fig.add_subplot(spec5[0, 0])         
plt.plot(data[0,:, 0], 'k')
plt.xlim(0, 5984)
x = np.arange(5984)
plt.ylabel('Amplitude Counts', fontsize=23) 
plt.xticks([0,1000,2000,3000,4000,5000],[0,10,20,30,40,50])
ax.tick_params(axis='both',labelsize=17)
plt.ylim(-1.2,1.2)
plt.rcParams["figure.figsize"] = (10, 10)
legend_properties = {'weight':'bold'} 
ymin, ymax = ax.get_ylim()
pt=int((tp-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
st=int((ts-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
ax = fig.add_subplot(spec5[0, 1])                 
custom_lines = [Line2D([0], [0], color='k', lw=0),
                Line2D([0], [0], color='c', lw=2),
                Line2D([0], [0], color='m', lw=2)]
plt.legend(custom_lines, ['E', 'Picked P', 'Picked S'], fancybox=True, shadow=True,loc='center left', bbox_to_anchor=(-0.47, 0.5))
plt.axis('off')
plt.rcParams["figure.figsize"] = (10, 10)

ax = fig.add_subplot(spec5[1, 0])         
plt.plot(data[0,:, 1], 'k')
plt.xlim(0, 5984)
x = np.arange(5984)

ax.set_xticks([])
plt.rcParams["figure.figsize"] = (10, 10)
legend_properties = {'weight':'bold'} 
ymin, ymax = ax.get_ylim()
pt=int((tp-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
st=int((ts-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.ylabel('Amplitude Counts', fontsize=23) 
plt.xticks([0,1000,2000,3000,4000,5000],[0,10,20,30,40,50])
ax.tick_params(axis='both',labelsize=17)
ax = fig.add_subplot(spec5[1, 1])                 
custom_lines = [Line2D([0], [0], color='k', lw=0),
                Line2D([0], [0], color='c', lw=2),
                Line2D([0], [0], color='m', lw=2)]
plt.legend(custom_lines,['N', 'Picked P', 'Picked S'], fancybox=True, shadow=True,loc='center left', bbox_to_anchor=(-0.47, 0.5))
plt.axis('off')
plt.rcParams["legend.fontsize"] =25
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.tick_params(axis='y',labelsize=17)

ax = fig.add_subplot(spec5[2, 0])         
plt.plot(data[0,:, 2], 'k')
plt.xlim(0, 5984)
x = np.arange(5984)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.ylabel('Amplitude Counts', fontsize=23) 
plt.xticks([0,1000,2000,3000,4000,5000],[0,10,20,30,40,50])
ax.tick_params(axis='both',labelsize=17)
plt.rcParams["figure.figsize"] = (10, 10)
legend_properties = {'weight':'bold'} 
ymin, ymax = ax.get_ylim()
pt=int((tp-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
st=int((ts-j0[0].stats.starttime)*sa)-(int((a-j0[0].stats.starttime)*sa)-7*sa)
plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
ax = fig.add_subplot(spec5[2, 1])                 
custom_lines = [Line2D([0], [0], color='k', lw=0),
                Line2D([0], [0], color='c', lw=2),
                Line2D([0], [0], color='m', lw=2)]
plt.legend(custom_lines, ['Z', 'Picked P', 'Picked S'], fancybox=True, shadow=True,loc='center left', bbox_to_anchor=(-0.47, 0.5))
plt.axis('off')
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))

ax.tick_params(axis='y',labelsize=17)

ax = fig.add_subplot(spec5[3, 0])
x = np.array(range(5984))+10

plt.plot(x, predp[:], '--', color='b', alpha = 0.5, linewidth=2, label='P_arrival')
plt.plot(x, preds[:], '--', color='r', alpha = 0.5, linewidth=2, label='S_arrival')
plt.tight_layout()       
plt.ylim((-0.1, 1.1)) 
plt.xlim(0, 5984)
plt.ylabel('Probability', fontsize=23) 
plt.xlabel('Time(s)', fontsize=23) 
plt.xticks([0,1000,2000,3000,4000,5000],[0,10,20,30,40,50])
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=7))
# plt.legend(loc='lower center', bbox_to_anchor=(0., 0, 1., .102), ncol=3, mode="expand",
#            prop=legend_properties,  borderaxespad=0., fancybox=True, shadow=True)
ax.tick_params(axis='both',labelsize=17)
axes = plt.gca()
axes.yaxis.grid(color='lightgray')
#             ax = fig.add_subplot(spec5[6, 1])  
#             custom_lines = [
#                             Line2D([0], [0], linestyle='--', color='b', lw=2),
#                             Line2D([0], [0], linestyle='--', color='r', lw=2)]
#             plt.legend(custom_lines, ['P_arrival', 'S_arrival'], fancybox=True, shadow=True)
#             plt.axis('off')
ax = fig.add_subplot(spec5[3, 1])  
custom_lines = [
                Line2D([0], [0], linestyle='--', color='b', lw=2),
                Line2D([0], [0], linestyle='--', color='r', lw=2)]
plt.legend(custom_lines, ['P_arrival', 'S_arrival'], fancybox=True, shadow=True,loc='center left', bbox_to_anchor=(-0.47, 0.7))
plt.axis('off')
plt.xlim(0, 5984)
import matplotlib.pyplot as plt
import os
import obspy
from obspy.core import UTCDateTime
r='229180'
k=(len(s[r])-1)*5
plt.figure(figsize=(20,13))
l=0
for i in s[r]:
    j=obspy.read('E:'+'/0/'+i.split('*')[1]+'/'+i.split('*')[1]+'_E_'+i.split('*')[0][0:3]+'.sac')
    sa=500
    j.filter(type='bandpass', freqmin = 0.1, freqmax =10)
    #r=UTCDateTime(i.split('*')[2])
    if k==(len(s[r])-1)*5:
        a=UTCDateTime(i.split('*')[2])
#             print
#             print(d[r][l])
#     print(r)
#     print(type(r))
#     print(r-j[0].stats.starttime)
    if d[r][l]!='0':
        if d[r][l][42:47]=='60.00':
            ts=UTCDateTime(int(d[r][l][18:22]),int(d[r][l][22:24]),int(d[r][l][24:26]),int(d[r][l][26:28]),int(d[r][l][28:30]),59.95)
        else:
            ts=UTCDateTime(int(d[r][l][18:22]),int(d[r][l][22:24]),int(d[r][l][24:26]),int(d[r][l][26:28]),int(d[r][l][28:30]),float(d[r][l][42:47]))
#             print(ts)
    else:
        ts=0
    if d[r][l+1]!='0':
        if d[r][l+1][30:35]=='60.00':
            tp=UTCDateTime(int(d[r][l+1][18:22]),int(d[r][l+1][22:24]),int(d[r][l+1][24:26]),int(d[r][l+1][26:28]),int(d[r][l+1][28:30]),59.95)
        else:
            tp=UTCDateTime(int(d[r][l+1][18:22]),int(d[r][l+1][22:24]),int(d[r][l+1][24:26]),int(d[r][l+1][26:28]),int(d[r][l+1][28:30]),float(d[r][l+1][30:35]))
#             print(tp)
    else:
        tp=0
    data=j[0].data[int((a-j[0].stats.starttime)*sa)-23*sa:int((a-j[0].stats.starttime)*sa)+50*sa]
    if len(data)==0:
        continue
    data=data/max(j[0].data[int((a-j[0].stats.starttime)*sa)-23*sa:int((a-j[0].stats.starttime)*sa)+50*sa])*2
    plt.plot(range(len(data)),data+k,c='k')
    if tp!=0:
#             print('tp')
#             print(tp)
        plt.plot([int((tp-j[0].stats.starttime)*sa)-(int((a-j[0].stats.starttime)*sa)-23*sa),int((tp-j[0].stats.starttime)*sa)-(int((a-j[0].stats.starttime)*sa)-23*sa)],[k-2.5,k+2.5],c='blue')


    if ts!=0:
#             print('ts')
#             print(ts)
        plt.plot([int((ts-j[0].stats.starttime)*sa)-(int((a-j[0].stats.starttime)*sa)-23*sa),int((ts-j[0].stats.starttime)*sa)-(int((a-j[0].stats.starttime)*sa)-23*sa)],[k-2.5,k+2.5],c='r')
    plt.xticks([0,10*sa,20*sa,30*sa,40*sa,50*sa,60*sa,70*sa],[0,10,20,30,40,50,60,70],size=20)
    plt.yticks([0,50,100,150,200,250,300],[0,10,20,30,40,50,60],size=20)
    plt.xlabel('Time(s)',fontsize=23)
    plt.ylabel('Station Counts',fontsize=23)
    plt.ylim((-5,len(s[r])*5))
    k-=5
    l+=2
