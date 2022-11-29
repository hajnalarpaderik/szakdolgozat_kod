#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import scipy.stats as stats

from statsmodels.stats.weightstats import ttest_ind

import pingouin as pg

from io import StringIO
import pprint

from os import listdir
from os.path import isfile, join


# In[3]:


def get_cameras():
    camera_list=[]
    
    onlyfiles = [f for f in listdir(".") if isfile(join(".", f))]
    for names in onlyfiles:
        if "Cam" in names:
            name_list=names.split("_")
            camera_list.append(int(name_list[0][3:]))
    return camera_list

def get_presets(camera):
    dis_char="[]'"
    preset_list=[]
    
    f = open('Cam'+str(camera)+'_list.txt')
    for line in f.readlines():
        substr="Preset:"
        fn=line.split(',')
        for words in fn:
            for character in dis_char:
                words=words.replace(character, "")
            if substr in words:
                id_=words.split(' ')
                if "Preset:" in id_[0]:
                    preset_list.append(int(id_[1]))
                else:
                    preset_list.append(int(id_[2]))
    return preset_list


def stat_test(camera, preset):
    return_dic = {}
    presets = {}
    dis_char="[]'"
    
    f = open('Cam'+str(camera)+'_list.txt')
    for line in f.readlines():
        substr="Preset:"
        fn=line.split(',')
        for words in fn:
            for character in dis_char:
                words=words.replace(character, "")
            if substr in words:
                id_=words.split(' ')
                if "Preset:" in id_[0]:
                    d_id=int(id_[1])
                else:
                    d_id=int(id_[2])
                presets[d_id]=""
            else:
                tmp=presets[d_id]
                if tmp != "'":
                    tmp+=words+','
                    presets[d_id]=tmp
    
                    
    for k, v in presets.items():
        presets[k]=presets[k][:-1]
    
    if presets[preset] != "":
        Test=StringIO(presets[preset])
    
        df_cam_pos=pd.DataFrame(pd.read_csv(Test,sep=",", header=None).values.reshape(-1,2))
        df_cam_pos.columns = ["run-id", "timestamp"]
        df_cam_pos['timestamp'] = pd.to_datetime(df_cam_pos['timestamp'])        
        
        df_cam_cleaned = pd.read_csv('cam_data_cleaned.csv')
        df_cam_cleaned['timestamp'] = pd.to_datetime(df_cam_cleaned['timestamp'])
        
        bus_data = pd.read_pickle("raw_data_with_corrigated_coords.pkl")
        
        run_ids = df_cam_pos['run-id']
    
        bus_data2 = bus_data.loc[run_ids]
            
        startTime = df_cam_pos.sort_values(by=['timestamp']).iloc[0]['timestamp']
        endTime = df_cam_pos.sort_values(by=['timestamp'], ascending=False).iloc[0]['timestamp']
        
        df_cam_cleaned2 = df_cam_cleaned.loc[(df_cam_cleaned['cam.id'] == "cam_"+str(camera)) & (df_cam_cleaned['cam.preset'] == preset) & (df_cam_cleaned['timestamp'] >= startTime) & (df_cam_cleaned['timestamp'] <= endTime)]
        
        bus_data_sorted = bus_data2['velocity'].to_numpy()
    
        cam_data_sorted = df_cam_cleaned2['avg_speed'].to_numpy()
        
        return_dic['bus_var'] = np.var(bus_data_sorted)
        
        return_dic['cam_var'] = np.var(cam_data_sorted)
        
        return_dic['f_test'] =  f_test(bus_data_sorted, cam_data_sorted)
        
        var_diff = abs(return_dic['bus_var'] - return_dic['cam_var'])
        
        var_ratio = max(return_dic['bus_var'] * 0.1, return_dic['cam_var'] * 0.1)
                
        if return_dic['f_test'][1] >= 0.05:
        
            return_dic['t_test'] = stats.ttest_ind(a=bus_data_sorted, b=cam_data_sorted, equal_var=True)
            
        else:
            
            return_dic['t_test'] = 'A F-teszt eredménye következtében, a két mintás T-teszt nem végezhető el.'
            
        return_dic['k_test'] = stats.ks_2samp(bus_data_sorted, cam_data_sorted)
        
        return return_dic


def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1)
    dfn = x.size-1
    dfd = y.size-1 
    p = stats.f.cdf(f, dfn, dfd)
    return f, p


# In[4]:


print(get_cameras())


# In[5]:


print(get_presets(115))


# In[6]:


stat_test(115,1)


# In[7]:


test_res = []

for camera in get_cameras():
    for preset in get_presets(camera):
        tmp = stat_test(camera, preset)
        if tmp != None:
            tmp['cam'] = camera
            tmp['pres'] = preset
        test_res.append(tmp)
        
test_res


# In[8]:


sort_res = []

for test in test_res:
    if test != None:
        if not isinstance(test.get('t_test'), str) and test.get('t_test').pvalue>=0.05:
            sort_res.append(test)
        
sort_res


# In[9]:


sort_res_k = []

for test in test_res:
    if test != None:
        if test.get('k_test').pvalue>=0.05:
            sort_res_k.append(test)
        
sort_res_k


# In[10]:


sort_cams = []

for cam in sort_res:
    sort_cams.append({'Camera': cam.get('cam'), 'Preset': cam.get('pres')})

print("Kiemelkedően jó kamerák és preszetek:")
sort_cams

