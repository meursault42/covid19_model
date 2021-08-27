# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:54:59 2020

@author: u6026797
"""
#%% libraries
import pandas as pd
import numpy as np
import math
#import pygam as pg
#from sklearn.preprocessing import MinMaxScaler
#import tsaug
#import random 
import warnings
from r_hybrid_mod_wrapper import r_hybrid_mod_wrapper
import sys

def shift2(arr,num):
    arr=np.roll(arr,num)
    num=min(num,np.shape(arr)[0])
    if num<0:
         np.put(arr,range(len(arr)+num,len(arr)),0)
    elif num > 0:
         np.put(arr,range(num),0)
    return arr
'''
def ts_aug_avg(ts_1,ts_2):
    #resize to target shape
    ts_2=tsaug.Resize(size=len(ts_1)).augment(ts_2)
    #reshape to x,1
    ts_2=np.reshape(ts_2, newshape=(len(ts_2),1))
    ts_1=np.reshape(ts_1,newshape=(len(ts_1),1))
    #train and fit scaler to target
    ts_aug_scale= MinMaxScaler()
    ts_aug_scale.fit(ts_1)
    ts_1_scaled= ts_aug_scale.transform(ts_1)
    ts_2_scaled=ts_aug_scale.fit_transform(ts_1)
    #generate new series as a weighted mean
    weight = np.mean(ts_1)/np.mean(ts_2)
    intercept_adjust = (ts_1[0,0]+(ts_2[0,0]*(1-weight)))/2
    new_ts_scaled=ts_1_scaled+(ts_2_scaled*(1-weight))/2
    unscaled=(ts_aug_scale.inverse_transform(new_ts_scaled)-intercept_adjust)
    unscaled=np.reshape(unscaled,newshape=(len(unscaled)))
    unscaled[unscaled<0]=1
    return unscaled

def ts_resampler(data_frame, n_times=5, kz_col = 4):#, exclude = 0):
    df = data_frame.copy()
    times = 0
    while times < n_times:
        #comp_ts_out = date_data[(date_data['LastDate']==last_date)]
        comp_ts_out = data_frame[(data_frame['LastDate']==last_date)]
        sample_ind = [i for i in range(0,len(comp_ts_out),1)]
        ind1, ind2 = random.sample(sample_ind,2)
        #filter for junk sequences
        #kz = ts_aug_avg(comp_ts_out.iloc[ind1,7],comp_ts_out.iloc[ind2,7])
        kz = ts_aug_avg(comp_ts_out.iloc[ind1,kz_col],comp_ts_out.iloc[ind2,kz_col])
        if np.sum(kz)!=len(kz) and sum(kz==1)/len(kz) < .3: #and ind1!=exclude and ind2!=exclude:
            #ind1 = comp_ts_out.iloc[ind1,0] 
            #ind2 = comp_ts_out.iloc[ind2,0] 
            for date in date_list:
                #comp_ts = date_data[(date_data['LastDate']==date)]
                comp_ts = data_frame[(data_frame['LastDate']==date)]
                try:
                    new_feat_meta = comp_ts.iloc[ind1,:]
                    new_feat_meta['Country_Region']='Augment_'+str(ind1)+'+'+str(ind2)
                    #was 52
                    new_feat_meta['Index']=np.broadcast_to(np.array((75+times)),shape=(len(new_feat_meta['Index']),1))
                    kz = ts_aug_avg(comp_ts.iloc[ind1,kz_col],comp_ts.iloc[ind2,kz_col])
                    new_feat_meta['kz']=kz
                    df=df.append(new_feat_meta,ignore_index=True)
                except:
                    pass
            times+=1
    return(df)
'''
def smoothed_der(ts,smooth=2):
    out = np.array([])
    for zero_append in range(smooth):
        ts_append = np.insert(ts,[0],values=1)
    for i in range(len(ts_append)):
        if(smooth==1):
            x = ts_append[(i+smooth):(i+smooth+1)]-ts_append[i:(i+smooth)]
            if len(x)==0 or math.isinf(x) or np.isnan(x):
                x = 1
            out = np.append(out,x)
        else:
            x = ts_append[(i+smooth-1):(i+smooth+1)]-ts_append[i:(i+smooth)]
            for val in range(len(x)):
                if math.isinf(x[val]) or np.isnan(x[val]):
                    x[val]=0
            x=np.mean(x)
            out = np.append(out,x)
    return(out[0:(len(out))-1])

warnings.filterwarnings("ignore")
#%% ts looper
#file_name_list = ['country_state','hrr','ards','80mi']
file_name_list = ['state','80mi']
for name in file_name_list:
    ts_output = pd.DataFrame(columns = ['param_name','loc','date','1','2','3','4','5','6',
                                    '7','8','9','10','11','12','13','14'])
    iterlist = ['kz_f7']
    param_name='kz_f7'
    f_name_1 = name+'_oracle_kz.pkl'
    f_name_2 = name+'_flat_kz.pkl'
    date_data_2 = pd.read_pickle(f_name_1)
    date_data_og = pd.read_pickle(f_name_2)
    
    for param_name in iterlist:
        out_file_name = param_name+'case_results_file.csv'
        date_data=date_data_og[['Country_Region','Date','Dt','DayOfWeek',param_name,'LastDate']]
        date_data =date_data.rename(columns={param_name: "kz"})
                
        date_list=pd.unique(date_data['LastDate'])
        date_list = pd.DataFrame(date_list)
        date_list = date_list.sort_values(by=0)
        date_list = date_list.to_numpy()
        date_list = np.reshape(date_list, newshape=(len(date_list),))
        
        loc_list=pd.DataFrame(pd.unique(date_data['Country_Region']))
        loc_list=loc_list.rename(columns={0: "Country_Region"})
        #generate int index of location
        loc_list.insert(0,'Index',range(0,len(loc_list)))
        
        date_data=date_data.merge(loc_list,how='left',on='Country_Region')
        
        val_2_vec = []
        for row in date_data.iterrows():
          val_2_vec.append(np.broadcast_to(np.array(row[1]['Index']),shape=(len(row[1].iloc[2]),1)))
        val_2_vec_df = pd.DataFrame(val_2_vec)
        date_data['Index']=val_2_vec_df
        
        date_data=date_data[['Country_Region','Date','Dt','DayOfWeek','kz','LastDate','Index']]
        #state_list = loc_list[24:]
        #date_data=date_data[date_data['Country_Region'].isin(state_list.iloc[:,1])]
        
        #gam modifications
        date_data = date_data[(date_data['LastDate']>='2020-07-01 00:00:00')]
        if len(date_list)> 30:
            date_list = date_list[31:]
        last_date = date_list[len(date_list)-1]
        for row in date_data.iterrows():        
          if row[1]['LastDate']==last_date:
            loc_in=row[1]['Country_Region']
            oracle_row=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]#['oracle_kz']
            #get min date
            min_date=row[1]['Date'][0]
            #get shorter sequence len
            new_len=len(oracle_row['Date'].to_list()[0][oracle_row['Date'].to_list()[0]>min_date])
            #truncate labels to input sequence length
            Oracle_kz=oracle_row['Oracle_kz'].to_list()[0][-new_len:]
            row[1]['kz']=Oracle_kz
        
        
        #for row in date_data.iterrows():
        #  if row[1]['LastDate']==last_date:
        #    loc_in=row[1]['Country_Region']
        #    row[1]['kz']=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]['oracle_kz']
        
        #temp date restriction
        #temp_start_date = date_list[62]
        #date_data = date_data[(date_data['LastDate']>=temp_start_date)]
        #date_list = date_list[62:]
        #loc_list = loc_list[:55]
    
        for loc in loc_list.iloc[:,1]:
            #loc=loc_list.iloc[35,1]
            for date in date_list:
                #date=date_list[0]
                comp_ts = date_data[(date_data['LastDate']==date) & (date_data['Country_Region']==loc)]
                if len(comp_ts) !=0:
                    comp_ts = comp_ts['kz'].tolist()[0]
                    hybrid_out = r_hybrid_mod_wrapper(ts=comp_ts)
                    hybrid_out_pred = hybrid_out['forecast'].to_numpy()
                    #gather label data and calc loss
                    label_set = date_data[(date_data['LastDate']==last_date) & (date_data['Country_Region']==loc)]
                    label_set = label_set['kz'].tolist()[0]
                    label_set = label_set[len(comp_ts):(len(comp_ts)+14)]
                    if len(label_set)<len(hybrid_out):
                        hybrid_out_pred=hybrid_out_pred[0:len(label_set)]
                    ###
                    hybrid_out_pred = np.round((abs(hybrid_out_pred-label_set)/label_set),4)
                    for pred in range(0,len(hybrid_out['forecast'])):
                        if hybrid_out['forecast'][pred]<1:
                            hybrid_out['forecast'][pred]= 1
                    for na_pad in range((len(hybrid_out_pred)-1),13):
                        hybrid_out_pred=np.append(hybrid_out_pred,np.NaN)
                        label_set=np.append(label_set,np.NaN)
                        #hybrid_out_pred=np.append(hybrid_out_pred,np.NaN)
                        
                    #old output sample
                    #[hybrid_out['forecast'][1],hybrid_out_pred[1],label_set[1]]
                    ts_output = ts_output.append({'param_name': param_name,\
                                                  'loc' : loc, \
                                                  'date' : date,\
                                                  '1' : [hybrid_out['lower_ci'][0],hybrid_out['forecast'][0],hybrid_out['upper_ci'][0]],\
                                                  '2' : [hybrid_out['lower_ci'][1],hybrid_out['forecast'][1],hybrid_out['upper_ci'][1]],\
                                                  '3' : [hybrid_out['lower_ci'][2],hybrid_out['forecast'][2],hybrid_out['upper_ci'][2]],\
                                                  '4' : [hybrid_out['lower_ci'][3],hybrid_out['forecast'][3],hybrid_out['upper_ci'][3]],\
                                                  '5' : [hybrid_out['lower_ci'][4],hybrid_out['forecast'][4],hybrid_out['upper_ci'][4]],\
                                                  '6' : [hybrid_out['lower_ci'][5],hybrid_out['forecast'][5],hybrid_out['upper_ci'][5]],\
                                                  '7' : [hybrid_out['lower_ci'][6],hybrid_out['forecast'][6],hybrid_out['upper_ci'][6]],\
                                                  '8' : [hybrid_out['lower_ci'][7],hybrid_out['forecast'][7],hybrid_out['upper_ci'][7]],\
                                                  '9' : [hybrid_out['lower_ci'][8],hybrid_out['forecast'][8],hybrid_out['upper_ci'][8]],\
                                                  '10': [hybrid_out['lower_ci'][9],hybrid_out['forecast'][9],hybrid_out['upper_ci'][9]],\
                                                  '11': [hybrid_out['lower_ci'][10],hybrid_out['forecast'][10],hybrid_out['upper_ci'][10]],\
                                                  '12': [hybrid_out['lower_ci'][11],hybrid_out['forecast'][11],hybrid_out['upper_ci'][11]],\
                                                  '13': [hybrid_out['lower_ci'][12],hybrid_out['forecast'][12],hybrid_out['upper_ci'][12]],\
                                                  '14': [hybrid_out['lower_ci'][13],hybrid_out['forecast'][13],hybrid_out['upper_ci'][13]]},
                                                 ignore_index=True)
                    sys.stdout.write('Done with {} and {}'.format(loc, date))
    #write raw output
    raw_name = name+'_pred_t.csv'                
    ts_output.to_csv(raw_name)
    #unroller
    results_output = pd.DataFrame(columns = ['param_name','loc','date'])
    
    col_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
    for row in ts_output.iterrows():
        woo = 1
        
        row_dict = {'model' : 'ts', 
                    'param_name' : row[1][0], 
                    'loc' : row[1][1], 
                    'date' : row[1][2], 
                    '1' : row[1][3][1],
                    '2' : row[1][4][1], 
                    '3' : row[1][5][1],
                    '4' : row[1][6][1],
                    '5' : row[1][7][1],                  
                    '6' : row[1][8][1],
                    '7' : row[1][9][1],
                    '8' : row[1][10][1],
                    '9' : row[1][11][1],
                    '10' : row[1][12][1],
                    '11' : row[1][13][1],
                    '12' : row[1][14][1],
                    '13' : row[1][15][1],
                    '14' : row[1][16][1],
                    '1_l' : row[1][3][0],
                    '2_l' : row[1][4][0], 
                    '3_l' : row[1][5][0],
                    '4_l' : row[1][6][0],
                    '5_l' : row[1][7][0],                  
                    '6_l' : row[1][8][0],
                    '7_l' : row[1][9][0],
                    '8_l' : row[1][10][0],
                    '9_l' : row[1][11][0],
                    '10_l' : row[1][12][0],
                    '11_l' : row[1][13][0],
                    '12_l' : row[1][14][0],
                    '13_l' : row[1][15][0],
                    '14_l' : row[1][16][0],
                    '1_u' : row[1][3][2],
                    '2_u' : row[1][4][2], 
                    '3_u' : row[1][5][2],
                    '4_u' : row[1][6][2],
                    '5_u' : row[1][7][2],                  
                    '6_u' : row[1][8][2],
                    '7_u' : row[1][9][2],
                    '8_u' : row[1][10][2],
                    '9_u' : row[1][11][2],
                    '10_u' : row[1][12][2],
                    '11_u' : row[1][13][2],
                    '12_u' : row[1][14][2],
                    '13_u' : row[1][15][2],
                    '14_u' : row[1][16][2]}

        results_output=results_output.append(row_dict, ignore_index=True)
    
    #write clean output
    clean_name = 'clean_'+name+'_results_t.csv'
    results_output.to_csv(clean_name)

#%% 80mi
'''
ts_output = pd.DataFrame(columns = ['param_name','loc','date','1','2','3','4','5','6',
                                    '7','8','9','10','11','12','13','14'])
#iterlist = ['loess_2', 'loess_25', 'loess_3', 'loess_35', 'loess_4',
#       'loess_adf', 'loess_adf_05', 'loess_adf_1', 'loess_adf_15',
#       'loess_adf_2', 'loess_adf_25']

iterlist = ['kz_f7']
param_name='kz_f7'
#iterlist = ['loess_f02', 'loess_f025', 'loess_f03', 'loess_f035', 'loess_f04',
#       'loess_f045', 'loess_f05']
date_data_2 = pd.read_pickle('C:\\Users\\u6026797\\80mi_oracle_kz.pkl')
date_data_og = pd.read_pickle('C:\\Users\\u6026797\\80mi_flat_kz.pkl')
#date_data_og = pd.read_pickle('C:\\Users\\u6026797\\downloads\\PyGAMFlatHyperparameter.pkl') 

for param_name in iterlist:
    out_file_name = param_name+'case_results_file.csv'
    date_data=date_data_og[['Country_Region','Date','Dt','DayOfWeek',param_name,'LastDate']]
    date_data =date_data.rename(columns={param_name: "kz"})
            
    date_list=pd.unique(date_data['LastDate'])
    date_list = pd.DataFrame(date_list)
    date_list = date_list.sort_values(by=0)
    date_list = date_list.to_numpy()
    date_list = np.reshape(date_list, newshape=(len(date_list),))
    
    loc_list=pd.DataFrame(pd.unique(date_data['Country_Region']))
    loc_list=loc_list.rename(columns={0: "Country_Region"})
    #generate int index of location
    loc_list.insert(0,'Index',range(0,len(loc_list)))
    
    date_data=date_data.merge(loc_list,how='left',on='Country_Region')
    
    val_2_vec = []
    for row in date_data.iterrows():
      val_2_vec.append(np.broadcast_to(np.array(row[1]['Index']),shape=(len(row[1].iloc[2]),1)))
    val_2_vec_df = pd.DataFrame(val_2_vec)
    date_data['Index']=val_2_vec_df
    
    date_data=date_data[['Country_Region','Date','Dt','DayOfWeek','kz','LastDate','Index']]
    #state_list = loc_list[24:]
    #date_data=date_data[date_data['Country_Region'].isin(state_list.iloc[:,1])]
    
    #gam modifications
    date_data = date_data[(date_data['LastDate']>='2020-07-01 00:00:00')]
    if len(date_list)> 30:
        date_list = date_list[31:]
    last_date = date_list[len(date_list)-1]
    for row in date_data.iterrows():        
      if row[1]['LastDate']==last_date:
        loc_in=row[1]['Country_Region']
        oracle_row=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]#['oracle_kz']
        #get min date
        min_date=row[1]['Date'][0]
        #get shorter sequence len
        new_len=len(oracle_row['Date'].to_list()[0][oracle_row['Date'].to_list()[0]>min_date])
        #truncate labels to input sequence length
        Oracle_kz=oracle_row['Oracle_kz'].to_list()[0][-new_len:]
        row[1]['kz']=Oracle_kz
    
    
    #for row in date_data.iterrows():
    #  if row[1]['LastDate']==last_date:
    #    loc_in=row[1]['Country_Region']
    #    row[1]['kz']=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]['oracle_kz']
    
    #temp date restriction
    #temp_start_date = date_list[62]
    #date_data = date_data[(date_data['LastDate']>=temp_start_date)]
    #date_list = date_list[62:]
    #loc_list = loc_list[:55]

    for loc in loc_list.iloc[:,1]:
        #loc=loc_list.iloc[288,1]
        for date in date_list:
            #date=date_list[0]
            comp_ts = date_data[(date_data['LastDate']==date) & (date_data['Country_Region']==loc)]
            if len(comp_ts) !=0:
                comp_ts = comp_ts['kz'].tolist()[0]
                hybrid_out = r_hybrid_mod_wrapper(ts=comp_ts)
                hybrid_out_pred = hybrid_out
                #gather label data and calc loss
                label_set = date_data[(date_data['LastDate']==last_date) & (date_data['Country_Region']==loc)]
                label_set = label_set['kz'].tolist()[0]
                label_set = label_set[len(comp_ts):(len(comp_ts)+14)]
                if len(label_set)<len(hybrid_out):
                    hybrid_out=hybrid_out[0:len(label_set)]
                hybrid_out = np.round((abs(hybrid_out-label_set)/label_set),4)
                for pred in range(0,len(hybrid_out)):
                    if hybrid_out[pred]<1:
                        hybrid_out[pred]= 1
                for na_pad in range((len(hybrid_out)-1),13):
                    hybrid_out=np.append(hybrid_out,np.NaN)
                    label_set=np.append(label_set,np.NaN)
                    hybrid_out_pred=np.append(hybrid_out_pred,np.NaN)
                    
                
                ts_output = ts_output.append({'param_name': param_name,\
                                              'loc' : loc, \
                                              'date' : date,\
                                              '1' : [hybrid_out[0],hybrid_out_pred[0],label_set[0]],\
                                              '2' : [hybrid_out[1],hybrid_out_pred[1],label_set[1]],\
                                              '3' : [hybrid_out[2],hybrid_out_pred[2],label_set[2]],\
                                              '4' : [hybrid_out[3],hybrid_out_pred[3],label_set[3]],\
                                              '5' : [hybrid_out[4],hybrid_out_pred[4],label_set[4]],\
                                              '6' : [hybrid_out[5],hybrid_out_pred[5],label_set[5]],\
                                              '7' : [hybrid_out[6],hybrid_out_pred[6],label_set[6]],\
                                              '8' : [hybrid_out[7],hybrid_out_pred[7],label_set[7]],\
                                              '9' : [hybrid_out[8],hybrid_out_pred[8],label_set[8]],\
                                              '10': [hybrid_out[9],hybrid_out_pred[9],label_set[9]],\
                                              '11': [hybrid_out[10],hybrid_out_pred[10],label_set[10]],\
                                              '12': [hybrid_out[11],hybrid_out_pred[11],label_set[11]],\
                                              '13': [hybrid_out[12],hybrid_out_pred[12],label_set[12]],\
                                              '14': [hybrid_out[13],hybrid_out_pred[13],label_set[13]]},
                                             ignore_index=True)
                sys.stdout.write('Done with {} and {}'.format(loc, date))
ts_output.to_csv('80mi_pred.csv')
#unroller
results_output = pd.DataFrame(columns = ['param_name','loc','date'])

col_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
for row in ts_output.iterrows():
    woo = 1
    
    row_dict = {'model' : 'ts', 
                'param_name' : row[1][0], 
                'loc' : row[1][1], 
                'date' : row[1][2], 
                '1' : row[1][3][1],
                '2' : row[1][4][1], 
                '3' : row[1][5][1],
                '4' : row[1][6][1],
                '5' : row[1][7][1],                  
                '6' : row[1][8][1],
                '7' : row[1][9][1],
                '8' : row[1][10][1],
                '9' : row[1][11][1],
                '10' : row[1][12][1],
                '11' : row[1][13][1],
                '12' : row[1][14][1],
                '13' : row[1][15][1],
                '14' : row[1][16][1]}
    '''
    
    for col in col_list:
        if row[1][col].strip('][').split(', ')[0]!='nan':
            temp_col_val = row[1][col].strip('][').split(', ')
            num = float(temp_col_val[1])
            if num<1:
                num=1
            true_val = abs(num - float(temp_col_val[2]))/float(temp_col_val[2])
            row[1][col]=true_val
        else:
            row[1][col] = np.nan
    row_dict = {'model' : 'ts', 
                'param_name' : row[1][1], 
                'loc' : row[1][2], 
                'date' : row[1][3], 
                '1' : row[1][4],
                '2' : row[1][5], 
                '3' : row[1][6], 
                '4' : row[1][7], 
                '5' : row[1][8],                     
                '6' : row[1][9], 
                '7' : row[1][10], 
                '8' : row[1][11], 
                '9' : row[1][12], 
                '10' : row[1][13], 
                '11' : row[1][14], 
                '12' : row[1][15], 
                '13' : row[1][16], 
                '14' : row[1][17]}
    '''
    results_output=results_output.append(row_dict, ignore_index=True)


results_output.to_csv('clean_80mi_results.csv')
    
#%% ards
ts_output = pd.DataFrame(columns = ['param_name','loc','date','1','2','3','4','5','6',
                                    '7','8','9','10','11','12','13','14'])
#iterlist = ['loess_2', 'loess_25', 'loess_3', 'loess_35', 'loess_4',
#       'loess_adf', 'loess_adf_05', 'loess_adf_1', 'loess_adf_15',
#       'loess_adf_2', 'loess_adf_25']

iterlist = ['kz_f7']
param_name='kz_f7'
#iterlist = ['loess_f02', 'loess_f025', 'loess_f03', 'loess_f035', 'loess_f04',
#       'loess_f045', 'loess_f05']
date_data_2 = pd.read_pickle('C:\\Users\\u6026797\\ards_oracle_kz.pkl')
date_data_og = pd.read_pickle('C:\\Users\\u6026797\\ards_flat_kz.pkl')
#date_data_og = pd.read_pickle('C:\\Users\\u6026797\\downloads\\PyGAMFlatHyperparameter.pkl') 

for param_name in iterlist:
    out_file_name = param_name+'case_results_file.csv'
    date_data=date_data_og[['Country_Region','Date','Dt','DayOfWeek',param_name,'LastDate']]
    date_data =date_data.rename(columns={param_name: "kz"})
            
    date_list=pd.unique(date_data['LastDate'])
    date_list = pd.DataFrame(date_list)
    date_list = date_list.sort_values(by=0)
    date_list = date_list.to_numpy()
    date_list = np.reshape(date_list, newshape=(len(date_list),))
    
    loc_list=pd.DataFrame(pd.unique(date_data['Country_Region']))
    loc_list=loc_list.rename(columns={0: "Country_Region"})
    #generate int index of location
    loc_list.insert(0,'Index',range(0,len(loc_list)))
    
    date_data=date_data.merge(loc_list,how='left',on='Country_Region')
    
    val_2_vec = []
    for row in date_data.iterrows():
      val_2_vec.append(np.broadcast_to(np.array(row[1]['Index']),shape=(len(row[1].iloc[2]),1)))
    val_2_vec_df = pd.DataFrame(val_2_vec)
    date_data['Index']=val_2_vec_df
    
    date_data=date_data[['Country_Region','Date','Dt','DayOfWeek','kz','LastDate','Index']]
    #state_list = loc_list[24:]
    #date_data=date_data[date_data['Country_Region'].isin(state_list.iloc[:,1])]
    
    #gam modifications
    date_data = date_data[(date_data['LastDate']>='2020-07-01 00:00:00')]
    if len(date_list)> 30:
        date_list = date_list[31:]
    last_date = date_list[len(date_list)-1]
    for row in date_data.iterrows():        
      if row[1]['LastDate']==last_date:
        loc_in=row[1]['Country_Region']
        oracle_row=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]#['oracle_kz']
        #get min date
        min_date=row[1]['Date'][0]
        #get shorter sequence len
        new_len=len(oracle_row['Date'].to_list()[0][oracle_row['Date'].to_list()[0]>min_date])
        #truncate labels to input sequence length
        Oracle_kz=oracle_row['Oracle_kz'].to_list()[0][-new_len:]
        row[1]['kz']=Oracle_kz
    
    
    #for row in date_data.iterrows():
    #  if row[1]['LastDate']==last_date:
    #    loc_in=row[1]['Country_Region']
    #    row[1]['kz']=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]['oracle_kz']
    
    #temp date restriction
    #temp_start_date = date_list[62]
    #date_data = date_data[(date_data['LastDate']>=temp_start_date)]
    #date_list = date_list[62:]
    #loc_list = loc_list[:55]

    for loc in loc_list.iloc[:,1]:
        #loc=loc_list.iloc[288,1]
        for date in date_list:
            #date=date_list[0]
            comp_ts = date_data[(date_data['LastDate']==date) & (date_data['Country_Region']==loc)]
            if len(comp_ts) !=0:
                comp_ts = comp_ts['kz'].tolist()[0]
                hybrid_out = r_hybrid_mod_wrapper(ts=comp_ts)
                hybrid_out_pred = hybrid_out
                #gather label data and calc loss
                label_set = date_data[(date_data['LastDate']==last_date) & (date_data['Country_Region']==loc)]
                label_set = label_set['kz'].tolist()[0]
                label_set = label_set[len(comp_ts):(len(comp_ts)+14)]
                if len(label_set)<len(hybrid_out):
                    hybrid_out=hybrid_out[0:len(label_set)]
                hybrid_out = np.round((abs(hybrid_out-label_set)/label_set),4)
                for pred in range(0,len(hybrid_out)):
                    if hybrid_out[pred]<1:
                        hybrid_out[pred]= 1
                for na_pad in range((len(hybrid_out)-1),13):
                    hybrid_out=np.append(hybrid_out,np.NaN)
                    label_set=np.append(label_set,np.NaN)
                    hybrid_out_pred=np.append(hybrid_out_pred,np.NaN)
                    
                
                ts_output = ts_output.append({'param_name': param_name,\
                                              'loc' : loc, \
                                              'date' : date,\
                                              '1' : [hybrid_out[0],hybrid_out_pred[0],label_set[0]],\
                                              '2' : [hybrid_out[1],hybrid_out_pred[1],label_set[1]],\
                                              '3' : [hybrid_out[2],hybrid_out_pred[2],label_set[2]],\
                                              '4' : [hybrid_out[3],hybrid_out_pred[3],label_set[3]],\
                                              '5' : [hybrid_out[4],hybrid_out_pred[4],label_set[4]],\
                                              '6' : [hybrid_out[5],hybrid_out_pred[5],label_set[5]],\
                                              '7' : [hybrid_out[6],hybrid_out_pred[6],label_set[6]],\
                                              '8' : [hybrid_out[7],hybrid_out_pred[7],label_set[7]],\
                                              '9' : [hybrid_out[8],hybrid_out_pred[8],label_set[8]],\
                                              '10': [hybrid_out[9],hybrid_out_pred[9],label_set[9]],\
                                              '11': [hybrid_out[10],hybrid_out_pred[10],label_set[10]],\
                                              '12': [hybrid_out[11],hybrid_out_pred[11],label_set[11]],\
                                              '13': [hybrid_out[12],hybrid_out_pred[12],label_set[12]],\
                                              '14': [hybrid_out[13],hybrid_out_pred[13],label_set[13]]},
                                             ignore_index=True)
                sys.stdout.write('Done with {} and {}'.format(loc, date))
ts_output.to_csv('ards_pred.csv')
#unroller
results_output = pd.DataFrame(columns = ['param_name','loc','date'])

col_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
for row in ts_output.iterrows():
    woo = 1
    
    row_dict = {'model' : 'ts', 
                'param_name' : row[1][0], 
                'loc' : row[1][1], 
                'date' : row[1][2], 
                '1' : row[1][3][1],
                '2' : row[1][4][1], 
                '3' : row[1][5][1],
                '4' : row[1][6][1],
                '5' : row[1][7][1],                  
                '6' : row[1][8][1],
                '7' : row[1][9][1],
                '8' : row[1][10][1],
                '9' : row[1][11][1],
                '10' : row[1][12][1],
                '11' : row[1][13][1],
                '12' : row[1][14][1],
                '13' : row[1][15][1],
                '14' : row[1][16][1]}
    '''
    
    for col in col_list:
        if row[1][col].strip('][').split(', ')[0]!='nan':
            temp_col_val = row[1][col].strip('][').split(', ')
            num = float(temp_col_val[1])
            if num<1:
                num=1
            true_val = abs(num - float(temp_col_val[2]))/float(temp_col_val[2])
            row[1][col]=true_val
        else:
            row[1][col] = np.nan
    row_dict = {'model' : 'ts', 
                'param_name' : row[1][1], 
                'loc' : row[1][2], 
                'date' : row[1][3], 
                '1' : row[1][4],
                '2' : row[1][5], 
                '3' : row[1][6], 
                '4' : row[1][7], 
                '5' : row[1][8],                     
                '6' : row[1][9], 
                '7' : row[1][10], 
                '8' : row[1][11], 
                '9' : row[1][12], 
                '10' : row[1][13], 
                '11' : row[1][14], 
                '12' : row[1][15], 
                '13' : row[1][16], 
                '14' : row[1][17]}
    '''
    results_output=results_output.append(row_dict, ignore_index=True)


results_output.to_csv('clean_ards_results.csv')

#%% hrr
ts_output = pd.DataFrame(columns = ['param_name','loc','date','1','2','3','4','5','6',
                                    '7','8','9','10','11','12','13','14'])
#iterlist = ['loess_2', 'loess_25', 'loess_3', 'loess_35', 'loess_4',
#       'loess_adf', 'loess_adf_05', 'loess_adf_1', 'loess_adf_15',
#       'loess_adf_2', 'loess_adf_25']

iterlist = ['kz_f7']
param_name='kz_f7'
#iterlist = ['loess_f02', 'loess_f025', 'loess_f03', 'loess_f035', 'loess_f04',
#       'loess_f045', 'loess_f05']
date_data_2 = pd.read_pickle('C:\\Users\\u6026797\\hrr_oracle_kz.pkl')
date_data_og = pd.read_pickle('C:\\Users\\u6026797\\hrr_flat_kz.pkl')
#date_data_og = pd.read_pickle('C:\\Users\\u6026797\\downloads\\PyGAMFlatHyperparameter.pkl') 

for param_name in iterlist:
    out_file_name = param_name+'case_results_file.csv'
    date_data=date_data_og[['Country_Region','Date','Dt','DayOfWeek',param_name,'LastDate']]
    date_data =date_data.rename(columns={param_name: "kz"})
            
    date_list=pd.unique(date_data['LastDate'])
    date_list = pd.DataFrame(date_list)
    date_list = date_list.sort_values(by=0)
    date_list = date_list.to_numpy()
    date_list = np.reshape(date_list, newshape=(len(date_list),))
    
    loc_list=pd.DataFrame(pd.unique(date_data['Country_Region']))
    loc_list=loc_list.rename(columns={0: "Country_Region"})
    #generate int index of location
    loc_list.insert(0,'Index',range(0,len(loc_list)))
    
    date_data=date_data.merge(loc_list,how='left',on='Country_Region')
    
    val_2_vec = []
    for row in date_data.iterrows():
      val_2_vec.append(np.broadcast_to(np.array(row[1]['Index']),shape=(len(row[1].iloc[2]),1)))
    val_2_vec_df = pd.DataFrame(val_2_vec)
    date_data['Index']=val_2_vec_df
    
    date_data=date_data[['Country_Region','Date','Dt','DayOfWeek','kz','LastDate','Index']]
    #state_list = loc_list[24:]
    #date_data=date_data[date_data['Country_Region'].isin(state_list.iloc[:,1])]
    
    #gam modifications
    date_data = date_data[(date_data['LastDate']>='2020-07-01 00:00:00')]
    if len(date_list)> 30:
        date_list = date_list[31:]
    last_date = date_list[len(date_list)-1]
    for row in date_data.iterrows():        
      if row[1]['LastDate']==last_date:
        loc_in=row[1]['Country_Region']
        oracle_row=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]#['oracle_kz']
        #get min date
        min_date=row[1]['Date'][0]
        #get shorter sequence len
        new_len=len(oracle_row['Date'].to_list()[0][oracle_row['Date'].to_list()[0]>min_date])
        #truncate labels to input sequence length
        Oracle_kz=oracle_row['Oracle_kz'].to_list()[0][-new_len:]
        row[1]['kz']=Oracle_kz
    
    
    #for row in date_data.iterrows():
    #  if row[1]['LastDate']==last_date:
    #    loc_in=row[1]['Country_Region']
    #    row[1]['kz']=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]['oracle_kz']
    
    #temp date restriction
    #temp_start_date = date_list[62]
    #date_data = date_data[(date_data['LastDate']>=temp_start_date)]
    #date_list = date_list[62:]
    #loc_list = loc_list[:55]

    for loc in loc_list.iloc[:,1]:
        #loc=loc_list.iloc[288,1]
        for date in date_list:
            #date=date_list[0]
            comp_ts = date_data[(date_data['LastDate']==date) & (date_data['Country_Region']==loc)]
            if len(comp_ts) !=0:
                comp_ts = comp_ts['kz'].tolist()[0]
                hybrid_out = r_hybrid_mod_wrapper(ts=comp_ts)
                hybrid_out_pred = hybrid_out
                #gather label data and calc loss
                label_set = date_data[(date_data['LastDate']==last_date) & (date_data['Country_Region']==loc)]
                label_set = label_set['kz'].tolist()[0]
                label_set = label_set[len(comp_ts):(len(comp_ts)+14)]
                if len(label_set)<len(hybrid_out):
                    hybrid_out=hybrid_out[0:len(label_set)]
                hybrid_out = np.round((abs(hybrid_out-label_set)/label_set),4)
                for pred in range(0,len(hybrid_out)):
                    if hybrid_out[pred]<1:
                        hybrid_out[pred]= 1
                for na_pad in range((len(hybrid_out)-1),13):
                    hybrid_out=np.append(hybrid_out,np.NaN)
                    label_set=np.append(label_set,np.NaN)
                    hybrid_out_pred=np.append(hybrid_out_pred,np.NaN)
                    
                
                ts_output = ts_output.append({'param_name': param_name,\
                                              'loc' : loc, \
                                              'date' : date,\
                                              '1' : [hybrid_out[0],hybrid_out_pred[0],label_set[0]],\
                                              '2' : [hybrid_out[1],hybrid_out_pred[1],label_set[1]],\
                                              '3' : [hybrid_out[2],hybrid_out_pred[2],label_set[2]],\
                                              '4' : [hybrid_out[3],hybrid_out_pred[3],label_set[3]],\
                                              '5' : [hybrid_out[4],hybrid_out_pred[4],label_set[4]],\
                                              '6' : [hybrid_out[5],hybrid_out_pred[5],label_set[5]],\
                                              '7' : [hybrid_out[6],hybrid_out_pred[6],label_set[6]],\
                                              '8' : [hybrid_out[7],hybrid_out_pred[7],label_set[7]],\
                                              '9' : [hybrid_out[8],hybrid_out_pred[8],label_set[8]],\
                                              '10': [hybrid_out[9],hybrid_out_pred[9],label_set[9]],\
                                              '11': [hybrid_out[10],hybrid_out_pred[10],label_set[10]],\
                                              '12': [hybrid_out[11],hybrid_out_pred[11],label_set[11]],\
                                              '13': [hybrid_out[12],hybrid_out_pred[12],label_set[12]],\
                                              '14': [hybrid_out[13],hybrid_out_pred[13],label_set[13]]},
                                             ignore_index=True)
                sys.stdout.write('Done with {} and {}'.format(loc, date))
ts_output.to_csv('hrr_pred.csv')
#unroller
results_output = pd.DataFrame(columns = ['param_name','loc','date'])

col_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
for row in ts_output.iterrows():
    woo = 1
    
    row_dict = {'model' : 'ts', 
                'param_name' : row[1][0], 
                'loc' : row[1][1], 
                'date' : row[1][2], 
                '1' : row[1][3][1],
                '2' : row[1][4][1], 
                '3' : row[1][5][1],
                '4' : row[1][6][1],
                '5' : row[1][7][1],                  
                '6' : row[1][8][1],
                '7' : row[1][9][1],
                '8' : row[1][10][1],
                '9' : row[1][11][1],
                '10' : row[1][12][1],
                '11' : row[1][13][1],
                '12' : row[1][14][1],
                '13' : row[1][15][1],
                '14' : row[1][16][1]}
    '''
    
    for col in col_list:
        if row[1][col].strip('][').split(', ')[0]!='nan':
            temp_col_val = row[1][col].strip('][').split(', ')
            num = float(temp_col_val[1])
            if num<1:
                num=1
            true_val = abs(num - float(temp_col_val[2]))/float(temp_col_val[2])
            row[1][col]=true_val
        else:
            row[1][col] = np.nan
    row_dict = {'model' : 'ts', 
                'param_name' : row[1][1], 
                'loc' : row[1][2], 
                'date' : row[1][3], 
                '1' : row[1][4],
                '2' : row[1][5], 
                '3' : row[1][6], 
                '4' : row[1][7], 
                '5' : row[1][8],                     
                '6' : row[1][9], 
                '7' : row[1][10], 
                '8' : row[1][11], 
                '9' : row[1][12], 
                '10' : row[1][13], 
                '11' : row[1][14], 
                '12' : row[1][15], 
                '13' : row[1][16], 
                '14' : row[1][17]}
    '''
    results_output=results_output.append(row_dict, ignore_index=True)


results_output.to_csv('hrr_80mi_results.csv')

#%% state
ts_output = pd.DataFrame(columns = ['param_name','loc','date','1','2','3','4','5','6',
                                    '7','8','9','10','11','12','13','14'])
#iterlist = ['loess_2', 'loess_25', 'loess_3', 'loess_35', 'loess_4',
#       'loess_adf', 'loess_adf_05', 'loess_adf_1', 'loess_adf_15',
#       'loess_adf_2', 'loess_adf_25']

iterlist = ['kz_f7']
param_name='kz_f7'
#iterlist = ['loess_f02', 'loess_f025', 'loess_f03', 'loess_f035', 'loess_f04',
#       'loess_f045', 'loess_f05']
date_data_2 = pd.read_pickle('C:\\Users\\u6026797\\country_state_oracle_kz.pkl')
date_data_og = pd.read_pickle('C:\\Users\\u6026797\\country_state_flat_kz.pkl')
#date_data_og = pd.read_pickle('C:\\Users\\u6026797\\downloads\\PyGAMFlatHyperparameter.pkl') 

for param_name in iterlist:
    out_file_name = param_name+'case_results_file.csv'
    date_data=date_data_og[['Country_Region','Date','Dt','DayOfWeek',param_name,'LastDate']]
    date_data =date_data.rename(columns={param_name: "kz"})
            
    date_list=pd.unique(date_data['LastDate'])
    date_list = pd.DataFrame(date_list)
    date_list = date_list.sort_values(by=0)
    date_list = date_list.to_numpy()
    date_list = np.reshape(date_list, newshape=(len(date_list),))
    
    loc_list=pd.DataFrame(pd.unique(date_data['Country_Region']))
    loc_list=loc_list.rename(columns={0: "Country_Region"})
    #generate int index of location
    loc_list.insert(0,'Index',range(0,len(loc_list)))
    
    date_data=date_data.merge(loc_list,how='left',on='Country_Region')
    
    val_2_vec = []
    for row in date_data.iterrows():
      val_2_vec.append(np.broadcast_to(np.array(row[1]['Index']),shape=(len(row[1].iloc[2]),1)))
    val_2_vec_df = pd.DataFrame(val_2_vec)
    date_data['Index']=val_2_vec_df
    
    date_data=date_data[['Country_Region','Date','Dt','DayOfWeek','kz','LastDate','Index']]
    #state_list = loc_list[24:]
    #date_data=date_data[date_data['Country_Region'].isin(state_list.iloc[:,1])]
    
    #gam modifications
    date_data = date_data[(date_data['LastDate']>='2020-07-01 00:00:00')]
    if len(date_list)> 30:
        date_list = date_list[31:]
    last_date = date_list[len(date_list)-1]
    for row in date_data.iterrows():        
      if row[1]['LastDate']==last_date:
        loc_in=row[1]['Country_Region']
        oracle_row=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]#['oracle_kz']
        #get min date
        min_date=row[1]['Date'][0]
        #get shorter sequence len
        new_len=len(oracle_row['Date'].to_list()[0][oracle_row['Date'].to_list()[0]>min_date])
        #truncate labels to input sequence length
        Oracle_kz=oracle_row['Oracle_kz'].to_list()[0][-new_len:]
        row[1]['kz']=Oracle_kz
    
    
    #for row in date_data.iterrows():
    #  if row[1]['LastDate']==last_date:
    #    loc_in=row[1]['Country_Region']
    #    row[1]['kz']=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]['oracle_kz']
    
    #temp date restriction
    #temp_start_date = date_list[62]
    #date_data = date_data[(date_data['LastDate']>=temp_start_date)]
    #date_list = date_list[62:]
    #loc_list = loc_list[:55]

    for loc in loc_list.iloc[:,1]:
        #loc=loc_list.iloc[288,1]
        for date in date_list:
            #date=date_list[0]
            comp_ts = date_data[(date_data['LastDate']==date) & (date_data['Country_Region']==loc)]
            if len(comp_ts) !=0:
                comp_ts = comp_ts['kz'].tolist()[0]
                hybrid_out = r_hybrid_mod_wrapper(ts=comp_ts)
                hybrid_out_pred = hybrid_out
                #gather label data and calc loss
                label_set = date_data[(date_data['LastDate']==last_date) & (date_data['Country_Region']==loc)]
                label_set = label_set['kz'].tolist()[0]
                label_set = label_set[len(comp_ts):(len(comp_ts)+14)]
                if len(label_set)<len(hybrid_out):
                    hybrid_out=hybrid_out[0:len(label_set)]
                hybrid_out = np.round((abs(hybrid_out-label_set)/label_set),4)
                for pred in range(0,len(hybrid_out)):
                    if hybrid_out[pred]<1:
                        hybrid_out[pred]= 1
                for na_pad in range((len(hybrid_out)-1),13):
                    hybrid_out=np.append(hybrid_out,np.NaN)
                    label_set=np.append(label_set,np.NaN)
                    hybrid_out_pred=np.append(hybrid_out_pred,np.NaN)
                    
                
                ts_output = ts_output.append({'param_name': param_name,\
                                              'loc' : loc, \
                                              'date' : date,\
                                              '1' : [hybrid_out[0],hybrid_out_pred[0],label_set[0]],\
                                              '2' : [hybrid_out[1],hybrid_out_pred[1],label_set[1]],\
                                              '3' : [hybrid_out[2],hybrid_out_pred[2],label_set[2]],\
                                              '4' : [hybrid_out[3],hybrid_out_pred[3],label_set[3]],\
                                              '5' : [hybrid_out[4],hybrid_out_pred[4],label_set[4]],\
                                              '6' : [hybrid_out[5],hybrid_out_pred[5],label_set[5]],\
                                              '7' : [hybrid_out[6],hybrid_out_pred[6],label_set[6]],\
                                              '8' : [hybrid_out[7],hybrid_out_pred[7],label_set[7]],\
                                              '9' : [hybrid_out[8],hybrid_out_pred[8],label_set[8]],\
                                              '10': [hybrid_out[9],hybrid_out_pred[9],label_set[9]],\
                                              '11': [hybrid_out[10],hybrid_out_pred[10],label_set[10]],\
                                              '12': [hybrid_out[11],hybrid_out_pred[11],label_set[11]],\
                                              '13': [hybrid_out[12],hybrid_out_pred[12],label_set[12]],\
                                              '14': [hybrid_out[13],hybrid_out_pred[13],label_set[13]]},
                                             ignore_index=True)
                sys.stdout.write('Done with {} and {}'.format(loc, date))
ts_output.to_csv('country_state_pred.csv')
#unroller
results_output = pd.DataFrame(columns = ['param_name','loc','date'])

col_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
for row in ts_output.iterrows():
    woo = 1
    
    row_dict = {'model' : 'ts', 
                'param_name' : row[1][0], 
                'loc' : row[1][1], 
                'date' : row[1][2], 
                '1' : row[1][3][1],
                '2' : row[1][4][1], 
                '3' : row[1][5][1],
                '4' : row[1][6][1],
                '5' : row[1][7][1],                  
                '6' : row[1][8][1],
                '7' : row[1][9][1],
                '8' : row[1][10][1],
                '9' : row[1][11][1],
                '10' : row[1][12][1],
                '11' : row[1][13][1],
                '12' : row[1][14][1],
                '13' : row[1][15][1],
                '14' : row[1][16][1]}
    '''
    
    for col in col_list:
        if row[1][col].strip('][').split(', ')[0]!='nan':
            temp_col_val = row[1][col].strip('][').split(', ')
            num = float(temp_col_val[1])
            if num<1:
                num=1
            true_val = abs(num - float(temp_col_val[2]))/float(temp_col_val[2])
            row[1][col]=true_val
        else:
            row[1][col] = np.nan
    row_dict = {'model' : 'ts', 
                'param_name' : row[1][1], 
                'loc' : row[1][2], 
                'date' : row[1][3], 
                '1' : row[1][4],
                '2' : row[1][5], 
                '3' : row[1][6], 
                '4' : row[1][7], 
                '5' : row[1][8],                     
                '6' : row[1][9], 
                '7' : row[1][10], 
                '8' : row[1][11], 
                '9' : row[1][12], 
                '10' : row[1][13], 
                '11' : row[1][14], 
                '12' : row[1][15], 
                '13' : row[1][16], 
                '14' : row[1][17]}
    '''
    results_output=results_output.append(row_dict, ignore_index=True)


results_output.to_csv('country_state_results.csv')
'''