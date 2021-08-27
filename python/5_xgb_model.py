# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:55:22 2020

@author: Chris Wilson
"""

#%% Libraries 
import xgboost as xgb
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
#import pygam as pg
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
import tsaug
import random 
from sklearn.model_selection import GridSearchCV
import warnings
from datetime import datetime

def shift2(arr,num):
    '''
    Convience backshift operator function
    '''
    arr=np.roll(arr,num)
    num=min(num,np.shape(arr)[0])
    if num<0:
         np.put(arr,range(len(arr)+num,len(arr)),0)
    elif num > 0:
         np.put(arr,range(num),0)
    return arr

def ts_aug_avg(ts_1,ts_2):
    '''
    Workhorse function for augmentation. Handles scaling, permutation and shape.
    '''
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

def ts_resampler(data_frame, n_times=5, kz_col = 4, count_start = 76):#, exclude = 0):
    '''
    Parent function reads in and permutes time series data to enrich training set and
    improve predictions. Logic ensures perumations meet non-zero requirements 
    and appropriate dimensionality.
    '''
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
        if np.sum(kz)!=len(kz) and sum(kz==1)/len(kz) < .2: #and ind1!=exclude and ind2!=exclude:
            #ind1 = comp_ts_out.iloc[ind1,0] 
            #ind2 = comp_ts_out.iloc[ind2,0] 
            for date in date_list:
                #comp_ts = date_data[(date_data['LastDate']==date)]
                comp_ts = data_frame[(data_frame['LastDate']==date)]
                try:
                    new_feat_meta = comp_ts.iloc[ind1,:]
                    new_feat_meta['Country_Region']='Augment_'+str(ind1)+'+'+str(ind2)
                    #was 52
                    new_feat_meta['Index']=np.broadcast_to(np.array((count_start+times)),shape=(len(new_feat_meta['Index']),1))
                    kz = ts_aug_avg(comp_ts.iloc[ind1,kz_col],comp_ts.iloc[ind2,kz_col])
                    new_feat_meta['kz']=kz
                    df=df.append(new_feat_meta,ignore_index=True)
                except:
                    pass
            times+=1
    return(df)

def smoothed_der(ts,smooth=2):
    '''
    Convenience function for generating smoothed backshift estimated, discrete
    derivatives.
    '''
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
#%% Data
xgb_output = pd.DataFrame(columns = ['param_name','loc','date'])
iterlist = ['Denoised_Curve_Opt', 'Denoised_Curve_Half', 'Denoised_Curve_Quarter',
            'Denoised_Curve_Twice', 'Denoised_Curve_Four',
            'Denoised_Curve_Point', 'Denoised_Curve_Ten']

date_data_2 = pd.read_pickle('/home/ec2-user/rona_folder/new_kz_adf.pkl')
date_data_og = pd.read_pickle('/home/ec2-user/rona_folder/all_loess.pkl')
#date_data_og = pd.read_pickle('/home/ec2-user/rona_folder/PyGAMFlatHyperparameter.pkl')
#date_data_og = pd.read_pickle('/home/ec2-user/rona_folder/kz_flat_params.pkl')
for param_name in iterlist:
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
    last_date = date_list[len(date_list)-1]
    
    for row in date_data.iterrows():
      if row[1]['LastDate']==last_date:
        loc_in=row[1]['Country_Region']
        row[1]['kz']=date_data_2[(date_data_2['Country_Region']==loc_in) & (date_data_2['LastDate']==last_date)]['oracle_kz']
    
    #state list subset
    state_list = loc_list[:55]
    date_data=date_data[date_data['Country_Region'].isin(state_list.iloc[:,1])]
    
    
    
    #date reduction
    date_data = date_data[(date_data['LastDate']>='2020-06-01 00:00:00')]
    date_data_aug = ts_resampler(date_data, n_times=len(loc_list),count_start = len(loc_list))
    #date_list = date_list[62:]
    
    #state list subset
    #state_list = loc_list[21:]
    
    ####data integrity check###
    #for i in range(0,82):
    #    temp_x = date_data[(date_data['Country_Region']=='WY')].iloc[i,:]
    #    print(max(temp_x['Dt'])==len(temp_x['Dt'])-1)
        
    for loc in state_list.iloc[:,0]:
      #loc=state_list.iloc[8,0]
      print(datetime.now().strftime("%H:%M:%S"))
      #date_data_aug = ts_resampler(date_data, n_times=len(loc_list))
      for date in date_list:
        print('date done: {}, time taken: {}'.format(date,datetime.now().strftime("%H:%M:%S")))
        #date=date_list[0]
        comp_ts = date_data_aug[(date_data_aug['LastDate']==date)]
        comp_ts = comp_ts[['Index','Dt','DayOfWeek','kz']]
        
        ####test code###
        #comp_ts = comp_ts[['Index','Dt','DayOfWeek','kz','Open','Dew','Dense']]
        ##data augmentation experimental run
        
        #n_times = round(len(comp_ts)/2)
        #comp_ts = ts_resampler(comp_ts, n_times=n_times)
        
        comp_ts_np = np.hstack((np.reshape(comp_ts.iloc[0,0],newshape=(len(comp_ts.iloc[0,1]),1)), #state intercept
                               np.reshape(comp_ts.iloc[0,1],newshape=(len(comp_ts.iloc[0,1]),1)), #index
                               np.reshape(comp_ts.iloc[0,2],newshape=(len(comp_ts.iloc[0,2]),1)), #dow
                               #add diffs
                               np.reshape(np.diff(comp_ts.iloc[0,3],prepend=0),newshape=(len(comp_ts.iloc[0,3]),1)), #diff
                               np.reshape(smoothed_der(shift2(
                                   np.reshape(comp_ts.iloc[0,3],newshape=(len(comp_ts.iloc[0,3]),1)),2)),newshape=(len(comp_ts.iloc[0,3]),1)),
                               np.reshape(smoothed_der(shift2(
                                   np.reshape(comp_ts.iloc[0,3],newshape=(len(comp_ts.iloc[0,3]),1)),3)),newshape=(len(comp_ts.iloc[0,3]),1)),
                               np.reshape(smoothed_der(shift2(
                                   np.reshape(comp_ts.iloc[0,3],newshape=(len(comp_ts.iloc[0,3]),1)),4)),newshape=(len(comp_ts.iloc[0,3]),1)),
                               ###add lags
                               shift2(np.reshape(comp_ts.iloc[0,3],newshape=(len(comp_ts.iloc[0,3]),1)),1),
                               shift2(np.reshape(comp_ts.iloc[0,3],newshape=(len(comp_ts.iloc[0,3]),1)),2),
                               shift2(np.reshape(comp_ts.iloc[0,3],newshape=(len(comp_ts.iloc[0,3]),1)),3),
                               shift2(np.reshape(comp_ts.iloc[0,3],newshape=(len(comp_ts.iloc[0,3]),1)),4),
                               #val
                               np.reshape(comp_ts.iloc[0,3],newshape=(len(comp_ts.iloc[0,3]),1)) #value
                               ))
        comp_ts = comp_ts.tail(len(comp_ts)-1)
        for row in comp_ts.iterrows():
          temp_row = row[1]
          chunk = np.hstack((np.reshape(temp_row[0],newshape=(len(temp_row[0]),1)),
                             np.reshape(temp_row[1],newshape=(len(temp_row[1]),1)),
                             np.reshape(temp_row[2],newshape=(len(temp_row[2]),1)),
                             #diffs
                             np.reshape(np.diff(temp_row[3],prepend=0),newshape=(len(temp_row[3]),1)),
                             np.reshape(smoothed_der(shift2(
                                   np.reshape(temp_row[3],newshape=(len(temp_row[3]),1)),2)),newshape=(len(temp_row[3]),1)),
                             np.reshape(smoothed_der(shift2(
                                   np.reshape(temp_row[3],newshape=(len(temp_row[3]),1)),3)),newshape=(len(temp_row[3]),1)),
                             np.reshape(smoothed_der(shift2(
                                   np.reshape(temp_row[3],newshape=(len(temp_row[3]),1)),4)),newshape=(len(temp_row[3]),1)),
                             #lags
                             shift2(np.reshape(temp_row[3],newshape=(len(temp_row[3]),1)),1),
                             shift2(np.reshape(temp_row[3],newshape=(len(temp_row[3]),1)),2),
                             shift2(np.reshape(temp_row[3],newshape=(len(temp_row[3]),1)),3),
                             shift2(np.reshape(temp_row[3],newshape=(len(temp_row[3]),1)),4),
                             ####test code###
                             ###add additional data###
                             #np.reshape(temp_row[4],newshape=(len(temp_row[4]),1)),
                             #np.reshape(temp_row[5],newshape=(len(temp_row[5]),1)),
                             #np.reshape(temp_row[6],newshape=(len(temp_row[6]),1)),
                             #np.reshape(temp_row[4],newshape=(len(temp_row[4]),1)),
                             np.reshape(temp_row[3],newshape=(len(temp_row[3]),1))
                             ))
          comp_ts_np = np.vstack((comp_ts_np,chunk))
          
        #for i in np.unique(comp_ts_np[:,0]):
        #    print('loc {} is good {}'.format(i,max(comp_ts_np[comp_ts_np[:,0]==i][:,1])==len(comp_ts_np[comp_ts_np[:,0]==i][:,1])-1))
            
        if len(comp_ts_np[comp_ts_np[:,0]==loc])!=0:
            #generate iterative label columns
        
            ##test code ##
            #generate labels from oracle gam
            results = {'param_name':param_name,'loc':[loc_list[loc_list['Index']==loc]['Country_Region'][loc]],
                       'date':[date]}
            for pred_step in range(1,15):
              #gather index data
              loc_index = np.where(loc_list == loc)[0][0]
              lazy_date = 86400000000000 * pred_step 
              #ensure date is available
              if sum(date_list==(date+lazy_date))==0:
                #pred cannot be made, output NaN
                  results[pred_step]=[np.NAN]
              else:
                #subset for date step  
                comp_ts = date_data_aug[(date_data_aug['LastDate']==last_date)]
                #comp_ts = date_data_aug[(date_data_aug['LastDate']==date)]
                #comp_ts_0 = comp_ts[['kz']]
                comp_ts=comp_ts[['Index','kz']]
                #comp_ts.insert(0,'Index',range(0,len(loc_list)))
                #generate empty array
                label_ts_np=np.zeros((1,1))
                
                for row in comp_ts.iterrows():
                  #get loc id
                  loc_id = row[1][0][0][0]
                  #retrieve start and stop len
                  if np.sum(comp_ts_np[:,0]==loc_id)!=0:
                      start_ts = int(min(comp_ts_np[comp_ts_np[:,0]==loc_id][:,1])+pred_step)
                      end_ts = int(len(comp_ts_np[comp_ts_np[:,0]==loc_id][:,1])+pred_step)#int(max(comp_ts_np[comp_ts_np[:,0]==loc_id][:,1])+pred_step+1)
                      temp_row= row[1][1][start_ts:end_ts]
                      temp_row = np.reshape(temp_row,newshape=(len(temp_row),1))
                      #if len(temp_row)!=len(comp_ts_np[comp_ts_np[:,0]==loc_id][:,1]):
                      #    temp_row=np.vstack((temp_row,temp_row[len(temp_row)-1]))
                      label_ts_np=np.vstack((label_ts_np,temp_row))
                     
                        ################################################### 
                      
                      ##test code##
                      if len(temp_row)!=len(comp_ts_np[comp_ts_np[:,0]==loc_id][:,1]):
                          idea=row
                
                #merge data with label
                label_ts_np = label_ts_np[1:len(label_ts_np)]
                comp_ts_np = np.hstack((comp_ts_np,label_ts_np))
            
            #take the last 5 of each location and put it into the val set
            #create np.array
            val_ts=np.zeros(shape=(1,np.shape(comp_ts_np)[1]))
            updated_comp_ts=np.zeros(shape=(1,np.shape(comp_ts_np)[1]))
            for loc_sub in loc_list.iloc[:,0]:
              svt = comp_ts_np[comp_ts_np[:,0]==loc_sub]
              val_ts = np.vstack((val_ts, svt[(len(svt)-8):(len(svt)-1),:]))
              updated_comp_ts = np.vstack((updated_comp_ts, svt[0:(len(svt)-8),:]))
            
            val_ts = val_ts[1:len(val_ts)]
            updated_comp_ts = updated_comp_ts[1:len(updated_comp_ts)]
            #split target ts from comp ts set
            target_ts = comp_ts_np[comp_ts_np[:,0]==loc]
            comp_ts_np = updated_comp_ts
            
            #get recent slope to determine if recently flat to increase gamma for xgb
            recent_slope = np.mean(target_ts[len(target_ts)-1][3:7])
            
            
            #comp_ts_np = updated_comp_ts[updated_comp_ts[:,0]!=loc]
    
            #train and run model
            end_len = 12
            for label_col in range(0,(np.shape(comp_ts_np)[1]-end_len)):
              #print(label_col)
              #divide into test, train, val
              train_x, train_y=comp_ts_np[:,0:end_len], comp_ts_np[:,(end_len+label_col):((end_len+1)+label_col)]
              #val_x, val_y=target_ts[0:(len(target_ts)-1),0:5],target_ts[0:(len(target_ts)-1),(5+label_col):(6+label_col)]
              ##test code ##
              val_x, val_y=val_ts[0:(len(val_ts)-1),0:end_len],val_ts[0:(len(val_ts)-1),(end_len+label_col):((end_len+1)+label_col)]          
              
              #train_x = np.vstack((train_x, val_x))
              #train_y = np.vstack((train_y, val_y))
              
              test_x, test_y=target_ts[(len(target_ts)-1),0:end_len],target_ts[(len(target_ts)-1),(end_len+label_col):((end_len+1)+label_col)]
              #reshape point test val    
              test_x = np.reshape(test_x,newshape=(1,len(test_x)))
              test_y = np.reshape(test_y,newshape=(1,len(test_y)))
              #train model
              
              gamma = .2
              max_depth = 3
              n_estimators = 150
              if 0 <= label_col < 4:
                  gamma = .1
                  max_depth = 5
                  n_estimators = 200
                  if  3<= recent_slope <= 4 or -3>= recent_slope >= -4:
                      gamma = 2
                      max_depth = 3
                      n_estimators = 150 
                  elif 2<= recent_slope <= 3 or -2>= recent_slope >= -3:
                      gamma = 4
                      max_depth = 3
                      n_estimators = 125
                  elif 1 >= recent_slope >= -1:
                      gamma = 6
                      max_dpeth = 2
                      n_estimators = 100
                  xgb1 = xgb.XGBRegressor(n_estimators=n_estimators,#600
                                      max_depth=max_depth,
                                      min_child_weight=7,
                                      gamma=gamma,
                                      colsample_bytree=.7,
                                      subsample=.7
                                      )
                  xgb1.fit(train_x, train_y,
                       eval_set=[(val_x, val_y)],
                       verbose=False)
                  #produce prediction
                  y_hat = xgb1.predict(test_x)
              elif label_col >= 10:
                  #generate sample weight values
                  #train_weight=np.reshape((train_x[:,0]==0)+1*(train_x[:,1]*1000),newshape=(len(train_x),1))
                  gamma = .2
                  max_depth = 3
                  n_estimators = 150
                  
                  idea = (train_x[:,0]==0)+1
                  idea[idea==2]=10
                  idea = np.reshape(idea*((train_x[:,1]+2)*100),newshape=(len(train_x),1))
                  
                  weight_scale = MinMaxScaler(feature_range=(1, 10000))
                  train_weight = weight_scale.fit_transform(idea)
                  train_weight=np.reshape(train_weight,newshape=(len(train_weight),))
                  #train_weight=train_weight*.1
                  #generate sample val weight values
                  #val_weight=np.reshape((val_x[:,0]==0)+1*(val_x[:,1]*1000),newshape=(len(val_x),1))
                  
                  idea_v = (val_x[:,0]==0)+1
                  idea_v[idea_v==2]=10
                  idea_v = np.reshape(idea_v*((val_x[:,1]+2)*100),newshape=(len(val_x),1))
                  
                  weight_scale = MinMaxScaler(feature_range=(1, 10000))
                  val_weight = weight_scale.fit_transform(idea_v)
                  val_weight=np.reshape(val_weight,newshape=(len(val_weight),))
                  #val_weight = val_weight*.1
                  xgb1 = xgb.XGBRegressor(n_estimators=n_estimators,#600
                                      max_depth=max_depth,
                                      min_child_weight=7,
                                      gamma=gamma,
                                      colsample_bytree=.7,
                                      subsample=.7
                                      )
                  xgb1.fit(train_x, train_y,
                           eval_set=[(val_x, val_y)],
                           verbose=False,
                           sample_weight=train_weight,
                           sample_weight_eval_set=[val_weight])
                  #produce prediction
                  y_hat = xgb1.predict(test_x)
              else:
                  gamma = .2
                  max_depth = 3
                  n_estimators = 150
                  xgb1 = xgb.XGBRegressor(n_estimators=n_estimators,#600
                                      max_depth=max_depth,
                                      min_child_weight=7,
                                      gamma=gamma,
                                      colsample_bytree=.7,
                                      subsample=.7
                                      )
                  
                  xgb1.fit(train_x, train_y,
                       eval_set=[(val_x, val_y)],
                       verbose=False)
                  y_hat = xgb1.predict(test_x)
                  
              #gen % error
              point_error = round(float(abs(y_hat-test_y)/test_y),4)
              #point_error = y_hat
              #scaler = scale_dict['11_scaler']
              #point_error = scaler.inverse_transform(y_hat)
              if point_error<0:
                  point_error = point_error*-1
              results[label_col]=[point_error,y_hat[0],test_y[0,0]]
              #results[label_col]=[point_error]
              #np.array([point_error,y_hat[0],test_y[0,0]])
            
            #xgb_output=pd.DataFrame.from_dict(results)
            #xgb_output=xgb_output.append(pd.DataFrame.from_dict(results), ignore_index=True)
            xgb_output=xgb_output.append(results, ignore_index=True)
      print('gam for {} is done'.format(loc_list[loc_list['Index']==loc]['Country_Region'][loc]))
      print(datetime.now().strftime("%H:%M:%S"))


xgb_output.to_csv('xgb_loess_output.csv')
