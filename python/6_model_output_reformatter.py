# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:08:26 2020

@author: u6026797
"""


#%% libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#%% oracle file reformatter
file_name_list = ['state','80mi']
for name in file_name_list:
    file_n = name+'_oracle_kz.pkl'
    data_file = pd.read_pickle(file_n)
    output_df = pd.DataFrame(columns=['Country_Region','Date'])
    for row in data_file.iterrows():
        resize_size = len(row[1][1])
        strArr = np.empty(resize_size, dtype="U30")
        for i in range(0,len(strArr)):
            strArr[i]=row[1][0]
        
        row_dict = {'Location_Name' : strArr,
                    'Date' : row[1][1],
                    'Total_Cases' : row[1][4],
                    'New_Cases' : row[1][5],
                    'Oracle_Smooth' : row[1][6]}
        output_df = output_df.append(pd.DataFrame(row_dict),ignore_index=False)
    out_n = name+'_unrolled_oracle_kz.csv'
    output_df.to_csv(out_n)

#%% data
for name in file_name_list:
    oracle_name = name+'_unrolled_oracle_kz.csv'
    model_name = 'clean_'+name+'_results_t.csv'
    model_file = pd.read_csv(model_name)
    data_file = pd.read_csv(oracle_name)
    
    model_file = model_file[['loc','date','1','2','3','4','5','6','7','8','9','10','11','12','13','14',
                             '1_u','2_u','3_u','4_u','5_u','6_u','7_u','8_u','9_u','10_u','11_u',
                             '12_u','13_u','14_u','1_l','2_l','3_l','4_l','5_l','6_l','7_l','8_l','9_l',
                             '10_l','11_l','12_l','13_l','14_l']]
    model_file=model_file.rename(columns={"date":"Date",
                                          "loc": "Location_Name"})
                                 #, "date_1": "Date"})
    #change col type to date
    model_file['Date']=model_file['Date'].astype('datetime64')
    data_file['Date']=data_file['Date'].astype('datetime64')
    
    date_vec = model_file['Date'].unique()
    loc_vec = model_file['Location_Name'].unique()
    
    results_output = pd.DataFrame(columns = ['Location_Name'])
    
    for state in loc_vec:
        small_state = data_file[data_file['Location_Name']==state]
        for date in date_vec:
            npday = date+(14*86400000000000)
            small_date = small_state[small_state['Date']<npday]
            if len(small_date)<(len(small_state[small_state['Date']<=date])+14):
                internal_date_df = pd.DataFrame(columns=['Date'])
                internal_date_df['Date']=pd.date_range(min(small_date['Date']), periods=(len(small_state[small_state['Date']<=date])+14))
                internal_date_df=internal_date_df.set_index('Date').join(small_date.set_index('Date'))
                internal_date_df['Location_Name'] = internal_date_df['Location_Name'].fillna(state)
                internal_date_df=internal_date_df.reset_index()
                small_date = internal_date_df
                
            if len(small_date)>0:
                small_date = small_date[['Location_Name','Date','New_Cases','Total_Cases','Oracle_Smooth']]
                model_data_sub = model_file[(model_file['Date']==date) & (model_file['Location_Name']==state)]
                model_data = model_data_sub.iloc[:,2:16]
                model_data_u = model_data_sub.iloc[:,16:30]
                model_data_l = model_data_sub.iloc[:,30:44]
                if len(model_data )>0:
                    model_data = model_data.T
                    model_data_u = model_data_u.T
                    model_data_l = model_data_l.T                    
                    model_data  = model_data.iloc[:,0].to_numpy()
                    model_data_u  = model_data_u.iloc[:,0].to_numpy()
                    model_data_l  = model_data_l.iloc[:,0].to_numpy()
                    
                    new_len = len(small_date)-len(model_data)
                    a = np.empty(new_len)
                    a[:] = np.nan
                    na_pred=np.append(a,model_data)
                    na_up=np.append(a,model_data_u)
                    na_low=np.append(a,model_data_l)
                    small_date['Prediction']=na_pred
                    small_date['Pred_Upper']=na_up
                    small_date['Pred_Lower']=na_low
                    #small_date['New_Cases'][(len(small_date)-14):]=model_data.iloc[:,0]
                    small_date['Pred_Date']=date
                    
                    results_output=results_output.append(small_date, ignore_index=True)
    
    output_name = name+"_unrolled_model_predictions.csv"
    results_output.to_csv(output_name)
    

