# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:00:42 2020

@author: u6026797
"""

#%% prep
from r_func_script_v3 import r_smoother_wrapper

import numpy as np
import pandas as pd

import DatamanagerState as StateDM
import datamanager as DM
import DatamanagerHRR as DMHRR
import DatamanagerSta6a as DMSta6a
import sys
sys.path.insert(1, 'C:\\Users\\u6026797\\Desktop\\covid_data')

flat_date = np.datetime64('2020-10-28T00:00:00.00')
oracle_date = np.datetime64('2020-11-10T00:00:00.00')

def clean_zeors(x):
    for i in range(1,len(x)-1):
        if(x[i]<=0):
            if x[i+1] != 0:
                x[i] = (x[i-1] +  x[i+1])/2.0
            else :
                x[i] = x[i-1]
    if x[-1] <= 0:
        x[-1] = x[-2]
    return x

def func(x, a,b):
    return a*x+b

#%% data processing step
#Country_Data = DM.Data_Manager('C:\\Users\\u6026797\\Documents\\Local_work\\covid_model\\data\\JHU_data\\COVID19_country.csv')
#Country_Data.Set_Min_Confirmed(10)
#country_list = Country_Data.Get_Available_Countries()

df = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','count'])

State_Data = StateDM.Data_Manager('C:\\Users\\u6026797\\Documents\\Local_work\\covid_model\\data\\JHU_data\\COVID19_USTracking.csv')
State_Data.Set_Min_Confirmed(10)
State_list = State_Data.Get_Available_States()
#%% state
for state in State_list:
    State_Data.Set_State(state)
    temp_val = State_Data.Get_Confirm_Count()
    if temp_val!= 0:
      Total_Conf_Time, Total_Conf_Count, Total_DayOfWeek = State_Data.Get_Confirm_Count()
      Total_Conf_Time  = np.flip(Total_Conf_Time)
      Total_Conf_Count = np.flip(Total_Conf_Count)
      Total_DayOfWeek  = np.flip(Total_DayOfWeek)
      if len(Total_Conf_Count) > 21:        
          Dt = (Total_Conf_Time-Total_Conf_Time[0])/(60.0*60.0*24.0)/1000000000
          df = df.append({'Country_Region' : state , 'Date' : Total_Conf_Time,'Dt': Dt,'DayOfWeek':Total_DayOfWeek,'count': Total_Conf_Count} , ignore_index=True)

df_kz = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','Count',
                                   'NewPat','Oracle_kz','LastDate'])

for index,row in df.iterrows():
    iter_for_days = np.where(row['Date'] >= oracle_date )[0]
    for i in iter_for_days:
        Country_Region = row['Country_Region']
        print('Starting {}, day {}'.format(Country_Region,i))
        Date = row['Date'][:i]
        if len(Date)>=25: #21
          Dt = row['Dt'][:i].astype(int)
          DayOfWeek = row['DayOfWeek'][:i].astype(int)
          count = row['count'][:i]          
          new_pat = count[1:] - count[:-1]
          new_pat = new_pat.astype('float64')
          new_pat = np.insert(new_pat,0,count[0])
          new_pat[new_pat<0] = 0
          new_pat[np.isnan(new_pat)] = 0
          
          combined = np.vstack((Dt, DayOfWeek)).T      
          new_pat = clean_zeors(new_pat)          
          kz = r_smoother_wrapper(new_pat,
                         combined[:,0],
                         smoother='kz')
          kz = np.round(kz[1])
                    
          df_kz = df_kz.append({'Country_Region' : Country_Region, \
                          'Date' : Date,\
                          'Dt' : Dt,\
                          'DayOfWeek' : DayOfWeek,\
                          'Count' : count,\
                          'NewPat' : new_pat,\
                          'Oracle_kz' : kz,\
                          'LastDate' : Date[-1]}, ignore_index=True)


LastDate=df_kz['LastDate'].max()
df_last=df_kz[df_kz['LastDate']==LastDate]
df_last.to_pickle("country_state_oracle_kz.pkl", protocol=4)

df_kz = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','Count',
                                   'NewPat','kz_f7','kz_f9','LastDate'])
permute_vec=[7,9]
for index,row in df.iterrows():
    iter_for_days = np.where(row['Date'] >= flat_date)[0]
    for i in iter_for_days:
        Country_Region = row['Country_Region']

        print('Starting {}, day {}'.format(Country_Region,i))
        Date = row['Date'][:i]
        if len(Date)>=25: #21
          Dt = row['Dt'][:i].astype(int)
          DayOfWeek = row['DayOfWeek'][:i].astype(int)
          count = row['count'][:i]          
          new_pat = count[1:] - count[:-1]
          new_pat = new_pat.astype('float64')
          new_pat = np.insert(new_pat,0,count[0])
          new_pat[new_pat<0] = 0
          new_pat[np.isnan(new_pat)] = 0
          
          combined = np.vstack((Dt, DayOfWeek)).T      
          new_pat = clean_zeors(new_pat)
          annoying_iterator = {}
          for permute in permute_vec:
              param, kz = r_smoother_wrapper(new_pat,
                         combined[:,0],
                         smoother='dkz',
                         dparam=[permute])
              annoying_iterator['kz_f{}'.format(str(permute))] = np.round(kz)
                    
          df_kz = df_kz.append({'Country_Region' : Country_Region, \
                          'Date' : Date,\
                          'Dt' : Dt,\
                          'DayOfWeek' : DayOfWeek,\
                          'Count' : count,\
                          'NewPat' : new_pat,\
                          'kz_f7' : annoying_iterator['kz_f7'],\
                          'kz_f9' : annoying_iterator['kz_f9'],\
                          'LastDate' : Date[-1]}, ignore_index=True)
              
df_kz.to_pickle("country_state_flat_kz.pkl", protocol=4)
#%% HRR
#HRR
df = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','count'])
HRR_Data = DMHRR.Data_Manager('C:\\Users\\u6026797\\Documents\\Local_work\\covid_model\\data\\JHU_data\\HRR_county_agg.csv')
HRR_Data.Set_Min_Confirmed(10)
HRR_list = HRR_Data.Get_Available_HRR()

for hrr in HRR_list:
    HRR_Data.Set_HRRName(hrr)
    temp_val = HRR_Data.Get_Confirm_Count()
    if temp_val!= 0:
      Total_Conf_Time, Total_Conf_Count, Total_DayOfWeek = HRR_Data.Get_Confirm_Count()
      Total_Conf_Time  = np.flip(Total_Conf_Time)
      Total_Conf_Count = np.flip(Total_Conf_Count)
      Total_DayOfWeek  = np.flip(Total_DayOfWeek)
      if len(Total_Conf_Count) > 21:        
          Dt = (Total_Conf_Time-Total_Conf_Time[0])/(60.0*60.0*24.0)/1000000000
          df = df.append({'Country_Region' : hrr , 'Date' : Total_Conf_Time,'Dt': Dt,'DayOfWeek':Total_DayOfWeek,'count': Total_Conf_Count} , ignore_index=True)

df_kz = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','Count',
                                   'NewPat','Oracle_kz','LastDate'])

for index,row in df.iterrows():
    iter_for_days = np.where(row['Date'] >= oracle_date )[0]
    for i in iter_for_days:
        Country_Region = row['Country_Region']
        print('Starting {}, day {}'.format(Country_Region,i))
        Date = row['Date'][:i]
        if len(Date)>=25: #21
          Dt = row['Dt'][:i].astype(int)
          DayOfWeek = row['DayOfWeek'][:i].astype(int)
          count = row['count'][:i]          
          new_pat = count[1:] - count[:-1]
          new_pat = new_pat.astype('float64')
          new_pat = np.insert(new_pat,0,count[0])
          new_pat[new_pat<0] = 0
          new_pat[np.isnan(new_pat)] = 0
          
          combined = np.vstack((Dt, DayOfWeek)).T      
          new_pat = clean_zeors(new_pat)          
          kz = r_smoother_wrapper(new_pat,
                         combined[:,0],
                         smoother='kz')
          kz = np.round(kz[1])
                    
          df_kz = df_kz.append({'Country_Region' : Country_Region, \
                          'Date' : Date,\
                          'Dt' : Dt,\
                          'DayOfWeek' : DayOfWeek,\
                          'Count' : count,\
                          'NewPat' : new_pat,\
                          'Oracle_kz' : kz,\
                          'LastDate' : Date[-1]}, ignore_index=True)


LastDate=df_kz['LastDate'].max()
df_last=df_kz[df_kz['LastDate']==LastDate]
df_last.to_pickle("hrr_oracle_kz.pkl", protocol=4)

df_kz = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','Count',
                                   'NewPat','kz_f7','kz_f9','LastDate'])
permute_vec=[7,9]
for index,row in df.iterrows():
    iter_for_days = np.where(row['Date'] >= flat_date)[0]
    for i in iter_for_days:
        Country_Region = row['Country_Region']

        print('Starting {}, day {}'.format(Country_Region,i))
        Date = row['Date'][:i]
        if len(Date)>=25: #21
          Dt = row['Dt'][:i].astype(int)
          DayOfWeek = row['DayOfWeek'][:i].astype(int)
          count = row['count'][:i]          
          new_pat = count[1:] - count[:-1]
          new_pat = new_pat.astype('float64')
          new_pat = np.insert(new_pat,0,count[0])
          new_pat[new_pat<0] = 0
          new_pat[np.isnan(new_pat)] = 0
          
          combined = np.vstack((Dt, DayOfWeek)).T      
          new_pat = clean_zeors(new_pat)
          annoying_iterator = {}
          for permute in permute_vec:
              param, kz = r_smoother_wrapper(new_pat,
                         combined[:,0],
                         smoother='dkz',
                         dparam=[permute])
              annoying_iterator['kz_f{}'.format(str(permute))] = np.round(kz)
                    
          df_kz = df_kz.append({'Country_Region' : Country_Region, \
                          'Date' : Date,\
                          'Dt' : Dt,\
                          'DayOfWeek' : DayOfWeek,\
                          'Count' : count,\
                          'NewPat' : new_pat,\
                          'kz_f7' : annoying_iterator['kz_f7'],\
                          'kz_f9' : annoying_iterator['kz_f9'],\
                          'LastDate' : Date[-1]}, ignore_index=True)
              
df_kz.to_pickle("hrr_flat_kz.pkl", protocol=4)
#%% ARDS
#ards
df = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','count'])
Sta6a_Data = DMSta6a.Data_Manager('C:\\Users\\u6026797\\Documents\\Local_work\\covid_model\\data\\JHU_data\\ards_county_aggv2.csv')
Sta6a_Data.Set_Min_Confirmed(10)
Sta6a_list = Sta6a_Data.Get_Available_Sta6a()

for Sta6a in Sta6a_list:
    Sta6a_Data.Set_Sta6a(Sta6a)
    temp_val = Sta6a_Data.Get_Confirm_Count()
    if temp_val!= 0:
      Total_Conf_Time, Total_Conf_Count, Total_DayOfWeek = Sta6a_Data.Get_Confirm_Count()
      Total_Conf_Time  = np.flip(Total_Conf_Time)
      Total_Conf_Count = np.flip(Total_Conf_Count)
      Total_DayOfWeek  = np.flip(Total_DayOfWeek)
      if len(Total_Conf_Count) > 21:        
          Dt = (Total_Conf_Time-Total_Conf_Time[0])/(60.0*60.0*24.0)/1000000000
          df = df.append({'Country_Region' : Sta6a , 'Date' : Total_Conf_Time,'Dt': Dt,'DayOfWeek':Total_DayOfWeek,'count': Total_Conf_Count} , ignore_index=True)

df_kz = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','Count',
                                   'NewPat','Oracle_kz','LastDate'])

for index,row in df.iterrows():
    iter_for_days = np.where(row['Date'] >= oracle_date )[0]
    for i in iter_for_days:
        Country_Region = row['Country_Region']
        print('Starting {}, day {}'.format(Country_Region,i))
        Date = row['Date'][:i]
        if len(Date)>=25: #21
          Dt = row['Dt'][:i].astype(int)
          DayOfWeek = row['DayOfWeek'][:i].astype(int)
          count = row['count'][:i]          
          new_pat = count[1:] - count[:-1]
          new_pat = new_pat.astype('float64')
          new_pat = np.insert(new_pat,0,count[0])
          new_pat[new_pat<0] = 0
          new_pat[np.isnan(new_pat)] = 0
          
          combined = np.vstack((Dt, DayOfWeek)).T      
          new_pat = clean_zeors(new_pat)          
          kz = r_smoother_wrapper(new_pat,
                         combined[:,0],
                         smoother='kz')
          kz = np.round(kz[1])
                    
          df_kz = df_kz.append({'Country_Region' : Country_Region, \
                          'Date' : Date,\
                          'Dt' : Dt,\
                          'DayOfWeek' : DayOfWeek,\
                          'Count' : count,\
                          'NewPat' : new_pat,\
                          'Oracle_kz' : kz,\
                          'LastDate' : Date[-1]}, ignore_index=True)


LastDate=df_kz['LastDate'].max()
df_last=df_kz[df_kz['LastDate']==LastDate]
df_last.to_pickle("ards_oracle_kz.pkl", protocol=4)

df_kz = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','Count',
                                   'NewPat','kz_f7','kz_f9','LastDate'])
permute_vec=[7,9]
for index,row in df.iterrows():
    iter_for_days = np.where(row['Date'] >= flat_date)[0]
    for i in iter_for_days:
        Country_Region = row['Country_Region']

        print('Starting {}, day {}'.format(Country_Region,i))
        Date = row['Date'][:i]
        if len(Date)>=25: #21
          Dt = row['Dt'][:i].astype(int)
          DayOfWeek = row['DayOfWeek'][:i].astype(int)
          count = row['count'][:i]          
          new_pat = count[1:] - count[:-1]
          new_pat = new_pat.astype('float64')
          new_pat = np.insert(new_pat,0,count[0])
          new_pat[new_pat<0] = 0
          new_pat[np.isnan(new_pat)] = 0
          
          combined = np.vstack((Dt, DayOfWeek)).T      
          new_pat = clean_zeors(new_pat)
          annoying_iterator = {}
          for permute in permute_vec:
              param, kz = r_smoother_wrapper(new_pat,
                         combined[:,0],
                         smoother='dkz',
                         dparam=[permute])
              annoying_iterator['kz_f{}'.format(str(permute))] = np.round(kz)
                    
          df_kz = df_kz.append({'Country_Region' : Country_Region, \
                          'Date' : Date,\
                          'Dt' : Dt,\
                          'DayOfWeek' : DayOfWeek,\
                          'Count' : count,\
                          'NewPat' : new_pat,\
                          'kz_f7' : annoying_iterator['kz_f7'],\
                          'kz_f9' : annoying_iterator['kz_f9'],\
                          'LastDate' : Date[-1]}, ignore_index=True)
              
df_kz.to_pickle("ards_flat_kz.pkl", protocol=4)
#%% 80mi
#80mi
df = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','count'])
Sta6a_Data = DMSta6a.Data_Manager('C:\\Users\\u6026797\\Documents\\Local_work\\covid_model\\data\\JHU_data\\Sta6a_county_agg.csv')
Sta6a_Data.Set_Min_Confirmed(10)
Sta6a_list = Sta6a_Data.Get_Available_Sta6a()

for Sta6a in Sta6a_list:
    Sta6a_Data.Set_Sta6a(Sta6a)
    temp_val = Sta6a_Data.Get_Confirm_Count()
    if temp_val!= 0:
      Total_Conf_Time, Total_Conf_Count, Total_DayOfWeek = Sta6a_Data.Get_Confirm_Count()
      Total_Conf_Time  = np.flip(Total_Conf_Time)
      Total_Conf_Count = np.flip(Total_Conf_Count)
      Total_DayOfWeek  = np.flip(Total_DayOfWeek)
      if len(Total_Conf_Count) > 21:        
          Dt = (Total_Conf_Time-Total_Conf_Time[0])/(60.0*60.0*24.0)/1000000000
          df = df.append({'Country_Region' : Sta6a , 'Date' : Total_Conf_Time,'Dt': Dt,'DayOfWeek':Total_DayOfWeek,'count': Total_Conf_Count} , ignore_index=True)
          
df_kz = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','Count',
                         'NewPat','Oracle_kz','LastDate'])
          
for index,row in df.iterrows():
    iter_for_days = np.where(row['Date'] >= oracle_date )[0]
    for i in iter_for_days:
        Country_Region = row['Country_Region']
        print('Starting {}, day {}'.format(Country_Region,i))
        Date = row['Date'][:i]
        if len(Date)>=25: #21
          Dt = row['Dt'][:i].astype(int)
          DayOfWeek = row['DayOfWeek'][:i].astype(int)
          count = row['count'][:i]          
          new_pat = count[1:] - count[:-1]
          new_pat = new_pat.astype('float64')
          new_pat = np.insert(new_pat,0,count[0])
          new_pat[new_pat<0] = 0
          new_pat[np.isnan(new_pat)] = 0
          
          combined = np.vstack((Dt, DayOfWeek)).T      
          new_pat = clean_zeors(new_pat)          
          kz = r_smoother_wrapper(new_pat,
                         combined[:,0],
                         smoother='kz')
          kz = np.round(kz[1])
                    
          df_kz = df_kz.append({'Country_Region' : Country_Region, \
                          'Date' : Date,\
                          'Dt' : Dt,\
                          'DayOfWeek' : DayOfWeek,\
                          'Count' : count,\
                          'NewPat' : new_pat,\
                          'Oracle_kz' : kz,\
                          'LastDate' : Date[-1]}, ignore_index=True)

LastDate=df_kz['LastDate'].max()
df_last=df_kz[df_kz['LastDate']==LastDate]
df_last.to_pickle("80mi_oracle_kz.pkl", protocol=4)

df_kz = pd.DataFrame(columns = ['Country_Region','Date','Dt','DayOfWeek','Count',
                                   'NewPat','kz_f7','kz_f9','LastDate'])
permute_vec=[7,9]
for index,row in df.iterrows():
    iter_for_days = np.where(row['Date'] >= flat_date)[0]
    for i in iter_for_days:
        Country_Region = row['Country_Region']

        print('Starting {}, day {}'.format(Country_Region,i))
        Date = row['Date'][:i]
        if len(Date)>=25: #21
          Dt = row['Dt'][:i].astype(int)
          DayOfWeek = row['DayOfWeek'][:i].astype(int)
          count = row['count'][:i]          
          new_pat = count[1:] - count[:-1]
          new_pat = new_pat.astype('float64')
          new_pat = np.insert(new_pat,0,count[0])
          new_pat[new_pat<0] = 0
          new_pat[np.isnan(new_pat)] = 0
          
          combined = np.vstack((Dt, DayOfWeek)).T      
          new_pat = clean_zeors(new_pat)
          annoying_iterator = {}
          for permute in permute_vec:
              param, kz = r_smoother_wrapper(new_pat,
                         combined[:,0],
                         smoother='dkz',
                         dparam=[permute])
              annoying_iterator['kz_f{}'.format(str(permute))] = np.round(kz)
                    
          df_kz = df_kz.append({'Country_Region' : Country_Region, \
                          'Date' : Date,\
                          'Dt' : Dt,\
                          'DayOfWeek' : DayOfWeek,\
                          'Count' : count,\
                          'NewPat' : new_pat,\
                          'kz_f7' : annoying_iterator['kz_f7'],\
                          'kz_f9' : annoying_iterator['kz_f9'],\
                          'LastDate' : Date[-1]}, ignore_index=True)
              
df_kz.to_pickle("80mi_flat_kz.pkl", protocol=4)