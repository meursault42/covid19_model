# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:19:03 2020

@author: u6026797
"""

#%% Libraries
import numpy as np
import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
#%% function  
rstring="""
library(kza)
library(stats)
library(tseries)
r_smoother_wrapper_auto<-function(ts_data,index,smoother,param_list_loess,
                                  param_list_kz,param_jitter,dparam){
  
ts_PerInc <- function(ts,smoothing_n=1){
  out<-c(1)
  ts_append<-c(rep_len(ts[1],smoothing_n),ts)
  for(i in 1:length(ts_append)){
    if(smoothing_n==1){
      x<-mean(ts_append[(i+smoothing_n):(i+smoothing_n)]/ts_append[i:(i+smoothing_n-1)])
      if(is.na(x)==TRUE || is.infinite(x)==TRUE){
        x<-0
      }
      out<-c(out,x)
    }
    else{
      x<-mean(ts_append[(i+smoothing_n-1):(i+smoothing_n)]/ts_append[i:(i+smoothing_n-1)])
      if(is.na(x)==TRUE || is.infinite(x)==TRUE){
        x<-0
      }
      out<-c(out,x)
    }
  }
  #get rid of init 1
  out<-tail(out,(length(out)-1))
  #get rid of index error
  out<-head(out,(length(out)-smoothing_n))
  return(out)
}

kz_grid_search<-function(df,param_sweep,param_jitter){
  output_df_colnames<-c('param','adf')
  kz_grid_df<-data.frame(matrix(ncol=2,nrow=0, dimnames=list(NULL,output_df_colnames)), 
                         stringsAsFactors = FALSE)
  for(param in param_sweep){
    func_out<-kza(x=df$noise_seq,m=param,k=4,impute_tails = TRUE)
    resid<-df$noise_seq-func_out$kz
    adf_out<- -999
    for(k in seq.int(7,14)){
      adf_in<-adf.test(resid,alternative = 'stationary',k = k)
      adf_in<- unname(adf_in$p.value)
      #adf_in<- unname(adf_in$statistic)
      if(is.na(adf_in)){
        adf_in<- 0
      }
      if(adf_in>adf_out){
        adf_out<-adf_in
      }
    }
    outlist<-list('param'=param,
                  'adf_out'=adf_out)
    kz_grid_df<-rbind.data.frame(kz_grid_df,outlist,stringsAsFactors = FALSE)
  }
  kz_grid_df$adf_perinc<-ts_PerInc(kz_grid_df$adf_out)
  best_param<-c(1)
  test_seq<-kz_grid_df$adf_perinc
  for(i in seq.int(1,length(test_seq)-1)){
    if(test_seq[i+1]>test_seq[i] & test_seq[i+1]>1){
      best_param<-c(best_param,i)
    }
  }
  if(length(best_param)==1){
    best_param<-1
  }
  else{
    best_param<-best_param[2]
  }
  best_param<-best_param+param_jitter
  if(dim(kz_grid_df)[1]<best_param){
          best_param<-dim(kz_grid_df)[1]
          }
  best_param<-kz_grid_df$param[best_param]
  temp_kz<-kza(x=df$noise_seq,m=best_param,k=4,impute_tails = TRUE)
  df$kz<-temp_kz$kz
  df<-subset.data.frame(df,select=c('index','kz'))
  return(c(best_param,df))
}

loess_grid_search<-function(df,param_sweep,param_jitter){
  output_df_colnames<-c('param','adf')
  loess_grid_df<-data.frame(matrix(ncol=2,nrow=0, dimnames=list(NULL,output_df_colnames)), 
                     stringsAsFactors = FALSE)
  for(param in param_sweep){
    func_out<-loess(log(noise_seq)~index, span = param, data=df)
    loess_pred<-exp(predict(func_out,df))
    resid<-loess_pred-df$noise_seq
    adf_out<- -999
    for(k in seq.int(7,14)){
        adf_in<-adf.test(resid,alternative = 'stationary',k = k)
        adf_in<- unname(adf_in$p.value)
        #adf_in<- unname(adf_in$statistic)
        if(is.nan(adf_in)){
          adf_in<- -999
        }
        if(adf_in>adf_out){
          adf_out<-adf_in
        }
    }
    outlist<-list('param'=param,
                    'adf_out'=adf_out)
    loess_grid_df<-rbind.data.frame(loess_grid_df,outlist,stringsAsFactors = FALSE)
  }
  loess_grid_df$adf_perinc<-ts_PerInc(loess_grid_df$adf_out)
  best_param<-c(1)
  test_seq<-loess_grid_df$adf_perinc
  for(i in seq.int(1,length(test_seq)-1)){
    if(test_seq[i+1]>test_seq[i] & test_seq[i+1]>1){
      best_param<-c(best_param,i)
    }
  }
  if(length(best_param)==1){
    best_param<-1
  }
  else{
    best_param<-best_param[2]
  }
  best_param<-best_param+param_jitter
  if(dim(loess_grid_df)[1]<best_param){
          best_param<-dim(loess_grid_df)[1]
          }
  best_param<-loess_grid_df$param[best_param]
  
  temp_low<-loess(log(noise_seq)~index, span = best_param, data=df)
  loess_pred<-exp(predict(temp_low,df))
  df$loess<-loess_pred
  df<-subset.data.frame(df,select=c('index','loess'))
  return(c(best_param,df))
}

dumb_loess<-function(df,param){
  func_out<-loess(log(noise_seq)~index, span = param, data=df)
  loess_pred<-exp(predict(func_out,df))
  df$loess<-loess_pred
  df<-subset.data.frame(df,select=c('index','loess'))
  return(c(param,df))
}
  #bind up dataset in df
  df<-data.frame(ts_data,stringsAsFactors = FALSE)
  df<-cbind.data.frame(df,index)
  colnames(df)<-c('noise_seq','index')
  #run smoothers
  if(smoother=='kz'){
      return(kz_grid_search(df,param_sweep = param_list_kz,param_jitter=param_jitter))
    }
  if(smoother=='loess'){
    return(loess_grid_search(df,param_sweep = param_list_loess,param_jitter=param_jitter))
    }
  if(smoother=='dloess'){
    return(dumb_loess(df,param=dparam))          
  }
                                      
dumb_kz<-function(df,param){
  temp_kz<-kza(x=df$noise_seq,m=param,k=4,impute_tails = TRUE)
  df$kz<-temp_kz$kz
  df<-subset.data.frame(df,select=c('index','kz'))
  return(c(param,df))
    
  func_out<-loess(log(noise_seq)~index, span = param, data=df)
  loess_pred<-exp(predict(func_out,df))
  df$loess<-loess_pred
  df<-subset.data.frame(df,select=c('index','loess'))
  return(c(param,df))
}
  #bind up dataset in df
  df<-data.frame(ts_data,stringsAsFactors = FALSE)
  df<-cbind.data.frame(df,index)
  colnames(df)<-c('noise_seq','index')
  #run smoothers
  if(smoother=='kz'){
      return(kz_grid_search(df,param_sweep = param_list_kz,param_jitter=param_jitter))
    }
  if(smoother=='loess'){
    return(loess_grid_search(df,param_sweep = param_list_loess,param_jitter=param_jitter))
    }
  if(smoother=='dloess'){
    return(dumb_loess(df,param=dparam))          
  }
  if(smoother=='dkz'){
    return(dumb_kz(df,param=dparam))          
  }
}
"""
rfunc=robjects.r(rstring)

def r_smoother_wrapper(ts_data,index,smoother,
                       param_list_loess=np.arange(.2,.76,.05),
                       param_list_kz=list(range(3, 25, 2)),
                       param_jitter = [0],
                       dparam = [.2]):
  '''
  Wrapper function for R smoother functions via rpy2. Be sure that you have rpy2 
  and it/'s dependencies installed. Specifically:
  
  import rpy2.robjects.packages as rpackages
  utils = rpackages.importr('utils')
  utils.chooseCRANmirror(ind=1)
  packnames = ('kza', 'stats','tseries')
  from rpy2.robjects.vectors import StrVector
  names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
  if len(names_to_install) > 0:
      utils.install_packages(StrVector(names_to_install))
  
  Parameters
  ----------
  ts_data : int/float numeric vector
    Vector to be smoothed
  index : integer vector
    Index of ts_data values. Usually 1,2...len(ts_data)
  smoother : str
    str literals kz or loess
  param_list_loess : float or int vector, optional
    A vector of hyperparameter values to be optimized over. The default is .2:.75.
  param_list_kz : int vector, optional
    A vector of hyperparameter values to be optimized over. The default is 3:26.
  param_jitter : int, optional
    An integer that permutes the 'best' value found by adf.

  Returns
  -------
  kza_out : TYPE
    DESCRIPTION.

  '''
  index = robjects.IntVector(index)
  ts_data = robjects.IntVector(ts_data)
  param_jitter = robjects.IntVector(param_jitter)
  dparam= robjects.FloatVector(dparam)
  
  #output_dict = dict()
  if smoother=='kz':
      kza_out = rfunc(ts_data=ts_data,index=index,smoother=smoother, 
                      param_list_loess=param_list_loess, 
                      param_list_kz=param_list_kz,
                      param_jitter=param_jitter,
                      dparam=dparam)
      return kza_out.rx2(1),kza_out.rx2(3)
  if smoother=='loess':
      loess_out = rfunc(ts_data=ts_data,index=index,smoother=smoother, 
                      param_list_loess=param_list_loess, 
                      param_list_kz=param_list_kz,
                      param_jitter=param_jitter,
                      dparam=dparam)
      return loess_out.rx2(1),loess_out.rx2(3)
  if smoother=='dloess':
      dloess_out = rfunc(ts_data=ts_data,index=index,smoother=smoother, 
                      param_list_loess=param_list_loess, 
                      param_list_kz=param_list_kz,
                      param_jitter=param_jitter,
                      dparam=dparam)
      return dloess_out.rx2(1),dloess_out.rx2(3)
  if smoother=='dkz':
      dloess_out = rfunc(ts_data=ts_data,index=index,smoother=smoother, 
                      param_list_loess=param_list_loess, 
                      param_list_kz=param_list_kz,
                      param_jitter=param_jitter,
                      dparam=dparam)
      return dloess_out.rx2(1),dloess_out.rx2(3)
  #smoother = 'dloess'
  #p = pandas2ri.ri2py(kza_out)[0]
  #p = pandas2ri.ri2py(p)[0]
  #kz = pandas2ri.ri2py(kza_out)[2]
  #kz = pandas2ri.ri2py(kz)
  #return p, kz

#%% thingy
'''
import random
index = list(range(1,101,1))
ts_data= list(range(1,101,1))

ts_data = [12,  2,  1, 12,  2,  3,  5,  8,  6,  4,  1,  2, 11,
        8,  5,  2,  9, 19,  1,  8,  4,  3,  2,  3,  3,  3,
        3,  3,  3,  3,  3]
index = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
for i in range(len(ts_data)):
  ts_data[i]=ts_data[i]*random.randint(1,10)

param_list_loess=np.arange(.2,.76,.05)
param_list_kz=list(range(3,26,1))



p,kz = r_smoother_wrapper(ts_data=ts_data,
                  index=index,
                  smoother='dkz',
                  dparam=[15],
                  param_jitter=[0])

#param_list_loess=np.arange(.2,.76,.05),
#param_list_kz=list(range(3, 25, 2))

#kza_out = rfunc(ts_data=ts_data,
#                index=index,
#                smoother='kz',  
#                  param_list_kz=param_list_kz)

#pandas2ri.ri2py(kza_out.rx2((i+1)).rx2('kz'))
'''