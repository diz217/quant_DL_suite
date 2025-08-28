#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:32:10 2025

@author: diz217
"""
import pandas as pd 
from pandas.tseries.offsets import BDay
import numpy as np
import matplotlib.pyplot as plt
ticker = 'FORD'
Pred_len = 40
start,end = '2019-08-20','2020-08-20'
#start,end = '2013-08-20','2014-08-20'
df = pd.read_csv(f'{ticker}_{start}_{Pred_len}_v1',sep=',',header=0)
dates = df['dates'].values
dates = np.append(dates,end)
hashmap = {}
money = 1
for i,tp in enumerate(df['tp_real'].values):
    Bdays = np.busday_count(dates[i],dates[i+1])
    log_tp = np.log(1+tp)
    log_tp_dy = log_tp/Bdays
    tp_dy = np.exp(log_tp_dy)-1
    hashmap[tp_dy] = hashmap.get(tp_dy,0)+Bdays
    money *= (1+tp)
ret_dy_list = []
for value,count in hashmap.items():
    ret_dy_list.extend([float(value)]*int(count))
ret_dy_list = np.array(ret_dy_list)

mu = ret_dy_list.mean()
std = ret_dy_list.std()
#--risk-free yield
Y_free = 0.041
df_free = Y_free/252
#-- daiy sharpe to annual sharpe
sharpe = (mu-df_free)/std*np.sqrt(252)
#--largest retrace
largest_retrace = 1
def dp(i,retrace):
    global largest_retrace
    if i==len(df['tp_real']):
        return retrace
    if df['tp_real'].iloc[i]>0:
        return dp(i+1,1)
    else:
        retrace *= (1+df['tp_real'].iloc[i])
        largest_retrace = min(retrace,largest_retrace)
        return dp(i+1,retrace)
dp(0,1)
max_loss = largest_retrace-1


#-- plot yield curve 
datetime = pd.to_datetime(df['dates'].values)
start_ = pd.to_datetime(start)+pd.DateOffset(years=5)
end_ = pd.to_datetime(end)+pd.DateOffset(years=5)
plt.figure(figsize=(16,6))
plt.plot(datetime,df['tp_real'].values,linestyle='--',marker='o',markersize=3,linewidth=1)
plt.title(f'{start_} to {end_}: sharpe = {sharpe:.2f} tot yield = {money:.2f}, Max DD = {max_loss:.2f}')
plt.savefig(f'{ticker}_yield_sharpe_{Pred_len}_{start}_v1')