# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 09:11:08 2025

@author: Ding Zhang
"""

from TST_v0p81 import TST_v0p8
import time 
import pandas as pd 
from pandas.tseries.offsets import BDay
import os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

Trn_years = 5
Seq_len = 365
Pred_len = 40
Stop_loss = 0.94
Seed = 0
money = 1
ticker = ['GM','FORD','AAPL','AMD']

def generate_date_ranges(start_str, end_str, step_bd=40):
    start_date = pd.to_datetime(start_str)
    end_date = pd.to_datetime(end_str)
    dates = []
    cur = start_date
    
    while cur < end_date:
        dates.append(cur)
        cur += BDay(step_bd+1)
    
    return dates

date_ranges = [('2019-08-20','2020-08-20'),('2013-08-20','2014-08-20')]
#date_list = generate_date_ranges(start, end, Pred_len)

for stk in ticker:
    for start,end in date_ranges:
        end_date = pd.to_datetime(end)
        tp_pred_list,tp_prce_list,tp_true_list,tp_real_list, tp_stgy_list = [],[],[],[],[]
        date = pd.to_datetime(start)
        date_list = [date]
        while date <= end_date:
            tst = TST_v0p8()
            tst.load_data(stk,date)
            tst.prepare(pred_len=Pred_len, seq_len=Seq_len,stop_loss=Stop_loss)
            tst.build(seed = Seed, embed_dim=128, ff_dim = 256, head_base = 4, dropout_rate = 0.1, 
                      embed_sig = 32, head_sig = 2, ff_sig = 64,
                      base_lr = 1e-4, clipnorm = 1.0, batch_size=32, epochs=100,warmup_epoch=10)
            tst.fit(batch_size=32, epochs=100)
            tst.predict(seed=Seed,path='./',i=0,ema_span=21)
            tp_pred = np.float32(tst.tp_pred); tp_pred_list.append(tp_pred)
            tp_prce = tst.tp_pred_;tp_prce_list.append(tp_prce)
            tp_true = np.float32(tst.tp_true[0]); tp_true_list.append(tp_true)
            tp_stgy = tp_pred;
            
            cls_true = tst.true_cls
            opn_true = tst.true_opn
            hgh_true = tst.true_hgh
            low_true = tst.true_low
            
            step_bd = Pred_len
            #--tp_strategy adjusted by price------------------ 
            if tp_pred*tp_prce<0 or abs(tp_pred)<1-Stop_loss:
                tp_stgy = 0
            tp_stgy_list.append(tp_stgy)
            #--decision_making, 9 cases-----------------------
            if tp_stgy == 0 and tp_true == 0: 
                print('###: Case 1')
                tp_real_list.append(0)
                date += BDay(step_bd+1)
                date_list.append(date)
                continue
            if tp_stgy == 0 and tp_true>0: # miss
                print('###: Case 2')
                entry = opn_true[0]                   
                for j in range(len(opn_true)):
                    if hgh_true[j]/entry>2-Stop_loss:
                        step_bd = j+1
                        break 
                tp_real_list.append(0)
                date += BDay(step_bd+1)
                date_list.append(date)
                continue
            if tp_stgy == 0 and tp_true<0: # miss
                print('###: Case 3')    
                entry = opn_true[0]                   
                for j in range(len(opn_true)):
                    if low_true[j]/entry<Stop_loss:
                        step_bd = j+1
                        break 
                tp_real_list.append(0)
                date += BDay(step_bd+1)
                date_list.append(date)
                continue
            if tp_stgy > 0 and tp_true<=0: # stopped
                print('###: Case 4&5')    
                entry = opn_true[0]                   
                for j in range(len(opn_true)):
                    if low_true[j]/entry<Stop_loss:
                        step_bd = j+1
                        break
                tp_real_list.append(Stop_loss-1)
                date += BDay(step_bd+1)
                money *= Stop_loss
                date_list.append(date)
                continue
            if tp_stgy < 0 and tp_true>=0: # stopped
                print('###: Case 6&7')  
                entry = opn_true[0]                   
                for j in range(len(opn_true)):
                    if hgh_true[j]/entry>2-Stop_loss:
                        step_bd = j+1
                        break
                tp_real_list.append(Stop_loss-1)
                date += BDay(step_bd+1)
                money *= Stop_loss
                date_list.append(date)
                continue
            if tp_stgy>0 and tp_true>0: # thru
                print('###: Case 8')  
                if tp_stgy>tp_true:
                    ret = max(cls_true[-1]/opn_true[0]-1,0)
                else:
                    ret = tp_stgy
                tp_real_list.append(ret)
                date += BDay(Pred_len+1)
                money *= (1+ret)
                date_list.append(date)
                continue
            if tp_stgy<0 and tp_true<0: # thru
                print('###: Case 9')  
                if tp_stgy<tp_true:
                    ret = -min(cls_true[-1]/opn_true[0]-1,0)
                else:
                    ret = -tp_stgy
                tp_real_list.append(ret)
                date += BDay(Pred_len+1)
                money *= (1+ret)
                date_list.append(date)
                continue
            
        df = pd.DataFrame()
        df['dates'] = date_list[:-1]
        df['tp_real'] = tp_real_list
        df['tp_stgy'] = tp_stgy_list
        df['tp_true'] = tp_true_list
        df['tp_pred'] = tp_pred_list
        df['tp_prce'] = tp_prce_list
        df.to_csv(f'{stk}_{start}_{Pred_len}_v1')
    
        
        
    
     