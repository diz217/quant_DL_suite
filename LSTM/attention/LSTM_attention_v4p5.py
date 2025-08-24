#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 13:15:40 2025

@author: diz217
"""
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ta
import random
import matplotlib
matplotlib.use('Agg')
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import time
import pickle
import math
#import os
from scipy.stats import linregress


class BatchLossLogger(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"\nBatch {batch}: loss = {logs['loss']:.6f}")
        
class LSTM_attention_v4:
    def __init__(self):
        self.encoder_lstm = None
        self.decoder_lstm = None
        self.gate = None
        self.decoder_dense = None
        self.q_dense = None
        self.k_dense = None
        self.v_dense = None
        self.scaler_logcls = StandardScaler()
        self.scaler_rsi = MinMaxScaler(feature_range=(0,1))
        self.scaler_vol = MinMaxScaler(feature_range=(0,1))
        self.scaler_vix = MinMaxScaler(feature_range=(0,1))
        self.dataset = None
        self.x_encoder = []
        self.x_decoder = []
        self.y_target = []
        self.x_test = []
        self.y_test = []
        self.rmse = None
        self.trn_len = None
        self.trn_rat = None
        self.in_stp = None
        self.out_stp = None
        self.y0_pred = []
        self.y_pred = []
        self.att_weights = None
        self.num_heads = None
        self.depth = None
    def hashem_rsi(self,close_series, rsilen=14,smth=1):
        rsi0 = ta.momentum.rsi(close_series,window=rsilen,fillna=False)
        rsi_sma = rsi0.rolling(smth).mean()
        rsi_ema = rsi0.ewm(span=smth,adjust=False).mean()
        rsi_avg = (rsi_ema+rsi_sma)/2
        rsi_avg = rsi_avg.bfill()
        return rsi_avg
    def load_data(self,ticker,start,end=datetime.now()):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.dataset = yf.download(ticker,start,end)
    
        self.dataset[('Vix',ticker)] = (self.dataset['High']-self.dataset['Low'])/self.dataset['Close']       
        self.dataset[('Rsi_14',ticker)] = self.hashem_rsi(self.dataset[('Close',ticker)])
        
    def prepare(self,out_stp =60, in_stp=181,trn_rat=0.9):
        self.in_stp = in_stp
        self.out_stp = out_stp
        self.trn_rat = trn_rat
        close = self.dataset['Close'].values
        logcls = np.log(close)
        rsi = self.dataset['Rsi_14'].values
        vix = self.dataset['Vix'].values
        vol = self.dataset['Volume'].values
        
        trn_len = int(np.ceil(len(close)*trn_rat)) 
        self.trn_len = trn_len
        
        
        norm_data = self.scaler_logcls.fit_transform(logcls)
        norm_rsi = self.scaler_rsi.fit_transform(rsi)
        norm_vol = self.scaler_vol.fit_transform(vol)
        norm_vix = self.scaler_vix.fit_transform(vix)
        trn_data = norm_data[:trn_len]
        trn_rsi = norm_rsi[:trn_len]
        trn_vol = norm_vol[:trn_len]
        trn_vix = norm_vix[:trn_len]
        trn_set = np.hstack([trn_data,trn_rsi,trn_vol,trn_vix])
        
        
        for i in range(in_stp,trn_len-out_stp):
            self.x_encoder.append(trn_set[i-in_stp:i,:])
            self.x_decoder.append(trn_data[i-1:i+out_stp-1])
            self.y_target.append(trn_data[i:i+out_stp])
        self.x_encoder = np.array(self.x_encoder)
        self.x_decoder = np.array(self.x_decoder)
        self.y_target = np.array(self.y_target)
        
        tst_data = norm_data[trn_len-in_stp:]
        tst_rsi = norm_rsi[trn_len-in_stp:]
        tst_vol = norm_vol[trn_len-in_stp:]
        tst_vix = norm_vix[trn_len-in_stp:]
        tst_set = np.hstack([tst_data,tst_rsi,tst_vol,tst_vix])
        for i in range(in_stp,len(tst_data)):
            self.x_test.append(tst_set[i-in_stp:i,:])
        self.x_test = np.array(self.x_test)
    def split_heads(self,x,batch_size): #batch,time/1,hidden
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,perm=[0,2,1,3])
    def merge_head(self,x,batch_size):  #batch*head*1*depth
        x = tf.transpose(x,perm=[0,2,1,3]) #batch*1*head*depth
        return tf.reshape(x,(batch_size,-1,self.num_heads*self.depth))
    def buildfit(self,seed = 0, batch_size=250,n_unit1=128,n_head = 4,epochs=50,callbacks=[BatchLossLogger()]):
        tf.keras.utils.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.losses = []
        self.loss_precision = []
        self.recur_loss = []
        self.gate_list = []
        self.att_weights = None
        
        self.encoder_lstm = LSTM(n_unit1,return_sequences = True, return_state=True)           
        self.gate = Dense(n_unit1,activation = 'sigmoid')
        self.decoder_lstm = LSTM(n_unit1,return_state=True) #layer instantiation 实例化
        self.decoder_dense = Dense(1)        
        
        self.q_dense = Dense(n_unit1)
        self.k_dense = Dense(n_unit1)
        self.v_dense = Dense(n_unit1)
        
        self.num_heads = n_head
        self.depth = n_unit1//n_head
        
        optimizer = optimizers.Adam()
        smpl_prb = 0.8
        seq_max = int(np.ceil(self.x_encoder.shape[0]/batch_size))
        start_time = time.time()
            
        best_loss_ever = np.inf
        best_encoder_weights = None
        best_decoder_weights = None
        best_gate_weights = None
        best_dense_weights = None
        best_q_weights = None
        best_k_weights = None
        best_v_weights = None
        
        for epo in range(epochs):
            loss_precision = []
            for seq in range(seq_max):
                batch_start = time.time()
                start,end = seq*batch_size,min(self.x_decoder.shape[0],(seq+1)*batch_size)
                batch_size = end-start
                with tf.GradientTape() as tape: 
                    encoder_inputs = tf.convert_to_tensor(self.x_encoder[start:end,:,:],dtype=tf.float32)
                    decoder_target = tf.convert_to_tensor(self.y_target[start:end,:,:],dtype=tf.float32)
                    enc_output,h_enc,c_enc = self.encoder_lstm(encoder_inputs)
                    #batch*time*hidden, batch*hidden,batch*hidden
                    preds = []
                    decoder_output = None
                    for t in range(self.out_stp):
                        if t == 0 or np.random.rand()<smpl_prb:
                            last_close = tf.convert_to_tensor(self.x_decoder[start:end,t:t+1,:],dtype=tf.float32) #batch*1*1                          
                        else:
                            last_close = decoder_output
                       
                        q = self.q_dense(tf.expand_dims(h_enc,axis=1)) # batch,1,hidden
                        k = self.k_dense(enc_output) # batch, time, hidden
                        v = self.v_dense(enc_output)
                        q = self.split_heads(q,batch_size) #batch,head,1,depth
                        k = self.split_heads(k,batch_size) #batch,head,time,depth
                        v = self.split_heads(v,batch_size) #batch,head,time,depth
                        
                        score = tf.matmul(q,k,transpose_b=True) #batch*head*1*time
                        score = score/math.sqrt(n_unit1)
                        att_weights = tf.nn.softmax(score,axis=-1) #batch*head*1*time
                        context = tf.matmul(att_weights,v) #batch*head*1*depth
                        context = tf.squeeze(self.merge_head(context,batch_size),axis=1) #batch*hidden
                        
                        gate = self.gate(tf.concat([context,h_enc],axis=-1)) #batch*hidden
                        gt_mean = gate.numpy().mean()
                        h_enc = gate*context+(1-gate)*h_enc
                        
                        decoder_output,h_enc,c_enc = self.decoder_lstm(last_close,initial_state=(h_enc,c_enc))
                        decoder_output = self.decoder_dense(decoder_output)
                        decoder_output = tf.expand_dims(decoder_output,axis=1)#batch*1*1
                        preds.append(decoder_output)
                    pred = tf.concat(preds, axis=1)
                    loss = tf.reduce_mean(tf.square(pred-decoder_target))
                    self.losses.append(loss.numpy())
                
                trainable_weights = (self.encoder_lstm.trainable_weights + self.decoder_lstm.trainable_weights 
                                     +self.gate.trainable_weights+self.decoder_dense.trainable_weights
                                     +self.q_dense.trainable_weights+self.k_dense.trainable_weights+self.v_dense.trainable_weights)
                
                grads = tape.gradient(loss, trainable_weights)
                optimizer.apply_gradients(zip(grads, trainable_weights))
                
                recur_pred = self.my_predict_lite(encoder_inputs)
                recur_loss = tf.reduce_mean(tf.square(recur_pred - decoder_target))
                self.recur_loss.append(recur_loss.numpy())
                
                
                batch_end = time.time()
                batch_dur = batch_end-batch_start
                
                avg_batch_tim = (batch_end-start_time)/(epo*seq_max+seq+1)
                batches_left = epochs*seq_max-(epo*seq_max+seq+1)
                time_left = (avg_batch_tim*batches_left)/60
                
                print(f'Epoch: {epo}, Batch: {seq}, loss = {loss.numpy():.3f}, recur loss: {recur_loss.numpy():.3f}, Att_use: {gt_mean*100:.1f}%, SmPrb={smpl_prb:.2f}, Teach: {int(np.random.rand()<smpl_prb)}, Bat tim: {batch_dur:.2f}, Tim left: {time_left:.1f} min')
                
                if smpl_prb <1 and epo>=0.8*epochs and seq>=0.7*seq_max:
                    loss_precision.append(recur_loss)
                    if recur_loss.numpy() < best_loss_ever: 
                        print(">> enter best fit update")
                        best_loss_ever = loss.numpy()
                        best_encoder_weights = self.encoder_lstm.get_weights()
                        best_decoder_weights = self.decoder_lstm.get_weights()
                        best_gate_weights = self.gate.get_weights()
                        best_dense_weights = self.decoder_dense.get_weights()
                        best_q_weights = self.q_dense.get_weights()
                        best_k_weights = self.k_dense.get_weights()
                        best_v_weights = self.v_dense.get_weights()
                        print(f'best fit: Epoch: {epo}, Batch: {seq}, loss = {loss.numpy():.4f}, recur loss: {recur_loss.numpy():.3f}')
            if loss_precision:
                loss_precision = np.array(loss_precision)
                self.loss_precision.append(loss_precision.mean())   
            smpl_prb = max(0.05,0.8*(1-epo/epochs))
        self.encoder_lstm.set_weights(best_encoder_weights)
        self.decoder_lstm.set_weights(best_decoder_weights)
        self.gate.set_weights(best_gate_weights)
        self.decoder_dense.set_weights(best_dense_weights)
        self.q_dense.set_weights(best_q_weights)
        self.k_dense.set_weights(best_k_weights)
        self.v_dense.set_weights(best_v_weights)
        
        val_mean = np.mean(self.loss_precision)
        ema,alpha = 0,0.2
        for v in self.loss_precision:
            ema = alpha*v+(1-alpha)*ema
        val_ema = ema
        slope,*_ = linregress(np.arange(len(self.loss_precision)),self.loss_precision)
        slopee = max(0,slope)
        
        score = val_mean+1.2*val_ema+0.1*slopee
        return val_mean,val_ema,slope,score
    
    def my_predict_lite(self,x_encoder_tst):
        enc_output,h,c = self.encoder_lstm(x_encoder_tst)
        d_k = tf.cast(tf.shape(enc_output)[-1],tf.float32)       
        d_b = tf.shape(enc_output)[0]   
        decode_input = x_encoder_tst[:,-1:,0:1]
        self.decode_input = decode_input
        output_sequence = []
        
        for t in range(self.out_stp):
            
            q = self.q_dense(tf.expand_dims(h,axis=1)) 
            k = self.k_dense(enc_output) 
            v = self.v_dense(enc_output)
            q = self.split_heads(q,d_b) 
            k = self.split_heads(k,d_b) 
            v = self.split_heads(v,d_b) 
            
            score = tf.matmul(q,k,transpose_b=True) #batch*head*1*time
            score = score/tf.sqrt(d_k)
            att_weights = tf.nn.softmax(score,axis=-1) #batch*head*1*time
            cxt_vec = tf.matmul(att_weights,v) #batch*head*1*depth
            cxt_vec = tf.squeeze(self.merge_head(cxt_vec,d_b),axis=1) #batch*hidden
            
            gat = self.gate(tf.concat([cxt_vec,h],axis=-1)) 
            h = gat*cxt_vec+(1-gat)*h
            
            decode_input, h, c = self.decoder_lstm(decode_input,initial_state=(h,c)) 
            decode_input = self.decoder_dense(decode_input)
            decode_input = tf.expand_dims(decode_input,axis=1)
            output_sequence.append(decode_input)
        predictions = tf.concat(output_sequence,axis=1)
        return predictions
    def my_predict(self,x_encoder_tst):
        enc_output,h,c = self.encoder_lstm(x_encoder_tst)
        d_k = tf.cast(tf.shape(enc_output)[-1],tf.float32)
        d_b = tf.shape(enc_output)[0]  
        decode_input = x_encoder_tst[:,-1:,0:1]
        self.decode_input = decode_input
        output_sequence = []
        att_weights_seq = []
        gat_list = []
        for t in range(self.out_stp):      
            q = self.q_dense(tf.expand_dims(h,axis=1)) 
            k = self.k_dense(enc_output) 
            v = self.v_dense(enc_output)
            q = self.split_heads(q,d_b) 
            k = self.split_heads(k,d_b) 
            v = self.split_heads(v,d_b) 
            
            score = tf.matmul(q,k,transpose_b=True) #batch*head*1*time
            score = score/tf.sqrt(d_k)
            att_weights = tf.nn.softmax(score,axis=-1) #batch*head*1*time
            att_weights_seq.append(tf.squeeze(att_weights,axis=2))#[batch*head*time]
            cxt_vec = tf.matmul(att_weights,v) 
            cxt_vec = tf.squeeze(self.merge_head(cxt_vec,d_b),axis=1) #batch*hidden
            
            gat = self.gate(tf.concat([cxt_vec,h],axis=-1)) 
            h = gat*cxt_vec+(1-gat)*h
            gat_list.append(gat.numpy().mean())

            decode_input, h, c = self.decoder_lstm(decode_input,initial_state=(h,c))  
            decode_input = self.decoder_dense(decode_input)
            decode_input = tf.expand_dims(decode_input,axis=1)
            output_sequence.append(decode_input)
            
        predictions = tf.concat(output_sequence,axis=1)
        att_weights_seq = tf.stack(att_weights_seq,axis=2)
        self.att_weights = att_weights_seq.numpy()[0]
        self.gate_list = np.array(gat_list)
        return predictions        
    def plot(self,seed=0, path='./',i=0,ema_span=21):
        x = []; x.append(self.x_test[i]); x = np.array(x)
        y_pred = self.my_predict(x)
        self.y_pred = y_pred[0,:,0:1].numpy()
        y0_pred = self.scaler_logcls.inverse_transform(self.y_pred).flatten()
        self.y0_pred = y0_pred.copy()
        y0_pred = np.exp(y0_pred)
        
        trn_dataset = self.dataset[:self.trn_len]     
        tst_dataset = self.dataset[self.trn_len+i:self.trn_len+i+self.out_stp]
        tst_dataset = tst_dataset.copy()
        tst_dataset[('Pred',self.ticker)] = y0_pred
        y0 = tst_dataset['Close'].values
        self.rmse = np.sqrt(np.mean((y0_pred-y0)**2))

        EMA_21 = self.dataset['Close'].ewm(span=ema_span,adjust=False).mean() 
        self.dataset[('ema21',self.ticker)] = EMA_21

        plt.figure(figsize=(16,6))
        plt.title(f'Attention4 {seed} {self.ticker} pred {self.in_stp}-{self.out_stp}Day {self.trn_rat*100}% prediction')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(trn_dataset['Close'],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.plot(tst_dataset[['Close','Pred']],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.plot(self.dataset['ema21'],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.legend(['Train', 'Test Vals', 'Predictions','EMA 21 Days'], loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{path}LSTM pred att4 {seed} {self.ticker} {self.in_stp}-Day {int(self.out_stp*100)} Panel')
        #plt.show(block=False)
        
        plt.figure(figsize=(16,6))
        plt.plot(self.losses,marker='o',linestyle='--',markersize=3,linewidth=1,label='step_loss')
        plt.plot(self.recur_loss,marker='o',linestyle='--',markersize=3,linewidth=1,label='recur_loss')
        plt.legend()
        plt.tight_layout()
        #plt.show(block=False)
        plt.savefig(f'{path}Loss_plot att4 {seed} {self.ticker} {self.in_stp} {self.out_stp}')
        
        fig,axes = plt.subplots(2,self.num_heads//2,figsize=(20,10))
        axes = axes.flatten() 
        for h in range(self.num_heads):
            ax = axes[h]
            sns.heatmap(self.att_weights[h],ax=ax,cmap='viridis',cbar=True)
            ax.set_title(f'Att4 {seed} {h}-head {self.ticker} weights heatmap')
            ax.set_xlabel('Encoder Time Step')
            ax.set_ylabel('Decoder Time Step')
        plt.tight_layout()
        #plt.show(block=False)
        plt.savefig(f'{path}Att4_weights_plot {seed} {self.ticker} {self.in_stp} {self.out_stp}')
        
        plt.figure(figsize=(12,6))
        plt.plot(self.gate_list,marker='o',linestyle='--',markersize=3,linewidth=1,label='gate mean')
        plt.legend()
        plt.tight_layout()
        #plt.show(block=False)
        plt.savefig(f'{path}Gate_plot att4 {seed} {self.ticker} {self.in_stp} {self.out_stp}')
    def save(self,seed=0, path='./'):
        with open(f'{path}att4_{seed}_{self.ticker}_enc_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','wb') as f:
            pickle.dump(self.encoder_lstm.get_weights(),f)
        with open(f'{path}att4_{seed}_{self.ticker}_dec_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','wb') as f:
            pickle.dump(self.decoder_lstm.get_weights(),f)
        with open(f'{path}att4_{seed}_{self.ticker}_gat_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','wb') as f:
            pickle.dump(self.gate.get_weights(),f)
        with open(f'{path}att4_{seed}_{self.ticker}_den_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','wb') as f:
            pickle.dump(self.decoder_dense.get_weights(),f)
        with open(f'{path}att4_{seed}_{self.ticker}_q_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','wb') as f:
            pickle.dump(self.q_dense.get_weights(),f)
        with open(f'{path}att4_{seed}_{self.ticker}_k_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','wb') as f:
            pickle.dump(self.k_dense.get_weights(),f)
        with open(f'{path}att4_{seed}_{self.ticker}_v_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','wb') as f:
            pickle.dump(self.v_dense.get_weights(),f)
    def load(self,seed=0, path='./'):
        with open(f'{path}att4_{seed}_{self.ticker}_enc_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','rb') as f:
            enc_weights = pickle.load(f)
        with open(f'{path}att4_{seed}_{self.ticker}_dec_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','rb') as f:
            dec_weights = pickle.load(f)
        with open(f'{path}att4_{seed}_{self.ticker}_gat_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','rb') as f:
            gat_weights = pickle.load(f)
        with open(f'{path}att4_{seed}_{self.ticker}_den_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','rb') as f:
            den_weights = pickle.load(f)
        with open(f'{path}att4_{seed}_{self.ticker}_q_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','rb') as f:
            q_weights = pickle.load(f)
        with open(f'{path}att4_{seed}_{self.ticker}_k_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','rb') as f:
            k_weights = pickle.load(f)
        with open(f'{path}att4_{seed}_{self.ticker}_v_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','rb') as f:
            v_weights = pickle.load(f)
        self.encoder_lstm.set_weights(enc_weights)
        self.decoder_lstm.set_weights(dec_weights)
        self.gate.set_weights(gat_weights)
        self.decoder_dense.set_weights(den_weights)
        self.q_dense.set_weights(q_weights)
        self.k_dense.set_weights(k_weights)
        self.v_dense.set_weights(v_weights)