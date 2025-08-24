# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 01:58:57 2025

@author: Ding Zhang
"""
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ta
import matplotlib
matplotlib.use('Qt5Agg')
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import time
import pickle

class BatchLossLogger(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"\nBatch {batch}: loss = {logs['loss']:.6f}")
        
class LSTM_ssmpl_v1:
    def __init__(self):
        self.encoder_lstm = None
        self.decoder_lstm = None
        self.decoder_dense = None
        self.scaler_delta = StandardScaler()
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
        self.losses = []
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
        
    def prepare(self,out_stp =60, in_stp=180,trn_rat=0.9):
        self.in_stp = in_stp
        self.out_stp = out_stp
        self.trn_rat = trn_rat
        close = self.dataset['Close'].values
        rsi = self.dataset['Rsi_14'].values
        vix = self.dataset['Vix'].values
        vol = self.dataset['Volume'].values
        delta = close[1:]-close[:-1]
        
        trn_len = int(np.ceil(len(close)*trn_rat)) 
        self.trn_len = trn_len
        
        norm_delta = self.scaler_delta.fit_transform(delta)
        norm_rsi = self.scaler_rsi.fit_transform(rsi)
        norm_vol = self.scaler_vol.fit_transform(vol)
        norm_vix = self.scaler_vix.fit_transform(vix)
        
        trn_delta = norm_delta[:trn_len-1]
        trn_rsi = norm_rsi[1:trn_len]
        trn_vol = norm_vol[1:trn_len]
        trn_vix = norm_vix[1:trn_len]
        trn_set = np.hstack([trn_delta,trn_rsi,trn_vol,trn_vix])
        
        
        for i in range(in_stp,trn_len-out_stp-1):
            self.x_encoder.append(trn_set[i-in_stp:i,:])
            self.x_decoder.append(trn_delta[i-1:i+out_stp-1])
            self.y_target.append(trn_delta[i:i+out_stp])
        self.x_encoder = np.array(self.x_encoder)
        self.x_decoder = np.array(self.x_decoder)
        self.y_target = np.array(self.y_target)
        

        tst_delta = norm_delta[trn_len-1-in_stp:]
        tst_rsi = norm_rsi[trn_len-in_stp:]
        tst_vol = norm_vol[trn_len-in_stp:]
        tst_vix = norm_vix[trn_len-in_stp:]
        tst_set = np.hstack([tst_delta,tst_rsi,tst_vol,tst_vix])
        for i in range(in_stp,len(tst_delta)):
            self.x_test.append(tst_set[i-in_stp:i,:])
        self.x_test = np.array(self.x_test)
    def buildfit(self,batch_size=250,n_unit1=128,epochs=50,callbacks=[BatchLossLogger()]):
        self.encoder_lstm = LSTM(n_unit1,return_state=True)           
        self.decoder_lstm = LSTM(n_unit1,return_state=True) #layer instantiation 实例化
        self.decoder_dense = Dense(1)        
        
        optimizer = optimizers.Adam()
        smpl_prb = 1
        seq_max = int(np.ceil(self.x_encoder.shape[0]/batch_size))
        start_time = time.time()
            
        best_loss_ever = np.inf
        best_encoder_weights = None
        best_decoder_weights = None
        best_dense_weights = None
        
        for epo in range(epochs):
            for seq in range(seq_max):
                batch_start = time.time()
                start,end = seq*batch_size,min(self.x_decoder.shape[0],(seq+1)*batch_size)
                with tf.GradientTape() as tape: 
                    encoder_inputs = tf.convert_to_tensor(self.x_encoder[start:end,:,:],dtype=tf.float32)
                    decoder_target = tf.convert_to_tensor(self.y_target[start:end,:,:],dtype=tf.float32)
                    enc_output,h_enc,c_enc = self.encoder_lstm(encoder_inputs)
                    last_close = tf.convert_to_tensor(self.x_decoder[start:end,0:1,:],dtype=tf.float32)
                    decoder_output,h_enc,c_enc = self.decoder_lstm(last_close,initial_state=(h_enc,c_enc))
                    decoder_output = self.decoder_dense(decoder_output)
                    decoder_output = tf.expand_dims(decoder_output,axis=1)
                    preds = []
                    preds.append(decoder_output)
                    for t in range(1,self.out_stp):
                        last_close = self.x_decoder[start:end,t:t+1,:] if np.random.rand()<smpl_prb else decoder_output
                        decoder_output,h_enc,c_enc = self.decoder_lstm(last_close,initial_state=(h_enc,c_enc))
                        decoder_output = self.decoder_dense(decoder_output)
                        decoder_output = tf.expand_dims(decoder_output,axis=1)
                        preds.append(decoder_output)
                    pred = tf.concat(preds, axis=1)
                    loss = tf.reduce_mean(tf.square(pred-decoder_target))
                    self.losses.append(loss.numpy())
                grads = tape.gradient(loss, self.encoder_lstm.trainable_weights + self.decoder_lstm.trainable_weights +self.decoder_dense.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.encoder_lstm.trainable_weights + self.decoder_lstm.trainable_weights + self.decoder_dense.trainable_weights))
                
                recur_pred = self.my_predict(encoder_inputs)
                recur_loss = tf.reduce_mean(tf.square(recur_pred - decoder_target))
                
                
                batch_end = time.time()
                batch_dur = batch_end-batch_start
                
                avg_batch_tim = (batch_end-start_time)/(epo*seq_max+seq+1)
                batches_left = epochs*seq_max-(epo*seq_max+seq+1)
                time_left = (avg_batch_tim*batches_left)/60
                
                print(f'Epoch: {epo}, Batch: {seq}, loss = {loss.numpy():.3f}, recur loss: {recur_loss.numpy():.3f}, SamplingProb={smpl_prb:.2f}, Teach: {np.random.rand()<smpl_prb}, Batch time: {batch_dur:.2f}, Time left: {time_left:.1f} min')
                
                if smpl_prb <1 and seq>=2 and recur_loss.numpy() < best_loss_ever:
                    best_loss_ever = loss.numpy()
                    best_encoder_weights = self.encoder_lstm.get_weights()
                    best_decoder_weights = self.decoder_lstm.get_weights()
                    best_dense_weights = self.decoder_dense.get_weights()
                    print(f'best fit: Epoch: {epo}, Batch: {seq}, loss = {loss.numpy():.4f}, recur loss: {recur_loss.numpy():.3f}')
           
            if epo<epochs*0.5:
                smpl_prb = 1
            else:
                smpl_prb = max(0.1,0.5*(1-epo/epochs))
        self.encoder_lstm.set_weights(best_encoder_weights)
        self.decoder_lstm.set_weights(best_decoder_weights)
        self.decoder_dense.set_weights(best_dense_weights)
    def my_predict(self,x_encoder_tst):
        _,h,c = self.encoder_lstm(x_encoder_tst)
                      
        decode_input = x_encoder_tst[:,-1:,0:1]
        self.decode_input = decode_input
        output_sequence = []
        
        for t in range(self.out_stp):
            output_token, h, c = self.decoder_lstm(decode_input,initial_state=(h,c))
            decode_input = self.decoder_dense(output_token)
            decode_input = tf.expand_dims(decode_input,axis=1)
            output_sequence.append(decode_input)
        predictions = tf.concat(output_sequence,axis=1)
        return predictions
            
    def plot(self,i=0,ema_span=21):
        x = []; x.append(self.x_test[i]); x = np.array(x)
        y_pred = self.my_predict(x)
        self.y_pred = y_pred[0,:,0:1].numpy()
        y0_pred = self.scaler_delta.inverse_transform(self.y_pred).flatten()
        self.y0_pred = y0_pred.copy()
        self.last_close = self.dataset['Close'].iloc[self.trn_len+i-1]
        y0_pred[0] +=self.last_close
        for j in range(1,len(self.y0_pred)):
            y0_pred[j] += y0_pred[j-1]
            
        trn_dataset = self.dataset[:self.trn_len]     
        tst_dataset = self.dataset[self.trn_len+i:self.trn_len+i+self.out_stp]
        tst_dataset = tst_dataset.copy()
        tst_dataset[('Pred',self.ticker)] = y0_pred.flatten()
        y0 = tst_dataset['Close'].values
        self.rmse = np.sqrt(np.mean((y0_pred-y0)**2))

        EMA_21 = self.dataset['Close'].ewm(span=ema_span,adjust=False).mean() 
        self.dataset[('ema21',self.ticker)] = EMA_21

        plt.figure(figsize=(16,6))
        plt.title(f'LSTM {self.ticker} pred {self.in_stp}-{self.out_stp} Day {self.trn_rat*100}% prediction')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(trn_dataset['Close'],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.plot(tst_dataset[['Close','Pred']],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.plot(self.dataset['ema21'],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.legend(['Train', 'Test Vals', 'Predictions','EMA 21 Days'], loc='lower right')
        plt.savefig(f'LSTM pred ssmpl1 {self.ticker} {self.in_stp}-Day {int(self.out_stp*100)} Panel')
        plt.show()
        
        plt.figure(figsize=(16,6))
        plt.plot(self.losses,marker='o',linestyle='--',markersize=3,linewidth=1)
        plt.show()
        plt.savefig(f'Loss_plot ssmpl {self.ticker} {self.in_stp} {self.out_stp}')
    def save(self,path='./'):
        with open(f'{path}ssmpl1{self.ticker}_enc_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','wb') as f:
            pickle.dump(self.encoder_lstm.get_weights(),f)
        with open(f'{path}ssmpl1{self.ticker}_dec_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','wb') as f:
            pickle.dump(self.decoder_lstm.get_weights(),f)
        with open(f'{path}ssmpl1{self.ticker}_den_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','wb') as f:
            pickle.dump(self.decoder_dense.get_weights(),f)
    def load(self,path='./'):
        with open(f'{path}ssmpl1{self.ticker}_enc_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','rb') as f:
            enc_weights = pickle.load(f)
        with open(f'{path}ssmpl1{self.ticker}_dec_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','rb') as f:
            dec_weights = pickle.load(f)
        with open(f'{path}ssmpl1{self.ticker}_den_{self.in_stp}Day_{self.out_stp}p{int(self.trn_rat*10)}.keras','rb') as f:
            den_weights = pickle.load(f)
        self.encoder_lstm.set_weights(enc_weights)
        self.decoder_lstm.set_weights(dec_weights)
        self.decoder_dense.set_weights(den_weights)