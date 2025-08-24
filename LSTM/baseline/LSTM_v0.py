# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 21:32:06 2025

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
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tensorflow.keras import Input
from sklearn.preprocessing import MinMaxScaler

class BatchLossLogger(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"\nBatch {batch}: loss = {logs['loss']:.6f}")
        
class LSTM_tforce:
    def __init__(self):
        self.model = None
        self.scaler_cls = MinMaxScaler(feature_range=(0,1))
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
        
        trn_len = int(np.ceil(len(close)*trn_rat)) 
        self.trn_len = trn_len
        
        
        norm_data = self.scaler_cls.fit_transform(close)
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
    def build(self,n_unit1=128):
        #symbolic graph
        encoder_inputs = Input(shape=(self.in_stp,4))
        encoder_lstm = LSTM(n_unit1,return_state=True)
        _,state_h,state_c = encoder_lstm(encoder_inputs)
        
        decoder_inputs = Input(shape=(None,1))
        decoder_lstm = LSTM(n_unit1,return_sequences=True,return_state=True) #layer instantiation 
        decoder_outputs,_,_ = decoder_lstm(decoder_inputs,initial_state=[state_h,state_c])
        decoder_dense = Dense(1)
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.model = Model([encoder_inputs,decoder_inputs],decoder_outputs)
        self.model.compile(optimizer='adam',loss='mean_squared_error')
        
    def fit(self,epochs=20,batch_size=1100,callbacks=[BatchLossLogger()]):
        self.model.fit([self.x_encoder,self.x_decoder],self.y_target,batch_size=batch_size,epochs=epochs,callbacks=callbacks)
        
        
    def predict(self,x_encoder_tst,start_token=0):
        encoder_model = Model(self.model.input[0],self.model.layers[2].output[1:])
        
        n_unit1 = self.model.layers[2].output[1].shape[-1]
        state_h = Input(shape=(n_unit1,))
        state_c = Input(shape=(n_unit1,))
        single_input = Input(shape=(1,1))
        
        decoder_lstm = self.model.layers[3]
        decoder_dense = self.model.layers[4]
        
        decoder_outputs,state_h1,state_c1 = decoder_lstm(single_input,initial_state=[state_h,state_c])
        decoder_outputs = decoder_dense(decoder_outputs)
        
        decoder_model = Model([single_input,state_h,state_c],[decoder_outputs,state_h1,state_c1])
        
        one_value = encoder_model.predict(x_encoder_tst)
        
        target_seq = np.zeros((1,1,1))
        target_seq[0,0,0] = start_token
        
        output_sequence = []
        
        for t in range(self.out_stp):
            output_tokens, h, c = decoder_model.predict([target_seq] + one_value)
            y_pred = output_tokens[0,0,0]
            output_sequence.append(y_pred)
            target_seq[0,0,0] = y_pred
            one_value = [h,c]
        return np.array(output_sequence)
            
    def plot(self,i=0,ema_span=21):
        x = []; x.append(self.x_test[i]); x = np.array(x)
        y_pred = self.predict(x,x[0,-1,0])
        
        y0_pred = self.scaler_cls.inverse_transform(y_pred.reshape(-1, 1))
        trn_dataset = self.dataset[:self.trn_len]
        
        tst_dataset = self.dataset[self.trn_len+i:self.trn_len+i+self.out_stp]
        tst_dataset = tst_dataset.copy()
        tst_dataset[('Pred',self.ticker)] = y0_pred.flatten()
        y0 = tst_dataset['Close'].values
        self.rmse = np.sqrt(np.mean((y0_pred-y0)**2))

        EMA_21 = self.dataset['Close'].ewm(span=ema_span,adjust=False).mean() 
        self.dataset[('ema21',self.ticker)] = EMA_21

        plt.figure(figsize=(16,6))
        plt.title(f'LSTM {self.ticker} 60pred {self.in_stp}-Day {self.trn_rat*100}% prediction')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(trn_dataset['Close'],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.plot(tst_dataset[['Close','Pred']],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.plot(self.dataset['ema21'],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.legend(['Train', 'Test Vals', 'Predictions','EMA 21 Days'], loc='lower right')
        plt.savefig(f'LSTM 60pred {self.ticker} {self.in_stp}-Day {int(self.trn_rat*100)} Panel')
        plt.show()
    def save(self,path='./'):
        self.model.save(f'{path}{self.ticker}_f4_{self.in_stp}Day_p{int(self.trn_rat*10)}.keras')
    def load(self,path):
        self.model = load_model(path)
        