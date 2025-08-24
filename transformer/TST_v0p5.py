# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 22:40:52 2025

@author: Ding Zhang
"""
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import ta
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

import random
import matplotlib
matplotlib.use('Agg')
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
from tensorflow.keras.layers import Dense,Input,MultiHeadAttention,Embedding,Add,LayerNormalization,Dropout,Lambda,Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential,Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import time
import pickle
import math

def debug_wrapper(x,name):
    tf.print(f"🧪 {name} — mean:", tf.reduce_mean(x), 
             "max:", tf.reduce_max(x), 
             "min:", tf.reduce_min(x))
    return x

class BatchLossLogger(Callback):
    def __init__(self,schedule,tot_batch,batchs_per_epoch,epoch):
        super().__init__()
        self.schedule = schedule
        self.tot_batch = tot_batch
        self.batch_times = []
        self.batchs_per_epoch = batchs_per_epoch
        self.epoch = epoch
        self.batch_losses = []
    def on_train_batch_begin(self,batch,logs=None):
        self.batch_start_time = time.time()
    def on_train_batch_end(self, batch, logs=None):
        step = tf.keras.backend.get_value(self.model.optimizer.iterations)
        current_lr = self.schedule(step).numpy()
                
        duration = time.time() - self.batch_start_time
        self.batch_times.append(duration)
        avg_time = sum(self.batch_times) / len(self.batch_times)
        remaining_batches = self.tot_batch - (batch + 1 + self.epoch * self.batchs_per_epoch)
        remaining_time = remaining_batches * avg_time
        mins, secs = divmod(remaining_time, 60)
        self.batch_losses.append(logs.get('loss'))
        print(f"\n Epoch:{self.epoch}, Batch {batch}: loss = {logs['loss']:.3f}, learning = {current_lr:.3e}, Est. time left: {int(mins):02d}:{int(secs):02d}")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        
class TST_v0p5:
    def __init__(self):
        self.model = None
        self.history = None
        self.cb = None
        
        self.scaler_logcls = StandardScaler() #standard
        self.scaler_logema = StandardScaler()  # standard

        self.scaler_bb = MinMaxScaler(feature_range=(0,1)) # min max
        self.scaler_body = MinMaxScaler(feature_range=(0,1)) #min max
        self.scaler_wick = MinMaxScaler(feature_range=(0,1)) #min max
        
        self.dataset = None
        self.trn_len = None
        self.trn_rat = None
        
        self.x_encoder = []
        self.x_decoder = []
        self.y_target = []
        
        self.x_tst = []
        self.y_tst = []
        self.rmse = None


        self.seq_len = None
        self.pred_len = None
        
        self.nums_feat = None
        self.nums_head = None
        self.embed_dim = None
        self.ff_dim = None
        self.dropout_rate = None
        self.tn_layers = None
        self.base_lr = None
        self.clipnorm = None
        self.lr_schedule = None
    
        self.y_pred = []
        
        self.tot_batch = None
        self.steps_per_epoch = None
        self.loss = None
        self.val_loss = None
        
        
        self.importance_df = None
        self.importance = None
        self.base_mse = None
    def hashem_rsi(self,close_series, rsilen=14,smth=1):
        rsi0 = ta.momentum.rsi(close_series,window=rsilen,fillna=False)
        rsi_sma = rsi0.rolling(smth).mean()
        rsi_ema = rsi0.ewm(span=smth,adjust=False).mean()
        rsi_avg = (rsi_ema+rsi_sma)/2
        rsi_avg = rsi_avg.bfill()
        return rsi_avg
    def rsi_zones(self,rsi):
        rsi_zone = np.zeros_like(rsi,dtype=float)
        rsi_zone[(rsi >= 65) & (rsi < 85)]  =  1.0   # Bull zone
        rsi_zone[(rsi >= 85)]               = -1.0  # Overbought reversal
        rsi_zone[(rsi >= 25) & (rsi < 45)]  = -1.0  # Bear zone
        rsi_zone[(rsi < 25)]                =  1.0  # Oversold reversal
        return rsi_zone
    def bb_squeeze(self,bw,thr,window=5):
        return bw.le(thr).rolling(window,min_periods=1).mean()  
    def obv_slope_tanh(self,obv,w=50,eps=1e-6):
        obv = np.asarray(obv, float)
        d1 = np.diff(obv, prepend=obv[0])               # 一阶差分：当天净流入方向
        vol = pd.Series(d1).rolling(w, min_periods=w//2).std().bfill().values + eps
        z = d1 / vol                                     # 按近期波动缩放，保留正负
        return np.tanh(z).astype(np.float32)             # 压到 [-1,1] 
    def load_data(self,ticker,start,end=datetime(2025,8,7,5,0,0)):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.dataset = yf.download(ticker,start,end)
        # heikin ashi candle
        ha_close = (self.dataset[('Open',ticker)]+self.dataset[('Close',ticker)]+self.dataset[('High',ticker)]+self.dataset[('Low',ticker)])/4
        ha_open = [(self.dataset[('Open',ticker)].iloc[0]+self.dataset[('Close',ticker)].iloc[0])/2]
        for i in range(1,len(self.dataset[('Open',ticker)].values)):
            ha_open.append((ha_open[i-1]+ha_close.iloc[i-1])/2)
        self.dataset[('ha_close',ticker)] = ha_close
        self.dataset[('ha_open',ticker)] = ha_open
        self.dataset[('ha_high',ticker)] = self.dataset[[('ha_open', ticker), ('ha_close', ticker), ('High', ticker)]].max(axis=1)
        self.dataset[('ha_low',ticker)] = self.dataset[[('ha_open', ticker), ('ha_close', ticker), ('Low', ticker)]].min(axis=1)
            
        # EMA, trend
        ema = EMAIndicator(close=self.dataset[('Close',ticker)],window=10)
        self.dataset[('ema',ticker)] = ema.ema_indicator()
        # macd_diff, trend differentiation
        macd = MACD(close=self.dataset[('Close',ticker)])
        self.dataset[('macd_diff',ticker)] = macd.macd_diff()
        max_macd = self.dataset[('macd_diff',ticker)].abs().max()
        
        # rsi, momentum
        self.dataset[('rsi_14',ticker)] = self.hashem_rsi(self.dataset[('Close',ticker)])
        
        # bollinger band width, volatility
        bb = BollingerBands(close=self.dataset[('Close',ticker)])
        self.dataset[('bb_bandwidth',ticker)] = (bb.bollinger_hband()-bb.bollinger_lband())/bb.bollinger_mavg()
        thr = self.dataset[('bb_bandwidth',ticker)].quantile(0.3)
        
        # obv, energy
        obv = OnBalanceVolumeIndicator(close=self.dataset[('Close',ticker)],volume=self.dataset[('Volume',ticker)])
        self.dataset[('obv',ticker)] = obv.on_balance_volume()
        
        # heikin ashi candle, body 
        self.dataset[('ha_body',ticker)] =  self.dataset[('ha_close',ticker)]-self.dataset[('ha_open',ticker)]
        # heikin ashi candle, tick ratio
        self.dataset[('ha_wick_rat',ticker)] = (self.dataset[('ha_high',ticker)]-self.dataset[('ha_low',ticker)])/abs(1e-6+self.dataset[('ha_body',ticker)])
        
        columns = self.dataset.columns.to_list()
        any_nan_mask = self.dataset[columns].isna().any(axis=1)
        last_nan_idx = self.dataset[any_nan_mask].index.max()
        last_nan_pos = self.dataset.index.get_loc(last_nan_idx)
        self.dataset = self.dataset.iloc[last_nan_pos+1:].reset_index(drop=True)
        
        self.dataset[('macd_scl',ticker)] = self.dataset[('macd_diff',ticker)]/max_macd
        self.dataset[("rsi_zone",ticker)] = self.rsi_zones(self.dataset[('rsi_14',ticker)].values)
        self.dataset[('bb_squeeze',ticker)] = self.bb_squeeze(self.dataset[('bb_bandwidth',ticker)],thr,window=5)
        self.dataset[('obv_slope',ticker)] = self.obv_slope_tanh(self.dataset[('obv',ticker)],w=50,eps=1e-6)
    def prepare(self,pred_len =60, seq_len=181,trn_rat=0.9):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.trn_rat = trn_rat

        close = self.dataset[('Close',self.ticker)].values
        logcls = np.log(close)
        ema = self.dataset[('ema',self.ticker)].values
        logema = np.log(ema)
        macd_scl = self.dataset[('macd_scl',self.ticker)].values
        rsi_z = self.dataset[('rsi_zone',self.ticker)].values
        bb_band = self.dataset[('bb_squeeze',self.ticker)].values
        obv_slope = self.dataset[('obv_slope',self.ticker)].values
        #ha_body = self.dataset[('ha_body',self.ticker)].values
        #ha_wick = self.dataset[('ha_wick_rat',self.ticker)].values
        
        trn_len = int(np.ceil(len(close)*trn_rat)) 
        self.trn_len = trn_len
        
        
        norm_data = self.scaler_logcls.fit_transform(logcls.reshape(-1, 1)) #standard
        norm_ema =self.scaler_logema.fit_transform(logema.reshape(-1, 1)) # standard
        norm_macd = macd_scl.reshape(-1, 1)
        norm_rsi = rsi_z.reshape(-1, 1)
        norm_bb = self.scaler_bb.fit_transform(bb_band.reshape(-1, 1)) # min max
        norm_obv = obv_slope.reshape(-1, 1)
        #norm_body = self.scaler_body.fit_transform(ha_body.reshape(-1, 1)) #min max
        #norm_wick = self.scaler_wick.fit_transform(ha_wick.reshape(-1, 1)) #min max

        trn_data = norm_data[:trn_len]
        trn_ema = norm_ema[:trn_len]
        trn_macd = norm_macd[:trn_len]
        trn_rsi = norm_rsi[:trn_len]
        trn_bb = norm_bb[:trn_len]
        trn_obv = norm_obv[:trn_len]
        #trn_body = norm_body[:trn_len]
        #trn_wick = norm_wick[:trn_len]
        #trn_set = np.hstack([trn_data,trn_ema,trn_macd,trn_rsi,trn_bb,trn_obv,trn_body,trn_wick])
        trn_set = np.hstack([trn_data,trn_ema,trn_macd,trn_rsi,trn_bb,trn_obv])
        
        for i in range(seq_len,trn_len-pred_len):
            self.x_encoder.append(trn_set[i-seq_len:i,:])
            self.x_decoder.append(trn_data[i-1:i+pred_len-1])
            self.y_target.append(trn_data[i:i+pred_len])
        self.x_encoder = np.array(self.x_encoder) # 700,365,8
        self.nums_feat = self.x_encoder.shape[-1]
        self.x_decoder = np.array(self.x_decoder) # 700,40,1
        self.y_target = np.array(self.y_target) # 700,40,1
        
        tst_data = norm_data[trn_len-seq_len:]
        tst_ema = norm_ema[trn_len-seq_len:]
        tst_macd = norm_macd[trn_len-seq_len:]
        tst_rsi = norm_rsi[trn_len-seq_len:]
        tst_bb = norm_bb[trn_len-seq_len:]
        tst_obv = norm_obv[trn_len-seq_len:]
        #tst_body = norm_body[trn_len-seq_len:]
        #tst_wick = norm_wick[trn_len-seq_len:]
        #tst_set = np.hstack([tst_data,tst_ema,tst_macd,tst_rsi,tst_bb,tst_obv,tst_body,tst_wick])
        tst_set = np.hstack([tst_data,tst_ema,tst_macd,tst_rsi,tst_bb,tst_obv])
        for i in range(seq_len,len(tst_data)-pred_len):
            self.x_tst.append(tst_set[i-seq_len:i,:])
            self.y_tst.append(tst_data[i:i+pred_len])
        self.x_tst = np.array(self.x_tst) # 122,365,8
        self.y_tst = np.array(self.y_tst)
    def build(self, seed = 0, embed_dim=128, ff_dim = 256, nums_head = 4, dropout_rate = 0.1, 
              base_lr = 1e-3, clipnorm = 1.0, batch_size=32, epochs=50,warmup_epoch=10):
        tf.keras.utils.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.nums_head = nums_head
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.base_lr = base_lr
        self.clipnorm = clipnorm
        
        inputs = Input(shape=(self.seq_len,self.nums_feat)) #none, seq_len,features
        #inputs = Lambda(lambda t: debug_wrapper(t, "model_inputs"))(inputs)
        
        x = Dense(embed_dim)(inputs) # none, seq_len,embed_dim
        #x = Lambda(lambda t: debug_wrapper(t, "dense_proj_out"))(x)
        
        # Positional encoding
        positions = tf.range(start=0,limit=self.seq_len,delta=1)
        pos_encoding = Embedding(input_dim=self.seq_len,output_dim=embed_dim)(positions)
        x = x+pos_encoding 
        #x = Lambda(lambda t: debug_wrapper(t, "pos_encoding_added"))(x)
        
        # Multi-head self-attention+ Causal mask
        attn_output = MultiHeadAttention(num_heads=nums_head,key_dim=embed_dim,dropout=dropout_rate)(x,x,use_causal_mask=True)
        x = Add()([x,attn_output])
        x = LayerNormalization()(x) 
        #x = Lambda(lambda t: debug_wrapper(t, "after_mha_norm"))(x)
        
        # Feed-forward network
        ff_output = Sequential([Dense(ff_dim,activation='relu'),Dropout(dropout_rate),Dense(embed_dim)])(x)
        x = Add()([x,ff_output])
        x = LayerNormalization()(x)
        #x = Lambda(lambda t: debug_wrapper(t, "after_ffn_norm"))(x)
        
        # Output
        last_token = Lambda(lambda t: t[:, -1, :])(x)  # 取最后一天的表示
        pred_vec   = Dense(self.pred_len)(last_token)  # (batch, 40)
        outputs    = Reshape((self.pred_len, 1))(pred_vec)  # (batch, 40, 1)
        
        # warm up + gradient clipping 
        self.steps_per_epoch = self.x_encoder.shape[0]//batch_size
        self.lr_schedule = Warmupcosine(base_lr=1e-4, total_steps = epochs * self.steps_per_epoch, warmup_steps=warmup_epoch* self.steps_per_epoch)
        
        # model 封装
        model = Model(inputs=inputs,outputs=outputs)
        opt = optimizers.Adam(learning_rate=self.lr_schedule,clipnorm=clipnorm)
        model.compile(loss='mse',optimizer=opt)
        self.model = model
    def fit(self,batch_size=32, epochs=50):
        #cb = BatchLossLogger(self.x_encoder,self.y_target,batch_size)
        #steps_per_epoch = math.ceil(self.x_encoder.shape[0]/batch_size)
        self.tot_batch = epochs*self.steps_per_epoch
        self.cb = BatchLossLogger(self.lr_schedule,self.tot_batch,self.steps_per_epoch,epochs)
        #self.cb = BatchLossLogger1(self.x_encoder,self.y_target,batch_size)
        self.history = self.model.fit(self.x_encoder,self.y_target,validation_data=[self.x_tst,self.y_tst],epochs=epochs,
                                      batch_size=batch_size,verbose=2,callbacks=[self.cb])
    def predict(self,seed=0,path='./',i=0,ema_span=21):
        x_tst = self.x_tst[i]
        if x_tst.ndim==2:
            x_tst = np.expand_dims(x_tst,axis=0) # 1,365,4
        y_pred = self.model.predict(x_tst,verbose=0) # 1,40,1
        self.y_pred = y_pred[0,:,0:1]
        y0_pred = self.scaler_logcls.inverse_transform(self.y_pred).flatten()
        self.y0_pred = np.exp(y0_pred)
        
        tst_dataset = self.dataset[self.trn_len+i:self.trn_len+i+self.pred_len]
        tst_dataset = tst_dataset.copy()
        tst_dataset[('Pred',self.ticker)] = self.y0_pred
        y0_true = tst_dataset['Close'].values
        self.rmse = np.sqrt(np.mean((self.y0_pred-y0_true)**2))

        EMA_21 = self.dataset['Close'].ewm(span=ema_span,adjust=False).mean() 
        self.dataset[('ema21',self.ticker)] = EMA_21

        trn_dataset = self.dataset[:self.trn_len]  
        
        plt.figure(figsize=(16,6))
        plt.title(f'Self-Attention {seed} {self.ticker} pred {self.seq_len}-{self.pred_len}Day {self.trn_rat*100}% prediction')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(trn_dataset['Close'],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.plot(tst_dataset[['Close','Pred']],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.plot(self.dataset['ema21'],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.legend(['Train', 'Test Vals', 'Predictions','EMA 21 Days'])
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'{path}Transformer6 pred v0 {seed} {self.ticker} {self.seq_len}-{self.pred_len} Panel')
        
        
        #self.loss = self.cb.batch_losses
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history.get('val_loss',None)
        epochs = [i*self.steps_per_epoch for i in range(1,len(self.val_loss)+1)]
        #tot_batches = range(1,len(self.loss)+1)
        plt.figure(figsize=(16,6))
        #plt.plot(tot_batches, self.loss,marker='o',linestyle='--',markersize=3,linewidth=1,label='train_loss')
        plt.plot(epochs, self.loss,marker='o',linestyle='--',markersize=3,linewidth=1,label='train_loss')
        plt.plot(epochs, self.val_loss,marker='o',linestyle='--',markersize=3,linewidth=1,label='validation_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'{path}Loss_plot transformer6 {seed} {self.ticker} {self.seq_len} {self.pred_len}')
        
    def save(self,seed=0, path='./'):
        self.model.save(f'{path}Transformer6_v0_{self.ticker}_{self.seq_len}_{self.pred_len}.keras')
        
        with open(f'{path}Transformer6_history_{self.ticker}_{self.seq_len}_{self.pred_len}.keras','wb') as f:
            pickle.dump(self.history,f)
    def load(self,seed=0, path='./'):
        self.model = load_model(f'{path}Transformer6_v0_{self.ticker}_{self.seq_len}_{self.pred_len}.keras',compile=False,safe_mode=False)
        
        with open(f'{path}Transformer6_history_{self.ticker}_{self.seq_len}_{self.pred_len}.keras','rb') as f:
            self.history = pickle.load(f)
    def compute_importance(self, i=0, seed=0, scoring='neg_mean_squared_error'): 
        feature_cols = ['close', 'ema', 'macd_scl', 'rsi_zone','bb_squeeze', 'obv_slope']
        rng = np.random.RandomState(seed)
        results = []
        for j,feat_name in enumerate(feature_cols):
            deltas = []
            mses = []
            for ii in range(i,min(i+30,self.x_tst.shape[0])):  
                y0_pred = self.model.predict(self.x_tst[ii:ii+1],verbose=0)
                base_mse = mean_squared_error(self.y_tst[ii],y0_pred[0])
                
                x_perturb = self.x_tst[ii].copy()
                x_perturb[:,j] = rng.permutation(x_perturb[:, j])
                y_pred = self.model.predict(x_perturb.reshape(1,self.seq_len,x_perturb.shape[-1]),verbose=0)
                deltas.append(mean_squared_error(self.y_tst[ii],y_pred[0,:,0:1])-base_mse)
                mses.append(base_mse)
            results.append({'feature': feat_name,'base_mse':np.mean(mses),'importance_mean': np.mean(deltas),'importance_std': np.std(deltas)})
        
        #self.importance_df = pd.DataFrame(results).sort_values(by='importance_mean',ascending=False)
        self.importance_df = pd.DataFrame(results)
        self.importance = self.importance_df['importance_mean'].values
        return self.importance, self.importance_df['base_mse'].values
                        
class Warmupcosine(optimizers.schedules.LearningRateSchedule):
    def __init__(self,base_lr,total_steps,warmup_steps,min_lr=1e-5):
        super().__init__()
        self.base_lr = base_lr
        self.tot_stps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
    def __call__(self,step):
        step = tf.cast(step, tf.float32)
        def warmup():
            return self.base_lr*step/self.warmup_steps
        def decay():
            progress = (step-self.warmup_steps)/(self.tot_stps-self.warmup_steps)
            cosine_decay  =0.5*(1+tf.cos(np.pi*progress))
            return self.min_lr+(self.base_lr-self.min_lr)*cosine_decay

        return tf.cond(step < self.warmup_steps, warmup, decay)
    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.tot_stps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }

class BatchLossLogger1(Callback):
    def __init__(self,x_train,y_train,batch_size):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
    def on_train_batch_end(self, batch, logs=None):
       # print(f"\nBatch {batch}: loss = {logs['loss']:.6f}")
        start = batch * self.batch_size
        end = min(start + self.batch_size,self.y_train.shape[0])
        print(f"\n=== Gradient Debug at Batch {batch} ===")
        x = self.x_train[start:end]
        y = self.y_train[start:end]
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = tf.keras.losses.MeanSquaredError()(y,y_pred)
        grads = tape.gradient(loss, self.model.trainable_weights)
        i = 0
        for layer in self.model.layers:
            weights = layer.trainable_weights
            if not weights:
                continue
            for var in weights:
                grad = grads[i]
                if grad is not None:
                    gnorm = tf.norm(grad).numpy()
                    print(f'{layer.name}--{var.name}: grad norm = {gnorm:.3e}')
                else:
                    print(f'{layer.name}--{var.name}: grad=None')
                i +=1
        #for i, (grad, var) in enumerate(zip(grads, self.model.trainable_weights)):
            #if grad is not None:
           #     gnorm = tf.norm(grad).numpy()
          #      print(f"{var.name} grad norm = {gnorm:.3e}")
         #   else:
        #        print(f"{var.name} grad = None")

       # print("=== End Gradient Debug ===\n")
        for i,layer in enumerate(self.model.layers):
            if isinstance(layer,Dense):
                w = layer.kernel
                b = layer.bias
                norm_w = tf.norm(w).numpy()
                norm_b = tf.norm(b).numpy()
                print(f"Batch {batch}, Dense layer {i}: weight norm = {norm_w:.4e}, bias norm = {norm_b:.4e}")
                #break  