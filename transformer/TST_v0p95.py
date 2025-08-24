# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 12:59:23 2025

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
from tensorflow.keras.layers import Dense,Input,MultiHeadAttention,Embedding,Add,LayerNormalization,Dropout,Lambda,Reshape,Activation,Concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
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
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
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
        self.batch_losses = defaultdict(list)
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
        self.batch_losses['loss'].append(logs.get('loss'))
        #self.batch_losses['val_loss'].append(logs.get('val_loss'))
        self.batch_losses['tp_bins_loss'].append(logs.get('tp_bins_loss'))
        #self.batch_losses['val_tp_bins_loss'].append(logs.get('val_tp_bins_loss'))
        self.batch_losses['y_hat_loss'].append(logs.get('y_hat_loss'))
        #self.batch_losses['val_y_hat_loss'].append(logs.get('val_y_hat_loss'))
        print(f"\n Epoch:{self.epoch}, Batch {batch}: tp_loss = {logs['tp_bins_loss']:.3f}, y_loss = {logs['y_hat_loss']:.3f}, learning = {current_lr:.3e}, Est. time left: {int(mins):02d}:{int(secs):02d}")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def set_steps_per_epoch(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch
        
class TST_v0p95:
    def __init__(self):
        self.model = None
        self.history = None
        self.cb = None
        
        self.scaler_logcls = StandardScaler() #standard
        self.scaler_logopn = StandardScaler() #standard
        self.scaler_loghigh = StandardScaler() #standard
        self.scaler_loglow = StandardScaler() #standard
        self.scaler_logema = StandardScaler()  # standard
        self.scaler_loghahigh = StandardScaler() #standard
        self.scaler_loghalow = StandardScaler() #standard
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
        self.y_signal = []
        
        self.x_tst = []
        self.y_tst = []
        self.y_tsi = []
        self.rmse = None


        self.seq_len = None
        self.pred_len = None
        
        self.nums_feat = None
        self.head_base = None
        self.head_sig = None
        self.embed_dim = None
        self.embed_sig = None
        self.ff_dim = None
        self.ff_sig = None
        
        self.dropout_rate = None
        self.tn_layers = None
        self.base_lr = None
        self.clipnorm = None
        self.lr_schedule = None
    
        self.y_pred = []
        
        self.tot_batch = None
        self.steps_per_epoch = None
        
        self.yhat_loss = None
        self.val_yhat_loss = None
        self.tps_loss = None
        self.val_tps_loss = None
        
        self.importance_df = None
        self.importance = None
        self.base_mse = None
        
        self.tp_lvls=None
        self.edges = None
        self.tp_labels = None
        self.stop_loss=None
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
    def load_data(self,ticker,start,end=datetime(2025,8,14,5,0,0)):
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
    def compute_confident_levels(self,pred_len=60,stop_loss=0.94):
        self.stop_loss = stop_loss
        ha_high = self.dataset[('ha_high',self.ticker)].to_numpy()
        ha_low = self.dataset[('ha_low',self.ticker)].to_numpy()
        highs = self.dataset[('High',self.ticker)].to_numpy()
        lows = self.dataset[('Low',self.ticker)].to_numpy()
        
        short_loss = 2-stop_loss
        profits = []
        for i,entry in enumerate(self.dataset[('Open',self.ticker)].to_numpy()):
            line = 1.0;add = 0         
            rng_max = min(pred_len,len(highs)-i)
            
            for j in range(0,rng_max):
                if lows[i+j]/entry<=stop_loss:
                    profits.append(line-1)
                    add = 1
                    break
                line = max(line,ha_high[i+j]/entry)
            if not add: profits.append(line-1)
            if line+stop_loss<2: 
                profits.pop()
                line = 1.0; add = 0
                for j in range(0,rng_max):
                    if highs[i+j]/entry>=short_loss:
                        profits.append(line-1)
                        add = 1
                        break
                    line = min(line,ha_low[i+j]/entry)
                if not add: profits.append(line-1)
                if line+short_loss>2: 
                    profits.pop()
                    profits.append(0)
        self.dataset[('tp',self.ticker)] = profits
    def compute_tp_labels(self,pred_len=60,qs=(1/6,1/3,1/2,2/3,5/6),gamma = 2, eps=1e-4):
        tp_yield = self.dataset[('tp',self.ticker)].to_numpy()
        self.tp_lvls=np.quantile(tp_yield,qs)
        #tp_binned = np.digitize(self.dataset[('tp',self.ticker)],self.tp_lvls,right=True)
        #self.tp_labels = to_categorical(tp_binned,num_classes=len(qs)+1)
        
        tp_labels = np.zeros((len(tp_yield),len(qs)+1))
        edges = np.insert(self.tp_lvls,0,tp_yield.min()-0.1)
        edges = np.insert(edges,len(edges),tp_yield.max()+0.1)
        self.edges = edges
        for i,tp in enumerate(tp_yield):
            idx = np.searchsorted(edges, tp, side='right') - 1
            left = edges[idx]
            right = edges[idx+1]
            width = np.maximum(right-left,eps)
            
            dL = np.clip(tp-left,0,None)
            dR = np.clip(right-tp,0,None)
            to_left = dL<dR; to_right = dR<dL
            s = 2*np.minimum(dL,dR)/width
            s = np.clip(s,0,1)
            
            p_cur = 0.5+0.5*(s**gamma)
            if (to_left and idx ==0) or (to_right and idx==len(edges)-2): 
                p_cur = 1
            tp_labels[i][idx] = p_cur
            p_nb = 1-p_cur
            if p_nb and to_left:
                tp_labels[i][idx-1] = p_nb
            if p_nb and to_right:
                tp_labels[i][idx+1] = p_nb
        smoothed = gaussian_filter1d(tp_labels, sigma=2, axis=0, mode="nearest")
        smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)
        self.tp_labels = tp_labels
    def prepare(self,pred_len =60, seq_len=181,trn_rat=0.9,stop_loss=0.94,tp_lvls=[1.1,1.15,1.2]):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.trn_rat = trn_rat
        
        self.compute_confident_levels(pred_len=pred_len,stop_loss=stop_loss)
        self.compute_tp_labels(pred_len=pred_len)
        #======predict header=====
        close = self.dataset[('Close',self.ticker)].values
        logcls = np.log(close)
        opn = self.dataset[('Open',self.ticker)].values
        logopn = np.log(opn)
        high = self.dataset[('High',self.ticker)].values
        loghgh = np.log(high)
        low = self.dataset[('Low',self.ticker)].values
        loglow = np.log(low)
        ema = self.dataset[('ema',self.ticker)].values
        logema = np.log(ema)
        #======tp header=====
        ha_high = self.dataset[('ha_high',self.ticker)].values
        logha_h = np.log(ha_high)
        ha_low = self.dataset[('ha_low',self.ticker)].values
        logha_l = np.log(ha_low)
        macd_scl = self.dataset[('macd_scl',self.ticker)].values
        rsi_z = self.dataset[('rsi_zone',self.ticker)].values
        bb_band = self.dataset[('bb_squeeze',self.ticker)].values
        obv_slope = self.dataset[('obv_slope',self.ticker)].values
              
        trn_len = int(np.ceil(len(close)*trn_rat)) 
        self.trn_len = trn_len
          
        norm_data = self.scaler_logcls.fit_transform(logcls.reshape(-1, 1)) #standard
        norm_opn = self.scaler_logopn.fit_transform(logopn.reshape(-1, 1)) #standard
        norm_hgh = self.scaler_loghigh.fit_transform(loghgh.reshape(-1, 1)) #standard
        norm_low = self.scaler_loglow.fit_transform(loglow.reshape(-1, 1)) #standard
        norm_ema =self.scaler_logema.fit_transform(logema.reshape(-1, 1)) # standard
        norm_hah = self.scaler_loghahigh.fit_transform(logha_h.reshape(-1, 1)) #min max
        norm_hal = self.scaler_loghalow.fit_transform(logha_l.reshape(-1, 1)) #min max
        norm_macd = macd_scl.reshape(-1, 1)
        norm_rsi = rsi_z.reshape(-1, 1)
        norm_bb = self.scaler_bb.fit_transform(bb_band.reshape(-1, 1)) # min max
        norm_obv = obv_slope.reshape(-1, 1)
        norm_tp_blks = self.tp_labels  


        trn_data = norm_data[:trn_len]
        trn_opn = norm_opn[:trn_len]
        trn_hgh = norm_hgh[:trn_len]
        trn_low = norm_low[:trn_len]
        trn_ema = norm_ema[:trn_len]
        trn_hah = norm_hah[:trn_len]
        trn_hal = norm_hal[:trn_len]
        trn_macd = norm_macd[:trn_len]
        trn_rsi = norm_rsi[:trn_len]
        trn_bb = norm_bb[:trn_len]
        trn_obv = norm_obv[:trn_len]
        
        trn_set = np.hstack([trn_data,trn_opn,trn_hgh,trn_low,trn_ema,trn_hah,trn_hal,trn_macd,trn_rsi,trn_bb,trn_obv])
        trn_yset = norm_tp_blks[:trn_len]
        
        for i in range(seq_len,trn_len-pred_len):
            self.x_encoder.append(trn_set[i-seq_len:i,:])
            self.x_decoder.append(trn_data[i-1:i+pred_len-1])
            self.y_target.append(trn_data[i:i+pred_len])
            self.y_signal.append(trn_yset[i,:])
        self.x_encoder = np.array(self.x_encoder) # 700,365,12
        self.nums_feat = self.x_encoder.shape[-1]
        self.x_decoder = np.array(self.x_decoder) # 700,40,1
        self.y_target = np.array(self.y_target) # 700,40,1
        self.y_signal = np.array(self.y_signal) # 700,6
        
        tst_data = norm_data[trn_len-seq_len:]
        tst_opn = norm_opn[trn_len-seq_len:]
        tst_hgh = norm_hgh[trn_len-seq_len:]
        tst_low = norm_low[trn_len-seq_len:]
        tst_ema = norm_ema[trn_len-seq_len:]
        tst_hah = norm_hah[trn_len-seq_len:]
        tst_hal = norm_hal[trn_len-seq_len:]
        tst_macd = norm_macd[trn_len-seq_len:]
        tst_rsi = norm_rsi[trn_len-seq_len:]
        tst_bb = norm_bb[trn_len-seq_len:]
        tst_obv = norm_obv[trn_len-seq_len:]

        tst_set = np.hstack([tst_data,tst_opn,tst_hgh,tst_low,tst_ema,tst_hah,tst_hal,tst_macd,tst_rsi,tst_bb,tst_obv])
        tst_yset = norm_tp_blks[trn_len-seq_len:]
        for i in range(seq_len,len(tst_data)-pred_len):
            self.x_tst.append(tst_set[i-seq_len:i,:])
            self.y_tst.append(tst_data[i:i+pred_len])
            self.y_tsi.append(tst_yset[i,:])
        self.x_tst = np.array(self.x_tst) # 122,365,12
        self.y_tst = np.array(self.y_tst)
        self.y_tsi = np.array(self.y_tsi)
    def build(self, seed = 0, embed_dim=128, ff_dim = 256, head_base = 4, dropout_rate = 0.1, 
              embed_sig = 32, head_sig = 2, ff_sig = 64,
              base_lr = 1e-4, clipnorm = 1.0, batch_size=32, epochs=50,warmup_epoch=10):
        tf.keras.utils.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.head_base = head_base
        self.head_sig = head_sig
        self.embed_dim = embed_dim
        self.embed_sig = embed_sig
        self.ff_dim = ff_dim
        self.ff_sig = ff_sig
        self.dropout_rate = dropout_rate
        self.base_lr = base_lr
        self.clipnorm = clipnorm
       
        K = self.y_signal.shape[-1]
        
        self.base_idx = [0,1,2,3,4]; self.signal_idx = [5,6,7,8,9,10]
        inputs = Input(shape=(self.seq_len,self.nums_feat)) #none, seq_len,features
        x_base =Lambda(lambda t: tf.gather(t,self.base_idx,axis=-1),name='slice_base',output_shape=(self.seq_len, len(self.base_idx)))(inputs)
        x_signal =Lambda(lambda t: tf.gather(t,self.signal_idx,axis=-1),name='slice_signal',output_shape=(self.seq_len, len(self.signal_idx)))(inputs)
        #inputs = Lambda(lambda t: debug_wrapper(t, "model_inputs"))(inputs)
     
        # ============================target head===========================
        x = Dense(embed_dim,name='base_embed')(x_base) # none, seq_len,embed_dim
        #x = Lambda(lambda t: debug_wrapper(t, "dense_proj_out"))(x)
        
        # Positional encoding
        pos = tf.range(start=0,limit=self.seq_len,delta=1)
        pos_base_encode = Embedding(input_dim=self.seq_len,output_dim=embed_dim,name='pos_base_emb')(pos)
        x = x+pos_base_encode 
        #x = Lambda(lambda t: debug_wrapper(t, "pos_encoding_added"))(x)
        
        # Multi-head self-attention+ Causal mask
        attn_base = MultiHeadAttention(num_heads=head_base,key_dim=embed_dim,dropout=dropout_rate,name='mha_base')(x,x,use_causal_mask=True)
        x = Add()([x,attn_base])
        x = LayerNormalization(name='mha_base_ln')(x) 
        #x = Lambda(lambda t: debug_wrapper(t, "after_mha_norm"))(x)
        
        # Feed-forward network
        ff_output = Sequential([Dense(ff_dim,activation='relu'),Dropout(dropout_rate),Dense(embed_dim)],name='ffn_base')(x)
        x = Add()([x,ff_output])
        x = LayerNormalization(name='ffn_base_ln')(x)
        #x = Lambda(lambda t: debug_wrapper(t, "after_ffn_norm"))(x)
        last_token = Lambda(lambda t: t[:, -1, :],name='base_last')(x)  # last day of seq len
        
        pred_vec   = Dense(self.pred_len,name='y_vec')(last_token)  # (batch, 40)
        y_hat    = Reshape((self.pred_len, 1),name='y_hat')(pred_vec)  # (batch, 40, 1)
        
        ## ============================signal head===========================
        x_sig = Dense(embed_sig,name='sig_embed')(x_signal)
        
        pos_sig_encode = Embedding(input_dim=self.seq_len,output_dim=embed_sig,name='pos_sig_emb')(pos) #reuse position
        x_sig = x_sig+pos_sig_encode
        
        attn_sig = MultiHeadAttention(num_heads=head_sig,key_dim=embed_sig,dropout=dropout_rate,name='mha_sig')(x_sig,x_sig,use_causal_mask=True)
        x_sig = Add()([x_sig,attn_sig])
        x_sig = LayerNormalization(name='mha_sig_ln')(x_sig)
        
        ff_sig = Sequential([Dense(ff_sig,activation='relu'),Dropout(dropout_rate),Dense(embed_sig)],name='ffn_sig')(x_sig)
        x_sig = Add()([x_sig,ff_sig])
        x_sig = LayerNormalization(name='ff_sig_ln')(x_sig)
        
        last_sig = Lambda(lambda t: t[:, -1, :], name="sig_last")(x_sig) # batch, embed_sig
        #token_detached = Lambda(lambda t:tf.stop_grdient(t),name='last_detached')(last_token)
        fused = Concatenate(name='prob_fuse')([last_token,last_sig]) # batch, embed_sig+embed_dim
        #fused = Concatenate(name='prob_fuse')([token_detached,last_sig]) # batch, embed_sig+embed_dim
        tp_bins = Activation("softmax", name="tp_bins")(Dense(K, name="tp_logits")(fused)) #batch,6
        
        # warm up + gradient clipping 
        self.steps_per_epoch = self.x_encoder.shape[0]//batch_size
        self.lr_schedule = Warmupcosine(base_lr=base_lr, total_steps = epochs * self.steps_per_epoch, warmup_steps=warmup_epoch* self.steps_per_epoch)
        
        # model 封装
        model = Model(inputs=inputs,outputs=[tp_bins,y_hat])
        opt = optimizers.Adam(learning_rate=self.lr_schedule,clipnorm=clipnorm)
        model.compile(loss={'tp_bins':'categorical_crossentropy','y_hat':'mse'},optimizer=opt,
                      loss_weights={'tp_bins':0.3,'y_hat':1},metrics={'tp_bins':['accuracy']})
        self.model = model
    def fit(self,batch_size=32, epochs=50):
        #cb = BatchLossLogger(self.x_encoder,self.y_target,batch_size)
        #steps_per_epoch = math.ceil(self.x_encoder.shape[0]/batch_size)
        self.tot_batch = epochs*self.steps_per_epoch
        self.cb = BatchLossLogger(self.lr_schedule,self.tot_batch,self.steps_per_epoch,epochs)
        #self.cb = BatchLossLogger1(self.x_encoder,self.y_target,batch_size)
        self.history = self.model.fit(self.x_encoder,{'tp_bins':self.y_signal,'y_hat':self.y_target},
                                      validation_data=[self.x_tst[0:1],{'tp_bins':self.y_tsi[0:1],'y_hat':self.y_tst[0:1]}],
                                      epochs=epochs,batch_size=batch_size,
                                      verbose=2,callbacks=[self.cb])
    def predict(self,seed=0,path='./',i=0,ema_span=21):
        #=========================predict==================================
        x_tst = self.x_tst[i:i+1]
        tp_pred, y_pred = self.model.predict(x_tst,verbose=0) # 1X3, 1X40X1
        tp_true = self.y_tsi[i]
        tp_yield = self.dataset[('tp',self.ticker)].iloc[self.trn_len+i]
        tp_pred = tp_pred.squeeze()
        K = tp_true.shape[-1]
        #=========================yhat plot==================================
        self.y_pred = y_pred[0] # 40X1
        y0_pred = self.scaler_logcls.inverse_transform(self.y_pred).flatten() #40,
        self.y0_pred = np.exp(y0_pred)
        
        tst_dataset = self.dataset[self.trn_len+i:self.trn_len+i+self.pred_len]
        tst_dataset = tst_dataset.copy()
        tst_dataset[('Pred',self.ticker)] = self.y0_pred
        y0_true = tst_dataset['Close'].values
        self.rmse = np.sqrt(np.mean((self.y0_pred-y0_true)**2))

        EMA_21 = self.dataset['Close'].ewm(span=ema_span,adjust=False).mean() 
        self.dataset[('ema21',self.ticker)] = EMA_21
        trn_dataset = self.dataset[:self.trn_len]  
        
        label_true = ''
        label_pred = ''
        for i in range(K):
            if i==0:
                label_true +=f'<{self.tp_lvls[i]:.3f}: {tp_true[i]:.3f}  '
                label_pred +=f'<{self.tp_lvls[i]:.3f}: {tp_pred[i]:.3f}  '
            elif i==K-1:
                label_true +=f'>{self.tp_lvls[i-1]:.3f}: {tp_true[i]:.3f}  '
                label_pred +=f'>{self.tp_lvls[i-1]:.3f}: {tp_pred[i]:.3f}  '
            else:
                label_true +=f'({self.tp_lvls[i-1]:.3f},{self.tp_lvls[i]:.3f}): {tp_true[i]:.3f}  '
                label_pred +=f'({self.tp_lvls[i-1]:.3f},{self.tp_lvls[i]:.3f}): {tp_pred[i]:.3f}  '
        plt.figure(figsize=(16,6))
        plt.title(f'yield next: {tp_yield:.3f}')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(trn_dataset['Close'],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.plot(tst_dataset[['Close','Pred']],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.plot(self.dataset['ema21'],linestyle='--',marker='o',markersize=3,linewidth=1)
        plt.legend(['Train','Test Vals:'+label_true,'Pred:'+label_pred],ncol=1,loc='upper left')
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'{path}Transformer6 pred v0p95 {seed} {self.ticker} {self.seq_len}-{self.pred_len} Panel')
        #=========================Loss plot==================================
        self.yhat_loss = self.history.history['y_hat_loss']
        self.val_yhat_loss = self.history.history.get('val_y_hat_loss',None)
        self.tps_loss = self.history.history.get('tp_bins_loss',None)
        self.val_tps_loss = self.history.history.get('val_tp_bins_loss',None)
        epochs = [i*self.steps_per_epoch for i in range(1,len(self.yhat_loss)+1)]
        #tot_batches = range(1,len(self.loss)+1)
        fig,axes = plt.subplots(2,1,figsize=(16,12))
        
        #axes[0].plot(tot_batches, self.loss,marker='o',linestyle='--',markersize=3,linewidth=1,label='trn_cls_loss')
        axes[0].plot(epochs, self.yhat_loss,marker='o',linestyle='--',markersize=3,linewidth=1,label='trn_cls_loss')
        axes[0].plot(epochs, self.val_yhat_loss,marker='o',linestyle='--',markersize=3,linewidth=1,label='tst_cls_loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Close Loss')
        axes[0].legend()
        axes[0].grid(True)
    
        axes[1].plot(epochs, self.tps_loss,marker='o',linestyle='--',markersize=3,linewidth=1,label='trn_tps_loss')
        axes[1].plot(epochs, self.val_tps_loss,marker='o',linestyle='--',markersize=3,linewidth=1,label='tst_tps_loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Tps Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'{path}Loss_plot transformer6 v0p95 {seed} {self.ticker} {self.seq_len} {self.pred_len}')
        #=========================TP Bins plot==================================
        x = np.concatenate([self.x_encoder,self.x_tst[:i+1]])
        tp_preds, y_preds = self.model.predict(x,verbose=0)
        tp_trues = np.concatenate([self.y_signal,self.y_tsi[:i+1]])
        
        fig,axes = plt.subplots(K,1,figsize=(16,4*K))
        for i in range(K):
            if i==0:
                lbl = f'<{self.tp_lvls[i]:.3f}'
            elif i==K-1:
                lbl = f'>{self.tp_lvls[i-1]:.3f}'
            else:
                lbl = f'({self.tp_lvls[i-1]:.3f},{self.tp_lvls[i]:.3f})'
            axes[i].set_xlabel('Date', fontsize=18)
            axes[i].set_ylabel('40 Day TP yield probability %', fontsize=18)
            axes[i].plot(100*tp_trues[:,i],linestyle='--',marker='o',markersize=3,linewidth=1)
            axes[i].plot(100*tp_preds[:,i],linestyle='--',marker='o',markersize=3,linewidth=1)
            axes[i].legend(['True '+lbl,'Pred '+lbl])
            axes[i].grid(True)
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(f'{path}Tps_loss transformer6 v0p95 {seed} {self.ticker} {self.seq_len} {self.pred_len}')
    def save(self,seed=0, path='./'):
        self.model.save(f'{path}Transformer6_v0p95_{self.ticker}_{self.seq_len}_{self.pred_len}.keras')
        
        with open(f'{path}Transformer6_history_v0p95_{self.ticker}_{self.seq_len}_{self.pred_len}.pkl','wb') as f:
            pickle.dump(self.history,f)
    def load(self,seed=0, path='./'):
        self.model = load_model(f'{path}Transformer6_v0p95_{self.ticker}_{self.seq_len}_{self.pred_len}.keras',compile=False,safe_mode=False)
        
        with open(f'{path}Transformer6_history_v0p95_{self.ticker}_{self.seq_len}_{self.pred_len}.pkl','rb') as f:
            self.history = pickle.load(f)
    def compute_importance(self, i=0, seed=0, scoring='neg_mean_squared_error'): 
        feature_cols = ['close', 'open','high','low','ema', 'ha_hgh','ha_low','macd_scl', 'rsi_zone','bb_squeeze', 'obv_slope']
        rng = np.random.RandomState(seed)
        results = []
        for j,feat_name in enumerate(feature_cols):
            deltas = []
            mses = []
            for ii in range(i,min(i+5,self.x_tst.shape[0])):  
                tps0_pred,_ = self.model.predict(self.x_tst[ii:ii+1],verbose=0)
                base_mse = mean_squared_error(self.y_tsi[ii],tps0_pred[0])
                
                x_perturb = self.x_tst[ii].copy()
                x_perturb[:,j] = rng.permutation(x_perturb[:, j])
                tps_pred,_ = self.model.predict(x_perturb.reshape(1,self.seq_len,x_perturb.shape[-1]),verbose=0)
                deltas.append(mean_squared_error(self.y_tsi[ii],tps_pred[0])-base_mse)
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