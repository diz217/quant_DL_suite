# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 23:21:21 2025

Created on Mon Jul 21 00:25:26 2025

@author: diz217
"""

from TST_v0 import TST_v0
from TST_v0p1 import TST_v0p1
from TST_v0p8 import TST_v0p8
from TST_v0p95 import TST_v0p95
import time 
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#tst = TST_v0()
#tst = TST_v0p1()
#tst = TST_v0p8()
tst = TST_v0p95()
start = '2020-04-20'
tst.load_data('AMD',start)
tst.prepare(40,365)
#tst.prepare(40,365,[1.0715,1.157,1.338])

tst.build(seed = 0, embed_dim=128, ff_dim = 256, head_base = 4, dropout_rate = 0.1, 
          embed_sig = 32, head_sig = 2, ff_sig = 64,
          base_lr = 1e-4, clipnorm = 1.0, batch_size=32, epochs=100,warmup_epoch=10)

tst.fit(batch_size=32, epochs=100)
  
tst.predict(seed=0,path='./',i=0,ema_span=21)

tst.compute_importance() 
