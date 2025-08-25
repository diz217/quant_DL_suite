# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 23:23:59 2025

@author: Ding Zhang
"""

import sys
import os

for entry in os.scandir():
    if entry.is_dir():
        sys.path.append(entry.path)


from LSTM_v0 import LSTM_tforce
from LSTM_v0p5 import LSTM_ssmpl
from LSTM_v0p6 import LSTM_ssmpl_v1
from LSTM_v1 import LSTM_ssmpl_v2
from LSTM_attention_v0 import LSTM_attention
from LSTM_attention_v1 import LSTM_attention_v1
from LSTM_attention_v2 import LSTM_attention_v2
from LSTM_attention_v2p5 import LSTM_attention_v2
from LSTM_attention_v3 import LSTM_attention_v3
from LSTM_attention_v4 import LSTM_attention_v4
from LSTM_attention_v4p5 import LSTM_attention_v4
from datetime import datetime
import time 

starttime = time.time()
#lstm = LSTM_tforce()
#lstm = LSTM_ssmpl()
#lstm = LSTM_ssmpl_v1()
#lstm = LSTM_ssmpl_v2()
#lstm = LSTM_attention()
#lstm = LSTM_attention_v1()
#lstm = LSTM_attention_v2()
#lstm = LSTM_attention_v3()
lstm = LSTM_attention_v4()
start='2020-08-20'
lstm.load_data('AMD',start)

lstm.prepare(40,365)
lstm.buildfit(32,64,50)
#lstm.buildfit(32,64,50,128)
#lstm.buildfit(32,64,4,50)
#lstm.build()
#lstm.fit()
#lstm.predict()
lstm.plot('./temp/')