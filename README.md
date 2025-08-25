# Quantitative Deep Learning Suite

## Overview
This repository hosts experimental deep learning architectures for financial time-series forecasting.  
The focus is on sequential models (LSTM, Transformer) applied to stock market data with engineered technical indicators.  
The repository is organized by model family, with each directory containing multiple versioned implementations that highlight different design choices.

## Quick Start
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements
```
## Directory 
├── LSTM/
│   ├── attention/
│   │   ├── LSTM_attention_v0.py
│   │   ├── LSTM_attention_v1.py
│   │   ├── LSTM_attention_v2.py
│   │   ├── LSTM_attention_v2p5.py
│   │   ├── LSTM_attention_v3.py
│   │   ├── LSTM_attention_v4p5.py
│   │   └── readme.md                
│   └── baseline/
│       ├── LSTM_v0.py
│       ├── LSTM_v0p5.py
│       ├── LSTM_v0p6.py
│       ├── LSTM_v1.py
│       └── readme.md               
│
├── transformer/
│   ├── TST_v0.py
│   ├── TST_v0p1.py
│   ├── TST_v0p5.py
│   ├── TST_v0p8.py
│   ├── TST_v0p95.py
│   └── readme.md                   
│
├── LICENSE
├── README.md                     
└── requirements

## Documentation
- [LSTM/attention/readme.md](./LSTM/attention/readme.md) — details for attention-based LSTM  
- [LSTM/baseline/readme.md](./LSTM/baseline/readme.md) — details for baseline LSTM  
- [transformer/readme.md](./transformer/readme.md) — details for Transformer models           

## License
This project is released under the MIT License. See [LICENSE](./LICENSE) for details
