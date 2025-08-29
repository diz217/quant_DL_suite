# Quantitative Deep Learning Suite

## Overview
This repository hosts experimental deep learning architectures for financial time-series forecasting. The models combine LSTM and Transformer backbones with attention mechanisms and engineered technical indicators.
Beyond the prediction of pricec and probabilities, the framework integrates **backtested trading strategies** that evaluate:

- **Risk-Adjusted Performance:** Sharpe ratio analysis

- **Downside Risk Assessment:** Maximum drawdown calculation  

- **Return Analysis:** Total yield (cumulative return) across different time horizons

These metrics allow not only inspection of model fit (losses, predictions, attention maps) but also assessment of practical trading performance over rolling windows.

## Quick Start
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements
```
### Run an example (choose a model family):

Transformer:
```
    python transformer/TST_run_.py
```   
LSTM:
```
    python LSTM/LSTM_run_.py
```
To change the stock ticker (e.g. AAPL, MSFT, TSLA), edit the corresponding `.py` file and modify the ticker symbol inside the code. The default ticker is 'AMD'.

Results over a forecast horizon include plots of predicted prices, profit-taking returns, and quantile-binned profit-taking probabilities. Training losses and attention weights are also plotted for quality control.

One can compare the results from:
- `TST_v0p95.py` (self-attention MHA, take-profit returns)  
- `LSTM_attention_v2p5.py` (LSTM simple dot-product attention) 
- `LSTM_v1.py` (pure LSTM)

Parameters can be adjusted directly in the scripts (about 30-40 parameters), such as:
- `pred_len` prediction horizon (e.g. 40 days)  
- `seq_len` training data sequence length  
- `seed` random seed    

For detailed explanations of each version and design change, see the README files inside the `LSTM/` and `transformer/` subfolders.

### Backtesting (Sharpe, Yield, Drawdown):
```
    python transformer/TST_run_yr_yield.py
    python transformer/cal_sharpe.py
```
To backtest the trading strategy based on the transformer model, run the two scripts in sequence. The first script (```TST_run_yr_yield.py```) generate trading signals from price predictions and take-profit recommendations, including: 
- positions (long/short)
- target prices
- no actions if the risk-reward ratio is unfavorable.
 
Based on the actual price actions in the following days, the algorithm updates decisions across different scenarios:
- stop-outs
- missed opportunities
- realized gains
- unmet targets.

The first script runs Training, predicting, strategy devising and feedback run sequentially in a rolling window across the full year. 

The second script (```cal_sharpe.py```) aggregates performance by computing cumulative yield and statistical metrics such as Sharpe ratio and peak-to-trough drawdown.

The script contains very important superparameters: 
- `start`,`end` training data window
- `Stop_loss` hard stop loss threshold
- `Pred_len` prediction horizon (e.g. 40 days)  

The trading algorithm parameters directly affect the performance of the profit-taking strategy. Sensible prediction horizion + stop_loss combinations are crucial to a large profit and stable performance. 

## Results

**Backtesting**

Sharpe ratio, total yield, maximum drawback from running 'cal_sharpe.py'. Representative results include two negatively-correlated stocks over the same time period:

2024-2025:
!['TSLA' Backtested Per-Trade Yields with transformer-based trading strategy](results/TSLA_yield_sharpe_40_2019-08-20_v1.png)
!['GM' Backtested Per-Trade Yields with transformer-based trading strategy](results/GM_yield_sharpe_40_2019-08-20_v1.png)

2018-2019:
!['TSLA' Backtested Per-Trade Yields with transformer-based trading strategy](results/TSLA_yield_sharpe_40_2013-08-20_v1.png)
!['GM' Backtested Per-Trade Yields with transformer-based trading strategy](results/GM_yield_sharpe_40_2013-08-20_v1.png)

**Transformer (TST_v0p95 & TST_v0p8)** 

Predictive outputs from running `TST_run_.py`:
![Transformer windowed Predicted Prices](results/Price_prediction_TST_v0p8.png)  
![Transformer windowed Profit-taking returns](results/TP_return_TST_v0p8.png)  
![Transformer Training Loss](results/Loss_training_TST_v0p8.png)  

## Directory 
```
├── LSTM/
|   ├── LSTM_run_.py
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
|   ├── TST_v0p81.py
│   ├── TST_v0p95.py
|   ├── TST_run_.py
|   ├── TST_run_yr_yield.py
|   ├── cal_sharpe.py
│   └── readme.md                   
│
├── LICENSE
├── README.md                     
└── requirements.txt
```

## Documentation
- [LSTM/attention/readme.md](./LSTM/attention/readme.md) — details for attention-based LSTM  
- [LSTM/baseline/readme.md](./LSTM/baseline/readme.md) — details for baseline LSTM  
- [transformer/readme.md](./transformer/readme.md) — details for Transformer models           

## License
This project is released under the MIT License. See [LICENSE](./LICENSE) for details
