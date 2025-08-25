# Version Comparison
|  Version  |       Architecture      | Features                                  | Task                                        | Key Characteristics                                |
|-----------|-------------------------|-------------------------------------------|---------------------------------------------|----------------------------------------------------|
| **v0**    | Single-head Transformer | Basic financial features                  | Price prediction only                       | Baseline self-attention implementation             |
| **v0.1**  | Single-head Transformer | Extended technical features + Heikin Ashi | Price prediction only                       | Enhanced feature engineering                       |
| **v0.8**  | Dual-head Transformer   | Features split into base/signal groups    | Multi-task: price + TP yield with stop loss | Accurate TP yield prediction                       |
| **v0.95** | Dual-head Transformer   | Same feature groups as v0.8               | Multi-task: price + TP probability bins     | TP probability bins based on TP yield distribution |

# Technical Architecture
## Single-Task Models (v0, v0.1)
### Standard Transformer encoder 
inputs → Dense(embed_dim) → Positional Encoding → 
MultiHeadAttention(causal mask) → LayerNorm → FFN → LayerNorm → 
last_token → Dense(pred_len) → price_predictions
model learning rate: Warmup cosine decay scheduler 
## Multi-Task Models (v0.8, v0.95)
### Dual processing streams
inputs → split into [base_features, signal_features]
### Base stream (price prediction)
base_features → Transformer_base → price_head
### Signal stream (trading signals)  
signal_features → Transformer_signal → trading_head
### Final fusion
fused = Concatenate([base_last_token, signal_last_token])
trading_output = Dense(fused)

# Key Technical Innovations
v0 → v0.1: Feature Engineering
Added Heikin Ashi candles for smoother price representation
Introduced technical indicators for better market context
Improved feature normalization strategies

v0.1 → v0.8: Multi-Task Architecture
Feature separation: Distinct processing of price vs signal features
Dual-stream design: Independent Transformer heads for different tasks
Take-profit integration: Accurate continuous TP yield prediction

v0.8 → v0.95: Classification Experiment
Probability binning: Attempted to convert TP yields to risk categories
Soft labeling: Gaussian smoothing for label assignment
Result: Classification approach showed limited effectiveness

# Model Pairing Strategy
Recommended Usage
v0.8 + v0.95: Use together as complementary models
v0.8: Primary model for accurate TP yield prediction
v0.95: Secondary model for rough probability estimates
Combined approach: v0.8 provides precise signals, v0.95 adds risk context
