# Version Comparison
|  Version | Attention Type | Key Features | Architecture Highlights |
|----------|---------------|--------------|------------------------|
| **v0**   | Dot-product              | Price delta features, Post-processing attention, Simple scheduled sampling           | Basic attention as output layer |
| **v1**   | Dot-product              | Log price features, Pre-processing attention, Linear scheduled sampling decay        | Attention for input projection |
| **v2**   | Dot-product              | Gate mechanism introduction, Real-time attention, Fixed seed (42)                    | **Major breakthrough**: Gated attention fusion |
| **v2.5** | Dot-product              | Configurable seed parameter, Model evaluation metrics, Enhanced training monitoring  | Improved usability and evaluation |
| **v3**   | Bahdanau (Additive)      | Classical additive attention, LayerNormalization, GELU activation                    | **Research exploration**: Traditional seq2seq attention |
| **v4**   | Multi-head (Dot-product) | Multi-head attention, Q-K-V decomposition, Parallel attention heads                  | **Modern architecture**: Scaled dot-product attention |
| **v4.5** | Multi-head (Dot-product) | Configurable seed management, Restored evaluation pipeline, Training stability fixes | Production-ready multi-head attention |

#Key Improvements by Version
### v0 → v1: Foundation Improvements
Switched from price deltas to log prices
Attention used to preprocess inputs before feeding to LSTM
Improved teacher forcing schedule

### v1 → v2: Breakthrough Innovation
Gate mechanism: Learned fusion of attention and hidden states
Significantly improved model capacity

### v2+:
Attention computed at each decoding timestep for dynamic context

### v2 → v2.5: Practical Improvements
Parameterized random seed for reproducible experiments
Added comprehensive evaluation metrics (val_mean, val_ema, slope, score)
Enhanced model monitoring and selection

### v2.5 → v3: Research Exploration
Classical Bahdanau attention mechanism
Added LayerNormalization for training stability
GELU activation for better gradient flow
Smaller default parameters (batch_size=32, n_units=64)

### v3 → v4: Modern Architecture
Multi-head attention (default: 4 heads)
Transformer-style Q-K-V decomposition
Parallel attention computation
Scaled dot-product attention

### v4 → v4.5: Production Readiness
Restored flexible seed management
Fixed training stability issues
Added a comprehensive evaluation pipeline
Better model monitoring and visualization
