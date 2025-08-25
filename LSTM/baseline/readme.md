## Version Comparison
| Version  | Architecture | Data Processing | Scheduled Sampling | Key Features |
|----------|-------------|-----------------|-------------------|--------------|
| **v0**    | Keras Functional API                        | Close prices (MinMaxScaler), Pure teacher forcing, No decoder iteration during training |
| **v0.5**  | Custom training loop with decoder iteration | Close prices (MinMaxScaler), Two-stage scheduled sampling (50% threshold)               |
| **v0.6**  | Custom training loop with decoder iteration | **Price deltas** (StandardScaler), Two-stage scheduled sampling (50% threshold)         | **Failed experiment**: Delta features proved ineffective |
| **v1**    | Custom training loop with decoder iteration | **Log prices** (StandardScaler), Enhanced scheduled sampling (30% threshold)            | The data pipeline used for attention models |

## Key Innovations by Version
### v0 → v0.5: From Static to Iterative Training
- **Architecture Change**: From Keras Functional API (fixed graph) to custom training loops with explicit decoder iteration
- **Introduced scheduled sampling**: Gradually reduce teacher forcing during training to avoid exposure bias
- **Added iterative decoding**: Each output timestep computed in a for-loop, enabling dynamic decision making
- Added real-time loss monitoring and time estimation

### v0.5 → v0.6: Data Processing Experiment ❌
- **Experimented with price delta features** instead of absolute prices
- **Critical Finding**: Price deltas behave like white noise - no meaningful correlation with technical indicators (RSI, VIX, Volume)
- **Root Cause**: Delta features lose the price level context that technical indicators depend on
- **Model couldn't learn**: Without correlation between inputs and targets, the model had no signal to learn from
- Led to the insight that preserving price level information is crucial

### v0.6 → v1: Stabilization and Modernization
- **Reverted to log prices** based on v0.6 findings
- **Log price advantage**: Log changes represent percentage changes, which better align with how financial markets behave
- **Improved scheduled sampling**: 30% threshold vs 50% - faster transition to model predictions
- **Enhanced model selection with recur_loss tracking**: Saves best weights based on inference-time performance, not training loss
- **Critical improvement**: Best weight selection is activated in the final 20% of the training to select the optimal weights. 
- **Alignment with attention models**: Identical data preprocessing pipeline enables fair comparison
- Better training stability and convergence
