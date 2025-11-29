# Uniform Renewal Process – RNN/GRU Experiments

This repository contains code for training and evaluating simple RNN and GRU models on uniform renewal processes. Models are assessed using both theoretical and empirical KL divergence metrics.

---

## Overview

The codebase implements a systematic comparison of recurrent neural network architectures on renewal process prediction tasks. Each experiment trains models on different slices of a binary sequence and evaluates their performance against both ground-truth analytical distributions and empirical measurements.

---

## Repository Structure
```
renewal_rnns/
├── processes.py          # Renewal FSM, stationary distribution, true P(y|context)
├── data.py               # Dataset loader and empirical distribution estimation
├── models.py             # GRUModel and VanillaRNNModel implementations
├── training.py           # Training loop with Adam optimizer, NLL loss, gradient clipping
├── evaluation.py         # True and empirical KL divergence computation
└── analysis.py           # PCA and visualization utilities

experiments/
├── base_experiment.py    # Shared experiment infrastructure
├── uniform_gru.py        # GRU experiment configuration
└── uniform_rnn.py        # RNN experiment configuration

scripts/
├── run_gru_sweep.py      # Execute GRU experiments across training slices
└── run_rnn_sweep.py      # Execute RNN experiments across training slices

configs/
├── gru_default.json      # Default GRU hyperparameters
└── rnn_default.json      # Default RNN hyperparameters

results/                  # Output directory for experiment results
```

The modular structure allows independent inspection and modification of individual components.

---

## Experiment Output

### GRU Sweep (`run_gru_sweep.py`)

Trains GRU models across multiple training slices and computes:

- **True KL Divergence**: Analytical computation using the renewal process FSM
- **Empirical KL Divergence**: Measured from empirical distributions P(C) and P(y|C)

**Output**: `results/gru_results.json`
```json
{
  "range": "start-end",
  "true_KL_divergence": <value>,
  "empirical_KL_divergence": <value>
}
```

### RNN Sweep (`run_rnn_sweep.py`)

Identical to GRU sweep using vanilla RNN architecture.

**Output**: `results/rnn_results.json`

Results from both sweeps are directly comparable under identical experimental conditions.

---

## Experimental Pipeline

Each experiment performs the following steps:

1. **Data Preparation**: Extract training slice from binary sequence
2. **Dataset Construction**: Generate sliding-window contexts of length *m*
3. **Model Training**:
   - Loss: Negative log-likelihood (cross-entropy)
   - Optimizer: Adam with weight decay
   - Gradient clipping for stability
   - Fixed random seeds for reproducibility
4. **Evaluation**:
   - Compute P(y|context) from FSM → **True KL divergence**
   - Estimate P(y|context) empirically → **Empirical KL divergence**
5. **Results Storage**: Save metrics to JSON

---

## Usage

### Run GRU Experiments
```bash
python scripts/run_gru_sweep.py --config configs/gru_default.json
```

### Run RNN Experiments
```bash
python scripts/run_rnn_sweep.py --config configs/rnn_default.json
```

Both commands generate JSON files in `results/` summarizing model performance across training slices.

---

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Additional dependencies listed in `requirements.txt`

---

## Notes

This repository is designed for internal collaboration and result reproduction. The codebase prioritizes clarity and modularity to facilitate review and extension by team members.
