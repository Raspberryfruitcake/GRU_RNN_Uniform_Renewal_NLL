# Uniform Renewal Process – RNN/GRU Experiments

This repository contains the code used to train simple RNN and GRU models on the uniform renewal process and evaluate them via true and empirical KL divergence.  
It is intended for internal use so collaborators can quickly review and reproduce results.

---

## Repository Structure

renewal_rnns/
│
├── processes.py # Renewal FSM, stationary distribution, true conditional P(y|context)
├── data.py # Dataset loader and empirical P(C), P(y|C)
├── models.py # GRUModel and VanillaRNNModel definitions
├── training.py # Generic training loop, Adam, NLL, seeds, grad clipping
├── evaluation.py # Functions for true KL and empirical KL
├── analysis.py # Optional PCA helpers
│
└── experiments/
├── base_experiment.py # Shared experiment logic
├── uniform_gru.py # GRU-specific experiment wrapper
└── uniform_rnn.py # RNN-specific experiment wrapper

scripts/
├── run_gru_sweep.py
└── run_rnn_sweep.py

configs/
├── gru_default.json
└── rnn_default.json

results/

The structure is minimal: every component is separated so it can be inspected or modified independently.

---

## What the Experiments Produce

### **run_gru_sweep.py**
Runs GRU models over multiple training slices.  
For each slice it computes:

- **True KL divergence**  
  Computed analytically from the renewal-process FSM.

- **Empirical KL divergence**  
  Computed from empirical P(C) and P(y|C) measured on the same slice.

Output is written to a JSON file:

results/gru_results.json

Each entry in the JSON contains:

{
"range": "start-end",
"true_KL_divergence": ...,
"empirical_KL_divergence": ...
}


### **run_rnn_sweep.py**
Same as the GRU sweep, but using the vanilla RNN model.  
Outputs:

results/rnn_results.json


The two outputs are directly comparable: same slices, same conditions, different architectures.

---

## What a Single Experiment Does Internally

When running either RNN or GRU:

1. **Read a slice** of the binary sequence  
   (`training_start`, `training_length`).

2. **Create sliding-window dataset**  
   Context length = `training_context_length`.

3. **Train the model**  
   - NLL (cross-entropy)  
   - Adam  
   - Weight decay  
   - Gradient clipping  
   - Deterministic seeds  

4. **Evaluate on all possible contexts of length m**  
   - Compute exact P(y|context) from FSM → **true KL**  
   - Compute empirical P(y|context) → **empirical KL**

5. **Store results in JSON**

---

## Minimal Usage

### GRU sweep
python scripts/run_gru_sweep.py --config configs/gru_default.json


### RNN sweep
python scripts/run_rnn_sweep.py --config configs/rnn_default.json


Both produce a JSON file summarizing KL performance across slices.

---

This README is intentionally simple so colleagues can navigate the repo and understand exactly what the experiment scripts produce.
