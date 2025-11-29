#!/usr/bin/env python
from __future__ import annotations
import argparse
import torch

from renewal_rnns.experiments.uniform_rnn import run_sweep
from renewal_rnns.utils import load_config, save_json


def main():
    parser = argparse.ArgumentParser(description="Run vanilla RNN sweep on uniform renewal process.")
    parser.add_argument("--config", type=str, default="configs/rnn_default.json",
                        help="Path to JSON config file.")
    parser.add_argument("--output", type=str, default="results/rnn_results.json",
                        help="Where to save result JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if "device" not in cfg or cfg["device"] is None:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    results = run_sweep(**cfg)

    payload = {
        "experiment": "VanillaRNN_uniform_renewal",
        "config": cfg,
        "results": results,
    }
    save_json(args.output, payload)
    print(f"Saved RNN results to {args.output}")


if __name__ == "__main__":
    main()
