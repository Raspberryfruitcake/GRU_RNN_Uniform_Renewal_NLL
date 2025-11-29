from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .processes import compute_true_conditional_distribution


def evaluate_true_kl(
    model: nn.Module,
    context_length: int,
    T0: np.ndarray,
    T1: np.ndarray,
    pi: np.ndarray,
    device: str,
) -> float:
    """
    Compute the true KL divergence rate (in bits) between the model predictions
    and the true uniform renewal process for contexts of length `context_length`.
    """
    model.eval()
    contexts = [
        [int(bit) for bit in format(i, f"0{context_length}b")]
        for i in range(2 ** context_length)
    ]

    kl_sum = 0.0
    total_prob_mass = 0.0
    epsilon = 1e-12

    with torch.no_grad():
        for context in tqdm(contexts, desc=f"True KL (m={context_length})"):
            p_c, (p0, p1) = compute_true_conditional_distribution(context, T0, T1, pi)
            if p_c == 0.0:
                continue

            context_tensor = torch.tensor([context], dtype=torch.float32).unsqueeze(-1).to(device)
            logits = model(context_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            q0 = float(max(probs[0], epsilon))
            q1 = float(max(probs[1], epsilon))

            kl = 0.0
            if p0 > 0:
                kl += p0 * np.log(p0 / q0)
            if p1 > 0:
                kl += p1 * np.log(p1 / q1)

            kl_sum += p_c * kl
            total_prob_mass += p_c

    kl_rate_nats = kl_sum / total_prob_mass if total_prob_mass > 0 else 0.0
    return kl_rate_nats / np.log(2.0)


def evaluate_empirical_kl(
    model: nn.Module,
    context_length: int,
    P_c: np.ndarray,
    P_y_given_c: np.ndarray,
    device: str,
) -> float:
    """
    Compute empirical KL divergence rate (in bits) between the model predictions
    and empirical conditionals P_emp(y|C) estimated from data.
    """
    model.eval()
    m = context_length
    num_contexts = 2 ** m
    epsilon = 1e-12
    kl_sum = 0.0
    total_mass = 0.0

    with torch.no_grad():
        for idx in tqdm(range(num_contexts), desc=f"Empirical KL (m={m})"):
            p_c = float(P_c[idx])
            if p_c == 0.0:
                continue

            ctx_bits = [int(b) for b in format(idx, f"0{m}b")]
            context_tensor = torch.tensor([ctx_bits], dtype=torch.float32).unsqueeze(-1).to(device)
            logits = model(context_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            q0 = float(max(probs[0], epsilon))
            q1 = float(max(probs[1], epsilon))

            p0 = float(P_y_given_c[idx, 0])
            p1 = float(P_y_given_c[idx, 1])

            kl = 0.0
            if p0 > 0:
                kl += p0 * np.log(p0 / q0)
            if p1 > 0:
                kl += p1 * np.log(p1 / q1)

            kl_sum += p_c * kl
            total_mass += p_c

    kl_rate_nats = kl_sum / total_mass if total_mass > 0 else 0.0
    return kl_rate_nats / np.log(2.0)
