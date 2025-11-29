from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def read_tokens_slice(filepath: str, start: int, length: int) -> List[int]:
    """
    Read a contiguous slice of binary tokens from a CSV file.

    Args:
        filepath: path to CSV file containing comma-separated 0/1.
        start: starting index (inclusive).
        length: number of tokens to read.

    Returns:
        List[int] of length `length`.
    """
    with open(filepath, "r") as f:
        data = f.read()
    tokens_str = data.strip().split(",")
    if len(tokens_str) < start + length:
        raise ValueError(
            f"The sequence in {filepath} does not have tokens [{start}:{start+length})."
        )
    return [int(tok) for tok in tokens_str[start:start + length]]


class BinarySequenceDataset(Dataset):
    """
    Sliding-window dataset over a binary sequence.

    Each example is:
        context: sequence[idx : idx + context_length]
        target: sequence[idx + context_length]
    """

    def __init__(self, sequence: List[int], context_length: int) -> None:
        super().__init__()
        self.sequence = sequence
        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, len(self.sequence) - self.context_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.sequence[idx: idx + self.context_length]
        target = self.sequence[idx + self.context_length]
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(-1)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return context_tensor, target_tensor


def compute_empirical_distributions(
    sequence: List[int],
    context_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a binary sequence, compute:

        P_c:        P(C) for all contexts C of length `context_length`
        P_y_given_c: P(y|C) for y in {0,1}

    using sliding windows of length context_length + 1.

    Returns:
        P_c: shape (2^m,)
        P_y_given_c: shape (2^m, 2)
    """
    m = context_length
    num_contexts = 2 ** m
    counts_c = np.zeros(num_contexts, dtype=np.int64)
    counts_cy = np.zeros((num_contexts, 2), dtype=np.int64)

    for i in range(len(sequence) - m):
        ctx_bits = sequence[i: i + m]
        y = sequence[i + m]
        idx = int("".join(str(b) for b in ctx_bits), 2)
        counts_c[idx] += 1
        counts_cy[idx, y] += 1

    total_windows = counts_c.sum()
    P_c = counts_c.astype(np.float64)
    if total_windows > 0:
        P_c /= total_windows

    P_y_given_c = np.zeros_like(counts_cy, dtype=np.float64)
    nonzero = counts_c > 0
    P_y_given_c[nonzero, 0] = counts_cy[nonzero, 0] / counts_c[nonzero]
    P_y_given_c[nonzero, 1] = counts_cy[nonzero, 1] / counts_c[nonzero]

    return P_c, P_y_given_c
