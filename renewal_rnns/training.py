from __future__ import annotations
from typing import List

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn


def set_seeds(seed: int = 42) -> None:
    """
    Set RNG seeds for Python, NumPy, and PyTorch for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 5.0,
) -> List[float]:
    """
    Train a model using cross-entropy (NLL) with Adam + gradient clipping.

    Returns:
        List of average training losses per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.to(device)
    history: List[float] = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_examples = 0

        for contexts, targets in data_loader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(contexts)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            batch_size = contexts.size(0)
            epoch_loss += loss.item() * batch_size
            n_examples += batch_size

        avg_loss = epoch_loss / max(1, n_examples)
        history.append(avg_loss)

    return history
