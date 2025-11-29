from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any

import torch
from torch.utils.data import DataLoader

from ..data import read_tokens_slice, BinarySequenceDataset, compute_empirical_distributions
from ..training import set_seeds, train_model
from ..processes import uniform_renewal_process_fsm, stationary_distribution_uniform_renewal
from ..evaluation import evaluate_true_kl, evaluate_empirical_kl


ModelCtor = Callable[..., torch.nn.Module]


def run_one_experiment(
    *,
    model_ctor: ModelCtor,
    data_file: str,
    training_start: int,
    training_length: int,
    training_context_length: int,
    eval_context_length: int,
    N_states: int,
    hidden_size: int,
    dropout_p: float,
    num_epochs: int,
    learning_rate: float,
    device: str | None = None,
    seed: int = 42,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """
    Generic experiment runner for a renewal process + RNN/GRU model.

    Returns:
        (true_KL_bits, empirical_KL_bits) for eval_context_length.
    """
    set_seeds(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    sequence = read_tokens_slice(data_file, training_start, training_length)

    dataset = BinarySequenceDataset(sequence, training_context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model_ctor(input_size=1, hidden_size=hidden_size, dropout_p=dropout_p)
    train_model(
        model=model,
        data_loader=loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    T0, T1 = uniform_renewal_process_fsm(N_states)
    pi = stationary_distribution_uniform_renewal(N_states)

    true_kl = evaluate_true_kl(
        model=model,
        context_length=eval_context_length,
        T0=T0,
        T1=T1,
        pi=pi,
        device=device,
    )

    P_c, P_y_given_c = compute_empirical_distributions(sequence, eval_context_length)
    emp_kl = evaluate_empirical_kl(
        model=model,
        context_length=eval_context_length,
        P_c=P_c,
        P_y_given_c=P_y_given_c,
        device=device,
    )

    return float(true_kl), float(emp_kl)


def run_sweep(
    *,
    model_ctor: ModelCtor,
    data_file: str,
    starts: List[int],
    training_length: int,
    training_context_length: int,
    eval_context_length: int,
    N_states: int,
    hidden_size: int,
    dropout_p: float,
    num_epochs: int,
    learning_rate: float,
    device: str | None = None,
    seed: int = 42,
    batch_size: int = 64,
) -> List[Dict[str, float]]:
    """
    Run a sweep over multiple starting offsets for the training slice.
    """
    results: List[Dict[str, float]] = []

    for s in starts:
        true_kl, emp_kl = run_one_experiment(
            model_ctor=model_ctor,
            data_file=data_file,
            training_start=s,
            training_length=training_length,
            training_context_length=training_context_length,
            eval_context_length=eval_context_length,
            N_states=N_states,
            hidden_size=hidden_size,
            dropout_p=dropout_p,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
            batch_size=batch_size,
        )
        results.append(
            {
                "range": f"{s}-{s + training_length - 1}",
                "true_KL_divergence": round(true_kl, 6),
                "empirical_KL_divergence": round(emp_kl, 6),
            }
        )

    return results
