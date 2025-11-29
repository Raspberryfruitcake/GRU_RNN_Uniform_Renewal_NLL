from __future__ import annotations
from typing import List, Tuple

import numpy as np


def uniform_renewal_process_fsm(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct symbol-resolved FSM transition matrices for a uniform renewal process
    with N internal states.

    Returns:
        T0, T1: each of shape (N, N), column-stochastic when summed (T0 + T1).
        These are defined such that v_new = T_b @ v, where v is a column vector
        of state probabilities.
    """
    T0 = np.zeros((N, N))
    T1 = np.zeros((N, N))

    # On a '0', transition from state i to state i+1
    for i in range(N - 1):
        T0[i, i + 1] = (N - i - 1) / (N - i)

    # On a '1', transition from any state i to state 0
    for i in range(N):
        T1[i, 0] = 1.0 / (N - i)

    # Transpose so v_new = T @ v for column vectors v
    return T0.T, T1.T


def stationary_distribution_uniform_renewal(N: int) -> np.ndarray:
    """
    Stationary distribution for the uniform renewal process with N states.

    pi[i] ∝ (N - i), properly normalised.
    """
    normalization = N * (N + 1)
    return np.array([2.0 * (N - i) / normalization for i in range(N)], dtype=float)


def compute_true_conditional_distribution(
    context: List[int],
    T0: np.ndarray,
    T1: np.ndarray,
    pi: np.ndarray,
) -> Tuple[float, Tuple[float, float]]:
    """
    Given a binary context (list of 0/1), compute:

        P(context), and P(y | context) for y in {0,1}

    using the known uniform renewal process.
    """
    v = pi.copy()
    for bit in context:
        v = (T0 @ v) if bit == 0 else (T1 @ v)

    p_c = float(np.sum(v))
    if p_c == 0.0:
        return 0.0, (0.0, 0.0)

    v0 = T0 @ v
    v1 = T1 @ v
    p_c0 = float(np.sum(v0))
    p_c1 = float(np.sum(v1))

    return p_c, (p_c0 / p_c, p_c1 / p_c)
