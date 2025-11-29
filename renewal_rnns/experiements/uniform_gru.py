from __future__ import annotations
from typing import List, Dict, Any

from .base_experiment import run_one_experiment as _run_one, run_sweep as _run_sweep
from ..models import GRUModel


def run_one_experiment(**kwargs) -> tuple[float, float]:
    """
    Run a single GRU experiment on the uniform renewal process.
    """
    return _run_one(model_ctor=GRUModel, **kwargs)


def run_sweep(**kwargs) -> List[Dict[str, float]]:
    """
    Run a sweep of GRU experiments over multiple training slices.
    """
    return _run_sweep(model_ctor=GRUModel, **kwargs)
