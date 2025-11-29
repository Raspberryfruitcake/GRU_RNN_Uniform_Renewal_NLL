from __future__ import annotations
from typing import Any, Dict

import json
import torch


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
