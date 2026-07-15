"""Small IO helpers for caching embeddings, checkpoints, and results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import config


def emb_path(dataset: str, encoder: str) -> Path:
    return config.EMB_DIR / f"emb_{dataset}_{encoder}.npy"


def meta_path(dataset: str) -> Path:
    return config.EMB_DIR / f"meta_{dataset}.parquet"


def ckpt_path(name: str) -> Path:
    return config.CKPT_DIR / f"{name}.pt"


def result_path(name: str, ext: str = "csv") -> Path:
    return config.RESULT_DIR / f"{name}.{ext}"


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
