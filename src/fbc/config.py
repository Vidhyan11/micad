"""Central configuration: environment detection, paths, dataset registry, and
default hyperparameters. Everything path-related flows through here so notebooks
stay identical across Kaggle and local runs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Environment detection
# --------------------------------------------------------------------------- #
ON_KAGGLE = Path("/kaggle").exists()

if ON_KAGGLE:
    INPUT_ROOT = Path("/kaggle/input")
    # /kaggle/working persists as notebook output and can be chained forward.
    ARTIFACT_ROOT = Path("/kaggle/working/artifacts")
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    INPUT_ROOT = REPO_ROOT / "data"          # local: drop datasets here
    ARTIFACT_ROOT = REPO_ROOT / "artifacts"

# Sub-locations under ARTIFACT_ROOT
EMB_DIR = ARTIFACT_ROOT / "embeddings"       # cached frozen-encoder embeddings
CKPT_DIR = ARTIFACT_ROOT / "checkpoints"     # trained MLP heads
RESULT_DIR = ARTIFACT_ROOT / "results"       # CSV/LaTeX tables, per-case records
FIG_DIR = ARTIFACT_ROOT / "figures"


def ensure_dirs() -> None:
    """Create all artifact sub-directories (idempotent)."""
    for d in (EMB_DIR, CKPT_DIR, RESULT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset registry — maps a logical name to its Kaggle mount directory.
# On Kaggle the mount dir is the dataset's *name* (slug minus the owner).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    name: str                # logical name used throughout the code
    kaggle_slug: str         # owner/name — for docs & kaggle CLI
    mount_dirname: str       # directory under INPUT_ROOT
    role: str                # primary | concept_validation | fairness

    @property
    def path(self) -> Path:
        return INPUT_ROOT / self.mount_dirname


DATASETS: dict[str, DatasetSpec] = {
    "derm7pt": DatasetSpec(
        name="derm7pt",
        kaggle_slug="menakamohanakumar/derm7pt",
        mount_dirname="derm7pt",
        role="primary",
    ),
    "ph2": DatasetSpec(
        name="ph2",
        kaggle_slug="jamesgoydos/melanoma-skin-lesion-id-ph2-data",
        mount_dirname="melanoma-skin-lesion-id-ph2-data",
        role="concept_validation",
    ),
    "fitzpatrick17k": DatasetSpec(
        name="fitzpatrick17k",
        kaggle_slug="mobaswiralfarabi/fitzpatrick17k",
        mount_dirname="fitzpatrick17k",
        role="fairness",
    ),
    "pad_ufes_20": DatasetSpec(
        name="pad_ufes_20",
        kaggle_slug="orvile/pad-ufes-20",
        mount_dirname="pad-ufes-20",
        role="fairness",
    ),
}


# --------------------------------------------------------------------------- #
# Encoder registry — frozen foundation encoders. HF/torch-hub ids resolved in
# encoders/*.py. Internet is ON in our Kaggle sessions, so weights download live.
# --------------------------------------------------------------------------- #
ENCODERS: dict[str, dict] = {
    # primary derm CLIP encoders
    "dermlip": {"hf_id": "redlessone/DermLIP_ViT-B-16", "kind": "open_clip"},
    "monet":   {"hf_id": "suinleelab/monet", "kind": "clip"},
    # general fallback encoder
    "dinov2":  {"hf_id": "facebook/dinov2-base", "kind": "timm_or_hf"},
}
DEFAULT_ENCODER = "dermlip"


# --------------------------------------------------------------------------- #
# Fitzpatrick grouping for the fairness audit
# --------------------------------------------------------------------------- #
FST_GROUPS: dict[str, tuple[int, ...]] = {
    "I-II": (1, 2),
    "III-IV": (3, 4),
    "V-VI": (5, 6),
}


# --------------------------------------------------------------------------- #
# Default hyperparameters (override per-run via experiments/configs/*.yaml)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    seed: int = 1337
    encoder: str = DEFAULT_ENCODER
    concept_hidden: int = 256
    diagnosis_hidden: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs_concept: int = 100
    epochs_diagnosis: int = 100
    training_mode: str = "sequential"        # sequential | joint
    concept_loss_weight: float = 1.0         # lambda for joint mode
    faithfulness_reg: float = 0.0            # optional I_c<->E_c alignment reg
    device: str = "cuda"


DEFAULT_TRAIN = TrainConfig()
