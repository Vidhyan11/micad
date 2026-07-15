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
# --------------------------------------------------------------------------- #
# Dataset path resolution — find datasets by MARKER FILES anywhere under
# INPUT_ROOT, so it works whether Kaggle mounts as /kaggle/input/<name> or the
# nested /kaggle/input/datasets/<owner>/<name> layout we actually have.
# --------------------------------------------------------------------------- #
def _first_match(*patterns: str) -> Path | None:
    """First path matching any rglob pattern under INPUT_ROOT (sorted)."""
    if not INPUT_ROOT.exists():
        return None
    for pat in patterns:
        hits = sorted(INPUT_ROOT.rglob(pat))
        if hits:
            return hits[0]
    return None


def dataset_paths(name: str) -> dict[str, Path] | None:
    """Return the key file/dir paths for a dataset, or None if not found.

    Keys vary per dataset; loaders know which they need. Resolved from marker
    files so the exact mount nesting is irrelevant.
    """
    if name == "derm7pt":
        meta = _first_match("**/release_v0/meta/meta.csv", "**/meta/meta.csv")
        if meta is None:
            return None
        release = meta.parents[1]                 # meta/ -> release_v0/
        return {
            "meta_csv": meta,
            "images_dir": release / "images",
            "train_idx": release / "meta" / "train_indexes.csv",
            "valid_idx": release / "meta" / "valid_indexes.csv",
            "test_idx": release / "meta" / "test_indexes.csv",
        }
    if name == "pad_ufes_20":
        md = _first_match("**/pad-ufes-20/metadata.csv", "**/PAD-UFES-20/**/metadata.csv",
                          "**/metadata.csv")
        if md is None:
            return None
        return {"metadata_csv": md, "images_root": md.parent / "images"}
    if name == "ph2":
        # NOTE: needs an images-bearing mirror. Markers probed once attached.
        ann = _first_match("**/PH2_dataset.xlsx", "**/PH2_dataset.txt")
        imgs = _first_match("**/IMD*/**/*.bmp", "**/*_Dermoscopic_Image/**/*.bmp",
                            "**/PH2*Dataset*images/**/*.bmp")
        if ann is None:
            return None
        return {"annotations": ann,
                "images_root": (imgs.parents[2] if imgs is not None else None)}
    if name == "fitzpatrick17k":
        csv = _first_match("**/fitzpatrick17k*.csv", "**/fitzpatrick*.csv")
        if csv is None:
            return None
        return {"csv": csv, "images_root": csv.parent}
    raise KeyError(f"unknown dataset {name!r}")


def available_datasets() -> dict[str, bool]:
    """Quick presence check for all registered datasets."""
    return {name: dataset_paths(name) is not None for name in DATASETS}


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
