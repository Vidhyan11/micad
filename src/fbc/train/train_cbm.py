"""Training loops over cached embeddings. All models are small MLP heads, so we
train full-batch (or large mini-batches) on GPU — seconds to a couple minutes.

Modes:
  sequential  — train concept head, freeze, train diagnosis head on predicted
                concept probs (most faithful; default).
  joint       — train concept + diagnosis heads jointly (ablation).
  image_only  — baseline emb->dx (no concepts).
"""
from __future__ import annotations

import numpy as np
import torch

from ..models import CBM, ImageOnly
from ..utils.seed import seed_everything
from . import losses
from .data import Assembled


def _to(device, *arrs):
    return [torch.as_tensor(a).to(device) for a in arrs]


def _iterate(n, batch_size, generator):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        yield perm[i:i + batch_size]


def _fit(model, params, step_fn, val_fn, epochs, lr, weight_decay, patience=15):
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    best, best_state, wait = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        step_fn(opt)
        model.eval()
        with torch.no_grad():
            score = val_fn()
        if score > best:
            best, best_state, wait = score, {k: v.detach().clone()
                                             for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def train_image_only(data: Assembled, cfg, device="cuda") -> ImageOnly:
    seed_everything(cfg.seed)
    Xtr, _, _, ytr = data.subset("train")
    Xva, _, _, yva = data.subset("val")
    Xtr, ytr = _to(device, Xtr, ytr)
    Xva, yva = _to(device, Xva, yva)
    model = ImageOnly(Xtr.shape[1], data.n_classes, cfg.diagnosis_hidden, cfg.dropout).to(device)
    g = torch.Generator().manual_seed(cfg.seed)

    def step(opt):
        for idx in _iterate(len(Xtr), cfg.batch_size, g):
            opt.zero_grad()
            loss = losses.diagnosis_ce(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()

    def val():
        pred = model(Xva).argmax(1)
        return (pred == yva).float().mean().item()

    _fit(model, model.parameters(), step, val, cfg.epochs_diagnosis, cfg.lr, cfg.weight_decay)
    return model


def _new_cbm(data, cfg, emb_dim, device):
    return CBM(emb_dim, len(data.keys), data.n_classes,
              cfg.concept_hidden, cfg.diagnosis_hidden, cfg.dropout).to(device)


def train_cbm(data: Assembled, cfg, device="cuda", mode="sequential") -> CBM:
    seed_everything(cfg.seed)
    Xtr, Ctr, Mtr, ytr = data.subset("train")
    Xva, Cva, Mva, yva = data.subset("val")
    Xtr, Ctr, Mtr, ytr = _to(device, Xtr, Ctr, Mtr, ytr)
    Xva, Cva, Mva, yva = _to(device, Xva, Cva, Mva, yva)
    model = _new_cbm(data, cfg, Xtr.shape[1], device)
    g = torch.Generator().manual_seed(cfg.seed)

    if mode == "sequential":
        # (1) concept head
        def cstep(opt):
            for idx in _iterate(len(Xtr), cfg.batch_size, g):
                opt.zero_grad()
                clogits = model.concept_head(Xtr[idx])
                loss = losses.masked_bce(clogits, Ctr[idx], Mtr[idx])
                loss.backward(); opt.step()

        def cval():
            clogits = model.concept_head(Xva)
            return -losses.masked_bce(clogits, Cva, Mva).item()   # higher=better

        _fit(model, model.concept_head.parameters(), cstep, cval,
             cfg.epochs_concept, cfg.lr, cfg.weight_decay)

        # (2) freeze concept head, train diagnosis head on predicted probs
        for p in model.concept_head.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            Ptr = model.predict_concepts(Xtr)
            Pva = model.predict_concepts(Xva)

        def dstep(opt):
            for idx in _iterate(len(Ptr), cfg.batch_size, g):
                opt.zero_grad()
                loss = losses.diagnosis_ce(model.diagnosis_head(Ptr[idx]), ytr[idx])
                loss.backward(); opt.step()

        def dval():
            pred = model.diagnosis_head(Pva).argmax(1)
            return (pred == yva).float().mean().item()

        _fit(model, model.diagnosis_head.parameters(), dstep, dval,
             cfg.epochs_diagnosis, cfg.lr, cfg.weight_decay)

    elif mode == "joint":
        def jstep(opt):
            for idx in _iterate(len(Xtr), cfg.batch_size, g):
                opt.zero_grad()
                clogits, cprobs, dxlogits = model(Xtr[idx])
                loss = (cfg.concept_loss_weight * losses.masked_bce(clogits, Ctr[idx], Mtr[idx])
                        + losses.diagnosis_ce(dxlogits, ytr[idx]))
                loss.backward(); opt.step()

        def jval():
            _, _, dxlogits = model(Xva)
            return (dxlogits.argmax(1) == yva).float().mean().item()

        _fit(model, model.parameters(), jstep, jval,
             max(cfg.epochs_concept, cfg.epochs_diagnosis), cfg.lr, cfg.weight_decay)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return model
