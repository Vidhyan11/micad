"""MM driver: train the two CBMs + image-only baselines -> Experiment 1 table.

    !python /kaggle/working/micad/scripts/train.py --encoder dermlip

Model A: derm7pt, dermoscopic concepts (REAL GT), 5-class dx.
Model B: Fitzpatrick17k, clinical concepts (foundation PSEUDO), 3-class dx.

Outputs (artifacts/):
    checkpoints/<name>.pt            trained heads
    results/exp1_diagnosis.csv       diagnosis + concept accuracy table
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import fbc.config as C  # noqa: E402
from fbc.eval import metrics as M  # noqa: E402
from fbc.train import train_cbm as T  # noqa: E402
from fbc.train.data import assemble  # noqa: E402
from fbc.utils import io  # noqa: E402

SPECS = {
    "A": {"ds": "derm7pt", "use_pseudo": False, "label": "ModelA-dermoscopic(GT)"},
    "B": {"ds": "fitzpatrick17k", "use_pseudo": True, "label": "ModelB-clinical(pseudo)"},
}


def _eval_cbm(model, data, device):
    Xte, Cte, Mte, yte = data.subset("test")
    with torch.no_grad():
        _, cprobs, dxlogits = model(torch.as_tensor(Xte).to(device))
    dx = M.dx_metrics(yte, dxlogits.cpu().numpy())
    con = M.concept_metrics(Cte, Mte, cprobs.cpu().numpy(), data.keys)
    return dx, con


def _eval_imageonly(model, data, device):
    Xte, _, _, yte = data.subset("test")
    with torch.no_grad():
        dxlogits = model(torch.as_tensor(Xte).to(device))
    return M.dx_metrics(yte, dxlogits.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dermlip")
    ap.add_argument("--models", nargs="*", default=["A", "B"])
    ap.add_argument("--joint", action="store_true", help="also train joint-mode CBM (ablation)")
    args = ap.parse_args()

    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = replace(C.DEFAULT_TRAIN, encoder=args.encoder, device=device)
    print(f"device={device} encoder={args.encoder}")

    rows = []
    for key in args.models:
        spec = SPECS[key]
        ds = spec["ds"]
        print(f"\n===== Model {key}: {spec['label']}  ({ds}) =====")
        data = assemble(ds, args.encoder, use_pseudo=spec["use_pseudo"])
        ntr = (data.split == "train").sum()
        nte = (data.split == "test").sum()
        print(f"  concepts={data.keys}")
        print(f"  classes={data.n_classes} | train={ntr} test={nte}")

        # baseline: image-only
        io_model = T.train_image_only(data, cfg, device)
        io_dx = _eval_imageonly(io_model, data, device)
        torch.save(io_model.state_dict(), io.ckpt_path(f"imageonly_{ds}_{args.encoder}"))
        print(f"  [image-only] dx: bal_acc={io_dx['bal_acc']:.3f} "
              f"f1={io_dx['f1_macro']:.3f} auroc={io_dx['auroc']:.3f}")
        rows.append({"model": key, "variant": "image-only", "dataset": ds,
                     **{f"dx_{k}": v for k, v in io_dx.items()}, "concept_mean_auroc": np.nan})

        # our CBM: sequential (+ optional joint)
        modes = ["sequential"] + (["joint"] if args.joint else [])
        for mode in modes:
            model = T.train_cbm(data, cfg, device, mode=mode)
            dx, con = _eval_cbm(model, data, device)
            torch.save(model.state_dict(), io.ckpt_path(f"cbm_{ds}_{args.encoder}_{mode}"))
            gt_note = "GT" if not spec["use_pseudo"] else "pseudo-fit"
            print(f"  [CBM-{mode}] dx: bal_acc={dx['bal_acc']:.3f} f1={dx['f1_macro']:.3f} "
                  f"auroc={dx['auroc']:.3f} | concept mean AUROC({gt_note})={con['mean_auroc']:.3f}")
            print(f"             per-concept AUROC: "
                  f"{{{', '.join(f'{k}:{v:.2f}' for k,v in con['per_concept'].items())}}}")
            rows.append({"model": key, "variant": f"CBM-{mode}", "dataset": ds,
                         **{f"dx_{k}": v for k, v in dx.items()},
                         "concept_mean_auroc": con["mean_auroc"]})

    tbl = pd.DataFrame(rows)
    out = io.result_path("exp1_diagnosis")
    tbl.to_csv(out, index=False)
    print(f"\n=== Experiment 1 table ===\n{tbl.to_string(index=False)}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
