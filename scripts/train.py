"""MM driver: train the two CBMs + image-only baselines -> Experiment 1 table.

    !python /kaggle/working/micad/scripts/train.py --encoder dermlip

Model A: derm7pt, dermoscopic concepts (REAL GT). Model B: Fitzpatrick17k, clinical
concepts (foundation PSEUDO). Runs over multiple seeds and reports mean+/-std.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import fbc.config as C  # noqa: E402
from fbc.eval import metrics as M  # noqa: E402
from fbc.eval.bootstrap import fmt_ms, mean_std  # noqa: E402
from fbc.train import train_cbm as T  # noqa: E402
from fbc.train.data import assemble  # noqa: E402
from fbc.utils import io  # noqa: E402

SPECS = {
    "A": {"ds": "derm7pt", "use_pseudo": False, "label": "ModelA-dermoscopic(GT)",
          "binary_positive": "MEL"},
    "B": {"ds": "fitzpatrick17k", "use_pseudo": True, "label": "ModelB-clinical(pseudo)",
          "binary_positive": "malignant"},
}
METRICS = ["dx_bal_acc", "dx_auroc", "dx_f1_macro", "concept_mean_auroc"]


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


def run_model_seed(spec, args, cfg, device):
    """One seed: returns {variant: {dx_bal_acc, dx_auroc, dx_f1_macro, concept_mean_auroc}}."""
    binpos = None if args.multiclass else spec["binary_positive"]
    data = assemble(spec["ds"], args.encoder, use_pseudo=spec["use_pseudo"],
                    binary_positive=binpos)
    out = {}

    def _pref(dxm, concept=np.nan):
        d = {f"dx_{k}": v for k, v in dxm.items()}
        d["concept_mean_auroc"] = concept
        return d

    io_model = T.train_image_only(data, cfg, device)
    out["image-only"] = _pref(_eval_imageonly(io_model, data, device))

    if not spec["use_pseudo"]:                       # oracle needs real GT concepts
        _, oracle_logits = T.train_dx_from_concepts(data, cfg, device, use_gt=True)
        out["CBM-oracle"] = _pref(M.dx_metrics(data.subset("test")[3], oracle_logits))

    for mode in ["sequential"] + (["joint"] if args.joint else []):
        model = T.train_cbm(data, cfg, device, mode=mode)
        dx, con = _eval_cbm(model, data, device)
        out[f"CBM-{mode}"] = _pref(dx, con["mean_auroc"])
    return out, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dermlip")
    ap.add_argument("--models", nargs="*", default=["A", "B"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--joint", action="store_true", help="also train joint-mode CBM (ablation)")
    ap.add_argument("--multiclass", action="store_true",
                    help="use full multiclass dx instead of binary detection")
    args = ap.parse_args()

    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = replace(C.DEFAULT_TRAIN, encoder=args.encoder, device=device)
    task = "multiclass" if args.multiclass else "binary detection"
    print(f"device={device} encoder={args.encoder} seeds={args.seeds} task={task}")

    rows = []
    for key in args.models:
        spec = SPECS[key]
        print(f"\n===== Model {key}: {spec['label']} ({spec['ds']}) =====")
        per = defaultdict(lambda: defaultdict(list))   # variant -> metric -> [per seed]
        for seed in args.seeds:
            out, data = run_model_seed(spec, args, replace(base, seed=seed), device)
            for variant, md in out.items():
                for m in METRICS:
                    per[variant][m].append(md.get(m, np.nan))
            seq = out.get("CBM-sequential", {})
            print(f"  seed {seed}: image-only AUROC={out['image-only']['dx_auroc']:.3f} "
                  f"| CBM AUROC={seq.get('dx_auroc', float('nan')):.3f} "
                  f"bal_acc={seq.get('dx_bal_acc', float('nan')):.3f}")

        for variant in per:
            row = {"model": key, "variant": variant, "dataset": spec["ds"],
                   "seeds": len(args.seeds)}
            for m in METRICS:
                mu, sd = mean_std(per[variant][m])
                row[m] = mu; row[f"{m}_std"] = sd
            rows.append(row)
            print(f"  [{variant:16s}] dx AUROC={fmt_ms(*mean_std(per[variant]['dx_auroc']))} "
                  f"bal_acc={fmt_ms(*mean_std(per[variant]['dx_bal_acc']))} "
                  f"concept={fmt_ms(*mean_std(per[variant]['concept_mean_auroc']))}")

    tbl = pd.DataFrame(rows)
    out_path = io.result_path("exp1_diagnosis")
    tbl.to_csv(out_path, index=False)
    print(f"\n=== Experiment 1 (mean+/-std over {len(args.seeds)} seeds) ===")
    print(tbl.to_string(index=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
