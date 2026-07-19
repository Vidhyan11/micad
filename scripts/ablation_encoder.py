"""Exp 4a: encoder ablation. Runs both CBMs across frozen encoders and reports how
encoder choice affects diagnosis, concept accuracy, AND faithfulness.

Prereq: extract embeddings for each encoder first, e.g.
    extract_embeddings.py --encoder dinov2   (and dermlip)
    make_splits.py --encoder dinov2          (and dermlip)
    pseudolabel.py --encoder dermlip         (pseudo-labels reused by all encoders)

    !python /kaggle/working/micad/scripts/ablation_encoder.py --encoders dinov2 dermlip
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import fbc.config as C  # noqa: E402
from fbc.eval import metrics as MET  # noqa: E402
from fbc.eval.bootstrap import fmt_ms, mean_std  # noqa: E402
from fbc.faithfulness import faithfulness_scores  # noqa: E402
from fbc.train import train_cbm as T  # noqa: E402
from fbc.train.data import assemble  # noqa: E402
from fbc.utils import io  # noqa: E402

METRICS = ["dx_bal_acc", "dx_auroc", "concept_auroc", "reliance"]

SPECS = {
    "A": {"ds": "derm7pt", "use_pseudo": False, "binary_positive": "MEL"},
    "B": {"ds": "fitzpatrick17k", "use_pseudo": True, "binary_positive": "malignant"},
}
PSEUDO_SOURCE = "dermlip"          # pseudo-labels always from a CLIP text tower


def run(model_key, encoder, cfg, device):
    spec = SPECS[model_key]
    data = assemble(spec["ds"], encoder, use_pseudo=spec["use_pseudo"],
                    binary_positive=spec["binary_positive"],
                    pseudo_encoder=PSEUDO_SOURCE if spec["use_pseudo"] else None)
    model = T.train_cbm(data, cfg, device, mode="sequential")
    Xtr = torch.as_tensor(data.emb[data.split_mask("train")]).to(device)
    Xte = torch.as_tensor(data.emb[data.split_mask("test")]).to(device)
    yte = data.y[data.split_mask("test")]
    Cte = data.concept_targets[data.split_mask("test")]
    Mte = data.concept_mask[data.split_mask("test")]
    with torch.no_grad():
        cp_te = model.predict_concepts(Xte)
        ref = model.predict_concepts(Xtr).mean(0)
        logits = model.diagnose_from_concepts(cp_te)
    dx = MET.dx_metrics(yte, logits.cpu().numpy())
    con = MET.concept_metrics(Cte, Mte, cp_te.cpu().numpy(), data.keys)
    fa = faithfulness_scores(cp_te, lambda c: model.diagnose_from_concepts(c), ref)
    return {"model": model_key, "encoder": encoder, "dx_bal_acc": dx["bal_acc"],
            "dx_auroc": dx["auroc"],
            "concept_auroc": con["mean_auroc"] if not spec["use_pseudo"] else float("nan"),
            "reliance": fa["reliance"], "comprehensiveness": fa["comprehensiveness"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=["dinov2", "dermlip"])
    ap.add_argument("--models", nargs="*", default=["A", "B"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={args.seeds}")

    rows = []
    for enc in args.encoders:
        for mk in args.models:
            print(f"  Model {mk} on {enc} ...")
            per = defaultdict(list)
            try:
                for seed in args.seeds:
                    cfg = replace(C.DEFAULT_TRAIN, encoder=enc, device=device, seed=seed)
                    r = run(mk, enc, cfg, device)
                    for m in METRICS:
                        per[m].append(r[m])
            except FileNotFoundError as e:
                print(f"    SKIP ({e}); extract embeddings/pseudo for {enc} first")
                continue
            row = {"model": mk, "encoder": enc, "seeds": len(args.seeds)}
            for m in METRICS:
                mu, sd = mean_std(per[m]); row[m] = mu; row[f"{m}_std"] = sd
            rows.append(row)
            print(f"    dx AUROC={fmt_ms(*mean_std(per['dx_auroc']))} "
                  f"concept={fmt_ms(*mean_std(per['concept_auroc']))} "
                  f"reliance={fmt_ms(*mean_std(per['reliance']))}")

    tbl = pd.DataFrame(rows)
    out = io.result_path("exp4a_encoder_ablation")
    tbl.to_csv(out, index=False)
    print(f"\n=== Experiment 4a: encoder ablation (mean+/-std over {len(args.seeds)} seeds) ===")
    print(tbl.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
