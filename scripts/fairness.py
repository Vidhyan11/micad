"""MF3 driver: fairness-of-reasoning audit + mitigation -> Experiment 3.

    !python /kaggle/working/micad/scripts/fairness.py --encoder dermlip                 # diverse
    !python /kaggle/working/micad/scripts/fairness.py --encoder dermlip --train_groups I-II  # biased

Model B (Fitzpatrick17k). Per Fitzpatrick group {I-II, III-IV, V-VI}, reports diagnosis
accuracy AND reasoning faithfulness (reliance/comprehensiveness), before/after
group-conditional concept calibration. Multi-seed: mean+/-std over --seeds.
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
from fbc.eval import metrics as MET  # noqa: E402
from fbc.eval.bootstrap import bootstrap_diff, fmt_ms, mean_std  # noqa: E402
from fbc.faithfulness import faithfulness_scores  # noqa: E402
from fbc.fairness import apply_group_temperature, fit_group_temperature  # noqa: E402
from fbc.train import train_cbm as T  # noqa: E402
from fbc.train.data import assemble  # noqa: E402
from fbc.utils import io  # noqa: E402

GROUPS = ["I-II", "III-IV", "V-VI"]
METRICS = ["dx_bal_acc", "reliance", "comprehensiveness"]


def audit(cp_te, y_te, fst_te, predict_fn, ref):
    """Per-group dx + faithfulness (scalar metrics + seed-0 per-sample arrays)."""
    out = {}
    for g in GROUPS:
        m = fst_te == g
        if int(m.sum()) < 20:
            continue
        cg = cp_te[m]
        with torch.no_grad():
            logits = predict_fn(cg)
        dx = MET.dx_metrics(y_te[m], logits.cpu().numpy())
        fa = faithfulness_scores(cg, predict_fn, ref, return_per_sample=True)
        out[g] = {"n": int(m.sum()), "dx_bal_acc": dx["bal_acc"], "dx_auroc": dx["auroc"],
                  "reliance": fa["reliance"], "comprehensiveness": fa["comprehensiveness"],
                  "_ps": fa["_per_sample"]}
    return out


def run_seed(args, cfg, device):
    data = assemble("fitzpatrick17k", args.encoder, use_pseudo=True,
                    binary_positive="malignant")
    if args.train_groups:                       # simulate biased (light-only) training
        tr = data.split_mask("train")
        excl = tr & ~np.isin(data.fst_group, list(set(args.train_groups)))
        data.split[excl] = ""
    model = T.train_cbm(data, cfg, device, mode="sequential")
    predict_fn = lambda c: model.diagnose_from_concepts(c)

    te, va = data.split_mask("test"), data.split_mask("val")
    Xte = torch.as_tensor(data.emb[te]).to(device)
    Xva = torch.as_tensor(data.emb[va]).to(device)
    y_te, fst_te, fst_va = data.y[te], data.fst_group[te], data.fst_group[va]
    Cva = torch.as_tensor(data.concept_targets[va]).to(device)
    with torch.no_grad():
        cp_te = model.predict_concepts(Xte)
        ref = model.predict_concepts(torch.as_tensor(data.emb[data.split_mask("train")]).to(device)).mean(0)
    res_before = audit(cp_te, y_te, fst_te, predict_fn, ref)

    with torch.no_grad():
        logit_va, logit_te = model.concept_head(Xva), model.concept_head(Xte)
    temps = fit_group_temperature(logit_va, Cva, fst_va, GROUPS)
    cp_te_cal = apply_group_temperature(logit_te, fst_te, temps)
    res_after = audit(cp_te_cal, y_te, fst_te, predict_fn, ref)
    return res_before, res_after, temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dermlip")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--train_groups", nargs="*", default=None,
                    help="restrict TRAINING to these Fitzpatrick groups (e.g. I-II) to "
                         "simulate biased training; audit stays on all groups.")
    args = ap.parse_args()
    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = replace(C.DEFAULT_TRAIN, encoder=args.encoder, device=device)
    regime = "diverse(all)" if not args.train_groups else f"biased({'+'.join(args.train_groups)})"
    print(f"device={device} encoder={args.encoder} seeds={args.seeds} | regime: {regime}")

    agg = {"before": defaultdict(lambda: defaultdict(list)),
           "after": defaultdict(lambda: defaultdict(list))}
    first = {}
    for si, seed in enumerate(args.seeds):
        rb, ra, temps = run_seed(args, replace(base, seed=seed), device)
        for phase, res in (("before", rb), ("after", ra)):
            for g, v in res.items():
                for m in METRICS:
                    agg[phase][g][m].append(v[m])
        if si == 0:
            first = {"before": rb, "after": ra, "temps": temps}
        vv = rb.get("V-VI", {})
        print(f"  seed {seed}: V-VI reliance(before)={vv.get('reliance', float('nan')):.3f}")

    rows = []
    for phase in ("before", "after"):
        print(f"\n  --- {phase} calibration ---")
        print(f"  {'group':8s} {'dx_bal(mean±std)':>18s} {'reliance':>16s} {'compreh':>16s}")
        for g in GROUPS:
            if g not in agg[phase]:
                continue
            r = {"phase": phase, "group": g, "seeds": len(args.seeds)}
            for m in METRICS:
                mu, sd = mean_std(agg[phase][g][m]); r[m] = mu; r[f"{m}_std"] = sd
            rows.append(r)
            print(f"  {g:8s} {fmt_ms(*mean_std(agg[phase][g]['dx_bal_acc'])):>18s} "
                  f"{fmt_ms(*mean_std(agg[phase][g]['reliance'])):>16s} "
                  f"{fmt_ms(*mean_std(agg[phase][g]['comprehensiveness'])):>16s}")

    # within-regime dark-vs-light (independent groups): seed-0 bootstrap difference
    rb0 = first["before"]
    if "V-VI" in rb0 and "I-II" in rb0:
        print("\n  Dark(V-VI) vs Light(I-II), seed 0:")
        for m in ("reliance", "comprehensiveness"):
            d = bootstrap_diff(rb0["V-VI"]["_ps"][m], rb0["I-II"]["_ps"][m])
            sig = "SIGNIF" if (d["lo"] > 0 or d["hi"] < 0) else "n.s."
            print(f"    Δ{m}(V-VI − I-II) = {d['diff']:+.3f} [{d['lo']:.3f},{d['hi']:.3f}] ({sig})")
    print(f"  seed-0 calibration temps: {{{', '.join(f'{g}:{t:.2f}' for g,t in first['temps'].items())}}}")

    tbl = pd.DataFrame(rows)
    out = io.result_path(f"exp3_fairness_{regime.split('(')[0]}")
    tbl.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
