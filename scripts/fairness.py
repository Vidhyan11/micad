"""MF3 driver: fairness-of-reasoning audit + mitigation -> Experiment 3.

    !python /kaggle/working/micad/scripts/fairness.py --encoder dermlip

Uses the clinical Model B (Fitzpatrick17k) — the only model with skin-tone labels.
For each Fitzpatrick group {I-II, III-IV, V-VI} on the test set, reports diagnosis
accuracy AND reasoning faithfulness (reliance, comprehensiveness). Then applies
group-conditional concept calibration (fit on val) and re-audits, reporting whether
the worst-vs-best group GAP shrinks.
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
from fbc.eval import metrics as MET  # noqa: E402
from fbc.faithfulness import faithfulness_scores  # noqa: E402
from fbc.fairness import apply_group_temperature, fit_group_temperature  # noqa: E402
from fbc.train import train_cbm as T  # noqa: E402
from fbc.train.data import assemble  # noqa: E402
from fbc.utils import io  # noqa: E402

GROUPS = ["I-II", "III-IV", "V-VI"]


def audit(cp_te, y_te, fst_te, predict_fn, ref):
    """Per-group dx + faithfulness metrics on precomputed test concept probs."""
    out = {}
    for g in GROUPS:
        m = fst_te == g
        n = int(m.sum())
        if n < 20:
            continue
        cg = cp_te[m]
        with torch.no_grad():
            logits = predict_fn(cg)
        dx = MET.dx_metrics(y_te[m], logits.cpu().numpy())
        fa = faithfulness_scores(cg, predict_fn, ref)
        out[g] = {"n": n, "dx_bal_acc": dx["bal_acc"], "dx_auroc": dx["auroc"],
                  "reliance": fa["reliance"], "comprehensiveness": fa["comprehensiveness"],
                  "ccf_corr": fa["ccf_corr"]}
    return out


def _gap(audit_res, key):
    vals = [v[key] for v in audit_res.values()]
    return max(vals) - min(vals) if vals else float("nan")


def _print_audit(title, res):
    print(f"\n  --- {title} ---")
    print(f"  {'group':8s} {'n':>5s} {'dx_bal':>7s} {'dx_auroc':>8s} "
          f"{'relian':>7s} {'compreh':>8s} {'ccf':>6s}")
    for g, v in res.items():
        print(f"  {g:8s} {v['n']:5d} {v['dx_bal_acc']:7.3f} {v['dx_auroc']:8.3f} "
              f"{v['reliance']:7.3f} {v['comprehensiveness']:8.3f} {v['ccf_corr']:6.3f}")
    print(f"  GAP(reliance)={_gap(res,'reliance'):.3f} "
          f"GAP(comprehensiveness)={_gap(res,'comprehensiveness'):.3f} "
          f"GAP(dx_bal_acc)={_gap(res,'dx_bal_acc'):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dermlip")
    args = ap.parse_args()
    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = replace(C.DEFAULT_TRAIN, encoder=args.encoder, device=device)
    print(f"device={device} encoder={args.encoder}  (fairness on Fitzpatrick17k / Model B)")

    data = assemble("fitzpatrick17k", args.encoder, use_pseudo=True,
                    binary_positive="malignant")
    model = T.train_cbm(data, cfg, device, mode="sequential")
    predict_fn = lambda c: model.diagnose_from_concepts(c)

    # test / val tensors
    te = data.split_mask("test"); va = data.split_mask("val")
    Xte = torch.as_tensor(data.emb[te]).to(device)
    Xva = torch.as_tensor(data.emb[va]).to(device)
    y_te = data.y[te]
    fst_te = data.fst_group[te]
    fst_va = data.fst_group[va]
    Cva = torch.as_tensor(data.concept_targets[va]).to(device)

    with torch.no_grad():
        cp_te = model.predict_concepts(Xte)
        ref = model.predict_concepts(torch.as_tensor(data.emb[data.split_mask("train")]).to(device)).mean(0)

    # ---------- BEFORE mitigation ----------
    res_before = audit(cp_te, y_te, fst_te, predict_fn, ref)
    _print_audit("BEFORE mitigation", res_before)

    # ---------- group-conditional concept calibration ----------
    with torch.no_grad():
        logit_va = model.concept_head(Xva)
        logit_te = model.concept_head(Xte)
    temps = fit_group_temperature(logit_va, Cva, fst_va, GROUPS)
    print(f"\n  fitted per-group temperatures: {{{', '.join(f'{g}:{t:.2f}' for g,t in temps.items())}}}")
    cp_te_cal = apply_group_temperature(logit_te, fst_te, temps)
    ref_cal = ref   # keep same reference for comparability
    res_after = audit(cp_te_cal, y_te, fst_te, predict_fn, ref_cal)
    _print_audit("AFTER group-conditional calibration", res_after)

    # ---------- summary ----------
    rows = []
    for phase, res in [("before", res_before), ("after", res_after)]:
        for g, v in res.items():
            rows.append({"phase": phase, "group": g, **v})
    tbl = pd.DataFrame(rows)
    out = io.result_path("exp3_fairness")
    tbl.to_csv(out, index=False)
    print(f"\n  Reliance gap: {_gap(res_before,'reliance'):.3f} -> {_gap(res_after,'reliance'):.3f}")
    print(f"  Comprehensiveness gap: {_gap(res_before,'comprehensiveness'):.3f} -> {_gap(res_after,'comprehensiveness'):.3f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
