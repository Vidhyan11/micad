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
from fbc.eval.bootstrap import bootstrap_ci, bootstrap_diff  # noqa: E402
from fbc.faithfulness import faithfulness_scores  # noqa: E402
from fbc.fairness import apply_group_temperature, fit_group_temperature  # noqa: E402
from fbc.train import train_cbm as T  # noqa: E402
from fbc.train.data import assemble  # noqa: E402
from fbc.utils import io  # noqa: E402

GROUPS = ["I-II", "III-IV", "V-VI"]


def _balacc_ci(y, pred, n_classes, n_boot=2000, seed=1337):
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.RandomState(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots.append(balanced_accuracy_score(y[idx], pred[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(balanced_accuracy_score(y, pred)), float(lo), float(hi)


def audit(cp_te, y_te, fst_te, predict_fn, ref):
    """Per-group dx + faithfulness metrics (with per-sample arrays for CIs)."""
    out = {}
    for g in GROUPS:
        m = fst_te == g
        n = int(m.sum())
        if n < 20:
            continue
        cg = cp_te[m]
        with torch.no_grad():
            logits = predict_fn(cg)
        pred = logits.argmax(1).cpu().numpy()
        dx = MET.dx_metrics(y_te[m], logits.cpu().numpy())
        fa = faithfulness_scores(cg, predict_fn, ref, return_per_sample=True)
        out[g] = {"n": n, "dx_bal_acc": dx["bal_acc"], "dx_auroc": dx["auroc"],
                  "reliance": fa["reliance"], "comprehensiveness": fa["comprehensiveness"],
                  "ccf_corr": fa["ccf_corr"], "_ps": fa["_per_sample"],
                  "_y": y_te[m], "_pred": pred}
    return out


def _gap(audit_res, key):
    vals = [v[key] for v in audit_res.values()]
    return max(vals) - min(vals) if vals else float("nan")


def _print_audit(title, res, n_classes):
    print(f"\n  --- {title} ---")
    print(f"  {'group':8s} {'n':>5s} {'dx_bal(95%CI)':>20s} "
          f"{'reliance(95%CI)':>22s} {'compreh(95%CI)':>22s}")
    for g, v in res.items():
        dbm, dbl, dbh = _balacc_ci(v["_y"], v["_pred"], n_classes)
        rm, rl, rh = bootstrap_ci(v["_ps"]["reliance"])
        cm, cl, ch = bootstrap_ci(v["_ps"]["comprehensiveness"])
        print(f"  {g:8s} {v['n']:5d} {dbm:.3f}[{dbl:.3f},{dbh:.3f}] "
              f"{rm:.3f}[{rl:.3f},{rh:.3f}] {cm:.3f}[{cl:.3f},{ch:.3f}]")
    print(f"  GAP(reliance)={_gap(res,'reliance'):.3f} "
          f"GAP(comprehensiveness)={_gap(res,'comprehensiveness'):.3f} "
          f"GAP(dx_bal_acc)={_gap(res,'dx_bal_acc'):.3f}")
    # key fairness test: dark (V-VI) vs light (I-II)
    if "V-VI" in res and "I-II" in res:
        for metric in ("reliance", "comprehensiveness"):
            d = bootstrap_diff(res["V-VI"]["_ps"][metric], res["I-II"]["_ps"][metric])
            sig = "SIGNIF" if (d["lo"] > 0 or d["hi"] < 0) else "n.s."
            print(f"    Δ{metric}(V-VI − I-II) = {d['diff']:+.3f} "
                  f"[{d['lo']:.3f},{d['hi']:.3f}] ({sig})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dermlip")
    ap.add_argument("--train_groups", nargs="*", default=None,
                    help="restrict TRAINING to these Fitzpatrick groups (e.g. I-II) "
                         "to simulate biased/light-skin-dominant training; audit stays "
                         "on all groups. Default: diverse training (all groups).")
    args = ap.parse_args()
    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = replace(C.DEFAULT_TRAIN, encoder=args.encoder, device=device)
    regime = "diverse(all groups)" if not args.train_groups else f"biased({'+'.join(args.train_groups)})"
    print(f"device={device} encoder={args.encoder} | training regime: {regime}")

    data = assemble("fitzpatrick17k", args.encoder, use_pseudo=True,
                    binary_positive="malignant")
    # Optionally simulate biased training: exclude non-selected groups from TRAIN
    # (val kept intact for early stopping + per-group calibration; test = all groups).
    if args.train_groups:
        tg = set(args.train_groups)
        tr = data.split_mask("train")
        excl = tr & ~np.isin(data.fst_group, list(tg))
        data.split[excl] = ""
        kept = data.split_mask("train")
        from collections import Counter
        print(f"  training composition (by FST): "
              f"{dict(Counter(data.fst_group[kept]))}")
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
    _print_audit("BEFORE mitigation", res_before, data.n_classes)

    # ---------- group-conditional concept calibration ----------
    with torch.no_grad():
        logit_va = model.concept_head(Xva)
        logit_te = model.concept_head(Xte)
    temps = fit_group_temperature(logit_va, Cva, fst_va, GROUPS)
    print(f"\n  fitted per-group temperatures: {{{', '.join(f'{g}:{t:.2f}' for g,t in temps.items())}}}")
    cp_te_cal = apply_group_temperature(logit_te, fst_te, temps)
    ref_cal = ref   # keep same reference for comparability
    res_after = audit(cp_te_cal, y_te, fst_te, predict_fn, ref_cal)
    _print_audit("AFTER group-conditional calibration", res_after, data.n_classes)

    # ---------- summary (drop bulky per-sample arrays from CSV) ----------
    rows = []
    for phase, res in [("before", res_before), ("after", res_after)]:
        for g, v in res.items():
            rows.append({"phase": phase, "group": g,
                         **{k: v[k] for k in ("n", "dx_bal_acc", "dx_auroc",
                                              "reliance", "comprehensiveness", "ccf_corr")}})
    tbl = pd.DataFrame(rows)
    out = io.result_path("exp3_fairness")
    tbl.to_csv(out, index=False)
    print(f"\n  Reliance gap: {_gap(res_before,'reliance'):.3f} -> {_gap(res_after,'reliance'):.3f}")
    print(f"  Comprehensiveness gap: {_gap(res_before,'comprehensiveness'):.3f} -> {_gap(res_after,'comprehensiveness'):.3f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
