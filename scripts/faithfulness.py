"""MF1 driver: concept-counterfactual faithfulness -> Experiment 2 table.

    !python /kaggle/working/micad/scripts/faithfulness.py --encoder dermlip

Compares the PURE bottleneck CBM against a LEAKY CBM (diagnosis sees concepts +
embedding). Runs over multiple seeds and reports mean+/-std, plus a paired Wilcoxon
signed-rank test (pure vs leaky) on per-case reliance/comprehensiveness.
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
from fbc.eval.bootstrap import fmt_ms, mean_std, wilcoxon_paired  # noqa: E402
from fbc.faithfulness import faithfulness_scores  # noqa: E402
from fbc.train import train_cbm as T  # noqa: E402
from fbc.train.data import assemble  # noqa: E402
from fbc.utils import io  # noqa: E402

METRICS = ["reliance", "comprehensiveness", "sufficiency", "ccf_corr", "decisive_hit"]

SPECS = {
    "A": {"ds": "derm7pt", "use_pseudo": False, "binary_positive": "MEL",
          "label": "ModelA-dermoscopic"},
    "B": {"ds": "fitzpatrick17k", "use_pseudo": True, "binary_positive": "malignant",
          "label": "ModelB-clinical"},
}


def _test_tensors(data, device):
    Xte, _, _, _ = data.subset("test")
    Xtr, _, _, _ = data.subset("train")
    return (torch.as_tensor(Xtr).to(device), torch.as_tensor(Xte).to(device))


def _reference(concept_probs_train, ref_mode):
    """Neutral reference for ablation: 'mean' = train-mean concepts (average case);
    'zero' = all concepts absent (the clinically natural 'remove all evidence')."""
    mean = concept_probs_train.mean(0)
    return torch.zeros_like(mean) if ref_mode == "zero" else mean


def run_seed(spec, args, cfg, device):
    """Train pure + leaky at one seed; return their faithfulness score dicts."""
    data = assemble(spec["ds"], args.encoder, use_pseudo=spec["use_pseudo"],
                    binary_positive=spec["binary_positive"])
    Xtr, Xte = _test_tensors(data, device)

    pure = T.train_cbm(data, cfg, device, mode="sequential")
    with torch.no_grad():
        ref = _reference(pure.predict_concepts(Xtr), args.ref_mode)
        cp_te = pure.predict_concepts(Xte)
    s_pure = faithfulness_scores(cp_te, lambda c: pure.diagnose_from_concepts(c),
                                 ref, args.importance, args.cf_mode, return_per_sample=True)

    leaky = T.train_leaky_cbm(data, cfg, device)
    with torch.no_grad():
        ref_l = _reference(leaky.predict_concepts(Xtr), args.ref_mode)
        lp_te = leaky.predict_concepts(Xte)
    s_leaky = faithfulness_scores(lp_te, lambda c: leaky.diagnose_from_concepts(c, Xte),
                                  ref_l, args.importance, args.cf_mode, return_per_sample=True)
    return s_pure, s_leaky


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dermlip")
    ap.add_argument("--models", nargs="*", default=["A", "B"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--importance", default="gradient", choices=["gradient", "loo"])
    ap.add_argument("--cf_mode", default="flip", choices=["flip", "ablate"])
    ap.add_argument("--ref_mode", default="zero", choices=["zero", "mean"],
                    help="ablation reference: 'zero'=all-absent (clinical, stable), "
                         "'mean'=train-average case")
    args = ap.parse_args()

    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = replace(C.DEFAULT_TRAIN, encoder=args.encoder, device=device)
    print(f"device={device} encoder={args.encoder} seeds={args.seeds} "
          f"importance={args.importance} cf={args.cf_mode} ref={args.ref_mode}")

    rows = []
    for key in args.models:
        spec = SPECS[key]
        print(f"\n===== Model {key}: {spec['label']} ({spec['ds']}) =====")
        per = {"pure-bottleneck": defaultdict(list), "leaky": defaultdict(list)}
        first_ps = {}
        for si, seed in enumerate(args.seeds):
            s_pure, s_leaky = run_seed(spec, args, replace(base, seed=seed), device)
            for variant, s in (("pure-bottleneck", s_pure), ("leaky", s_leaky)):
                for m in METRICS:
                    per[variant][m].append(s[m])
            if si == 0:
                first_ps = {"pure": s_pure["_per_sample"], "leaky": s_leaky["_per_sample"]}
            print(f"  seed {seed}: pure reliance={s_pure['reliance']:.3f} "
                  f"leaky reliance={s_leaky['reliance']:.3f}")

        # paired significance (pure vs leaky), computed on seed-0 per-case arrays
        sig = {}
        for m in ("reliance", "comprehensiveness"):
            sig[m] = wilcoxon_paired(first_ps["pure"][m], first_ps["leaky"][m])

        for variant in ("pure-bottleneck", "leaky"):
            row = {"model": key, "variant": variant, "dataset": spec["ds"],
                   "seeds": len(args.seeds)}
            for m in METRICS:
                mu, sd = mean_std(per[variant][m])
                row[m] = mu; row[f"{m}_std"] = sd
                row[f"{m}_lo"] = mu - sd; row[f"{m}_hi"] = mu + sd
            if variant == "pure-bottleneck":
                row["p_reliance"] = sig["reliance"]["p"]
                row["p_comprehensiveness"] = sig["comprehensiveness"]["p"]
            rows.append(row)
            rel = per[variant]["reliance"]; com = per[variant]["comprehensiveness"]
            print(f"  [{variant:16s}] reliance={fmt_ms(*mean_std(rel))} "
                  f"comprehensiveness={fmt_ms(*mean_std(com))}")
        print(f"  Wilcoxon pure>leaky:  reliance p={sig['reliance']['p']:.2e}  "
              f"comprehensiveness p={sig['comprehensiveness']['p']:.2e}  "
              f"(n={sig['reliance']['n']})")

    tbl = pd.DataFrame(rows)
    out = io.result_path("exp2_faithfulness")
    tbl.to_csv(out, index=False)
    print(f"\n=== Experiment 2: faithfulness (mean+/-std over {len(args.seeds)} seeds) ===")
    print(tbl.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
