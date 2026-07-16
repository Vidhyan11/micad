"""MF1 driver: concept-counterfactual faithfulness -> Experiment 2 table.

    !python /kaggle/working/micad/scripts/faithfulness.py --encoder dermlip

Compares the PURE bottleneck CBM against a LEAKY CBM (diagnosis sees concepts +
embedding). The pure bottleneck's decision depends only on concepts, so its stated
concept importance should match the causal (counterfactual) effect and it should
show high reliance/comprehensiveness; the leaky model should not.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import fbc.config as C  # noqa: E402
from fbc.eval.bootstrap import bootstrap_ci, bootstrap_diff, fmt_ci  # noqa: E402
from fbc.faithfulness import faithfulness_scores  # noqa: E402
from fbc.train import train_cbm as T  # noqa: E402
from fbc.train.data import assemble  # noqa: E402
from fbc.utils import io  # noqa: E402

CI_METRICS = ["reliance", "comprehensiveness", "sufficiency", "ccf_corr", "decisive_hit"]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dermlip")
    ap.add_argument("--models", nargs="*", default=["A", "B"])
    ap.add_argument("--importance", default="gradient", choices=["gradient", "loo"])
    ap.add_argument("--cf_mode", default="flip", choices=["flip", "ablate"])
    args = ap.parse_args()

    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = replace(C.DEFAULT_TRAIN, encoder=args.encoder, device=device)
    print(f"device={device} encoder={args.encoder} importance={args.importance} cf={args.cf_mode}")

    rows = []
    for key in args.models:
        spec = SPECS[key]
        ds = spec["ds"]
        print(f"\n===== Model {key}: {spec['label']} ({ds}) =====")
        data = assemble(ds, args.encoder, use_pseudo=spec["use_pseudo"],
                        binary_positive=spec["binary_positive"])
        Xtr, Xte = _test_tensors(data, device)

        # --- pure bottleneck CBM ---
        pure = T.train_cbm(data, cfg, device, mode="sequential")
        with torch.no_grad():
            cp_tr = pure.predict_concepts(Xtr)
            cp_te = pure.predict_concepts(Xte)
        ref = cp_tr.mean(0)                          # neutral reference (train mean)
        pure_fn = lambda c: pure.diagnose_from_concepts(c)
        s_pure = faithfulness_scores(cp_te, pure_fn, ref, args.importance, args.cf_mode,
                                     return_per_sample=True)
        _report(key, "pure-bottleneck", s_pure); rows.append(_row(key, "pure-bottleneck", ds, s_pure))

        # --- leaky CBM (concepts + embedding) ---
        leaky = T.train_leaky_cbm(data, cfg, device)
        with torch.no_grad():
            lp_tr = leaky.predict_concepts(Xtr)
            lp_te = leaky.predict_concepts(Xte)
        ref_l = lp_tr.mean(0)
        leaky_fn = lambda c: leaky.diagnose_from_concepts(c, Xte)
        s_leaky = faithfulness_scores(lp_te, leaky_fn, ref_l, args.importance, args.cf_mode,
                                      return_per_sample=True)
        _report(key, "leaky", s_leaky); rows.append(_row(key, "leaky", ds, s_leaky))

        # significance: does the pure bottleneck rely on concepts MORE than leaky?
        for metric in ("reliance", "comprehensiveness"):
            d = bootstrap_diff(s_pure["_per_sample"][metric], s_leaky["_per_sample"][metric])
            sig = "significant" if (d["lo"] > 0 or d["hi"] < 0) else "n.s."
            print(f"  Δ{metric}(pure−leaky) = {d['diff']:.3f} "
                  f"[{d['lo']:.3f},{d['hi']:.3f}] ({sig})")

    tbl = pd.DataFrame(rows)
    out = io.result_path("exp2_faithfulness")
    tbl.to_csv(out, index=False)
    print(f"\n=== Experiment 2: faithfulness ===\n{tbl.to_string(index=False)}")
    print(f"\nsaved -> {out}")


def _row(key, variant, ds, s):
    row = {"model": key, "variant": variant, "dataset": ds}
    for m in CI_METRICS:
        mean, lo, hi = bootstrap_ci(s["_per_sample"][m])
        row[m] = mean; row[f"{m}_lo"] = lo; row[f"{m}_hi"] = hi
    row["n"] = s["n"]
    return row


def _report(key, variant, s):
    parts = []
    for m in ("reliance", "comprehensiveness", "ccf_corr"):
        parts.append(f"{m}={fmt_ci(*bootstrap_ci(s['_per_sample'][m]))}")
    print(f"  [{variant:16s}] " + "  ".join(parts))


if __name__ == "__main__":
    main()
