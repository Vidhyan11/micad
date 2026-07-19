"""MR driver: regenerate all paper tables (LaTeX) + the qualitative figure.

    !python /kaggle/working/micad/scripts/make_report.py --encoder dermlip

Reads exp1/exp2/exp3 CSVs (run train.py, faithfulness.py, fairness.py first),
writes results/*.tex, and renders a qualitative concept-counterfactual panel for
a melanoma case from Model A.
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
from fbc.eval import figures, tables  # noqa: E402
from fbc.faithfulness.core import counterfactual_effect  # noqa: E402
from fbc.train import train_cbm as T  # noqa: E402
from fbc.train.data import assemble  # noqa: E402
from fbc.utils import io  # noqa: E402


def _emit(csv_name, fn, caption, label):
    p = io.result_path(csv_name)
    if not p.exists():
        print(f"  [skip] {csv_name}.csv not found — run its experiment first")
        return
    df = pd.read_csv(p)
    tbl = fn(df)
    tex = tables.to_latex(tbl, caption, label)
    out = io.result_path(csv_name + "_table", "tex")
    out.write_text(tex)
    print(f"\n=== {label} ===\n{tbl.to_string(index=False)}\n-> {out}")


def qualitative_figure(encoder, device, cfg):
    """Melanoma case: predicted concepts + counterfactual effects."""
    meta = io.meta_path("derm7pt"); emb = io.emb_path("derm7pt", encoder)
    if not meta.exists() or not emb.exists():
        print(f"  [skip figure] missing {meta.name}/{emb.name}. This script must run "
              f"where the artifacts live (Kaggle): run extract_embeddings + make_splits "
              f"+ train first. Nothing to do locally.")
        return
    data = assemble("derm7pt", encoder, use_pseudo=False, binary_positive="MEL")
    model = T.train_cbm(data, cfg, device, mode="sequential")
    Xte = torch.as_tensor(data.emb[data.split_mask("test")]).to(device)
    yte = data.y[data.split_mask("test")]
    with torch.no_grad():
        cp = model.predict_concepts(Xte)
        logits = model.diagnose_from_concepts(cp)
        prob_mel = torch.softmax(logits, 1)[:, 1]
        pred = logits.argmax(1).cpu().numpy()
    # pick a correctly-predicted melanoma with highest confidence
    cand = np.where((yte == 1) & (pred == 1))[0]
    if len(cand) == 0:
        print("  [fig] no correctly-predicted melanoma case; skipping figure")
        return
    i = cand[np.argmax(prob_mel[cand].cpu().numpy())]
    ci = cp[i:i + 1]
    yhat = torch.tensor([1], device=device)
    eff = counterfactual_effect(ci, lambda c: model.diagnose_from_concepts(c),
                                yhat, mode="flip").cpu().numpy()[0]
    out = C.FIG_DIR / "qualitative_melanoma.png"
    figures.concept_counterfactual_panel(
        ci.cpu().numpy()[0], list(data.keys), eff, "melanoma",
        f"derm7pt case (P(mel)={prob_mel[i].item():.2f})", out)
    print(f"\n=== Qualitative figure ===\n-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dermlip")
    args = ap.parse_args()
    C.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = replace(C.DEFAULT_TRAIN, encoder=args.encoder, device=device)

    _emit("exp1_diagnosis", tables.exp1_table,
          "Diagnosis and concept accuracy (detection task).", "tab:exp1")
    _emit("exp2_faithfulness", tables.exp2_table,
          "Concept-counterfactual faithfulness (95\\% CI). Higher reliance/"
          "comprehensiveness = more faithful.", "tab:exp2")
    _emit("exp3_fairness_diverse", tables.exp3_table,
          "Fairness of reasoning across Fitzpatrick groups (diverse-training regime, "
          "multi-seed), before/after group-conditional calibration.", "tab:exp3")
    faithfulness_figure()
    qualitative_figure(args.encoder, device, cfg)
    print("\nDONE.")


def faithfulness_figure():
    """Figure 3: faithfulness bar chart from exp2_faithfulness.csv."""
    p = io.result_path("exp2_faithfulness")
    if not p.exists():
        print("  [skip figure] exp2_faithfulness.csv not found — run faithfulness.py first")
        return
    out = C.FIG_DIR / "faithfulness_bars.png"
    figures.faithfulness_bar_chart(pd.read_csv(p), out)
    print(f"\n=== Faithfulness figure ===\n-> {out}")


if __name__ == "__main__":
    main()
