"""Turn the experiment result CSVs into paper-ready LaTeX tables."""
from __future__ import annotations

import pandas as pd


def _ci(row, base):
    lo, hi = f"{base}_lo", f"{base}_hi"
    if lo in row and pd.notna(row[lo]):
        return f"{row[base]:.3f} [{row[lo]:.3f}, {row[hi]:.3f}]"
    return f"{row[base]:.3f}"


def exp1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnosis + concept accuracy (melanoma / malignant detection)."""
    out = df.copy()
    out["dx bal-acc"] = out["dx_bal_acc"].map("{:.3f}".format)
    out["dx AUROC"] = out["dx_auroc"].map("{:.3f}".format)
    out["concept AUROC"] = out["concept_mean_auroc"].map(
        lambda x: "—" if pd.isna(x) else f"{x:.3f}")
    return out[["model", "variant", "dataset", "dx bal-acc", "dx AUROC", "concept AUROC"]]


def exp2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Faithfulness — magnitude-aware metrics lead; ccf_corr shown with caveat."""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "model": r["model"], "variant": r["variant"],
            "reliance (hi=better)": _ci(r, "reliance"),
            "comprehensiveness (hi=better)": _ci(r, "comprehensiveness"),
            "sufficiency (lo=better)": _ci(r, "sufficiency"),
            "ccf_corr (rank-only)": _ci(r, "ccf_corr"),
        })
    return pd.DataFrame(rows)


def exp3_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fairness of reasoning per Fitzpatrick group, before/after mitigation."""
    out = df.copy()
    for c in ("dx_bal_acc", "dx_auroc", "reliance", "comprehensiveness"):
        out[c] = out[c].map("{:.3f}".format)
    return out[["phase", "group", "n", "dx_bal_acc", "dx_auroc",
                "reliance", "comprehensiveness"]]


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=True, column_format="l" * df.shape[1])
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n{body}\\end{{table}}\n")
