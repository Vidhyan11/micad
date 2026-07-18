"""Bootstrap confidence intervals + group-difference tests over per-sample metric
arrays (fast: no model re-runs, just resample precomputed per-sample values)."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, seed: int = 1337,
                 ci: float = 95.0) -> tuple[float, float, float]:
    """Return (mean, lo, hi) percentile CI of the mean of `arr`."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"),) * 3
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(arr.mean()), float(lo), float(hi)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                   seed: int = 1337, ci: float = 95.0) -> dict:
    """Bootstrap the difference of means (a - b), independent resampling.
    Returns {diff, lo, hi, p_two_sided} where p is the fraction of bootstrap
    diffs crossing 0 (a rough two-sided significance proxy)."""
    a = np.asarray(a, np.float64); a = a[~np.isnan(a)]
    b = np.asarray(b, np.float64); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_two_sided": float("nan")}
    rng = np.random.RandomState(seed)
    da = a[rng.randint(0, len(a), size=(n_boot, len(a)))].mean(1)
    db = b[rng.randint(0, len(b), size=(n_boot, len(b)))].mean(1)
    d = da - db
    lo, hi = np.percentile(d, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    obs = a.mean() - b.mean()
    # fraction of bootstrap diffs on the opposite side of 0 from the observed diff
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(obs), "lo": float(lo), "hi": float(hi), "p_two_sided": float(p)}


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]"


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test on matched per-sample values (same cases).
    Returns {stat, p, n, median_diff}. Used for pure-vs-leaky (E2) and group
    comparisons (E3) where the two arrays are aligned case-by-case."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    out = {"stat": float("nan"), "p": float("nan"), "n": int(len(a)),
           "median_diff": float(np.median(a - b)) if len(a) else float("nan")}
    if len(a) < 1 or np.allclose(a, b):
        return out
    try:
        stat, p = wilcoxon(a, b)
        out["stat"], out["p"] = float(stat), float(p)
    except ValueError:
        pass
    return out


def mean_std(values) -> tuple[float, float]:
    """Mean and (population) std across seed runs."""
    v = np.asarray(values, np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std())


def fmt_ms(mean: float, std: float) -> str:
    return f"{mean:.3f}±{std:.3f}"
