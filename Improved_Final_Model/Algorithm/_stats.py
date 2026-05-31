"""
Small statistics helpers shared by the eval steps.
Bootstrap percentile CIs for proportions/means, and a paired-sign
test for comparing two models on the same set of paired examples.
"""
import numpy as np


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """
    Percentile bootstrap CI for the mean of `values` (numeric or 0/1).
    Returns dict with mean and ci_low/ci_high (CI bounds), plus n.
    """
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1 - alpha))
    return {
        "mean":    round(float(arr.mean()), 4),
        "ci_low":  round(lo, 4),
        "ci_high": round(hi, 4),
        "n": int(n),
    }
