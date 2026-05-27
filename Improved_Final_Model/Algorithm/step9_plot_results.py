"""
Step 9 — Aggregate all saved metrics and produce figures for the report.
Outputs (all in results/figures/):
  comparison_bias_metrics.png  — bar chart of all bias metrics
  comparison_utility.png       — perplexity and generation speed
  bold_gender_gap.png          — histogram of BOLD gender-term gap
  training_loss.png            — training loss curve (debiased model)
  all_results_table.csv        — flat table of every metric value
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import METRICS_DIR, FIGURES_DIR, MODELS_TO_EVAL
from Algorithm._wandb_log import start as wandb_start, finish as wandb_finish

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import re

from Algorithm.config import PROFESSION_LABELS, DATA_DIR

sns.set_theme(style="whitegrid", palette="Set2")
COLORS = {"base_gemma": "#c0a37b", "baseline": "#5591c7",
          "cda_only": "#e08850", "debiased": "#6daa45"}
EXPECTED_MODELS = set(MODELS_TO_EVAL)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def benchmark_eval_script(benchmark):
    """Map each summary benchmark to the eval script that should produce it."""
    if benchmark.startswith("CrowS-Pairs") or benchmark.startswith("StereoSet"):
        return "step5_eval_probability.py"
    if benchmark.startswith("Embedding cosine"):
        return "step6_eval_embedding.py"
    if benchmark in {"WinoBias", "BOLD"}:
        return "step7_eval_generated.py"
    if benchmark == "Utility":
        return "step8_eval_utility.py"
    return None


def summary_csv_path(summary):
    """Return the detail CSV path implied by a summary JSON filename."""
    source = summary.get("_source_file", "")
    csv_name = source.replace("_summary.json", ".csv")
    return os.path.join(METRICS_DIR, csv_name)


def validate_summary_files(summaries):
    """
    Fail fast if Step 9 is about to mix partial or stale metrics.
    Benchmarks should have both baseline/debiased summaries, matching n values,
    and matching detail CSV schemas when detail CSVs exist.
    """
    if not summaries:
        raise RuntimeError(f"No *_summary.json files found in {METRICS_DIR}")

    errors = []
    by_benchmark = {}
    for obj in summaries:
        bench = obj.get("benchmark", "")
        model = obj.get("model", "")
        source_path = os.path.join(METRICS_DIR, obj.get("_source_file", ""))
        if bench and model in EXPECTED_MODELS:
            by_benchmark.setdefault(bench, {})[model] = obj

        script_name = benchmark_eval_script(bench)
        if script_name:
            script_path = os.path.join(SCRIPT_DIR, script_name)
            if os.path.isfile(source_path) and os.path.isfile(script_path):
                if os.path.getmtime(source_path) < os.path.getmtime(script_path):
                    errors.append(
                        f"{obj.get('_source_file')}: older than {script_name}; "
                        "rerun that eval step"
                    )

        if "n" in obj:
            csv_path = summary_csv_path(obj)
            if not os.path.isfile(csv_path):
                errors.append(f"{obj.get('_source_file')}: missing detail CSV {os.path.basename(csv_path)}")
            else:
                csv_rows = len(pd.read_csv(csv_path))
                if csv_rows != int(obj["n"]):
                    errors.append(
                        f"{obj.get('_source_file')}: summary n={obj['n']} "
                        f"but {os.path.basename(csv_path)} has {csv_rows} rows"
                    )

    for bench, models in sorted(by_benchmark.items()):
        missing = EXPECTED_MODELS - set(models)
        if missing:
            errors.append(f"{bench}: missing summaries for {', '.join(sorted(missing))}")
            continue

        if all("n" in models[m] for m in EXPECTED_MODELS):
            ns = {m: models[m]["n"] for m in EXPECTED_MODELS}
            if len(set(ns.values())) > 1:
                errors.append(f"{bench}: per-model n disagrees: {ns}")

        csv_columns = {}
        for model, obj in models.items():
            if "n" not in obj:
                continue
            csv_path = summary_csv_path(obj)
            if os.path.isfile(csv_path):
                csv_columns[model] = list(pd.read_csv(csv_path, nrows=0).columns)
        if set(csv_columns) == EXPECTED_MODELS and len({tuple(v) for v in csv_columns.values()}) > 1:
            errors.append(f"{bench}: detail CSV columns disagree across models")

    if errors:
        msg = "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(f"Metric validation failed; rerun the affected eval step(s):\n{msg}")


# ── Collect all *_summary.json files ──────────────────────────────────────────
summaries = []
for fpath in glob.glob(os.path.join(METRICS_DIR, "*_summary.json")):
    with open(fpath) as f:
        obj = json.load(f)
    obj["_source_file"] = os.path.basename(fpath)
    summaries.append(obj)

validate_summary_files(summaries)

rows = []
ci_lookup = {}  # (model, benchmark, metric) -> (ci_low, ci_high)
for obj in summaries:
    model = obj.get("model", "")
    bench = obj.get("benchmark", "")
    for k, v in obj.items():
        if isinstance(v, (int, float)) and k not in ("n",):
            rows.append({"model": model, "benchmark": bench, "metric": k, "value": v})
        # Bootstrap CIs are stored as 2-element lists ending in _ci; index them
        # by the metric they describe so we can attach error bars later.
        if k.endswith("_ci") and isinstance(v, list) and len(v) == 2:
            base_metric = k[:-3]
            ci_lookup[(model, bench, base_metric)] = (v[0], v[1])

df_all = pd.DataFrame(rows)
df_all.to_csv(os.path.join(METRICS_DIR, "all_results_table.csv"), index=False)
print(f"[Step 9] Collected {len(df_all)} metric values")


# ── Plot 1: Bias metrics grouped bar chart ─────────────────────────────────────
bias_metrics = [
    "stereotype_preference_rate",
    "stereotype_preference_rate_avg",
    "score_gap_mean",
    "score_gap_avg_mean",
    "mean_cosine_distance",
    "stereotype_logprob_gap",
    "stereotype_logprob_gap_avg",
    "avg_abs_gender_gap",
]
df_bias = df_all[df_all["metric"].isin(bias_metrics)].copy()
if len(df_bias):
    df_bias["benchmark_metric"] = df_bias["benchmark"] + "\n" + df_bias["metric"]
    fig, ax = plt.subplots(figsize=(13, 5))
    metrics_order = list(dict.fromkeys(df_bias["benchmark_metric"]))
    models_order = [m for m in MODELS_TO_EVAL if m in df_bias["model"].unique()]
    n_models = max(len(models_order), 1)
    width = 0.8 / n_models
    for i, mname in enumerate(models_order):
        xs, ys, errs_lo, errs_hi = [], [], [], []
        for j, bm in enumerate(metrics_order):
            sub = df_bias[(df_bias["benchmark_metric"] == bm) & (df_bias["model"] == mname)]
            if len(sub) == 0:
                continue
            v = float(sub["value"].iloc[0])
            bench = sub["benchmark"].iloc[0]
            metric = sub["metric"].iloc[0]
            lo, hi = ci_lookup.get((mname, bench, metric), (None, None))
            # Offset each model's bars symmetrically around the metric tick.
            xs.append(j + (i - (n_models - 1) / 2) * width)
            ys.append(v)
            errs_lo.append(0 if lo is None else max(0, v - lo))
            errs_hi.append(0 if hi is None else max(0, hi - v))
        ax.bar(xs, ys, width=width, color=COLORS.get(mname, "#999"), label=mname,
               yerr=[errs_lo, errs_hi], capsize=3, ecolor="black")
    ax.set_xticks(range(len(metrics_order)))
    ax.set_xticklabels(metrics_order, rotation=25, ha="right")
    ax.set_title("Bias Metric Comparison (95% bootstrap CI)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Benchmark / Metric"); ax.set_ylabel("Value")
    ax.legend(title="Model")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "comparison_bias_metrics.png")
    plt.savefig(out, dpi=220); plt.close()
    print(f"  Saved: {out}")


# ── Plot 2: Utility metrics ────────────────────────────────────────────────────
util_keys = ["perplexity_wikitext2", "mean_generation_seconds", "train_seconds"]
df_util = df_all[df_all["metric"].isin(util_keys)].copy()
if len(df_util):
    keys_present = [k for k in util_keys if k in df_util["metric"].values]
    fig, axes = plt.subplots(1, len(keys_present), figsize=(5 * len(keys_present), 4))
    if len(keys_present) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys_present):
        sub = df_util[df_util["metric"] == key]
        ax.bar(sub["model"], sub["value"],
               color=[COLORS.get(m, "#999") for m in sub["model"]])
        ax.set_title(key.replace("_", " "), fontsize=10)
        ax.set_ylabel("Value")
    plt.suptitle("Utility Metrics: Baseline vs Debiased",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "comparison_utility.png")
    plt.savefig(out, dpi=220); plt.close()
    print(f"  Saved: {out}")


# ── Plot 3: BOLD gender gap histogram ─────────────────────────────────────────
bold_data = {}
for m in MODELS_TO_EVAL:
    p = os.path.join(METRICS_DIR, f"{m}_bold.csv")
    if os.path.isfile(p):
        bold_data[m] = pd.read_csv(p)
if bold_data:
    fig, axes = plt.subplots(1, len(bold_data), figsize=(6 * len(bold_data), 4), sharey=True)
    if len(bold_data) == 1:
        axes = [axes]
    for ax, (m, df_b) in zip(axes, bold_data.items()):
        if "abs_gender_gap" in df_b.columns:
            ax.hist(df_b["abs_gender_gap"].dropna(), bins=15,
                    color=COLORS.get(m, "#999"), alpha=0.85, edgecolor="white")
            ax.set_title(f"BOLD Gender Term Gap — {m}", fontsize=10)
            ax.set_xlabel("Absolute gender term count gap")
            ax.set_ylabel("Frequency")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "bold_gender_gap.png")
    plt.savefig(out, dpi=220); plt.close()
    print(f"  Saved: {out}")


# ── Plot 4: Debiased training loss curve ──────────────────────────────────────
# Prefer the explicit history JSON written by step 4 (which uses a custom
# loop and does not produce a trainer_state.json). Fall back to the
# Trainer state file if present (older runs).
history_json = os.path.join(METRICS_DIR, "train_history_debiased.json")
trainer_state = os.path.join(os.path.dirname(METRICS_DIR), "models", "debiased", "trainer_state.json")
losses = []
if os.path.isfile(history_json):
    with open(history_json) as f:
        hist = json.load(f)
    losses = [(i + 1, h.get("loss_total")) for i, h in enumerate(hist) if h.get("loss_total") is not None]
elif os.path.isfile(trainer_state):
    with open(trainer_state) as f:
        state = json.load(f)
    losses = [(e["step"], e["loss"]) for e in state.get("log_history", []) if "loss" in e]
if losses:
    steps, vals = zip(*losses)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, vals, color=COLORS["debiased"], linewidth=2)
    # Overlay the LM-only and CLP components if the history JSON has them.
    if os.path.isfile(history_json):
        with open(history_json) as f:
            hist = json.load(f)
        lm_orig = [(i + 1, h.get("loss_lm_orig")) for i, h in enumerate(hist) if h.get("loss_lm_orig") is not None]
        clp     = [(i + 1, h.get("loss_clp"))     for i, h in enumerate(hist) if h.get("loss_clp")     is not None]
        if lm_orig:
            xs, ys = zip(*lm_orig); ax.plot(xs, ys, alpha=0.6, label="LM(orig)")
        if clp:
            xs, ys = zip(*clp); ax.plot(xs, ys, alpha=0.6, label="CLP (raw)")
        ax.legend()
    ax.set_title("Debiased Model Training Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "training_loss.png")
    plt.savefig(out, dpi=220); plt.close()
    print(f"  Saved: {out}")


# ── Plot 5: Dataset — occupation distribution & gender imbalance ──────────────
bib_path = os.path.join(DATA_DIR, "bias_in_bios.csv")
if os.path.isfile(bib_path):
    df_bib = pd.read_csv(bib_path)
    if "profession" in df_bib.columns and "gender" in df_bib.columns:
        # 5a: Occupation counts (sorted)
        occ_counts = df_bib["profession"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 9))
        colors = ["#e08850" if c == "professor" else "#3182bd" for c in occ_counts.index]
        ax.barh(occ_counts.index, occ_counts.values, color=colors)
        ax.set_xlabel("Count (training set)")
        ax.set_title("Bias in Bios — Occupation Distribution\n(orange = professor, largest class)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "dataset_occupation_dist.png")
        plt.savefig(out, dpi=220); plt.close()
        print(f"  Saved: {out}")

        # 5b: Gender imbalance per occupation (% female)
        df_bib["is_female"] = (df_bib["gender"] == "female").astype(int)
        imb = (df_bib.groupby("profession")["is_female"].mean() * 100).sort_values()
        fig, ax = plt.subplots(figsize=(8, 9))
        bar_colors = ["#984ea3" if v > 50 else "#4daf4a" for v in imb.values]
        bars = ax.barh(imb.index, imb.values, color=bar_colors)
        ax.axvline(50, color="black", linewidth=1.2, linestyle="--", label="50% (neutral)")
        ax.set_xlabel("% Female in training set")
        ax.set_title("Gender Imbalance per Occupation\n(green = male-skewed, purple = female-skewed)",
                     fontsize=11, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "dataset_gender_imbalance.png")
        plt.savefig(out, dpi=220); plt.close()
        print(f"  Saved: {out}")


# ── Plot 6: In-domain gender bias (step4b) ────────────────────────────────────
indomain_path = os.path.join(METRICS_DIR, "indomain_gender_bias.json")
if os.path.isfile(indomain_path):
    with open(indomain_path) as f:
        indomain = json.load(f)
    models_present = [m for m in MODELS_TO_EVAL if m in indomain]
    if models_present:
        professions = sorted({p for m in models_present for p in indomain[m]})
        x = np.arange(len(professions))
        width = 0.8 / len(models_present)
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, m in enumerate(models_present):
            vals = [indomain[m].get(p, {}).get("male_bias", float("nan")) for p in professions]
            offset = (i - (len(models_present) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=m, color=COLORS.get(m, "#999"), alpha=0.85)
        ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--", label="0.5 (neutral)")
        ax.set_xticks(x)
        ax.set_xticklabels(professions, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("P(he) / (P(he) + P(she))")
        ax.set_title("In-domain Gender Bias per Profession\n"
                     "(0.5 = neutral, >0.5 = male-skewed, <0.5 = female-skewed)",
                     fontsize=11, fontweight="bold")
        ax.legend(title="Model")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "indomain_gender_bias.png")
        plt.savefig(out, dpi=220); plt.close()
        print(f"  Saved: {out}")


# ── Print summary table ────────────────────────────────────────────────────────
if len(df_all):
    pivot = df_all.pivot_table(index="metric", columns="model",
                                values="value", aggfunc="mean")
    print("\n=== Final Results Table ===")
    print(pivot.to_string())

print(f"\n[Step 9] Done. Figures saved to: {FIGURES_DIR}")


# ── Optional: log everything to W&B as one results-summary run ────────────────
def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "v"

wb_run = wandb_start(
    job_type="results",
    name="results_summary",
    config={"models": MODELS_TO_EVAL, "n_metric_rows": len(df_all)},
)
if wb_run is not None:
    # Log per-metric scalars under a stable namespace so the dashboard can
    # surface them as side-by-side panels across models.
    for _, row in df_all.iterrows():
        key = f"final/{_slug(row['benchmark'])}/{_slug(row['metric'])}/{_slug(row['model'])}"
        wb_run.log({key: float(row["value"])})
    # The aggregated table + each figure produced by this step.
    wb_run.log({"all_results_table": __import__("wandb").Table(dataframe=df_all)})
    for fig_name in ["comparison_bias_metrics.png", "comparison_utility.png",
                     "bold_gender_gap.png", "training_loss.png",
                     "dataset_occupation_dist.png", "dataset_gender_imbalance.png",
                     "indomain_gender_bias.png"]:
        fig_path = os.path.join(FIGURES_DIR, fig_name)
        if os.path.isfile(fig_path):
            wb_run.log({f"figures/{os.path.splitext(fig_name)[0]}":
                        __import__("wandb").Image(fig_path)})
    wb_run.summary["n_metric_rows"] = len(df_all)
    wb_run.summary["models_evaluated"] = MODELS_TO_EVAL
wandb_finish(wb_run)
