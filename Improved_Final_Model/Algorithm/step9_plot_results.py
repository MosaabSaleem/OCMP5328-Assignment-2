"""
Step 9 — Aggregate all saved metrics and produce figures for the report.
"""
import sys
import os
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import METRICS_DIR, FIGURES_DIR

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")
COLORS = {"baseline": "#5591c7", "debiased": "#6daa45"}

rows = []
for fpath in glob.glob(os.path.join(METRICS_DIR, "*_summary.json")):
    with open(fpath) as f:
        obj = json.load(f)
    model = obj.get("model", "")
    bench = obj.get("benchmark", "")
    for k, v in obj.items():
        if isinstance(v, (int, float)) and k not in ("n",):
            rows.append({"model": model, "benchmark": bench, "metric": k, "value": v})

df_all = pd.DataFrame(rows)
df_all.to_csv(os.path.join(METRICS_DIR, "all_results_table.csv"), index=False)
print(f"[Step 9] Collected {len(df_all)} metric values")

bias_metrics = [
    "stereotype_preference_rate",
    "score_gap_mean",
    "mean_cosine_distance",
    "stereotype_logprob_gap",
    "avg_abs_gender_gap",
]
df_bias = df_all[df_all["metric"].isin(bias_metrics)].copy()
if len(df_bias):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_bias, x="metric", y="value", hue="model", palette=COLORS, ax=ax)
    ax.set_title("Bias Metric Comparison: Baseline vs Debiased", fontsize=13, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="Model")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "comparison_bias_metrics.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f" Saved: {out}")

util_keys = ["perplexity_wikitext2", "mean_generation_seconds", "train_seconds"]
df_util = df_all[df_all["metric"].isin(util_keys)].copy()
if len(df_util):
    keys_present = [k for k in util_keys if k in df_util["metric"].values]
    fig, axes = plt.subplots(1, len(keys_present), figsize=(5 * len(keys_present), 4))
    if len(keys_present) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys_present):
        sub = df_util[df_util["metric"] == key]
        ax.bar(sub["model"], sub["value"], color=[COLORS.get(m, "#999") for m in sub["model"]])
        ax.set_title(key.replace("_", " "), fontsize=10)
        ax.set_ylabel("Value")
    plt.suptitle("Utility Metrics: Baseline vs Debiased", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "comparison_utility.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f" Saved: {out}")

bold_data = {}
for m in ["baseline", "debiased"]:
    p = os.path.join(METRICS_DIR, f"{m}_bold.csv")
    if os.path.isfile(p):
        bold_data[m] = pd.read_csv(p)

if bold_data:
    fig, axes = plt.subplots(1, len(bold_data), figsize=(6 * len(bold_data), 4), sharey=True)
    if len(bold_data) == 1:
        axes = [axes]
    for ax, (m, df_b) in zip(axes, bold_data.items()):
        if "abs_gender_gap" in df_b.columns:
            ax.hist(
                df_b["abs_gender_gap"].dropna(),
                bins=15,
                color=COLORS.get(m, "#999"),
                alpha=0.85,
                edgecolor="white",
            )
            ax.set_title(f"BOLD Gender Term Gap — {m}", fontsize=10)
            ax.set_xlabel("Absolute gender term count gap")
            ax.set_ylabel("Frequency")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "bold_gender_gap.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print(f" Saved: {out}")

hist_path = os.path.join(METRICS_DIR, "debiased_training_history.csv")
if os.path.isfile(hist_path):
    hist_df = pd.read_csv(hist_path)
    if len(hist_df):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hist_df["step"], hist_df["loss_total"], color=COLORS["debiased"], linewidth=2)
        ax.set_title("Debiased Training Loss", fontsize=12, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "training_loss.png")
        plt.savefig(out, dpi=220)
        plt.close()
        print(f" Saved: {out}")

if len(df_all):
    pivot = df_all.pivot_table(index="metric", columns="model", values="value", aggfunc="mean")
    print("\n=== Final Results Table ===")
    print(pivot.to_string())

print(f"\n[Step 9] Done. Figures saved to: {FIGURES_DIR}")