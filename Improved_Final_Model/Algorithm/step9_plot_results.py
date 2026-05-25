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
from Algorithm.config import METRICS_DIR, FIGURES_DIR

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")
COLORS = {"baseline": "#5591c7", "debiased": "#6daa45"}
EXPECTED_MODELS = {"baseline", "debiased"}
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
            n_base = models["baseline"]["n"]
            n_debiased = models["debiased"]["n"]
            if n_base != n_debiased:
                errors.append(f"{bench}: baseline n={n_base}, debiased n={n_debiased}")

        csv_columns = {}
        for model, obj in models.items():
            if "n" not in obj:
                continue
            csv_path = summary_csv_path(obj)
            if os.path.isfile(csv_path):
                csv_columns[model] = list(pd.read_csv(csv_path, nrows=0).columns)
        if set(csv_columns) == EXPECTED_MODELS and csv_columns["baseline"] != csv_columns["debiased"]:
            errors.append(f"{bench}: baseline/debiased detail CSV columns do not match")

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
for obj in summaries:
    model = obj.get("model", "")
    bench = obj.get("benchmark", "")
    for k, v in obj.items():
        if isinstance(v, (int, float)) and k not in ("n",):
            rows.append({"model": model, "benchmark": bench, "metric": k, "value": v})

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
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_bias, x="benchmark_metric", y="value", hue="model",
                palette=COLORS, ax=ax)
    ax.set_title("Bias Metric Comparison: Baseline vs Debiased",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Benchmark / Metric"); ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=25)
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
train_json = os.path.join(METRICS_DIR, "train_debiased.json")
if os.path.isfile(train_json):
    with open(train_json) as f:
        info = json.load(f)
    log_csv = os.path.join(os.path.dirname(METRICS_DIR),
                           "models", "debiased", "trainer_state.json")
    if os.path.isfile(log_csv):
        with open(log_csv) as f:
            state = json.load(f)
        log_hist = state.get("log_history", [])
        losses = [(e["step"], e["loss"]) for e in log_hist if "loss" in e]
        if losses:
            steps, vals = zip(*losses)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, vals, color=COLORS["debiased"], linewidth=2)
            ax.set_title("Debiased Model Training Loss", fontsize=12, fontweight="bold")
            ax.set_xlabel("Step"); ax.set_ylabel("Loss")
            plt.tight_layout()
            out = os.path.join(FIGURES_DIR, "training_loss.png")
            plt.savefig(out, dpi=220); plt.close()
            print(f"  Saved: {out}")


# ── Print summary table ────────────────────────────────────────────────────────
if len(df_all):
    pivot = df_all.pivot_table(index="metric", columns="model",
                                values="value", aggfunc="mean")
    print("\n=== Final Results Table ===")
    print(pivot.to_string())

print(f"\n[Step 9] Done. Figures saved to: {FIGURES_DIR}")
