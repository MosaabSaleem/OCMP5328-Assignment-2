"""
Step 9 — Aggregate all saved metrics and produce figures for the report.
Outputs (all in results/figures/):
  comparison_bias_metrics.png  — bar chart of all bias metrics
  comparison_utility.png       — perplexity and generation speed
  regard_gender_gap.png        — bar chart of Regard gender gaps
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

from Algorithm.config import PROFESSION_LABELS, DATA_DIR

sns.set_theme(style="whitegrid", palette="Set2")
COLORS = {"base_gemma": "#c0a37b", "baseline": "#5591c7",
          "cda_only": "#e08850", "debiased": "#6daa45"}
EXPECTED_MODELS = set(MODELS_TO_EVAL)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_STEP7_BENCHMARKS = {"WinoBias", "BOLD"}


def benchmark_eval_script(benchmark):
    """Map each summary benchmark to the eval script that should produce it."""
    if benchmark.startswith("CrowS-Pairs") or benchmark.startswith("StereoSet"):
        return "step5_eval_probability.py"
    if benchmark.startswith("Embedding cosine"):
        return "step6_eval_embedding.py"
    if benchmark == "Regard":
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
    Sanity-check the metrics directory before plotting.

    Hard failures (these usually mean an eval crashed mid-run or a script was
    edited without rerunning the eval; plotting on top of them produces
    misleading figures):
      * a summary file is older than the script that should have produced it
      * a summary's detail CSV is missing or has a different row count than `n`
      * detail CSVs for the same benchmark have different schemas across models
      * per-model `n` disagrees within a benchmark

    Soft warnings (we still plot what we have; this is the normal mode while
    iterating, e.g. when a single training arm has not finished yet):
      * a benchmark is missing a summary for one or more EXPECTED_MODELS
    """
    if not summaries:
        raise RuntimeError(f"No *_summary.json files found in {METRICS_DIR}")

    errors = []   # hard-fail conditions
    warnings = [] # soft conditions; report and continue
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
            warnings.append(f"{bench}: missing summaries for {', '.join(sorted(missing))}")

        present = [m for m in EXPECTED_MODELS if m in models]
        if present and all("n" in models[m] for m in present):
            ns = {m: models[m]["n"] for m in present}
            if len(set(ns.values())) > 1:
                errors.append(f"{bench}: per-model n disagrees: {ns}")

        csv_columns = {}
        for model, obj in models.items():
            if "n" not in obj:
                continue
            csv_path = summary_csv_path(obj)
            if os.path.isfile(csv_path):
                csv_columns[model] = list(pd.read_csv(csv_path, nrows=0).columns)
        if len(csv_columns) >= 2 and len({tuple(v) for v in csv_columns.values()}) > 1:
            errors.append(f"{bench}: detail CSV columns disagree across models")

    if warnings:
        msg = "\n".join(f"  - {w}" for w in warnings)
        print(f"[Step 9] WARNING — partial metrics, plotting what is available:\n{msg}")

    if errors:
        msg = "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(f"Metric validation failed; rerun the affected eval step(s):\n{msg}")


# ── Collect all *_summary.json files ──────────────────────────────────────────
summaries = []
skipped_legacy = []
for fpath in glob.glob(os.path.join(METRICS_DIR, "*_summary.json")):
    with open(fpath) as f:
        obj = json.load(f)
    if obj.get("benchmark") in LEGACY_STEP7_BENCHMARKS:
        skipped_legacy.append(os.path.basename(fpath))
        continue
    obj["_source_file"] = os.path.basename(fpath)
    summaries.append(obj)
if skipped_legacy:
    print(
        "[Step 9] Skipping legacy Step 7 summaries: "
        + ", ".join(sorted(skipped_legacy))
    )

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
    "regard_gap_negative",
    "regard_gap_positive",
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


# ── Plot 3: Regard gender gap bars ───────────────────────────────────────────
regard_data = {}
for m in MODELS_TO_EVAL:
    p = os.path.join(METRICS_DIR, f"{m}_regard_summary.json")
    if os.path.isfile(p):
        with open(p) as f:
            regard_data[m] = json.load(f)
if regard_data:
    metrics = ["regard_gap_negative", "regard_gap_positive"]
    x = np.arange(len(metrics))
    models_order = [m for m in MODELS_TO_EVAL if m in regard_data]
    width = 0.8 / max(len(models_order), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(models_order):
        vals = [regard_data[m].get(metric, 0.0) for metric in metrics]
        ax.bar(x + (i - (len(models_order) - 1) / 2) * width, vals,
               width=width, color=COLORS.get(m, "#999"), label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(["Negative regard gap", "Positive regard gap"])
    ax.set_title("Regard Gender Gap", fontsize=12, fontweight="bold")
    ax.set_ylabel("Absolute difference")
    ax.legend(title="Model")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "regard_gender_gap.png")
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


# ── Plot 6 & 7: held-out pronoun stereotype + gender-swap shift (step4b) ──────
# Step 4b writes stereotype_and_invariance.json with shape
#   {model: {pronoun_stereotype: {prof: {male_pronoun_share, ...}},
#            gender_swap_shift:  {prof: {mean_swap_shift_abs, ...}},
#            summary: {pronoun_skew, swap_shift_abs, swap_shift, *_ci, ...}}}
held_out_path = os.path.join(METRICS_DIR, "stereotype_and_invariance.json")
if os.path.isfile(held_out_path):
    with open(held_out_path) as f:
        held_out = json.load(f)
    models_present = [m for m in MODELS_TO_EVAL if m in held_out]

    # Plot 6: per-profession male_pronoun_share — does the model lean toward
    # "he" or "she" when continuing a bio for each occupation?
    if models_present:
        professions = sorted({p for m in models_present
                              for p in held_out[m].get("pronoun_stereotype", {})})
        if professions:
            x = np.arange(len(professions))
            width = 0.8 / len(models_present)
            fig, ax = plt.subplots(figsize=(14, 6))
            for i, m in enumerate(models_present):
                pron = held_out[m].get("pronoun_stereotype", {})
                vals = [pron.get(p, {}).get("male_pronoun_share", float("nan")) for p in professions]
                offset = (i - (len(models_present) - 1) / 2) * width
                ax.bar(x + offset, vals, width, label=m, color=COLORS.get(m, "#999"), alpha=0.85)
            ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--", label="0.5 (neutral)")
            ax.set_xticks(x)
            ax.set_xticklabels(professions, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("P(he) / (P(he) + P(she))")
            ax.set_title("Pronoun preference per profession\n"
                         "(0.5 = neutral, >0.5 = male-skewed, <0.5 = female-skewed)",
                         fontsize=11, fontweight="bold")
            ax.legend(title="Model")
            plt.tight_layout()
            out = os.path.join(FIGURES_DIR, "pronoun_stereotype_per_profession.png")
            plt.savefig(out, dpi=220); plt.close()
            print(f"  Saved: {out}")

    # Plot 7: per-model headline gender-swap shift — direct test of the
    # CDA+CLP training objective. Magnitude (left): how much the average
    # logprob shifts under gender swap (lower = more invariant). Direction
    # (right): which gender framing the model prefers on average.
    if models_present and any(held_out[m].get("summary", {}).get("swap_shift_abs") is not None
                              for m in models_present):
        fig, (ax_abs, ax_sgn) = plt.subplots(1, 2, figsize=(12, 4.5))
        abs_vals    = [held_out[m]["summary"].get("swap_shift_abs")    or 0 for m in models_present]
        abs_cis     = [held_out[m]["summary"].get("swap_shift_abs_ci") or [None, None] for m in models_present]
        signed_vals = [held_out[m]["summary"].get("swap_shift")        or 0 for m in models_present]
        signed_cis  = [held_out[m]["summary"].get("swap_shift_ci")     or [None, None] for m in models_present]

        def err_pair(v, ci):
            lo, hi = ci
            return (0 if lo is None else max(0, v - lo),
                    0 if hi is None else max(0, hi - v))

        errs_abs = list(zip(*[err_pair(v, ci) for v, ci in zip(abs_vals, abs_cis)]))
        errs_sgn = list(zip(*[err_pair(v, ci) for v, ci in zip(signed_vals, signed_cis)]))
        colors_present = [COLORS.get(m, "#999") for m in models_present]

        ax_abs.bar(models_present, abs_vals, color=colors_present,
                   yerr=errs_abs, capsize=4, ecolor="black")
        ax_abs.set_title("Mean |swap_shift| (lower = more gender-invariant)", fontsize=10)
        ax_abs.set_ylabel("|avg logprob(original) − avg logprob(swapped)|")

        ax_sgn.bar(models_present, signed_vals, color=colors_present,
                   yerr=errs_sgn, capsize=4, ecolor="black")
        ax_sgn.axhline(0, color="black", linewidth=1)
        ax_sgn.set_title("Mean signed swap_shift (0 = balanced)", fontsize=10)
        ax_sgn.set_ylabel("avg logprob(original) − avg logprob(swapped)")

        plt.suptitle("Gender-swap invariance on held-out bios (95% bootstrap CI)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "gender_swap_shift.png")
        plt.savefig(out, dpi=220); plt.close()
        print(f"  Saved: {out}")


# ── Print summary table ────────────────────────────────────────────────────────
if len(df_all):
    pivot = df_all.pivot_table(index="metric", columns="model",
                                values="value", aggfunc="mean")
    print("\n=== Final Results Table ===")
    print(pivot.to_string())

print(f"\n[Step 9] Done. Figures saved to: {FIGURES_DIR}")


# ── Log to W&B as one results-summary run ─────────────────────────────────────
# Dashboard shape:
#   * all_results: long-form Table (model, benchmark, metric, value)
#   * comparison:  pivoted Table — rows are (benchmark, metric), cols are models,
#                  with explicit debiased-vs-baseline delta columns so a single
#                  panel shows whether each metric moved in the right direction.
#   * summary:     a small dict of headline metrics surfaced as run summary
#                  columns in the project view.
#   * figures/*:   one Image panel per produced PNG.
HEADLINE_METRICS = [
    ("CrowS-Pairs (gender)",                  "stereotype_preference_rate"),
    ("StereoSet (gender)",                    "stereotype_preference_rate"),
    ("StereoSet (gender)",                    "icat_score"),
    ("Regard",                                "regard_gap_negative"),
    ("Regard",                                "regard_gap_positive"),
    ("Embedding cosine (CrowS-Pairs gender)", "mean_cosine_similarity"),
    ("Utility",                               "perplexity_wikitext2"),
]

wb_run = wandb_start(
    job_type="results",
    name="results_summary",
    config={"models": MODELS_TO_EVAL, "n_metric_rows": len(df_all)},
)
if wb_run is not None:
    import wandb  # local import keeps this branch lazy when W&B is disabled

    # 1) Long-form table — every (model, benchmark, metric) row.
    payload = {"all_results": wandb.Table(dataframe=df_all)}

    # 2) Pivoted comparison table — one row per (benchmark, metric), one column
    #    per model, plus deltas vs baseline and vs base_gemma for at-a-glance reads.
    if len(df_all):
        pivot = df_all.pivot_table(
            index=["benchmark", "metric"],
            columns="model",
            values="value",
            aggfunc="first",
        ).reset_index()
        pivot.columns.name = None
        for m in MODELS_TO_EVAL:
            if m not in pivot.columns:
                pivot[m] = None
        if "baseline" in pivot and "debiased" in pivot:
            pivot["delta_debiased_minus_baseline"] = pivot["debiased"] - pivot["baseline"]
        if "base_gemma" in pivot and "debiased" in pivot:
            pivot["delta_debiased_minus_base_gemma"] = pivot["debiased"] - pivot["base_gemma"]
        payload["comparison"] = wandb.Table(dataframe=pivot)

    # No static images — the matplotlib PNGs above are for the report,
    # not for W&B. The workspace can build interactive bar/line charts
    # directly from `all_results` and `comparison` tables, which gives
    # cross-run zoom, filtering, and per-model colour by default.
    wb_run.log(payload)

    # 4) Headline metrics → run summary. Each scalar shows up as a single
    #    column in the W&B project table so models can be compared side by side.
    for bench, metric in HEADLINE_METRICS:
        sub = df_all[(df_all["benchmark"] == bench) & (df_all["metric"] == metric)]
        for _, row in sub.iterrows():
            wb_run.summary[f"{metric}/{row['model']}"] = float(row["value"])
    wb_run.summary["n_metric_rows"]   = len(df_all)
    wb_run.summary["models_evaluated"] = MODELS_TO_EVAL
wandb_finish(wb_run)
