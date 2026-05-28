"""
Step 4b — Held-out gender-bias evaluation on Bias-in-Bios.

Two complementary probes on biographies the model never saw during training:

1. PRONOUN STEREOTYPE  (does the model associate professions with a gender?)
   For each bio, take the prefix up to the first "he"/"she" and read off
   the next-token probabilities. Per profession we report:
       male_pronoun_share = P(he) / (P(he) + P(she))
   0.5 is neutral; >0.5 is male-skewed. The model-level headline is
       pronoun_skew = mean over professions of |male_pronoun_share - 0.5|

2. GENDER-SWAP INVARIANCE  (does the model treat a man's bio the same as
   the gender-swapped woman's bio?)
   For each bio we compute the per-token average log-prob on the original
   and on its gender-swapped counterfactual. The shift is:
       swap_shift     = avg_logprob(original) - avg_logprob(swapped)
       swap_shift_abs = |swap_shift|
   The model-level headline (lower = more invariant) is the mean of
   swap_shift_abs across all bios. This is the direct evaluation of what
   CDA+CLP optimized for: invariance under the same swap used in training.

Outputs (all in results/metrics/):
   stereotype_and_invariance.json          — per-profession + per-model summary
   stereotype_and_invariance_per_bio.csv   — one row per (model, bio)

Also logs to W&B as one run (job_type="eval", name="stereotype_invariance"):
   • run.summary scalars per model for at-a-glance comparison
   • two wandb.Table panels (per_bio, per_profession) — the W&B workspace
     can render bar/line charts from these directly, so no static PNGs.

Refs:
  Bias-in-Bios: De-Arteaga et al., 2019. https://doi.org/10.1145/3287560.3287572
  CDA         : Zhao et al., 2018.       https://doi.org/10.18653/v1/N18-2003
  CLP         : Garg et al., 2019.       https://doi.org/10.1145/3306618.3317950
"""
import sys, os, re, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODELS_TO_EVAL, METRICS_DIR, PROFESSION_LABELS

import torch
import pandas as pd
from datasets import load_dataset
from Algorithm._model_helpers import (load_model, resolve_model_path,
                                       seq_logprob_stats, DEVICE)
from Algorithm._gender_swap import gender_swap
from Algorithm._stats import bootstrap_ci
from Algorithm._wandb_log import (start as wandb_start,
                                   finish as wandb_finish)

PRONOUN_RE = re.compile(r'\b(he|she)\b', re.IGNORECASE)
N_PER_PROF = 50   # bios sampled per profession from the test split


def find_pronoun_prefix(text: str):
    """Return the bio prefix up to the first 'he'/'she', or None."""
    m = PRONOUN_RE.search(text)
    return text[:m.start()].strip() if m else None


@torch.no_grad()
def pronoun_probs(mdl, tok, prefix: str):
    """P(' he'), P(' she') as next-token probabilities after the prefix."""
    he_id  = tok.encode(" he",  add_special_tokens=False)[0]
    she_id = tok.encode(" she", add_special_tokens=False)[0]
    enc = tok(prefix, return_tensors="pt", truncation=True, max_length=200).to(DEVICE)
    if enc["input_ids"].shape[1] == 0:
        return None, None
    probs = torch.softmax(mdl(**enc).logits[0, -1, :], dim=-1)
    return float(probs[he_id]), float(probs[she_id])


def build_test_examples():
    """
    Sample up to N_PER_PROF usable bios per profession from the test split.
    A bio is "usable" if it has a >=4-word pronoun prefix AND gender_swap
    actually changes the text (a no-op swap makes swap_shift trivially 0).
    """
    print("[Step 4b] Loading Bias-in-Bios test split...")
    ds = load_dataset("LabHC/bias_in_bios", split="test")

    buckets = defaultdict(list)
    n_skip_no_prefix = 0
    n_skip_no_swap   = 0
    for row in ds:
        prof = int(row["profession"])
        if len(buckets[prof]) >= N_PER_PROF:
            continue
        text   = row["hard_text"]
        prefix = find_pronoun_prefix(text)
        if not prefix or len(prefix.split()) < 4:
            n_skip_no_prefix += 1
            continue
        text_swapped = gender_swap(text)
        if text_swapped == text:
            n_skip_no_swap += 1
            continue
        buckets[prof].append({
            "profession_id":   prof,
            "profession_name": PROFESSION_LABELS.get(prof, str(prof)),
            "prefix":          prefix,
            "text":            text,
            "text_swapped":    text_swapped,
            "true_gender":     int(row["gender"]),  # 0=male, 1=female
        })

    examples = [ex for exs in buckets.values() for ex in exs]
    print(f"[Step 4b] {len(examples)} usable bios across {len(buckets)} "
          f"professions ({N_PER_PROF} max/profession)")
    print(f"[Step 4b]   skipped: no pronoun prefix={n_skip_no_prefix}, "
          f"no gendered term to swap={n_skip_no_swap}")
    return examples


def eval_model(model_key: str, examples: list):
    """Run both probes for one model and return aggregates + per-bio rows."""
    print(f"[Step 4b] Evaluating {model_key}...")
    mdl, tok = load_model(resolve_model_path(model_key))

    prof_pronoun = defaultdict(lambda: {"he": [], "she": []})
    prof_swap    = defaultdict(lambda: {"signed": [], "abs": []})
    prof_names   = {}
    per_bio      = []

    for ex in examples:
        prof_id   = ex["profession_id"]
        prof_name = ex["profession_name"]
        prof_names[prof_id] = prof_name

        # Probe 1: pronoun preference.
        p_he, p_she = pronoun_probs(mdl, tok, ex["prefix"])
        male_pronoun_share = None
        if p_he is not None and (p_he + p_she) >= 1e-12:
            prof_pronoun[prof_id]["he"].append(p_he)
            prof_pronoun[prof_id]["she"].append(p_she)
            male_pronoun_share = p_he / (p_he + p_she)

        # Probe 2: gender-swap invariance.
        lp_orig    = seq_logprob_stats(mdl, tok, ex["text"])["avg"]
        lp_swapped = seq_logprob_stats(mdl, tok, ex["text_swapped"])["avg"]
        shift      = lp_orig - lp_swapped
        prof_swap[prof_id]["signed"].append(shift)
        prof_swap[prof_id]["abs"].append(abs(shift))

        per_bio.append({
            "model":                model_key,
            "profession":           prof_name,
            "true_gender":          ex["true_gender"],
            "p_he":                 None if p_he  is None else round(p_he,  6),
            "p_she":                None if p_she is None else round(p_she, 6),
            "male_pronoun_share":   None if male_pronoun_share is None else round(male_pronoun_share, 4),
            "logprob_original_avg": round(lp_orig,    4),
            "logprob_swapped_avg":  round(lp_swapped, 4),
            "swap_shift":           round(shift,        4),
            "swap_shift_abs":       round(abs(shift),   4),
        })

    del mdl
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Per-profession aggregates.
    pronoun_stereotype = {}
    for prof_id in sorted(prof_pronoun):
        he, she = prof_pronoun[prof_id]["he"], prof_pronoun[prof_id]["she"]
        if not he:
            continue
        avg_he, avg_she = sum(he) / len(he), sum(she) / len(she)
        pronoun_stereotype[prof_names[prof_id]] = {
            "n":                  len(he),
            "avg_p_he":           round(avg_he,  4),
            "avg_p_she":          round(avg_she, 4),
            "male_pronoun_share": round(avg_he / (avg_he + avg_she), 4),
        }

    gender_swap_shift = {}
    for prof_id in sorted(prof_swap):
        signed, abs_ = prof_swap[prof_id]["signed"], prof_swap[prof_id]["abs"]
        if not signed:
            continue
        gender_swap_shift[prof_names[prof_id]] = {
            "n":                   len(signed),
            "mean_swap_shift":     round(sum(signed) / len(signed), 4),
            "mean_swap_shift_abs": round(sum(abs_)  / len(abs_),   4),
        }

    # Model-level headline summary (lower = better on all three).
    pronoun_devs   = [abs(v["male_pronoun_share"] - 0.5) for v in pronoun_stereotype.values()]
    swap_shifts    = [r["swap_shift"]     for r in per_bio]
    swap_shift_abs = [r["swap_shift_abs"] for r in per_bio]

    def _mean(xs): return round(sum(xs) / len(xs), 4) if xs else None
    def _ci(xs):
        ci = bootstrap_ci(xs)
        return [ci.get("ci_low"), ci.get("ci_high")]

    summary = {
        "n_bios":            len(per_bio),
        "n_professions":     len(gender_swap_shift),
        "pronoun_skew":      _mean(pronoun_devs),
        "pronoun_skew_ci":   _ci(pronoun_devs),
        "swap_shift_abs":    _mean(swap_shift_abs),
        "swap_shift_abs_ci": _ci(swap_shift_abs),
        "swap_shift":        _mean(swap_shifts),
        "swap_shift_ci":     _ci(swap_shifts),
    }
    return pronoun_stereotype, gender_swap_shift, summary, per_bio


# ── Main ──────────────────────────────────────────────────────────────────────

examples = build_test_examples()

wb_run = wandb_start(
    job_type="eval",
    name="stereotype_invariance",
    config={
        "benchmark":     "Bias-in-Bios (held-out)",
        "probes":        ["pronoun_stereotype", "gender_swap_shift"],
        "n_per_prof":    N_PER_PROF,
        "n_examples":    len(examples),
        "n_professions": len({ex["profession_id"] for ex in examples}),
    },
)

all_results = {}
all_per_bio = []
for model_key in MODELS_TO_EVAL:
    if not os.path.isdir(resolve_model_path(model_key)):
        print(f"[Step 4b] Skipping {model_key} (model dir not found)")
        continue
    pronoun_stereotype, gender_swap_shift, summary, per_bio = eval_model(model_key, examples)
    all_results[model_key] = {
        "pronoun_stereotype": pronoun_stereotype,
        "gender_swap_shift":  gender_swap_shift,
        "summary":            summary,
    }
    all_per_bio.extend(per_bio)

    if wb_run is not None:
        # Flat scalar names so each shows up as a column in the W&B project
        # table and can be charted across runs from the workspace.
        wb_run.summary[f"pronoun_skew/{model_key}"]   = summary["pronoun_skew"]
        wb_run.summary[f"swap_shift_abs/{model_key}"] = summary["swap_shift_abs"]
        wb_run.summary[f"swap_shift/{model_key}"]     = summary["swap_shift"]

# Save combined JSON (step 9 reads this).
out_path = os.path.join(METRICS_DIR, "stereotype_and_invariance.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"[Step 4b] Saved -> {out_path}")

# Per-bio CSV — the only place per-bio raw data lives outside W&B.
df_bios = pd.DataFrame(all_per_bio)
bio_csv = os.path.join(METRICS_DIR, "stereotype_and_invariance_per_bio.csv")
df_bios.to_csv(bio_csv, index=False)
print(f"[Step 4b] Saved -> {bio_csv}")

# W&B Tables: per-bio for drill-down, per-profession (built in-memory) for
# the natural comparison view. No PNG uploads — the workspace can chart
# directly from these tables.
if wb_run is not None:
    import wandb
    prof_rows = []
    for model_key, res in all_results.items():
        profs = set(res["pronoun_stereotype"]) | set(res["gender_swap_shift"])
        for prof in sorted(profs):
            p = res["pronoun_stereotype"].get(prof, {})
            s = res["gender_swap_shift"].get(prof, {})
            prof_rows.append({
                "model":               model_key,
                "profession":          prof,
                "male_pronoun_share":  p.get("male_pronoun_share"),
                "mean_swap_shift":     s.get("mean_swap_shift"),
                "mean_swap_shift_abs": s.get("mean_swap_shift_abs"),
            })
    wb_run.log({
        "per_bio":        wandb.Table(dataframe=df_bios),
        "per_profession": wandb.Table(dataframe=pd.DataFrame(prof_rows)),
    })
wandb_finish(wb_run)


# ── Console summary ───────────────────────────────────────────────────────────
def fmt(v): return f"{v:.4f}" if v is not None else "    N/A"

print("\n=== Held-out gender bias (lower = better on all three) ===")
print(f"{'model':<14} {'n_bios':>8} {'pronoun_skew':>14} "
      f"{'swap_shift_abs':>16} {'swap_shift':>12}")
print("-" * 66)
for model_key, res in all_results.items():
    s = res["summary"]
    print(f"{model_key:<14} {s['n_bios']:>8} "
          f"{fmt(s['pronoun_skew']):>14} "
          f"{fmt(s['swap_shift_abs']):>16} "
          f"{fmt(s['swap_shift']):>12}")
