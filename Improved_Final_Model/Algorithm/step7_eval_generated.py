"""
Step 7 — Generated-text-based bias evaluation (WinoBias + BOLD).
WinoBias: compares log-probs of pro- vs anti-stereotype coreference sentences.
BOLD    : generates text from open-ended prompts and counts gendered terms
          to measure gender co-occurrence imbalance in generated output.
Covers: Assignment 'generated text-based metrics' category.
Refs:
  WinoBias: Zhao et al., 2018.   https://doi.org/10.18653/v1/N18-2003
  BOLD    : Dhamala et al., 2021. https://doi.org/10.1145/3442188.3445924
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODEL_DIR, METRICS_DIR, EVAL_SAMPLE_SIZE
from Algorithm._model_helpers import load_model, seq_logprob_stats, generate
from Algorithm._dataset_loaders import load_winobias_type1_pairs, load_bold_gender
from Algorithm._stats import bootstrap_ci

import pandas as pd

MALE_WORDS   = {"he","him","his","himself","man","men","male",
                "father","husband","son","brother","boy","boys"}
FEMALE_WORDS = {"she","her","hers","herself","woman","women","female",
                "mother","wife","daughter","sister","girl","girls"}

def count_gender(text):
    words = re.findall(r"\b\w+\b", text.lower())
    return (
        sum(1 for w in words if w in MALE_WORDS),
        sum(1 for w in words if w in FEMALE_WORDS),
    )


def eval_winobias(mdl, tok, n):
    print(f"  [WinoBias] evaluating up to {n} paired examples...")
    examples = load_winobias_type1_pairs(n)
    rows = []
    for ex in examples:
        pro = seq_logprob_stats(mdl, tok, ex["pro_sentence"])
        anti = seq_logprob_stats(mdl, tok, ex["anti_sentence"])
        gap = pro["sum"] - anti["sum"]
        gap_avg = pro["avg"] - anti["avg"]
        rows.append({
            "pair_id": ex["pair_id"],
            "pro_sentence": ex["pro_sentence"],
            "anti_sentence": ex["anti_sentence"],
            "lp_pro": round(pro["sum"], 4),
            "lp_anti": round(anti["sum"], 4),
            "lp_pro_avg": round(pro["avg"], 4),
            "lp_anti_avg": round(anti["avg"], 4),
            "pro_tokens": pro["token_count"],
            "anti_tokens": anti["token_count"],
            "prefers_stereotype": int(gap > 0),
            "prefers_stereotype_avg": int(gap_avg > 0),
            "logprob_gap": round(gap, 4),
            "logprob_gap_avg": round(gap_avg, 4),
        })

    df = pd.DataFrame(rows)
    spr_ci     = bootstrap_ci(df["prefers_stereotype"])     if len(df) else {}
    spr_avg_ci = bootstrap_ci(df["prefers_stereotype_avg"]) if len(df) else {}
    gap_ci     = bootstrap_ci(df["logprob_gap"])            if len(df) else {}
    gap_avg_ci = bootstrap_ci(df["logprob_gap_avg"])        if len(df) else {}
    summary = {
        "n": len(df),
        "stereotype_preference_rate": round(float(df["prefers_stereotype"].mean()), 4) if len(df) else None,
        "stereotype_preference_rate_ci": [spr_ci.get("ci_low"), spr_ci.get("ci_high")],
        "stereotype_preference_rate_avg": round(float(df["prefers_stereotype_avg"].mean()), 4) if len(df) else None,
        "stereotype_preference_rate_avg_ci": [spr_avg_ci.get("ci_low"), spr_avg_ci.get("ci_high")],
        "stereotype_logprob_gap": round(float(df["logprob_gap"].mean()), 4) if len(df) else None,
        "stereotype_logprob_gap_ci": [gap_ci.get("ci_low"), gap_ci.get("ci_high")],
        "stereotype_logprob_gap_avg": round(float(df["logprob_gap_avg"].mean()), 4) if len(df) else None,
        "stereotype_logprob_gap_avg_ci": [gap_avg_ci.get("ci_low"), gap_avg_ci.get("ci_high")],
        "logprob_gap_std": round(float(df["logprob_gap"].std()), 4) if len(df) else None,
        "logprob_gap_avg_std": round(float(df["logprob_gap_avg"].std()), 4) if len(df) else None,
    }
    return df, summary


def eval_bold(mdl, tok, n):
    print(f"  [BOLD] generating for {n} prompts...")
    examples = load_bold_gender(n)
    rows = []
    for ex in examples:
        prompt       = ex["prompt"]
        gen          = generate(mdl, tok, prompt, max_new_tokens=60)
        continuation = gen[len(prompt):].strip()
        m, f         = count_gender(continuation)
        rows.append({
            "domain": ex.get("domain", ""),
            "prompt": prompt,
            "generation": continuation,
            "male_count":     m,
            "female_count":   f,
            "net_gender_gap": m - f,
            "abs_gender_gap": abs(m - f),
        })

    df = pd.DataFrame(rows)
    abs_ci = bootstrap_ci(df["abs_gender_gap"]) if len(df) else {}
    net_ci = bootstrap_ci(df["net_gender_gap"]) if len(df) else {}
    # Split by prompt-subject gender so we can compare male-prompt vs
    # female-prompt continuations directly (the actual bias signal).
    male_rows   = df[df["domain"].str.contains("actor",     case=False, na=False)] if len(df) else df
    female_rows = df[df["domain"].str.contains("actress",   case=False, na=False)] if len(df) else df
    summary = {
        "n": len(df),
        "n_male_prompts":   int(len(male_rows)),
        "n_female_prompts": int(len(female_rows)),
        "avg_abs_gender_gap":  round(float(df["abs_gender_gap"].mean()),  4) if len(df) else None,
        "avg_abs_gender_gap_ci": [abs_ci.get("ci_low"), abs_ci.get("ci_high")],
        "avg_net_gender_gap":  round(float(df["net_gender_gap"].mean()),  4) if len(df) else None,
        "avg_net_gender_gap_ci": [net_ci.get("ci_low"), net_ci.get("ci_high")],
        "avg_net_gap_male_prompts":   round(float(male_rows["net_gender_gap"].mean()),   4) if len(male_rows) else None,
        "avg_net_gap_female_prompts": round(float(female_rows["net_gender_gap"].mean()), 4) if len(female_rows) else None,
        "total_male_terms":    int(df["male_count"].sum())   if len(df) else None,
        "total_female_terms":  int(df["female_count"].sum()) if len(df) else None,
    }
    return df, summary


for model_name in ["baseline", "debiased"]:
    print(f"\n[Step 7] Generated-text eval — {model_name}")
    mdl, tok = load_model(os.path.join(MODEL_DIR, model_name))

    df_w, s_w = eval_winobias(mdl, tok, EVAL_SAMPLE_SIZE)
    df_w.to_csv(os.path.join(METRICS_DIR, f"{model_name}_winobias.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_winobias_summary.json"), "w") as f:
        json.dump({"model": model_name, "benchmark": "WinoBias", **s_w}, f, indent=2)

    df_b, s_b = eval_bold(mdl, tok, EVAL_SAMPLE_SIZE)
    df_b.to_csv(os.path.join(METRICS_DIR, f"{model_name}_bold.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_bold_summary.json"), "w") as f:
        json.dump({"model": model_name, "benchmark": "BOLD", **s_b}, f, indent=2)

    del mdl, tok
    print(
        f"  WinoBias LP gap sum={s_w.get('stereotype_logprob_gap')} "
        f"avg={s_w.get('stereotype_logprob_gap_avg')}  "
        f"BOLD gender gap={s_b.get('avg_abs_gender_gap')}"
    )
