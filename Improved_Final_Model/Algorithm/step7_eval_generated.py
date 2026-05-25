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
from Algorithm._dataset_loaders import load_winobias_type1, load_bold_gender

import pandas as pd

MALE_WORDS   = {"he","him","his","himself","man","men","male",
                "father","husband","son","brother","boy","boys"}
FEMALE_WORDS = {"she","her","hers","herself","woman","women","female",
                "mother","wife","daughter","sister","girl","girls"}

def count_gender(text):
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return len(words & MALE_WORDS), len(words & FEMALE_WORDS)


def eval_winobias(mdl, tok, n):
    print(f"  [WinoBias] evaluating {n} examples...")
    examples = load_winobias_type1(n // 2)
    rows = []
    for ex in examples:
        sent = ex["sentence"]
        lp = seq_logprob_stats(mdl, tok, sent)
        rows.append({
            "sentence": sent,
            "type": ex["type"],
            "lp": round(lp["sum"], 4),
            "lp_avg": round(lp["avg"], 4),
            "token_count": lp["token_count"],
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["sentence","type","lp","lp_avg","token_count"])
    pro_lp  = df[df["type"]=="type1_pro"]["lp"].mean()  if len(df) else float("nan")
    anti_lp = df[df["type"]=="type1_anti"]["lp"].mean() if len(df) else float("nan")
    pro_lp_avg  = df[df["type"]=="type1_pro"]["lp_avg"].mean()  if len(df) else float("nan")
    anti_lp_avg = df[df["type"]=="type1_anti"]["lp_avg"].mean() if len(df) else float("nan")
    summary = {
        "n": len(df),
        "mean_lp_pro_stereotype":   round(float(pro_lp),  4) if str(pro_lp)  != "nan" else None,
        "mean_lp_anti_stereotype":  round(float(anti_lp), 4) if str(anti_lp) != "nan" else None,
        "stereotype_logprob_gap":   round(float(pro_lp - anti_lp), 4) if (str(pro_lp) != "nan" and str(anti_lp) != "nan") else None,
        "mean_lp_avg_pro_stereotype":   round(float(pro_lp_avg),  4) if str(pro_lp_avg)  != "nan" else None,
        "mean_lp_avg_anti_stereotype":  round(float(anti_lp_avg), 4) if str(anti_lp_avg) != "nan" else None,
        "stereotype_logprob_gap_avg":   round(float(pro_lp_avg - anti_lp_avg), 4) if (str(pro_lp_avg) != "nan" and str(anti_lp_avg) != "nan") else None,
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
            "abs_gender_gap": abs(m - f),
        })

    df = pd.DataFrame(rows)
    summary = {
        "n": len(df),
        "avg_abs_gender_gap":  round(float(df["abs_gender_gap"].mean()),  4) if len(df) else None,
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
