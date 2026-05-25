"""
Step 5 — Probability-based bias evaluation (CrowS-Pairs + StereoSet).
Compares model log-probabilities for stereotypical vs anti-stereotypical
sentence pairs. A biased model will assign higher probability to the
stereotypical version more often (higher stereotype_preference_rate).
Covers: Assignment 'probability-based metrics' category.
Refs:
  CrowS-Pairs: Nangia et al., 2020. https://doi.org/10.18653/v1/2020.emnlp-main.154
  StereoSet  : Nadeem et al., 2021. https://doi.org/10.18653/v1/2021.acl-long.416
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODEL_DIR, METRICS_DIR, EVAL_SAMPLE_SIZE
from Algorithm._model_helpers import load_model, seq_logprob
from Algorithm._dataset_loaders import load_crowspairs, load_stereoset_intrasentence

import pandas as pd

BIAS_TYPE = "gender"


def eval_crowspairs(mdl, tok, n):
    """
    For each (sent_more, sent_less) pair, check which the model prefers.
    stereotype_preference_rate = fraction where model prefers sent_more.
    """
    print(f"  [CrowS-Pairs] evaluating up to {n} {BIAS_TYPE} examples...")
    ds = load_crowspairs(n, bias_type=BIAS_TYPE)

    rows = []
    for ex in ds:
        more = ex.get("sent_more") or ""
        less = ex.get("sent_less") or ""
        if not more or not less:
            continue
        lp_m = seq_logprob(mdl, tok, more)
        lp_l = seq_logprob(mdl, tok, less)
        rows.append({
            "sent_more": more, "sent_less": less,
            "lp_more": round(lp_m, 4), "lp_less": round(lp_l, 4),
            "prefers_stereotype": int(lp_m > lp_l),
            "logprob_gap": round(lp_m - lp_l, 4),
            "bias_type": ex.get("bias_type", ""),
        })

    df = pd.DataFrame(rows)
    summary = {
        "n": len(df),
        "bias_type": BIAS_TYPE,
        "stereotype_preference_rate": round(float(df["prefers_stereotype"].mean()), 4) if len(df) else None,
        "logprob_gap_mean": round(float(df["logprob_gap"].mean()), 4) if len(df) else None,
        "logprob_gap_std":  round(float(df["logprob_gap"].std()),  4) if len(df) else None,
    }
    return df, summary


def eval_stereoset(mdl, tok, n):
    """
    Score stereotype vs anti-stereotype sentence completions.
    stereotype_preference_rate = fraction where model scores stereotype higher.
    """
    print(f"  [StereoSet] evaluating up to {n} {BIAS_TYPE} examples...")
    ds = load_stereoset_intrasentence(n, bias_type=BIAS_TYPE)

    rows = []
    for ex in ds:
        ctx   = ex.get("context", "") or ""
        sents = ex.get("sentences") or []
        scores = {}
        for s in sents:
            lbl  = s.get("gold_label")
            sent = s.get("sentence", "")
            if lbl is None or not sent:
                continue
            suffix = sent[len(ctx):].strip() if sent.startswith(ctx) else sent
            scores[str(lbl)] = seq_logprob(mdl, tok, ctx + " " + suffix)

        if "stereotype" in scores and "anti-stereotype" in scores:
            gap = scores["stereotype"] - scores["anti-stereotype"]
            rows.append({
                "stereo_score": round(scores["stereotype"], 4),
                "anti_score":   round(scores["anti-stereotype"], 4),
                "score_gap":    round(gap, 4),
                "prefers_stereotype": int(gap > 0),
                "bias_type": ex.get("bias_type", ""),
            })

    df = pd.DataFrame(rows)
    summary = {
        "n": len(df),
        "bias_type": BIAS_TYPE,
        "stereotype_preference_rate": round(float(df["prefers_stereotype"].mean()), 4) if len(df) else None,
        "score_gap_mean": round(float(df["score_gap"].mean()), 4) if len(df) else None,
        "score_gap_std":  round(float(df["score_gap"].std()),  4) if len(df) else None,
    }
    return df, summary


for model_name in ["baseline", "debiased"]:
    print(f"\n[Step 5] Probability eval — {model_name}")
    mdl, tok = load_model(os.path.join(MODEL_DIR, model_name))

    df_c, s_c = eval_crowspairs(mdl, tok, EVAL_SAMPLE_SIZE)
    df_c.to_csv(os.path.join(METRICS_DIR, f"{model_name}_crowspairs.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_crowspairs_summary.json"), "w") as f:
        json.dump({"model": model_name, "benchmark": "CrowS-Pairs (gender)", **s_c}, f, indent=2)

    df_s, s_s = eval_stereoset(mdl, tok, EVAL_SAMPLE_SIZE)
    df_s.to_csv(os.path.join(METRICS_DIR, f"{model_name}_stereoset.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_stereoset_summary.json"), "w") as f:
        json.dump({"model": model_name, "benchmark": "StereoSet (gender)", **s_s}, f, indent=2)

    del mdl, tok
    print(f"  CrowS SPR={s_c['stereotype_preference_rate']}  StereoSet SPR={s_s['stereotype_preference_rate']}")
