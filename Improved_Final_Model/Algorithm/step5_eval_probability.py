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
from Algorithm.config import METRICS_DIR, EVAL_SAMPLE_SIZE, MODELS_TO_EVAL
from Algorithm._model_helpers import load_model, seq_logprob_stats, resolve_model_path
from Algorithm._dataset_loaders import load_crowspairs, load_stereoset_intrasentence
from Algorithm._stats import bootstrap_ci

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
        lp_m = seq_logprob_stats(mdl, tok, more)
        lp_l = seq_logprob_stats(mdl, tok, less)
        gap_sum = lp_m["sum"] - lp_l["sum"]
        gap_avg = lp_m["avg"] - lp_l["avg"]
        rows.append({
            "sent_more": more, "sent_less": less,
            "lp_more": round(lp_m["sum"], 4), "lp_less": round(lp_l["sum"], 4),
            "lp_more_avg": round(lp_m["avg"], 4), "lp_less_avg": round(lp_l["avg"], 4),
            "lp_more_tokens": lp_m["token_count"], "lp_less_tokens": lp_l["token_count"],
            "prefers_stereotype": int(gap_sum > 0),
            "prefers_stereotype_avg": int(gap_avg > 0),
            "logprob_gap": round(gap_sum, 4),
            "logprob_gap_avg": round(gap_avg, 4),
            "bias_type": ex.get("bias_type", ""),
        })

    df = pd.DataFrame(rows)
    spr_ci      = bootstrap_ci(df["prefers_stereotype"])     if len(df) else {}
    spr_avg_ci  = bootstrap_ci(df["prefers_stereotype_avg"]) if len(df) else {}
    gap_ci      = bootstrap_ci(df["logprob_gap"])            if len(df) else {}
    gap_avg_ci  = bootstrap_ci(df["logprob_gap_avg"])        if len(df) else {}
    summary = {
        "n": len(df),
        "bias_type": BIAS_TYPE,
        "stereotype_preference_rate": round(float(df["prefers_stereotype"].mean()), 4) if len(df) else None,
        "stereotype_preference_rate_ci": [spr_ci.get("ci_low"), spr_ci.get("ci_high")],
        "stereotype_preference_rate_avg": round(float(df["prefers_stereotype_avg"].mean()), 4) if len(df) else None,
        "stereotype_preference_rate_avg_ci": [spr_avg_ci.get("ci_low"), spr_avg_ci.get("ci_high")],
        "logprob_gap_mean": round(float(df["logprob_gap"].mean()), 4) if len(df) else None,
        "logprob_gap_mean_ci": [gap_ci.get("ci_low"), gap_ci.get("ci_high")],
        "logprob_gap_std":  round(float(df["logprob_gap"].std()),  4) if len(df) else None,
        "logprob_gap_avg_mean": round(float(df["logprob_gap_avg"].mean()), 4) if len(df) else None,
        "logprob_gap_avg_mean_ci": [gap_avg_ci.get("ci_low"), gap_avg_ci.get("ci_high")],
        "logprob_gap_avg_std":  round(float(df["logprob_gap_avg"].std()),  4) if len(df) else None,
    }
    return df, summary


def eval_stereoset(mdl, tok, n):
    """
    Score stereotype, anti-stereotype, and unrelated sentence completions.
    Reports the canonical StereoSet triplet:
      SS   (Stereotype Score)        = fraction where stereotype > anti-stereotype
      LMS  (Language Modelling Score) = fraction where max(stereo, anti) > unrelated
      ICAT = LMS * min(SS, 1-SS) * 2   (0 = bad, 1 = perfectly unbiased + meaningful)
    All three are needed: SS alone can be lowered by destroying language modelling,
    which ICAT penalises.
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
            # StereoSet intrasentence candidates are already full sentences
            # with the BLANK filled, so score the candidate sentence directly.
            scores[str(lbl)] = seq_logprob_stats(mdl, tok, sent)

        if "stereotype" in scores and "anti-stereotype" in scores:
            gap = scores["stereotype"]["sum"] - scores["anti-stereotype"]["sum"]
            gap_avg = scores["stereotype"]["avg"] - scores["anti-stereotype"]["avg"]
            row = {
                "stereo_score": round(scores["stereotype"]["sum"], 4),
                "anti_score":   round(scores["anti-stereotype"]["sum"], 4),
                "stereo_score_avg": round(scores["stereotype"]["avg"], 4),
                "anti_score_avg":   round(scores["anti-stereotype"]["avg"], 4),
                "stereo_tokens": scores["stereotype"]["token_count"],
                "anti_tokens":   scores["anti-stereotype"]["token_count"],
                "score_gap":    round(gap, 4),
                "score_gap_avg": round(gap_avg, 4),
                "prefers_stereotype": int(gap > 0),
                "prefers_stereotype_avg": int(gap_avg > 0),
                "bias_type": ex.get("bias_type", ""),
            }
            if "unrelated" in scores:
                row["unrelated_score"] = round(scores["unrelated"]["sum"], 4)
                row["unrelated_score_avg"] = round(scores["unrelated"]["avg"], 4)
                # LMS uses per-token-averaged log-prob so the comparison is not
                # dominated by sentence length (unrelated sentences are often longer).
                row["prefers_meaningful"] = int(
                    max(scores["stereotype"]["avg"], scores["anti-stereotype"]["avg"])
                    > scores["unrelated"]["avg"]
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    has_unrelated = "prefers_meaningful" in df.columns and df["prefers_meaningful"].notna().any()
    ss     = float(df["prefers_stereotype"].mean()) if len(df) else None
    lms    = float(df["prefers_meaningful"].mean()) if has_unrelated else None
    icat   = (lms * (1.0 - abs(2 * ss - 1.0))) if (ss is not None and lms is not None) else None
    ss_ci      = bootstrap_ci(df["prefers_stereotype"])     if len(df) else {}
    ss_avg_ci  = bootstrap_ci(df["prefers_stereotype_avg"]) if len(df) else {}
    lms_ci     = bootstrap_ci(df["prefers_meaningful"])     if has_unrelated else {}
    gap_ci     = bootstrap_ci(df["score_gap"])              if len(df) else {}
    gap_avg_ci = bootstrap_ci(df["score_gap_avg"])          if len(df) else {}
    summary = {
        "n": len(df),
        "bias_type": BIAS_TYPE,
        "stereotype_preference_rate": round(ss, 4) if ss is not None else None,
        "stereotype_preference_rate_ci": [ss_ci.get("ci_low"), ss_ci.get("ci_high")],
        "stereotype_preference_rate_avg": round(float(df["prefers_stereotype_avg"].mean()), 4) if len(df) else None,
        "stereotype_preference_rate_avg_ci": [ss_avg_ci.get("ci_low"), ss_avg_ci.get("ci_high")],
        "lms_language_modeling_score": round(lms, 4) if lms is not None else None,
        "lms_ci": [lms_ci.get("ci_low"), lms_ci.get("ci_high")] if has_unrelated else [None, None],
        "icat_score": round(icat, 4) if icat is not None else None,
        "score_gap_mean": round(float(df["score_gap"].mean()), 4) if len(df) else None,
        "score_gap_mean_ci": [gap_ci.get("ci_low"), gap_ci.get("ci_high")],
        "score_gap_std":  round(float(df["score_gap"].std()),  4) if len(df) else None,
        "score_gap_avg_mean": round(float(df["score_gap_avg"].mean()), 4) if len(df) else None,
        "score_gap_avg_mean_ci": [gap_avg_ci.get("ci_low"), gap_avg_ci.get("ci_high")],
        "score_gap_avg_std":  round(float(df["score_gap_avg"].std()),  4) if len(df) else None,
    }
    return df, summary


for model_name in MODELS_TO_EVAL:
    print(f"\n[Step 5] Probability eval — {model_name}")
    mdl, tok = load_model(resolve_model_path(model_name))

    df_c, s_c = eval_crowspairs(mdl, tok, EVAL_SAMPLE_SIZE)
    df_c.to_csv(os.path.join(METRICS_DIR, f"{model_name}_crowspairs.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_crowspairs_summary.json"), "w") as f:
        json.dump({"model": model_name, "benchmark": "CrowS-Pairs (gender)", **s_c}, f, indent=2)

    df_s, s_s = eval_stereoset(mdl, tok, EVAL_SAMPLE_SIZE)
    df_s.to_csv(os.path.join(METRICS_DIR, f"{model_name}_stereoset.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_stereoset_summary.json"), "w") as f:
        json.dump({"model": model_name, "benchmark": "StereoSet (gender)", **s_s}, f, indent=2)

    del mdl, tok
    print(
        f"  CrowS SPR sum={s_c['stereotype_preference_rate']} avg={s_c['stereotype_preference_rate_avg']}  "
        f"StereoSet SPR sum={s_s['stereotype_preference_rate']} avg={s_s['stereotype_preference_rate_avg']}"
    )
