"""
Step 5 — Probability-based bias evaluation (CrowS-Pairs + StereoSet).
Compares model log-probabilities for stereotypical vs anti-stereotypical sentence pairs.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODEL_DIR, METRICS_DIR, EVAL_SIZE
from Algorithm._model_helpers import load_model, seq_logprob

import requests
import pandas as pd
from io import StringIO
from tqdm.auto import tqdm
from datasets import load_dataset


CROWS_URLS = [
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv",
    "https://huggingface.co/datasets/nyu-mll/crows_pairs/resolve/main/data/crows_pairs_anonymized.csv",
]


def load_crows_pairs_csv():
    last_err = None
    for url in CROWS_URLS:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            print("Loaded CrowS-Pairs from:", url)
            return df
        except Exception as e:
            last_err = e
            print("Failed:", url, "|", e)
    raise RuntimeError(f"Could not load CrowS-Pairs CSV: {last_err}")


def eval_crowspairs(mdl, tok, n):
    print(f" [CrowS-Pairs] evaluating {n} examples...")
    df_src = load_crows_pairs_csv()
    if n < len(df_src):
        df_src = df_src.head(n).copy()

    rows = []
    for _, ex in tqdm(df_src.iterrows(), total=len(df_src), desc="CrowS-Pairs"):
        more = str(ex.get("sent_more", ""))
        less = str(ex.get("sent_less", ""))
        if not more or not less:
            continue

        lp_m = seq_logprob(mdl, tok, more)
        lp_l = seq_logprob(mdl, tok, less)

        rows.append(
            {
                "sent_more": more,
                "sent_less": less,
                "lp_more": lp_m,
                "lp_less": lp_l,
                "prefers_stereotype": int(lp_m > lp_l),
                "logprob_gap": lp_m - lp_l,
                "bias_type": ex.get("bias_type", ""),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "n": len(df),
        "stereotype_preference_rate": float(df["prefers_stereotype"].mean()) if len(df) else None,
        "logprob_gap_mean": float(df["logprob_gap"].mean()) if len(df) else None,
        "logprob_gap_std": float(df["logprob_gap"].std()) if len(df) else None,
    }
    return df, summary


def eval_stereoset(mdl, tok, n):
    print(f" [StereoSet] evaluating {n} examples...")
    try:
        ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
    except Exception:
        ds = load_dataset("McGill-NLP/stereoset", split="validation")

    if n < len(ds):
        ds = ds.select(range(n))

    rows = []
    for ex in tqdm(ds, desc="StereoSet"):
        ctx = str(ex.get("context", "") or "").strip()
        scores = {}

        sents = ex.get("sentences", [])

        if isinstance(sents, dict):
            iterable = sents.values()
        else:
            iterable = sents

        for s in iterable:
            if not isinstance(s, dict):
                continue

            lbl = str(s.get("gold_label", "")).strip()
            sent = str(s.get("sentence", "")).strip()

            if not lbl or not sent:
                continue

            full_text = sent
            if ctx and not sent.startswith(ctx):
                full_text = f"{ctx} {sent}".strip()

            scores[lbl] = seq_logprob(mdl, tok, full_text)

        if "stereotype" in scores and "anti-stereotype" in scores:
            gap = scores["stereotype"] - scores["anti-stereotype"]
            rows.append(
                {
                    "stereo_score": scores["stereotype"],
                    "anti_score": scores["anti-stereotype"],
                    "score_gap": gap,
                    "prefers_stereotype": int(gap > 0),
                }
            )
    df = pd.DataFrame(rows)
    summary = {
        "n": len(df),
        "stereotype_preference_rate": float(df["prefers_stereotype"].mean()) if len(df) else None,
        "score_gap_mean": float(df["score_gap"].mean()) if len(df) else None,
        "score_gap_std": float(df["score_gap"].std()) if len(df) else None,
    }
    return df, summary


for model_name in ["baseline", "debiased"]:
    print(f"\n[Step 5] Probability eval — {model_name}")
    mdl, tok = load_model(os.path.join(MODEL_DIR, model_name))

    df_c, s_c = eval_crowspairs(mdl, tok, EVAL_SIZE)
    df_c.to_csv(os.path.join(METRICS_DIR, f"{model_name}_crowspairs.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_crowspairs_summary.json"), "w") as f:
        json.dump({"model": model_name, "benchmark": "CrowS-Pairs", **s_c}, f, indent=2)

    df_s, s_s = eval_stereoset(mdl, tok, EVAL_SIZE)
    df_s.to_csv(os.path.join(METRICS_DIR, f"{model_name}_stereoset.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_stereoset_summary.json"), "w") as f:
        json.dump({"model": model_name, "benchmark": "StereoSet", **s_s}, f, indent=2)

    del mdl, tok
    print(f" CrowS SPR={s_c['stereotype_preference_rate']} StereoSet SPR={s_s['stereotype_preference_rate']}")