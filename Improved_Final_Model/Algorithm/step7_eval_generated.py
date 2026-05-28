"""
Step 7 — Generated-text-based bias evaluation (WinoBias + BOLD).
WinoBias: compares log-probs of pro- vs anti-stereotype coreference sentences.
BOLD: generates text from prompts and counts gendered terms in the continuation.
"""
import sys
import os
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODEL_DIR, METRICS_DIR, EVAL_SIZE
from Algorithm._model_helpers import load_model, seq_logprob, generate

import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset

MALE_WORDS = {
    "he", "him", "his", "himself", "man", "men", "male",
    "father", "husband", "son", "brother", "boy", "boys"
}
FEMALE_WORDS = {
    "she", "her", "hers", "herself", "woman", "women", "female",
    "mother", "wife", "daughter", "sister", "girl", "girls"
}


def count_gender(text):
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return len(words & MALE_WORDS), len(words & FEMALE_WORDS)


def eval_winobias(mdl, tok, n):
    print(f" [WinoBias] evaluating {n} examples...")
    rows = []

    for cfg in ["type1_pro", "type1_anti"]:
        try:
            ds = load_dataset("uclanlp/wino_bias", cfg, split="test")
        except Exception as e:
            print(f" Could not load {cfg}: {e}")
            continue

        if n // 2 < len(ds):
            ds = ds.select(range(n // 2))

        for ex in tqdm(ds, desc=f"WinoBias {cfg}"):
            tokens = ex.get("tokens") or []
            sent = " ".join(tokens).strip() if tokens else (ex.get("sentence") or "")
            if not sent:
                continue

            rows.append(
                {
                    "sentence": sent,
                    "type": cfg,
                    "lp": seq_logprob(mdl, tok, sent),
                }
            )

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["sentence", "type", "lp"])

    pro_lp = df[df["type"] == "type1_pro"]["lp"].mean() if len(df) else float("nan")
    anti_lp = df[df["type"] == "type1_anti"]["lp"].mean() if len(df) else float("nan")

    summary = {
        "n": len(df),
        "mean_lp_pro_stereotype": float(pro_lp) if str(pro_lp) != "nan" else None,
        "mean_lp_anti_stereotype": float(anti_lp) if str(anti_lp) != "nan" else None,
        "stereotype_logprob_gap": float(pro_lp - anti_lp)
        if (str(pro_lp) != "nan" and str(anti_lp) != "nan")
        else None,
    }
    return df, summary


def eval_bold(mdl, tok, n):
    print(f" [BOLD] generating for {n} prompts...")
    try:
        ds = load_dataset("AmazonScience/bold", split="train")
    except Exception as e:
        print(f" Could not load BOLD: {e}")
        return pd.DataFrame(), {}

    if n < len(ds):
        ds = ds.select(range(n))

    rows = []
    for ex in tqdm(ds, desc="BOLD"):
        raw = ex.get("prompts") or ex.get("prompt") or ex.get("text") or ""
        prompt = raw[0] if isinstance(raw, list) and raw else str(raw)
        if not prompt:
            continue

        gen = generate(mdl, tok, prompt, max_new_tokens=60)
        continuation = gen[len(prompt):].strip()
        m, f = count_gender(continuation)

        rows.append(
            {
                "domain": ex.get("domain", ""),
                "prompt": prompt,
                "generation": continuation,
                "male_count": m,
                "female_count": f,
                "abs_gender_gap": abs(m - f),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "n": len(df),
        "avg_abs_gender_gap": float(df["abs_gender_gap"].mean()) if len(df) else None,
        "total_male_terms": int(df["male_count"].sum()) if len(df) else None,
        "total_female_terms": int(df["female_count"].sum()) if len(df) else None,
    }
    return df, summary


for model_name in ["baseline", "debiased"]:
    print(f"\n[Step 7] Generated-text eval — {model_name}")
    mdl, tok = load_model(os.path.join(MODEL_DIR, model_name))

    df_w, s_w = eval_winobias(mdl, tok, EVAL_SIZE)
    df_w.to_csv(os.path.join(METRICS_DIR, f"{model_name}_winobias.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_winobias_summary.json"), "w") as f:
        json.dump({"model": model_name, "benchmark": "WinoBias", **s_w}, f, indent=2)

    df_b, s_b = eval_bold(mdl, tok, EVAL_SIZE)
    df_b.to_csv(os.path.join(METRICS_DIR, f"{model_name}_bold.csv"), index=False)
    with open(os.path.join(METRICS_DIR, f"{model_name}_bold_summary.json"), "w") as f:
        json.dump({"model": model_name, "benchmark": "BOLD", **s_b}, f, indent=2)

    del mdl, tok
    print(
        f" WinoBias LP gap={s_w.get('stereotype_logprob_gap')} "
        f"BOLD gender gap={s_b.get('avg_abs_gender_gap')}"
    )