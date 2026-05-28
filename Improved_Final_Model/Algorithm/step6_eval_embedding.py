"""
Step 6 — Embedding-based bias evaluation (anisotropy-corrected).

For each CrowS-Pairs gender pair (stereotype, anti-stereotype), we mean-pool
the model's last hidden state and compute the cosine similarity between the
two sentence embeddings. A more gender-invariant model should produce MORE
similar representations for paired sentences.

Anisotropy correction
---------------------
Transformer LMs (including Gemma) exhibit a well-documented "cone effect":
contextual embeddings cluster in a narrow region of the space, so raw cosine
similarities saturate near 1.0 across ALL pairs and the actual debiasing
signal is invisible (Ethayarajh, 2019; Mu & Viswanath, 2018).

We address this by subtracting the per-model mean embedding across all eval
sentences before computing cosine:

    cos_centered(v1, v2) = cos(v1 - mu, v2 - mu)

Both raw and centered cosine are reported. The centered value is the headline
metric; raw is kept so the report can demonstrate the cone effect.

Covers: Assignment 'embedding-based metrics' category.
Refs:
  Anisotropy: Ethayarajh, 2019.  https://doi.org/10.18653/v1/D19-1006
              Mu & Viswanath, 2018. https://openreview.net/forum?id=HkuGJ3kCb
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import METRICS_DIR, EVAL_SAMPLE_SIZE, MODELS_TO_EVAL
from Algorithm._model_helpers import load_model, last_hidden, resolve_model_path
from Algorithm._dataset_loaders import load_crowspairs
from Algorithm._stats import bootstrap_ci

import numpy as np
import pandas as pd

BIAS_TYPE = "gender"


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))


# Held-out CrowS-Pairs gender examples (not used for CDA training).
df_pairs = pd.DataFrame(load_crowspairs(EVAL_SAMPLE_SIZE, bias_type=BIAS_TYPE)).dropna(
    subset=["sent_more", "sent_less"]
)

for model_name in MODELS_TO_EVAL:
    print(f"\n[Step 6] Embedding eval — {model_name}  ({len(df_pairs)} pairs)")
    mdl, tok = load_model(resolve_model_path(model_name))

    # Pass 1: collect all sentence embeddings so we can compute the
    # per-model mean and subtract it before scoring.
    embeds_more, embeds_less = [], []
    for _, r in df_pairs.iterrows():
        embeds_more.append(last_hidden(mdl, tok, r["sent_more"]))
        embeds_less.append(last_hidden(mdl, tok, r["sent_less"]))
    embeds_more = np.stack(embeds_more)
    embeds_less = np.stack(embeds_less)
    mu = np.concatenate([embeds_more, embeds_less], axis=0).mean(axis=0)

    rows = []
    for i, (_, r) in enumerate(df_pairs.iterrows()):
        v1, v2 = embeds_more[i], embeds_less[i]
        v1c, v2c = v1 - mu, v2 - mu
        cos_raw      = cosine(v1, v2)
        cos_centered = cosine(v1c, v2c)
        rows.append({
            "sent_more": r["sent_more"],
            "sent_less": r["sent_less"],
            "bias_type": r.get("bias_type", BIAS_TYPE),
            # Headline metric is centered (anisotropy-corrected).
            "cosine_similarity":      round(cos_centered, 6),
            "cosine_distance":        round(1.0 - cos_centered, 6),
            # Raw is reported alongside so the cone-effect symptom is visible.
            "cosine_similarity_raw":  round(cos_raw, 6),
            "cosine_distance_raw":    round(1.0 - cos_raw, 6),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(METRICS_DIR, f"{model_name}_embedding.csv"), index=False)

    sim_ci      = bootstrap_ci(df_out["cosine_similarity"])     if len(df_out) else {}
    dist_ci     = bootstrap_ci(df_out["cosine_distance"])       if len(df_out) else {}
    sim_raw_ci  = bootstrap_ci(df_out["cosine_similarity_raw"]) if len(df_out) else {}
    summary = {
        "model":     model_name,
        "benchmark": "Embedding cosine (CrowS-Pairs gender)",
        "n":         len(df_out),
        "bias_type": BIAS_TYPE,
        "anisotropy_correction": "mean-centered across all eval embeddings (per model)",
        # Centered (headline)
        "mean_cosine_similarity":        round(float(df_out["cosine_similarity"].mean()), 4),
        "mean_cosine_similarity_ci":     [sim_ci.get("ci_low"), sim_ci.get("ci_high")],
        "std_cosine_similarity":         round(float(df_out["cosine_similarity"].std()), 4),
        "mean_cosine_distance":          round(float(df_out["cosine_distance"].mean()), 4),
        "mean_cosine_distance_ci":       [dist_ci.get("ci_low"), dist_ci.get("ci_high")],
        # Raw (kept to show the cone effect)
        "mean_cosine_similarity_raw":    round(float(df_out["cosine_similarity_raw"].mean()), 4),
        "mean_cosine_similarity_raw_ci": [sim_raw_ci.get("ci_low"), sim_raw_ci.get("ci_high")],
    }
    with open(os.path.join(METRICS_DIR, f"{model_name}_embedding_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    del mdl, tok
    print(
        f"  centered cos sim={summary['mean_cosine_similarity']}  "
        f"raw cos sim={summary['mean_cosine_similarity_raw']}"
    )
