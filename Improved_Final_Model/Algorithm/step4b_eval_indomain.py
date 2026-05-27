"""
Step 5b — In-domain gender bias evaluation on Bias-in-Bios test split.

For each biography in the held-out test set, we locate the first gendered
pronoun (he/she), feed the preceding prefix to the model, and compare
P(" he" | prefix) vs P(" she" | prefix).

Metric per profession:
  male_bias = P(he) / (P(he) + P(she))   (0.5 = neutral, >0.5 = male-biased)

This directly shows whether CDA+CLP reduced the model's tendency to predict
gender based on occupation — evaluated in the same domain it was trained on.
Uses the dataset's predefined test split (never seen during training).

Ref: De-Arteaga et al., 2019. https://doi.org/10.1145/3287560.3287572
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import MODELS_TO_EVAL, METRICS_DIR, PROFESSION_LABELS

import torch
from datasets import load_dataset
from Algorithm._model_helpers import load_model, resolve_model_path, DEVICE

PRONOUN_RE  = re.compile(r'\b(he|she)\b', re.IGNORECASE)
N_PER_PROF  = 50   # bios sampled per profession from the test split


def find_prefix(text: str):
    """Return (prefix, pronoun) at the first gendered pronoun, or (None, None)."""
    m = PRONOUN_RE.search(text)
    if m is None:
        return None, None
    return text[:m.start()].strip(), m.group(0).lower()


@torch.no_grad()
def pronoun_probs(mdl, tok, prefix: str):
    """
    Return (p_he, p_she) as next-token probabilities after the prefix.
    Uses the space-prefixed forms (' he', ' she') which are the typical
    mid-sentence token representations for Gemma.
    """
    he_id  = tok.encode(" he",  add_special_tokens=False)[0]
    she_id = tok.encode(" she", add_special_tokens=False)[0]
    enc = tok(prefix, return_tensors="pt", truncation=True,
              max_length=200).to(DEVICE)
    if enc["input_ids"].shape[1] == 0:
        return None, None
    logits = mdl(**enc).logits[0, -1, :]
    probs  = torch.softmax(logits, dim=-1)
    return float(probs[he_id]), float(probs[she_id])


def build_test_examples():
    """
    Load Bias-in-Bios test split, extract up to N_PER_PROF usable bios
    (those with a gendered pronoun) per profession.
    Returns list of {profession_id, profession_name, prefix, true_gender}.
    """
    print("[Step 4b] Loading Bias-in-Bios test split...")
    ds = load_dataset("LabHC/bias_in_bios", split="test")

    from collections import defaultdict
    buckets = defaultdict(list)
    for row in ds:
        prof = int(row["profession"])
        if len(buckets[prof]) >= N_PER_PROF:
            continue
        prefix, pronoun = find_prefix(row["hard_text"])
        if prefix and len(prefix.split()) >= 4:
            buckets[prof].append({
                "profession_id":   prof,
                "profession_name": PROFESSION_LABELS.get(prof, str(prof)),
                "prefix":          prefix,
                "true_gender":     int(row["gender"]),  # 0=male 1=female
            })

    examples = [ex for exs in buckets.values() for ex in exs]
    print(f"[Step 4b] {len(examples)} usable bios across "
          f"{len(buckets)} professions ({N_PER_PROF} max/profession)")
    return examples


def eval_model(model_key: str, examples: list):
    """Run pronoun probability eval for one model. Returns per-profession stats."""
    print(f"[Step 4b] Evaluating {model_key}...")
    mdl, tok = load_model(resolve_model_path(model_key))

    from collections import defaultdict
    prof_results = defaultdict(lambda: {"he": [], "she": [], "name": ""})

    for ex in examples:
        p_he, p_she = pronoun_probs(mdl, tok, ex["prefix"])
        if p_he is None:
            continue
        total = p_he + p_she
        if total < 1e-12:
            continue
        prof = ex["profession_id"]
        prof_results[prof]["he"].append(p_he)
        prof_results[prof]["she"].append(p_she)
        prof_results[prof]["name"] = ex["profession_name"]

    del mdl
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    summary = {}
    for prof_id, data in sorted(prof_results.items()):
        n = len(data["he"])
        if n == 0:
            continue
        avg_he  = sum(data["he"])  / n
        avg_she = sum(data["she"]) / n
        male_bias = avg_he / (avg_he + avg_she)   # 0.5 = neutral
        summary[data["name"]] = {
            "n":          n,
            "avg_p_he":   round(avg_he,    4),
            "avg_p_she":  round(avg_she,   4),
            "male_bias":  round(male_bias, 4),   # >0.5 = male-skewed
        }
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

examples = build_test_examples()

all_results = {}
for model_key in MODELS_TO_EVAL:
    model_path = resolve_model_path(model_key)
    if not os.path.isdir(model_path):
        print(f"[Step 4b] Skipping {model_key} (not found at {model_path})")
        continue
    all_results[model_key] = eval_model(model_key, examples)

# Save full results
out_path = os.path.join(METRICS_DIR, "indomain_gender_bias.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"[Step 4b] Saved -> {out_path}")

# Print summary table: male_bias per profession per model
print("\n=== In-domain Male Bias (0.5 = neutral) ===")
professions = sorted({p for res in all_results.values() for p in res})
header = f"{'Profession':<22}" + "".join(f"{m:>14}" for m in all_results)
print(header)
print("-" * len(header))
for prof in professions:
    row = f"{prof:<22}"
    for model_key in all_results:
        val = all_results[model_key].get(prof, {}).get("male_bias")
        row += f"{'N/A':>14}" if val is None else f"{val:>14.3f}"
    print(row)

# Overall mean absolute deviation from 0.5 per model (lower = less biased)
print("\n=== Mean |bias - 0.5| per model (lower = less biased) ===")
for model_key, res in all_results.items():
    biases = [v["male_bias"] for v in res.values()]
    mean_dev = sum(abs(b - 0.5) for b in biases) / len(biases) if biases else 0
    print(f"  {model_key:<20} {mean_dev:.4f}")
