"""
Step 1 — Download a subset of Bias in Bios and save to Input_data/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from Algorithm.config import DATA_DIR, SEED, TRAIN_SAMPLE_SIZE, METRICS_DIR
from datasets import load_dataset

print(
    f"[Step 1] Downloading Bias-in-Bios  train_sample={TRAIN_SAMPLE_SIZE}  seed={SEED}"
)
ds = load_dataset("LabHC/bias_in_bios", split="train")

cols = ds.column_names

def pick_col(candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

text_col = pick_col(["hard_text", "text", "bio", "biography"]) or cols[0]
label_col = pick_col(["profession", "title", "label", "p"])
gender_col = pick_col(["gender", "g"])

from Algorithm.config import PROFESSION_LABELS

GENDER_LABELS = {0: "male", 1: "female"}

data = {"text": [str(x) for x in ds[text_col]]}
if label_col:
    data["label"] = ds[label_col]
    data["profession"] = [PROFESSION_LABELS.get(p, str(p)) for p in ds[label_col]]
if gender_col:
    data["gender"] = [GENDER_LABELS.get(g, str(g)) for g in ds[gender_col]]

df = pd.DataFrame(data).dropna(subset=["text"])

out_data = os.path.join(DATA_DIR, "bias_in_bios.csv")
out_metrics = os.path.join(METRICS_DIR, "bias_in_bios_sample.csv")

# Stratified sample by profession so no single occupation dominates training.
# Each profession gets an equal share; professions with fewer rows than the
# per-class quota contribute all their rows.
if TRAIN_SAMPLE_SIZE < len(df) and label_col:
    n_classes = df["label"].nunique()
    per_class = max(1, TRAIN_SAMPLE_SIZE // n_classes)
    df = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_class), random_state=SEED))
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    print(f"[Step 1] Stratified to {per_class} rows/profession → {len(df)} total rows")
else:
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
out_path = os.path.join(DATA_DIR, "bias_in_bios.csv")
df.to_csv(out_path, index=False)
print(f"[Step 1] Saved {len(df)} rows -> {out_path}")
print(df.head(3).to_string())
