"""
Step 1 — Download a subset of Bias in Bios and save it for the pipeline.
This version matches the working notebook more closely by also saving
a sample copy into results/metrics.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import DATA_DIR, METRICS_DIR, SAMPLE_SIZE, SEED

from datasets import load_dataset
import pandas as pd

print(f"[Step 1] Downloading Bias-in-Bios sample={SAMPLE_SIZE} seed={SEED}")

ds = load_dataset("LabHC/bias_in_bios", split="train")
ds = ds.shuffle(seed=SEED)
if SAMPLE_SIZE < len(ds):
    ds = ds.select(range(SAMPLE_SIZE))

cols = ds.column_names


def pick_col(candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


text_col = pick_col(["hard_text", "text", "bio", "biography"]) or cols[0]
label_col = pick_col(["profession", "title", "label", "p"])
gender_col = pick_col(["gender", "g"])

df = pd.DataFrame(
    {
        "text": [str(x) for x in ds[text_col]],
        "label": ds[label_col] if label_col else ["unknown"] * len(ds),
        "gender": ds[gender_col] if gender_col else ["unknown"] * len(ds),
    }
).dropna(subset=["text"])

out_data = os.path.join(DATA_DIR, "bias_in_bios.csv")
out_metrics = os.path.join(METRICS_DIR, "bias_in_bios_sample.csv")

df.to_csv(out_data, index=False)
df.to_csv(out_metrics, index=False)

print(f"[Step 1] Saved {len(df)} rows -> {out_data}")
print(f"[Step 1] Saved notebook-style sample -> {out_metrics}")
print(df.head(3).to_string())