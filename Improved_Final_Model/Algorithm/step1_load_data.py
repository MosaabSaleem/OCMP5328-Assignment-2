"""
Step 1 — Download a subset of Bias in Bios and save to Input_data/.
Bias in Bios contains ~400k professional biographies labelled with occupation
and binary gender. We use it as our training/debiasing dataset.
Ref: De-Arteaga et al., 2019. https://doi.org/10.1145/3287560.3287572
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import DATA_DIR, TRAIN_SAMPLE_SIZE, SEED

from datasets import load_dataset
import pandas as pd

print(f"[Step 1] Downloading Bias-in-Bios  train_sample={TRAIN_SAMPLE_SIZE}  seed={SEED}")
ds = load_dataset("LabHC/bias_in_bios", split="train")
ds = ds.shuffle(seed=SEED)
if TRAIN_SAMPLE_SIZE < len(ds):
    ds = ds.select(range(TRAIN_SAMPLE_SIZE))

cols = ds.column_names

def pick_col(candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

text_col   = pick_col(["hard_text", "text", "bio", "biography"]) or cols[0]
label_col  = pick_col(["profession", "title", "label", "p"])
gender_col = pick_col(["gender", "g"])

data = {"text": [str(x) for x in ds[text_col]]}
if label_col:  data["label"]  = ds[label_col]
if gender_col: data["gender"] = ds[gender_col]

df = pd.DataFrame(data).dropna(subset=["text"])
out_path = os.path.join(DATA_DIR, "bias_in_bios.csv")
df.to_csv(out_path, index=False)
print(f"[Step 1] Saved {len(df)} rows -> {out_path}")
print(df.head(3).to_string())
