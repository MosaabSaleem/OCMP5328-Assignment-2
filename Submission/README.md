# Debiasing Gemma with LoRA + CDA + Counterfactual Logit Pairing

Trains and evaluates the models on `google/gemma-3-1b-pt` (M1 as the LoRA only, M2 as CDA only, M3 as the CDA + CLP debiased model), then produces figures and results.

`Algorithm/` holds the code, `data/` is empty (data downloads here on first run), and `run_all.py` runs the whole pipeline.

## Setup
This was developed and run on a Linux GPU (an NVIDIA T4, ~16 GB) with Python 3.11. A CUDA capable GPU is needed to train and evaluate in reasonable time.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r Algorithm/requirements.txt
```

Gemma is gated, so accept its licence on the model page and export a Hugging Face token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

## Running

> **⚠️ Run the dry run first:** `python run_all.py --dry-run`

This runs nothing, downloads nothing and trains nothing. It byte compiles every step as a syntax check and prints the execution plan, so you can confirm the code is sound before committing to a full run:

```bash
python run_all.py --dry-run
```

Each step is reported as `syntax:OK` or `syntax:FAIL` (with the error location), alongside whether it would run or be skipped. Then run the full pipeline (downloads data, trains, evaluates, plots):

```bash
python run_all.py
```
