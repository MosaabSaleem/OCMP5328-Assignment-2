"""
run_all.py
Runs the full assignment pipeline in order:
0. Set up base_gemma reference (tokenizer + model_reference.json)
1. Download data
2. Build CDA pairs
3. Train baseline (LoRA only)
3b. Train cda_only (LoRA on CDA-augmented data, no CLP)
4. Train debiased (LoRA + CDA + CLP)
4b. In-domain gender bias eval (Bias-in-Bios test split)
5. Probability-based evaluation (CrowS-Pairs, StereoSet)
6. Embedding-based evaluation
7. Generated-text evaluation (WinoBias, BOLD)
8. Utility evaluation (perplexity, generation speed)
9. Plot all figures + log final results to W&B
"""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from Algorithm._wandb_log import ensure_group, enabled as wandb_enabled

# Set the W&B run group once so every step's W&B run (baseline_train,
# cda_only_train, debiased_train, results_summary) is grouped together
# in the dashboard.
if wandb_enabled():
    group = ensure_group()
    print(f"[run_all] W&B group: {group}")

STEPS = [
    "Algorithm/step0_setup_base_gemma.py",
    "Algorithm/step1_load_data.py",
    "Algorithm/step2_make_cda_pairs.py",
    "Algorithm/step3_train_baseline.py",
    "Algorithm/step3b_train_cda_only.py",
    "Algorithm/step4_train_debiased.py",
    "Algorithm/step4b_eval_indomain.py",
    "Algorithm/step5_eval_probability.py",
    "Algorithm/step6_eval_embedding.py",
    "Algorithm/step7_eval_generated.py",
    "Algorithm/step8_eval_utility.py",
    "Algorithm/step9_plot_results.py",
]

def run_step(step_path):
    full_path = os.path.join(ROOT, step_path)
    print("\n" + "=" * 80)
    print(f"Running: {step_path}")
    print("=" * 80)
    result = subprocess.run([sys.executable, full_path], cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_path}")

def main():
    print("Starting full bias-mitigation pipeline...")
    print(f"Project root: {ROOT}")
    for step in STEPS:
        run_step(step)
    print("\nAll steps completed successfully.")

if __name__ == "__main__":
    main()