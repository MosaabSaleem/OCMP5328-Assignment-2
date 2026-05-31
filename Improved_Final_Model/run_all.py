"""
run_all.py
Runs the full assignment pipeline in order:
1. Download data
2. Build CDA pairs
3. Train baseline
4. Train debiased model
5. Probability-based evaluation
6. Embedding-based evaluation
7. Generated-text evaluation
8. Utility evaluation
9. Plot all figures
"""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    "Algorithm/step1_load_data.py",
    "Algorithm/step2_make_cda_pairs.py",
    "Algorithm/step3_train_baseline.py",
    "Algorithm/step4_train_debiased.py",
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
    print("Starting pipeline...")
    print(f"Project root: {ROOT}")
    for step in STEPS:
        run_step(step)
    print("\nAll steps completed successfully.")

if __name__ == "__main__":
    main()