"""
run_all.py — run the full debiasing pipeline in order.

Steps 0-9: set up the base model, download data, build CDA pairs, train the
three LoRA models (baseline / cda_only / debiased), run the bias and utility
evaluations, and plot the figures.

Each training step skips itself if its LoRA adapter already exists
(set FORCE_RETRAIN=1 to retrain from scratch).

Use --dry-run to byte-compile every step (a syntax check) and print the
execution plan, without running, downloading, or training anything.
"""

import argparse
import os
import py_compile
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from Algorithm._wandb_log import ensure_group, enabled as wandb_enabled

STEPS = [
    "Algorithm/step0_setup_base_gemma.py",
    "Algorithm/step1_load_data.py",
    "Algorithm/step2_make_cda_pairs.py",
    "Algorithm/step3_train_baseline.py",
    "Algorithm/step3b_train_cda_only.py",
    "Algorithm/step4_train_debiased.py",
    "Algorithm/step4b_eval_bib_test.py",
    "Algorithm/step5_eval_probability.py",
    "Algorithm/step6_eval_embedding.py",
    "Algorithm/step7_eval_generated.py",
    "Algorithm/step8_eval_utility.py",
    "Algorithm/step9_plot_results.py",
]

# Training steps that skip when their adapter already exists (model dir name).
TRAIN_GUARDS = {
    "Algorithm/step3_train_baseline.py": "baseline",
    "Algorithm/step3b_train_cda_only.py": "cda_only",
    "Algorithm/step4_train_debiased.py": "debiased",
}


def _adapter_exists(name):
    path = os.path.join(ROOT, "results", "models", name)
    return os.path.isfile(os.path.join(path, "adapter_config.json")) and (
        os.path.isfile(os.path.join(path, "adapter_model.safetensors"))
        or os.path.isfile(os.path.join(path, "adapter_model.bin"))
    )


def _syntax_ok(step):
    """Byte-compile a step to validate its syntax. No code is executed."""
    try:
        py_compile.compile(os.path.join(ROOT, step), doraise=True)
        return True, "OK"
    except py_compile.PyCompileError as e:
        return False, str(e).splitlines()[0]


def run_step(step):
    print(f"\n=== Running {step} ===")
    if subprocess.run([sys.executable, os.path.join(ROOT, step)], cwd=ROOT).returncode:
        raise RuntimeError(f"Step failed: {step}")


def main():
    parser = argparse.ArgumentParser(description="Run the debiasing pipeline.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Byte-compile every step (syntax check) and print the execution "
             "plan, without running anything.",
    )
    args = parser.parse_args()

    # Group all of this run's W&B runs together (no-op if W&B isn't configured).
    if not args.dry_run and wandb_enabled():
        ensure_group()

    for step in STEPS:
        skip = step in TRAIN_GUARDS and _adapter_exists(TRAIN_GUARDS[step])
        if args.dry_run:
            ok, msg = _syntax_ok(step)
            note = "  (adapter exists)" if skip else ""
            print(f"{'SKIP' if skip else 'RUN '}  syntax:{'OK' if ok else 'FAIL'}  {step}{note}")
            if not ok:
                print(f"        {msg}")
        elif skip:
            print(f"\n=== Skipping {step} (adapter exists; FORCE_RETRAIN=1 to retrain) ===")
        else:
            run_step(step)

    if not args.dry_run:
        print("\nAll steps completed.")


if __name__ == "__main__":
    main()
