"""
Step 10 — Register datasets/models and log completed evaluation results.

This step does not run model inference. It reads the finalized CSV/JSON outputs
from results/metrics, publishes reusable Weave datasets if needed, logs one
EvaluationLogger run per model/evaluation pair, and writes a compact W&B
summary run for dashboard comparison.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from typing import Any, Callable

import pandas as pd
import weave
from weave import EvaluationLogger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.config import FIGURES_DIR, METRICS_DIR  # noqa: E402
from Algorithm.weave_registry import (  # noqa: E402
    MANIFEST_PATH,
    DatasetSpec,
    build_dataset_specs,
    ensure_dataset,
    ensure_model,
    init_weave,
    jsonable,
    load_dotenv,
    model_details,
    read_json,
    result_fingerprint,
)


MODEL_NAME_MAP = {
    "base_gemma": "base_gemma",
    "baseline": "baseline_lora",
    "debiased": "debiased_lora_cda_clp",
}
WEAVE_MANIFEST_KEY = "weave_evals_logged"
WANDB_SUMMARY_MANIFEST_KEY = "wandb_summary_logged"
FIGURE_FILES = [
    "comparison_bias_metrics.png",
    "comparison_utility.png",
    "bold_gender_gap.png",
    "training_loss.png",
]


def csv_path(model_key: str, suffix: str) -> str:
    return os.path.join(METRICS_DIR, f"{model_key}_{suffix}.csv")


def summary_path(model_key: str, suffix: str) -> str:
    return os.path.join(METRICS_DIR, f"{model_key}_{suffix}_summary.json")


def all_results_path() -> str:
    return os.path.join(METRICS_DIR, "all_results_table.csv")


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing result CSV: {path}")
    return pd.read_csv(path)


def read_summary(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing summary JSON: {path}")
    return read_json(path)


def clean_record(row: pd.Series | dict[str, Any], keys: list[str]) -> dict[str, Any]:
    data = row.to_dict() if hasattr(row, "to_dict") else row
    return {key: jsonable(data.get(key)) for key in keys}


def numeric_summary(summary: dict[str, Any]) -> dict[str, Any]:
    skip = {"model", "benchmark", "bias_type"}
    return {k: jsonable(v) for k, v in summary.items() if k not in skip}


def dataframe_for_wandb(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(pd.notna(df), None)


def slugify(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug or "value"


def read_all_results_table() -> pd.DataFrame:
    path = all_results_path()
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing aggregate results table: {path}. Run step9_plot_results.py first.")
    df = pd.read_csv(path)
    require_columns(df, path, ["model", "benchmark", "metric", "value"])
    return df


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.pivot_table(
            index=["benchmark", "metric"],
            columns="model",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["benchmark", "metric"])
    )
    pivot.columns.name = None
    for model in ["baseline", "debiased"]:
        if model not in pivot.columns:
            pivot[model] = None
    pivot["delta_debiased_minus_baseline"] = pivot["debiased"] - pivot["baseline"]
    pivot["relative_delta_pct"] = pivot.apply(
        lambda row: None
        if pd.isna(row["baseline"]) or row["baseline"] == 0 or pd.isna(row["debiased"])
        else (row["delta_debiased_minus_baseline"] / abs(row["baseline"])) * 100,
        axis=1,
    )
    ordered = [
        "benchmark",
        "metric",
        "baseline",
        "debiased",
        "delta_debiased_minus_baseline",
        "relative_delta_pct",
    ]
    extra = [col for col in pivot.columns if col not in ordered]
    return pivot[ordered + extra]


def comparison_scalars(comparison: pd.DataFrame) -> dict[str, float]:
    scalars: dict[str, float] = {}
    value_cols = [
        "baseline",
        "debiased",
        "delta_debiased_minus_baseline",
        "relative_delta_pct",
    ]
    for _, row in comparison.iterrows():
        prefix = f"final/{slugify(row['benchmark'])}/{slugify(row['metric'])}"
        for col in value_cols:
            value = row.get(col)
            if pd.notna(value):
                scalars[f"{prefix}/{slugify(col)}"] = float(value)
    compared = pd.notna(comparison["baseline"]) | pd.notna(comparison["debiased"])
    scalars["final/metric_rows"] = float(compared.sum())
    return scalars


def log_wandb_summary_run(current_hash: str, dry_run: bool) -> None:
    df = read_all_results_table()
    comparison = build_comparison_table(df)
    if dry_run:
        print(
            f"[dry-run] wandb summary run: {len(df)} metric rows, "
            f"{len(comparison)} comparison rows"
        )
        return

    load_dotenv()
    entity = os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_TEAM")
    project = os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        raise RuntimeError("WANDB_ENTITY/WANDB_TEAM and WANDB_PROJECT must be set in the environment or .env")

    import wandb

    run = wandb.init(
        entity=entity,
        project=project,
        name=f"final-results-{current_hash[:8]}",
        job_type="final-results",
        tags=["step10", "final-results", "summary"],
        config={
            "result_fingerprint": current_hash,
            "models": jsonable(model_details()),
        },
    )
    try:
        payload: dict[str, Any] = {
            "all_results_table": wandb.Table(dataframe=dataframe_for_wandb(df)),
            "baseline_vs_debiased": wandb.Table(dataframe=dataframe_for_wandb(comparison)),
            **comparison_scalars(comparison),
        }

        figure_paths = []
        for filename in FIGURE_FILES:
            path = os.path.join(FIGURES_DIR, filename)
            if os.path.isfile(path):
                figure_paths.append(path)
                payload[f"figures/{os.path.splitext(filename)[0]}"] = wandb.Image(path)

        run.log(payload)
        run.summary["result_fingerprint"] = current_hash
        run.summary["comparison_rows"] = len(comparison)

        artifact = wandb.Artifact(
            name=f"final-results-{current_hash[:8]}",
            type="evaluation-results",
            metadata={"result_fingerprint": current_hash},
        )
        artifact.add_file(all_results_path(), name="all_results_table.csv")
        for path in figure_paths:
            artifact.add_file(path, name=f"figures/{os.path.basename(path)}")
        run.log_artifact(artifact)
        print(f"[wandb] summary run logged: {run.url}")
    finally:
        run.finish()


def require_columns(df: pd.DataFrame, path: str, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"{os.path.basename(path)} missing columns: {missing}")


def validate_count(eval_name: str, spec: DatasetSpec, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    if len(df) != len(spec.rows):
        raise RuntimeError(f"{eval_name}: result rows={len(df)} but dataset rows={len(spec.rows)}")
    if "n" in summary and summary["n"] is not None and int(summary["n"]) != len(df):
        raise RuntimeError(f"{eval_name}: summary n={summary['n']} but CSV rows={len(df)}")


def validate_crows_row(dataset_row: dict[str, Any], result_row: pd.Series) -> None:
    if dataset_row["sent_more"] != result_row["sent_more"] or dataset_row["sent_less"] != result_row["sent_less"]:
        raise RuntimeError("CrowS-Pairs result row does not match registered dataset row")


def validate_winobias_row(dataset_row: dict[str, Any], result_row: pd.Series) -> None:
    if int(dataset_row["pair_id"]) != int(result_row["pair_id"]):
        raise RuntimeError("WinoBias pair_id mismatch")
    if dataset_row["pro_sentence"] != result_row["pro_sentence"]:
        raise RuntimeError("WinoBias pro_sentence mismatch")
    if dataset_row["anti_sentence"] != result_row["anti_sentence"]:
        raise RuntimeError("WinoBias anti_sentence mismatch")


def validate_bold_row(dataset_row: dict[str, Any], result_row: pd.Series) -> None:
    if dataset_row["domain"] != result_row["domain"] or dataset_row["prompt"] != result_row["prompt"]:
        raise RuntimeError("BOLD result row does not match registered dataset row")


def get_input_rows(dataset: Any, spec: DatasetSpec, dry_run: bool) -> list[Any]:
    if dry_run or dataset is None:
        return spec.rows
    return list(dataset.rows)


def log_eval_from_csv(
    eval_name: str,
    model_key: str,
    model_obj: Any,
    dataset_obj: Any,
    spec: DatasetSpec,
    result_csv: str,
    result_summary: str,
    output_keys: list[str],
    score_keys: list[str],
    dry_run: bool,
    row_validator: Callable[[dict[str, Any], pd.Series], None] | None = None,
) -> None:
    df = read_csv(result_csv)
    summary = read_summary(result_summary)
    require_columns(df, result_csv, output_keys + score_keys)
    validate_count(eval_name, spec, df, summary)

    for idx, result_row in df.iterrows():
        if row_validator is not None:
            row_validator(spec.rows[idx], result_row)

    model_name = MODEL_NAME_MAP[model_key]
    if dry_run:
        print(f"[dry-run] eval {eval_name} / {model_name}: {len(df)} examples")
        return

    logger = EvaluationLogger(
        name=eval_name,
        model=model_obj,
        dataset=dataset_obj,
        scorers=score_keys,
        eval_attributes={
            "evaluation_name": eval_name,
            "model_name": model_name,
            "source_model_key": model_key,
            "source_csv": os.path.basename(result_csv),
            "source_summary": os.path.basename(result_summary),
        },
    )
    input_rows = get_input_rows(dataset_obj, spec, dry_run=False)
    for idx, result_row in df.iterrows():
        logger.log_example(
            inputs=input_rows[idx],
            output=clean_record(result_row, output_keys),
            scores=clean_record(result_row, score_keys),
        )
    logger.log_summary(numeric_summary(summary), auto_summarize=False)
    print(f"[weave] eval {eval_name} / {model_name}: logged {len(df)} examples")


def log_utility_eval(
    model_key: str,
    model_obj: Any,
    dataset_obj: Any,
    spec: DatasetSpec,
    dry_run: bool,
) -> None:
    result_summary = summary_path(model_key, "utility")
    summary = read_summary(result_summary)
    model_name = MODEL_NAME_MAP[model_key]
    if dry_run:
        print(f"[dry-run] eval eval_utility / {model_name}: summary-only")
        return

    score_keys = [k for k in numeric_summary(summary) if k != "n"]
    logger = EvaluationLogger(
        name="eval_utility",
        model=model_obj,
        dataset=dataset_obj,
        scorers=score_keys,
        eval_attributes={
            "evaluation_name": "eval_utility",
            "model_name": model_name,
            "source_model_key": model_key,
            "source_summary": os.path.basename(result_summary),
        },
    )
    logger.log_summary(numeric_summary(summary), auto_summarize=False)
    print(f"[weave] eval eval_utility / {model_name}: logged summary")


def read_log_manifest() -> dict[str, Any]:
    if not os.path.isfile(MANIFEST_PATH):
        return {}
    return read_json(MANIFEST_PATH)


def manifest_has_step(manifest: dict[str, Any], current_hash: str, step_key: str) -> bool:
    if manifest.get("result_fingerprint") != current_hash:
        return False
    if (
        step_key == WEAVE_MANIFEST_KEY
        and WEAVE_MANIFEST_KEY not in manifest
        and WANDB_SUMMARY_MANIFEST_KEY not in manifest
    ):
        return True
    return bool(manifest.get(step_key))


def check_manifest(dry_run: bool) -> tuple[str, dict[str, Any]]:
    current_hash = result_fingerprint()
    if dry_run:
        print(f"[dry-run] result fingerprint: {current_hash}")
    return current_hash, read_log_manifest()


def write_manifest(current_hash: str, updates: dict[str, bool]) -> None:
    existing = read_log_manifest()
    if existing.get("result_fingerprint") == current_hash:
        manifest = dict(existing)
    else:
        manifest = {}
    if manifest_has_step(existing, current_hash, WEAVE_MANIFEST_KEY):
        manifest[WEAVE_MANIFEST_KEY] = True
    manifest["result_fingerprint"] = current_hash
    manifest.update(updates)
    manifest["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[weave] wrote manifest: {MANIFEST_PATH}")


def register_all(specs: dict[str, DatasetSpec], dry_run: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    datasets = {name: ensure_dataset(spec, dry_run=dry_run) for name, spec in specs.items()}
    details = model_details()
    models = {
        key: ensure_model(details[key]["name"], details[key], dry_run=dry_run)
        for key in ["baseline", "debiased"]
    }
    return datasets, models


def log_all_evals(specs: dict[str, DatasetSpec], datasets: dict[str, Any], models: dict[str, Any], dry_run: bool) -> None:
    for model_key in ["baseline", "debiased"]:
        log_eval_from_csv(
            "eval_crows_prob",
            model_key,
            models[model_key],
            datasets["eval_crows_prob"],
            specs["eval_crows_prob"],
            csv_path(model_key, "crowspairs"),
            summary_path(model_key, "crowspairs"),
            output_keys=["lp_more", "lp_less", "lp_more_avg", "lp_less_avg", "lp_more_tokens", "lp_less_tokens"],
            score_keys=["prefers_stereotype", "prefers_stereotype_avg", "logprob_gap", "logprob_gap_avg"],
            dry_run=dry_run,
            row_validator=validate_crows_row,
        )
        log_eval_from_csv(
            "eval_crows_embed",
            model_key,
            models[model_key],
            datasets["eval_crows_embed"],
            specs["eval_crows_embed"],
            csv_path(model_key, "embedding"),
            summary_path(model_key, "embedding"),
            output_keys=[],
            score_keys=["cosine_similarity", "cosine_distance"],
            dry_run=dry_run,
            row_validator=validate_crows_row,
        )
        log_eval_from_csv(
            "eval_stereoset_prob",
            model_key,
            models[model_key],
            datasets["eval_stereoset_prob"],
            specs["eval_stereoset_prob"],
            csv_path(model_key, "stereoset"),
            summary_path(model_key, "stereoset"),
            output_keys=["stereo_score", "anti_score", "stereo_score_avg", "anti_score_avg", "stereo_tokens", "anti_tokens"],
            score_keys=["prefers_stereotype", "prefers_stereotype_avg", "score_gap", "score_gap_avg"],
            dry_run=dry_run,
        )
        log_eval_from_csv(
            "eval_winobias_prob",
            model_key,
            models[model_key],
            datasets["eval_winobias_prob"],
            specs["eval_winobias_prob"],
            csv_path(model_key, "winobias"),
            summary_path(model_key, "winobias"),
            output_keys=["lp_pro", "lp_anti", "lp_pro_avg", "lp_anti_avg", "pro_tokens", "anti_tokens"],
            score_keys=["prefers_stereotype", "prefers_stereotype_avg", "logprob_gap", "logprob_gap_avg"],
            dry_run=dry_run,
            row_validator=validate_winobias_row,
        )
        log_eval_from_csv(
            "eval_bold_gen",
            model_key,
            models[model_key],
            datasets["eval_bold_gen"],
            specs["eval_bold_gen"],
            csv_path(model_key, "bold"),
            summary_path(model_key, "bold"),
            output_keys=["generation"],
            score_keys=["male_count", "female_count", "net_gender_gap", "abs_gender_gap"],
            dry_run=dry_run,
            row_validator=validate_bold_row,
        )
        log_utility_eval(
            model_key,
            models[model_key],
            datasets["eval_utility"],
            specs["eval_utility"],
            dry_run=dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Register datasets/models and log completed eval results.")
    parser.add_argument("--dry-run", action="store_true", help="Validate mappings without writing to W&B/Weave.")
    parser.add_argument("--force", action="store_true", help="Log even if the local manifest says these results were already logged.")
    parser.add_argument("--skip-weave-evals", action="store_true", help="Do not create per-example Weave eval traces.")
    parser.add_argument("--skip-wandb-summary", action="store_true", help="Do not create the compact W&B final-results run.")
    args = parser.parse_args()

    if args.skip_weave_evals and args.skip_wandb_summary:
        raise RuntimeError("Nothing to do: both --skip-weave-evals and --skip-wandb-summary were set.")

    current_hash, manifest = check_manifest(dry_run=args.dry_run)
    log_weave_evals = not args.skip_weave_evals
    log_wandb_summary = not args.skip_wandb_summary

    if not args.dry_run and not args.force:
        if log_weave_evals and manifest_has_step(manifest, current_hash, WEAVE_MANIFEST_KEY):
            print(f"[weave] eval traces already logged for fingerprint {current_hash}; skipping")
            log_weave_evals = False
        if log_wandb_summary and manifest_has_step(manifest, current_hash, WANDB_SUMMARY_MANIFEST_KEY):
            print(f"[wandb] summary run already logged for fingerprint {current_hash}; skipping")
            log_wandb_summary = False

    completed: dict[str, bool] = {}

    if log_weave_evals:
        specs = build_dataset_specs()
        if not args.dry_run:
            init_weave()
        datasets, models = register_all(specs, dry_run=args.dry_run)
        log_all_evals(specs, datasets, models, dry_run=args.dry_run)
        if not args.dry_run:
            completed[WEAVE_MANIFEST_KEY] = True

    if log_wandb_summary:
        log_wandb_summary_run(current_hash, dry_run=args.dry_run)
        if not args.dry_run:
            completed[WANDB_SUMMARY_MANIFEST_KEY] = True

    if not args.dry_run and completed:
        write_manifest(current_hash, completed)

    if not completed and not args.dry_run:
        print("[Step 10] Nothing new to log for this result fingerprint.")
    print("[Step 10] Logging complete." if not args.dry_run else "[Step 10] Dry-run validation complete.")


if __name__ == "__main__":
    main()
