"""
Minimal W&B helpers for the assignment pipeline.

Design: every step (train, eval, results) creates its own W&B run, but they
all join the same WANDB_RUN_GROUP so the dashboard groups one pipeline
execution together. If WANDB_PROJECT is not set in the environment, every
helper here becomes a no-op so the pipeline still works without W&B.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Any


def _load_dotenv() -> None:
    """Best-effort .env loader. Looks one and two levels above this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", ".env"),
        os.path.join(here, "..", "..", ".env"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def enabled() -> bool:
    _load_dotenv()
    return bool(os.environ.get("WANDB_PROJECT"))


def ensure_group() -> str:
    """Set WANDB_RUN_GROUP to a timestamp if not already set, and return it."""
    _load_dotenv()
    if not os.environ.get("WANDB_RUN_GROUP"):
        os.environ["WANDB_RUN_GROUP"] = "pipeline_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.environ["WANDB_RUN_GROUP"]


def start(job_type: str, name: str, config: dict[str, Any] | None = None):
    """Init a W&B run, or return None if W&B is not configured."""
    if not enabled():
        return None
    import wandb
    entity = os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_TEAM")
    return wandb.init(
        entity=entity,
        project=os.environ["WANDB_PROJECT"],
        group=ensure_group(),
        job_type=job_type,
        name=name,
        config=config or {},
        reinit=True,
    )


def finish(run) -> None:
    if run is None:
        return
    import wandb
    run.finish()


# Files we want in the LoRA model artifact. The training output_dir also
# contains trainer-checkpoint subdirectories which we explicitly skip so the
# uploaded artifact stays small (one adapter ≈ 5.7 MB).
_LORA_ARTIFACT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)


def log_lora_artifact(run, name: str, save_path: str, metadata: dict[str, Any] | None = None) -> None:
    """
    Upload a trained LoRA adapter as a W&B model artifact so future
    eval-only runs can pull it from W&B instead of retraining locally.

    Download later via:
        api = wandb.Api()
        artifact = api.artifact(f"{entity}/{project}/{name}:latest")
        local_path = artifact.download()
        # load_model(local_path) from _model_helpers will then work unchanged.
    """
    if run is None:
        return
    import os as _os
    import wandb

    artifact = wandb.Artifact(
        name=name,
        type="model",
        description=f"LoRA adapter ({name}) on top of Gemma-3-1b-pt",
        metadata=metadata or {},
    )
    added = 0
    for fname in _LORA_ARTIFACT_FILES:
        fpath = _os.path.join(save_path, fname)
        if _os.path.isfile(fpath):
            artifact.add_file(fpath, name=fname)
            added += 1
    if added == 0:
        print(f"[wandb] log_lora_artifact: no adapter files found in {save_path}; skipping")
        return
    run.log_artifact(artifact)
    print(f"[wandb] uploaded {added} files as artifact {name}")
