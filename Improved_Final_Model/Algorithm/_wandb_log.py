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
