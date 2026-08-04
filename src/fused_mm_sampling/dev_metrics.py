"""Structured, low-overhead development-loop timing events."""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

EVENT_PREFIX = "FMMS_DEV_EVENT "
EVENT_SCHEMA_VERSION = 1


def emit_dev_event(event: str, **fields) -> None:
    """Print one machine-readable event when development metrics are enabled."""
    if os.environ.get("FMMS_DEV_METRICS") != "1":
        return
    payload = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(EVENT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


@contextmanager
def timed_dev_stage(stage: str, **fields) -> Iterator[None]:
    """Emit disjoint stage timing around a block when metrics are enabled."""
    start = time.perf_counter()
    emit_dev_event("stage_start", stage=stage, **fields)
    try:
        yield
    except BaseException as error:
        emit_dev_event(
            "stage_end",
            stage=stage,
            status="error",
            duration_seconds=time.perf_counter() - start,
            error_type=type(error).__name__,
            **fields,
        )
        raise
    emit_dev_event(
        "stage_end",
        stage=stage,
        status="success",
        duration_seconds=time.perf_counter() - start,
        **fields,
    )
