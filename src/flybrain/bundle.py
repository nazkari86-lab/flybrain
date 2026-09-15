"""Immutable filesystem bundles for reproducible experiment results."""

import json
import platform
import sys
from pathlib import Path

from flybrain.experiment import ExperimentResult


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def write_bundle(result: ExperimentResult, output: Path) -> Path:
    """Write a new experiment bundle without replacing existing data."""

    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", result.configuration)
    _write_json(
        output / "metrics.json",
        {
            "motor_spikes": result.motor_spikes,
            "lesioned_motor_spikes": result.lesioned_motor_spikes,
            "lesion_effect": result.lesion_effect,
            "runtime_seconds": result.runtime_seconds,
            "peak_memory_bytes": result.peak_memory_bytes,
        },
    )
    _write_json(
        output / "provenance.json",
        {
            "config_hash": result.config_hash,
            "dataset_id": result.dataset_id,
            "source_manifest_sha256": result.source_manifest_sha256,
            "software_revision": result.software_revision,
            "seed": result.seed,
        },
    )
    event_lines = "".join(event.model_dump_json() + "\n" for event in result.events)
    (output / "events.jsonl").write_text(event_lines)
    _write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    )
    return output
