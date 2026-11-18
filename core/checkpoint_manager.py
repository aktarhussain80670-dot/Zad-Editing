"""Fingerprint-protected atomic stage checkpoints for safe resume."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _file_fingerprint(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint input is not a file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


class CheckpointManager:
    VALID_STATUSES = {"complete", "failed"}

    def __init__(
        self,
        path: Path = Path("storage/metadata/checkpoint.json"),
    ) -> None:
        self.path = Path(path)

    def fingerprint(self, inputs: list[Path], settings: Any) -> str:
        config = asdict(settings) if is_dataclass(settings) else settings

        normalized = json.dumps(
            {
                "inputs": [_file_fingerprint(Path(item)) for item in inputs],
                "settings": config,
            },
            sort_keys=True,
            default=str,
            allow_nan=False,
        ).encode("utf-8")

        return hashlib.sha256(normalized).hexdigest()

    def load(self, fingerprint: str) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return None

            if data.get("fingerprint") != fingerprint:
                return None

            if data.get("status") not in self.VALID_STATUSES:
                return None

            checkpoint_data = data.get("data")
            generated_files = data.get("generated_files")
            generated_stats = data.get("generated_file_stats")

            if not isinstance(checkpoint_data, dict):
                return None

            if not isinstance(generated_files, list):
                return None

            if not isinstance(generated_stats, dict):
                return None

            last_stage = data.get("last_completed_stage")
            if last_stage is not None and not isinstance(last_stage, str):
                return None

            for raw_path in generated_files:
                if not isinstance(raw_path, str) or not raw_path:
                    return None

                artifact = Path(raw_path)
                if not artifact.is_file():
                    return None

                current = artifact.stat()
                saved = generated_stats.get(str(artifact.resolve()))

                if not isinstance(saved, dict):
                    return None

                if saved.get("size") != current.st_size:
                    return None

                if saved.get("mtime_ns") != current.st_mtime_ns:
                    return None

            return data

        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            AttributeError,
        ):
            return None

    def save(
        self,
        fingerprint: str,
        stage: str,
        generated_files: list[Path],
        status: str = "complete",
        error: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Unsupported checkpoint status: {status}")

        if not isinstance(stage, str) or not stage:
            raise ValueError("Checkpoint stage must be a non-empty string")

        if data is not None and not isinstance(data, dict):
            raise TypeError("Checkpoint data must be a dictionary")

        self.path.parent.mkdir(parents=True, exist_ok=True)

        resolved_files = [str(Path(item).resolve()) for item in generated_files]

        payload: dict[str, Any] = {
            "fingerprint": fingerprint,
            "completed_stage": stage,
            "last_completed_stage": stage if status == "complete" else None,
            "generated_files": resolved_files,
            "status": status,
            "error": error,
            "data": data or {},
        }

        payload["generated_file_stats"] = {
            str(Path(item).resolve()): {
                "size": Path(item).stat().st_size,
                "mtime_ns": Path(item).stat().st_mtime_ns,
            }
            for item in generated_files
            if Path(item).is_file()
        }

        if status == "failed":
            previous = self.load(fingerprint)
            if previous is not None:
                payload["last_completed_stage"] = previous.get(
                    "last_completed_stage"
                )
                payload["generated_files"] = previous.get(
                    "generated_files", []
                )
                payload["generated_file_stats"] = previous.get(
                    "generated_file_stats", {}
                )
                payload["data"] = previous.get("data", {})

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

                json.dump(
                    payload,
                    temporary_file,
                    indent=2,
                    allow_nan=False,
                )
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, self.path)
            temporary_path = None

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
