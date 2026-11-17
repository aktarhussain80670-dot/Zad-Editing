"""Fingerprint-protected atomic stage checkpoints for safe resume."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "sha256": digest.hexdigest()}


class CheckpointManager:
    def __init__(self, path: Path = Path("storage/metadata/checkpoint.json")) -> None:
        self.path = path

    def fingerprint(self, inputs: list[Path], settings: Any) -> str:
        config = asdict(settings) if is_dataclass(settings) else settings
        normalized = json.dumps(
            {"inputs": [_file_fingerprint(p) for p in inputs], "settings": config}, sort_keys=True, default=str
        ).encode()
        return hashlib.sha256(normalized).hexdigest()

    def load(self, fingerprint: str) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("fingerprint") != fingerprint or data.get("status") not in {"complete", "failed"}:
            return None
        for path in data.get("generated_files", []):
            artifact = Path(path)
            if not artifact.is_file():
                return None
            current = artifact.stat()
            saved = data.get("generated_file_stats", {}).get(str(artifact.resolve()))
            if saved != {"size": current.st_size, "mtime_ns": current.st_mtime_ns}:
                return None
        return data

    def save(
        self,
        fingerprint: str,
        stage: str,
        generated_files: list[Path],
        status: str = "complete",
        error: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": fingerprint,
            "completed_stage": stage,
            "last_completed_stage": stage if status == "complete" else None,
            "generated_files": [str(p.resolve()) for p in generated_files],
            "status": status,
            "error": error,
            "data": data or {},
        }
        payload["generated_file_stats"] = {
            str(path.resolve()): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in generated_files
            if path.is_file()
        }
        if status == "failed" and self.path.is_file():
            try:
                previous = json.loads(self.path.read_text(encoding="utf-8"))
                if previous.get("fingerprint") == fingerprint:
                    payload["last_completed_stage"] = previous.get("last_completed_stage")
                    payload["generated_files"] = previous.get("generated_files", [])
                    payload["generated_file_stats"] = previous.get("generated_file_stats", {})
                    payload["data"] = previous.get("data", {})
            except (OSError, json.JSONDecodeError):
                pass
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
