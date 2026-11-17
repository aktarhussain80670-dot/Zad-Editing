"""Delete only generated files in the designated temporary tree."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from config.settings import Settings

logger = logging.getLogger(__name__)


class CleanupManager:
    def __init__(self, temp_directory: Path = Path("temp")) -> None:
        root = Path("temp").absolute()
        candidate = temp_directory.absolute()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Cleanup is restricted to the project's temp/ directory")
        if root.is_symlink():
            raise ValueError("Refusing cleanup because temp/ is a symlink")
        self.temp_directory = candidate

    def cleanup(
        self,
        settings: Settings,
        success: bool,
        run_directory: Path | None = None,
        diagnostics: list[Path] | None = None,
    ) -> list[Path]:
        if not settings.delete_temp_files or not success:
            return []
        if run_directory is None:
            logger.warning("Cleanup skipped because no run-specific temporary directory was supplied")
            return []
        target = run_directory.absolute()
        if self.temp_directory not in target.parents or target.is_symlink():
            raise ValueError("Cleanup target must be a non-symlink run directory inside temp/")
        keep = {p.resolve() for p in (diagnostics or [])}
        if not target.exists():
            return []
        removed: list[Path] = []
        for item in target.iterdir():
            resolved = item.resolve()
            if resolved in keep or target not in resolved.parents or any(resolved in item.parents for item in keep):
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink(missing_ok=True)
            removed.append(item)
        if target.exists() and not any(target.resolve() in item.parents or item == target.resolve() for item in keep):
            target.rmdir()
            removed.append(target)
        return removed

    def remove_processed_voiceover(self, settings: Settings, files: list[Path], run_directory: Path) -> list[Path]:
        if not settings.delete_processed_voiceover:
            return []
        target = run_directory.absolute()
        if self.temp_directory not in target.parents or target.is_symlink():
            raise ValueError("Processed audio cleanup is restricted to a run directory inside temp/")
        removed: list[Path] = []
        for path in files:
            resolved = path.resolve()
            if target not in resolved.parents or not resolved.is_file():
                continue
            resolved.unlink()
            removed.append(resolved)
        return removed
