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

        if root.is_symlink():
            raise ValueError("Refusing cleanup because temp/ is a symlink")

        candidate = Path(temp_directory).absolute()
        resolved_root = root.resolve()
        resolved_candidate = candidate.resolve()

        if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
            raise ValueError(
                "Cleanup is restricted to the project's temp/ directory"
            )

        self.temp_directory = resolved_candidate

    def _validate_run_directory(self, run_directory: Path) -> Path:
        target = Path(run_directory).absolute()

        if target.is_symlink():
            raise ValueError("Cleanup target must not be a symlink")

        resolved = target.resolve()

        if self.temp_directory not in resolved.parents:
            raise ValueError(
                "Cleanup target must be inside the project's temp/ directory"
            )

        return resolved

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
            logger.warning(
                "Cleanup skipped because no run-specific temporary directory was supplied"
            )
            return []

        target = self._validate_run_directory(run_directory)

        if not target.exists():
            return []

        if not target.is_dir():
            raise ValueError("Cleanup target must be a directory")

        keep = {
            Path(item).absolute()
            for item in (diagnostics or [])
        }

        removed: list[Path] = []

        for item in target.iterdir():
            lexical_path = item.absolute()

            if lexical_path in keep:
                continue

            if any(parent in keep for parent in lexical_path.parents):
                continue

            try:
                if item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    if any(
                        kept == lexical_path or lexical_path in kept.parents
                        for kept in keep
                    ):
                        continue
                    shutil.rmtree(item)
                else:
                    item.unlink(missing_ok=True)

                removed.append(lexical_path)

            except OSError:
                logger.exception("Could not remove temporary item: %s", item)

        try:
            if target.exists() and not any(
                kept == target or target in kept.parents
                for kept in keep
            ):
                target.rmdir()
                removed.append(target)
        except OSError:
            logger.exception("Could not remove temporary directory: %s", target)

        return removed

    def remove_processed_voiceover(
        self,
        settings: Settings,
        files: list[Path],
        run_directory: Path,
    ) -> list[Path]:
        if not settings.delete_processed_voiceover:
            return []

        target = self._validate_run_directory(run_directory)
        removed: list[Path] = []

        for path in files:
            candidate = Path(path).absolute()

            if candidate.is_symlink():
                logger.warning(
                    "Skipping symlink during processed audio cleanup: %s",
                    candidate,
                )
                continue

            resolved = candidate.resolve()

            if target not in resolved.parents or not resolved.is_file():
                continue

            try:
                resolved.unlink()
                removed.append(resolved)
            except OSError:
                logger.exception(
                    "Could not remove processed voiceover: %s",
                    resolved,
                )

        return removed
