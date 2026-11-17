"""Report local runtime dependencies needed by the complete rendering path."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


def check(command: str) -> tuple[bool, str]:
    executable = shutil.which(command)
    if not executable:
        return False, "not found on PATH"
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False, "could not execute"
    version = (result.stdout or result.stderr).splitlines()
    return result.returncode == 0, version[0] if version else "version unavailable"


def main() -> int:
    checks = {name: check(name) for name in ("node", "ffmpeg", "ffprobe", "hyperframes")}
    modules = {name: importlib.util.find_spec(name) is not None for name in ("whisper", "cv2", "numpy")}
    failed = False
    for name, (ok, version) in checks.items():
        print(f"{'OK' if ok else 'MISSING'} {name}: {version}")
        failed |= not ok
    for name, ok in modules.items():
        print(f"{'OK' if ok else 'MISSING'} Python module {name}")
        failed |= not ok
    if failed:
        print("Install missing tools before running the complete pipeline.", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
