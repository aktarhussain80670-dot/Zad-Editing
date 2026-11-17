"""Command-line entry point for analysis and editing-guide generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from core.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voiceover", required=True, type=Path)
    parser.add_argument("--reference", required=True, action="append", type=Path, help="Provide 3 or 4 reference videos")
    parser.add_argument("--guide", type=Path, default=Path("storage/editing_guides/editing_guide.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_pipeline(args.voiceover, args.reference, guide_path=args.guide)


if __name__ == "__main__":
    main()
