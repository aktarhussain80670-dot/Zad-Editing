"""Command-line entry point for analysis and editing-guide generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from core.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voiceover", required=True, type=Path)
    parser.add_argument(
        "--reference", required=True, action="append", type=Path, help="Provide 3 or 4 reference videos"
    )
    parser.add_argument("--guide", type=Path, default=Path("storage/editing_guides/editing_guide.json"))
    parser.add_argument("--plan-only", action="store_true", help="Generate the editing guide without rendering video")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing valid stage checkpoints")
    parser.add_argument("--music", type=Path, help="Optional local background track")
    parser.add_argument(
        "--sound-effect",
        action="append",
        default=[],
        metavar="SECONDS=PATH",
        help="Optional local sound effect and placement; may be repeated",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    effects = []
    for item in args.sound_effect:
        try:
            at, path = item.split("=", 1)
            effects.append((Path(path), float(at)))
        except ValueError:
            parser.error(f"Invalid --sound-effect {item!r}; expected SECONDS=PATH")
    run_pipeline(
        args.voiceover,
        args.reference,
        guide_path=args.guide,
        render=not args.plan_only,
        resume=not args.no_resume,
        music=args.music,
        sound_effects=effects,
    )


if __name__ == "__main__":
    main()
