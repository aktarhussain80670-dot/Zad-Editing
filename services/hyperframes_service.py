"""Integration with the documented HyperFrames HTML composition and CLI."""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from services.ffmpeg_service import MediaToolError


class HyperFramesService:
    def __init__(self, executable: str = "hyperframes", timeout: int = 3600) -> None:
        self.executable = shutil.which(executable)
        self.timeout = timeout

    @staticmethod
    def node_version() -> tuple[int, int, int]:
        node = shutil.which("node")
        if not node:
            raise MediaToolError("Node.js is required for HyperFrames. Install Node.js 22 or newer.")
        result = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10, check=False)
        match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", result.stdout)
        if result.returncode or not match:
            raise MediaToolError("Could not determine Node.js version")
        return tuple(map(int, match.groups()))

    def ensure_available(self) -> str:
        version = self.node_version()
        if version < (22, 0, 0):
            raise MediaToolError(f"HyperFrames requires Node.js >=22; found {'.'.join(map(str, version))}")
        if not self.executable:
            raise MediaToolError("HyperFrames CLI is not installed. Install with `npm install --global hyperframes`.")
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise MediaToolError("HyperFrames rendering requires ffmpeg and ffprobe on PATH; install FFmpeg first.")
        try:
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env={
                    **os.environ,
                    "HYPERFRAMES_NO_UPDATE_CHECK": "1",
                    "HYPERFRAMES_SKIP_SKILLS": "1",
                    "HYPERFRAMES_NO_TELEMETRY": "1",
                    "DO_NOT_TRACK": "1",
                },
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaToolError("HyperFrames info check timed out") from exc
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
        if result.returncode or not match:
            raise MediaToolError(f"HyperFrames CLI version check failed: {result.stderr[-1500:].strip()}")
        if tuple(map(int, match.groups())) < (0, 9, 59):
            raise MediaToolError(f"HyperFrames >=0.9.59 is required; installed version is {match.group(0)}")
        return result.stdout.strip()

    def _composition_html(
        self,
        guide: dict[str, Any],
        assets: dict[int, Path],
        width: int,
        height: int,
        fps: int,
        composition_directory: Path = Path("."),
    ) -> str:
        duration = float(guide["duration"])
        video_layers: list[str] = []
        captions: list[str] = []
        gsap_setup: list[str] = []
        all_words = [
            word for segment in guide.get("transcript", {}).get("segments", []) for word in segment.get("words", [])
        ]
        for scene in guide["scenes"]:
            sid, start, end = int(scene["scene_id"]), float(scene["start"]), float(scene["end"])
            length = end - start
            if sid in assets and assets[sid].is_file():
                src = html.escape(
                    Path(os.path.relpath(assets[sid].resolve(), composition_directory.resolve())).as_posix(),
                    quote=True,
                )
                video_id = f"scene-video-{sid}"
                transition_type = scene.get("transition", {}).get("type", "cut")
                fade = 0.0 if not scene.get("transition", {}).get("enabled", True) or transition_type == "cut" else 0.3
                video_layers.append(
                    f'<video id="{video_id}" class="clip scene-media" src="{src}" muted playsinline data-start="{start}" data-duration="{length}" data-media-start="0" data-fade-in="{fade}" data-fade-out="{fade}" data-track-index="{sid}"></video>'
                )
                if scene.get("motion", {}).get("enabled", True):
                    gsap_setup.append(
                        f' tl.fromTo("#{video_id}", {{scale:1}}, {{scale:1.06, duration:{length}, ease:"none"}}, {start});'
                    )
            caption = scene.get("caption", {})
            text = str(caption.get("text", "")).strip()
            if caption.get("enabled") and text:
                words = [
                    word
                    for word in all_words
                    if float(word.get("end", -1)) > start
                    and float(word.get("start", duration)) < end
                    and str(word.get("word", "")).strip()
                ]
                if words:
                    word_markup = " ".join(
                        f'<span class="caption-word" id="word-{sid}-{i}">{html.escape(str(word["word"]).strip())}</span>'
                        for i, word in enumerate(words)
                    )
                else:
                    word_markup = html.escape(text)
                captions.append(
                    f'<div class="caption clip" id="caption-{sid}" data-start="{start}" data-duration="{length}" data-track-index="100">{word_markup}</div>'
                )
                gsap_setup.append(
                    f' tl.fromTo("#caption-{sid}", {{opacity:0, y:30, scale:0.98}}, {{opacity:1, y:0, scale:1, duration:0.22, ease:"power2.out"}}, {start}); tl.to("#caption-{sid}", {{opacity:0, duration:0.12}}, {max(start, end - 0.12)});'
                )
                for i, word in enumerate(words):
                    word_start = max(start, float(word.get("start", start)))
                    word_end = min(end, float(word.get("end", end)))
                    gsap_setup.append(
                        f' tl.to("#word-{sid}-{i}", {{color:"#ffc36d", scale:1.08, duration:0.08, transformOrigin:"center center"}}, {word_start}); tl.to("#word-{sid}-{i}", {{color:"#ffffff", scale:1, duration:0.08}}, {max(word_start, word_end)});'
                    )
            idea = html.escape(str(scene.get("visual_idea", "")))
            title = html.escape(str(scene.get("transcript", "")))
            if not assets.get(sid) or not assets[sid].is_file():
                fallback_id = f"fallback-{sid}"
                video_layers.append(
                    f'<div class="fallback clip" id="{fallback_id}" data-start="{start}" data-duration="{length}" data-track-index="{sid}">{idea} - {title}</div>'
                )
                if scene.get("motion", {}).get("enabled", True):
                    gsap_setup.append(
                        f' tl.fromTo("#{fallback_id}", {{scale:1}}, {{scale:1.04, duration:{length}, ease:"none"}}, {start});'
                    )
                transition = scene.get("transition", {})
                if transition.get("enabled", True) and transition.get("type", "cut") != "cut":
                    gsap_setup.append(
                        f' tl.fromTo("#{fallback_id}", {{opacity:0}}, {{opacity:1, duration:0.3, ease:"power1.out"}}, {start}); tl.to("#{fallback_id}", {{opacity:0, duration:0.3}}, {max(start, end - 0.3)});'
                    )
            if sid in assets and scene.get("motion", {}).get("enabled", True) and idea:
                title_id = f"scene-title-{sid}"
                video_layers.append(
                    f'<div class="scene-title clip" id="{title_id}" data-start="{start}" data-duration="{length}">{idea}</div>'
                )
                gsap_setup.append(
                    f' tl.fromTo("#{title_id}", {{opacity:0, x:-28}}, {{opacity:1, x:0, duration:0.3, ease:"power2.out"}}, {start});'
                )
        layer_markup = "\n".join(video_layers + captions)
        timeline = "".join(gsap_setup)
        template_root = Path(__file__).resolve().parent.parent / "templates"
        template_css = "\n".join(
            (template_root / folder / filename).read_text(encoding="utf-8")
            for folder, filename in (
                ("transitions", "fade.css"),
                ("motion_graphics", "motion.css"),
                ("typography", "title.css"),
                ("captions", "caption.css"),
            )
        )
        return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width={width},height={height}"><title>Universal Video Editor composition</title>
<script src="./gsap.min.js"></script>
<style>*{{box-sizing:border-box}}html,body{{margin:0;width:{width}px;height:{height}px;overflow:hidden;background:#10151b;font-family:Arial,sans-serif}}#root{{position:relative;width:{width}px;height:{height}px;overflow:hidden;background:linear-gradient(145deg,#172a3a,#10151b 55%,#403528)}}{template_css}
</style></head><body><div id="root" data-composition-id="main" data-start="0" data-duration="{duration}" data-width="{width}" data-height="{height}" data-fps="{fps}">{layer_markup}</div>
<script>window.__timelines=window.__timelines||{{}};const tl=gsap.timeline({{paused:true}});{timeline}window.__timelines.main=tl;tl.seek(0);</script></body></html>'''

    def create_composition(
        self,
        guide: dict[str, Any],
        assets: dict[int, Path],
        destination: Path,
        width: int = 1080,
        height: int = 1920,
        fps: int = 60,
    ) -> Path:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing HyperFrames composition: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        project_root = Path.cwd().resolve()
        resolved_destination = destination.resolve()
        if resolved_destination != project_root and project_root not in resolved_destination.parents:
            raise ValueError("HyperFrames composition must be inside the project directory")
        executable = Path(self.executable).resolve() if self.executable else None
        package_root = executable.parent.parent if executable else None
        gsap_candidates = (
            list(package_root.glob("dist/vendor/npm/gsap@*/dist/gsap.min.js"))
            if package_root and package_root.exists()
            else []
        )
        if not gsap_candidates:
            global_root = subprocess.run(
                [shutil.which("npm") or "npm", "root", "-g"], capture_output=True, text=True, timeout=15, check=False
            )
            if global_root.returncode == 0:
                gsap_candidates = list(
                    (Path(global_root.stdout.strip()) / "hyperframes").glob("dist/vendor/npm/gsap@*/dist/gsap.min.js")
                )
        if not gsap_candidates:
            raise MediaToolError("Could not locate HyperFrames' bundled GSAP runtime; verify the CLI installation")
        gsap_path = destination.parent / "gsap.min.js"
        if gsap_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing HyperFrames runtime asset: {gsap_path}")
        shutil.copyfile(gsap_candidates[0], gsap_path)
        destination.write_text(
            self._composition_html(guide, assets, width, height, fps, destination.parent), encoding="utf-8"
        )
        return destination

    def render(
        self,
        guide: dict[str, Any],
        assets: dict[int, Path],
        composition: Path,
        output: Path,
        width: int = 1080,
        height: int = 1920,
        fps: int = 60,
    ) -> Path:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing HyperFrames render: {output}")
        self.ensure_available()
        self.create_composition(guide, assets, composition, width, height, fps)
        env = {
            **os.environ,
            "HYPERFRAMES_NO_UPDATE_CHECK": "1",
            "HYPERFRAMES_SKIP_SKILLS": "1",
            "HYPERFRAMES_NO_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
        }
        composition_arg = str(composition.resolve().relative_to(Path.cwd().resolve()))
        lint = subprocess.run(
            [self.executable or "hyperframes", "lint", composition_arg],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=env,
        )
        if lint.returncode:
            raise MediaToolError(f"HyperFrames composition lint failed: {lint.stderr[-3000:] or lint.stdout[-3000:]}")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                [
                    self.executable or "hyperframes",
                    "render",
                    "-c",
                    composition_arg,
                    "-o",
                    str(output),
                    "--fps",
                    str(fps),
                    "--format",
                    "mp4",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaToolError("HyperFrames rendering timed out") from exc
        if result.returncode:
            raise MediaToolError(
                f"HyperFrames render failed (exit {result.returncode}): {result.stderr[-4000:].strip()}"
            )
        if not output.is_file() or output.stat().st_size == 0:
            raise MediaToolError("HyperFrames reported success but output is missing or empty")
        return output
