"""Direct universal voiceover-driven motion-graphics scene planning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ai.scene_planner import plan_scenes
from services.font_manager import FontManager


class EditingDirector:
    """Apply contextual typography and optional local icon decisions."""

    FONT_ROLES = ("heading", "title", "caption", "number", "label")

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = (project_root or Path.cwd()).resolve()
        self.font_manager = FontManager(self.project_root)
        self.icon_directory = self.project_root / "assets" / "icons"

    @staticmethod
    def _text_tokens(*values: Any) -> set[str]:
        text = " ".join(
            str(value) for value in values
            if isinstance(value, str)
        )
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+", text)
            if len(token) >= 3
        }

    @staticmethod
    def _style_hint(style: dict[str, Any]) -> str:
        profile = style.get("style_profile", {})
        if not isinstance(profile, dict):
            return ""

        typography = profile.get("typography", {})
        if not isinstance(typography, dict):
            return ""

        return " ".join(
            str(typography.get(key, ""))
            for key in ("family", "style", "classification", "weight")
        )

    def _select_fonts(
        self,
        style: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        hint = self._style_hint(style)
        selections = {}

        for role in self.FONT_ROLES:
            font = self.font_manager.select_font(role, hint)
            selections[role] = {
                "name": font["name"],
                "family": font["family"],
                "filename": font["filename"],
                "format": font["format"],
                "weight": font["weight"],
            }

        return selections

    def _select_icon(self, scene: dict[str, Any]) -> dict[str, str] | None:
        if not self.icon_directory.is_dir():
            return None

        plan = scene.get("visual_plan", {})
        if not isinstance(plan, dict):
            plan = {}

        primary = plan.get("primary_element", {})
        if not isinstance(primary, dict):
            primary = {}

        supporting = plan.get("supporting_elements", [])
        supporting_text = []

        if isinstance(supporting, list):
            for item in supporting:
                if isinstance(item, dict):
                    supporting_text.extend(
                        item.get(key, "")
                        for key in ("label", "text", "title")
                    )
                elif isinstance(item, str):
                    supporting_text.append(item)

        narration_tokens = self._text_tokens(
            scene.get("transcript"),
            scene.get("visual_idea"),
            plan.get("headline"),
            primary.get("label"),
            *supporting_text,
        )

        if not narration_tokens:
            return None

        candidates = []

        for path in self.icon_directory.iterdir():
            if not path.is_file() or path.suffix.lower() != ".svg":
                continue

            icon_tokens = self._text_tokens(path.stem)
            overlap = narration_tokens.intersection(icon_tokens)

            if overlap:
                candidates.append((len(overlap), path.name.lower(), path))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = candidates[0][2]

        return {
            "filename": selected.name,
            "alt": selected.stem.replace("_", " "),
        }

    def direct(
        self,
        transcript: dict[str, Any],
        style: dict[str, Any],
        features: dict[str, bool] | None = None,
        audio_analysis: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Create narration-led scenes with contextual local assets."""

        if not isinstance(transcript, dict):
            raise TypeError("Transcript must be a dictionary.")

        if not isinstance(style, dict):
            style = {}

        if features is not None and not isinstance(features, dict):
            raise TypeError("Features must be a dictionary or None.")

        scenes = plan_scenes(
            transcript=transcript,
            style=style,
            audio_analysis=audio_analysis,
        )

        if not isinstance(scenes, list):
            raise TypeError("Scene planner must return a list.")

        font_selections = self._select_fonts(style)

        for scene in scenes:
            if not isinstance(scene, dict):
                raise TypeError("Scene planner returned a malformed scene.")

            for key in (
                "caption",
                "motion",
                "transition",
                "sound_effect",
                "audio",
            ):
                if not isinstance(scene.get(key), dict):
                    scene[key] = {}

            enabled = features or {}

            scene["caption"]["enabled"] = enabled.get("subtitles", True)

            # Motion graphics are mandatory in this editor.
            scene["motion"]["enabled"] = True

            scene["transition"]["enabled"] = enabled.get(
                "transitions", True
            )

            scene["sound_effect"]["enabled"] = enabled.get(
                "sound_effects", False
            )
            scene["audio"]["music_enabled"] = enabled.get(
                "background_music", False
            )
            scene["audio"]["sound_effect_enabled"] = enabled.get(
                "sound_effects", False
            )

            scene["font_selection"] = {
                role: dict(font)
                for role, font in font_selections.items()
            }

            icon = self._select_icon(scene)
            if icon:
                scene["icon"] = icon
            else:
                scene.pop("icon", None)

            # External footage and images are permanently disabled.
            scene["external_media"] = {
                "required": False,
                "query": "",
            }

            scene["decision_rationale"] = [
                "Scene structure is derived from narration context and timing.",
                "Audio energy, silence, and emphasis inform visual synchronization.",
                "Reference videos provide appearance and motion guidance only.",
                "Typography is selected from the local font library.",
                "Icons are included only when their names match scene context.",
                "All scene visuals are generated through HyperFrames HTML/CSS/JavaScript.",
                "External clips, external images, and reference footage are excluded.",
                "The original voiceover remains the narration source.",
            ]

        return scenes


def direct(
    transcript: dict[str, Any],
    style: dict[str, Any],
    features: dict[str, bool] | None = None,
    audio_analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible Editing Director entry point."""

    return EditingDirector().direct(
        transcript=transcript,
        style=style,
        features=features,
        audio_analysis=audio_analysis,
    )
