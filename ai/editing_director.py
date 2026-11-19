"""Narration-driven motion graphics editing director."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ai.scene_planner import plan_scenes
from services.font_manager import FontManager


class EditingDirector:
    """Direct scene planning, visual direction, typography and local assets."""

    FONT_ROLES = ("heading", "title", "caption", "number", "label")

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = (project_root or Path.cwd()).resolve()
        self.font_manager = FontManager(self.project_root)
        self.icon_directory = self.project_root / "assets" / "icons"

    @staticmethod
    def _text_tokens(*values: Any) -> set[str]:
        text = " ".join(
            str(value)
            for value in values
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
        selections: dict[str, dict[str, Any]] = {}

        for role in self.FONT_ROLES:
            try:
                font = self.font_manager.select_font(role, hint)

                if not isinstance(font, dict):
                    continue

                selections[role] = {
                    "name": font.get("name", ""),
                    "family": font.get("family", ""),
                    "filename": font.get("filename", ""),
                    "format": font.get("format", ""),
                    "weight": font.get("weight", ""),
                }

            except (KeyError, TypeError, ValueError, OSError):
                selections[role] = {
                    "name": "",
                    "family": "",
                    "filename": "",
                    "format": "",
                    "weight": "",
                }

        return selections

    @staticmethod
    def _get_visual_plan(scene: dict[str, Any]) -> dict[str, Any]:
        plan = scene.get("visual_plan")

        if not isinstance(plan, dict):
            plan = {}

        scene["visual_plan"] = plan
        return plan

    @staticmethod
    def _get_primary_element(plan: dict[str, Any]) -> dict[str, Any]:
        primary = plan.get("primary_element")

        if not isinstance(primary, dict):
            primary = {}

        plan["primary_element"] = primary
        return primary

    @staticmethod
    def _get_supporting_elements(plan: dict[str, Any]) -> list[Any]:
        supporting = plan.get("supporting_elements")

        if not isinstance(supporting, list):
            supporting = []

        plan["supporting_elements"] = supporting
        return supporting

    def _select_icon(
        self,
        scene: dict[str, Any],
    ) -> dict[str, str] | None:
        if not self.icon_directory.is_dir():
            return None

        plan = self._get_visual_plan(scene)
        primary = self._get_primary_element(plan)
        supporting = self._get_supporting_elements(plan)

        supporting_text: list[str] = []

        for item in supporting:
            if isinstance(item, dict):
                supporting_text.extend(
                    str(item.get(key, ""))
                    for key in ("label", "text", "title", "type")
                )
            elif isinstance(item, str):
                supporting_text.append(item)

        narration_tokens = self._text_tokens(
            scene.get("transcript"),
            scene.get("visual_idea"),
            scene.get("summary"),
            scene.get("topic"),
            plan.get("headline"),
            plan.get("composition"),
            primary.get("label"),
            primary.get("type"),
            primary.get("description"),
            *supporting_text,
        )

        if not narration_tokens:
            return None

        candidates: list[tuple[int, str, Path]] = []

        try:
            paths = list(self.icon_directory.iterdir())
        except OSError:
            return None

        for path in paths:
            if not path.is_file() or path.suffix.lower() != ".svg":
                continue

            icon_tokens = self._text_tokens(path.stem)
            overlap = narration_tokens.intersection(icon_tokens)

            if overlap:
                candidates.append(
                    (len(overlap), path.name.lower(), path)
                )

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = candidates[0][2]

        return {
            "filename": selected.name,
            "alt": selected.stem.replace("_", " ").replace("-", " "),
        }

    @staticmethod
    def _derive_visual_metadata(
        scene: dict[str, Any],
    plan: dict[str, Any],
    primary: dict[str, Any],
    supporting: list[Any],
    audio_analysis: dict[str, Any] | None,
    style: dict[str, Any],
    scene_index: int,
    scene_count: int,
    ) -> dict[str, Any]:
        transcript = str(scene.get("transcript", "")).strip()
        visual_idea = str(scene.get("visual_idea", "")).strip()

        headline = str(
            plan.get("headline")
            or scene.get("headline")
            or ""
        ).strip()

        composition = str(
            plan.get("composition")
            or plan.get("layout")
            or ""
        ).strip()

        primary_type = str(
            primary.get("type")
            or primary.get("visual_type")
            or ""
        ).strip()

        primary_description = str(
            primary.get("description")
            or primary.get("label")
            or visual_idea
            or headline
            or transcript
        ).strip()

        primary.setdefault("type", primary_type or "concept_visual")
        primary.setdefault("description", primary_description)

        if not primary.get("label") and headline:
            primary["label"] = headline

        if not composition:
            composition = "narration_contextual"

        plan.setdefault("headline", headline)
        plan.setdefault("composition", composition)

        visual_roles = {
            "primary": {
                "role": "main_visual",
                "type": primary.get("type", "concept_visual"),
                "description": primary.get("description", ""),
                "label": primary.get("label", ""),
            },
            "supporting": [],
        }

        for item in supporting:
            if isinstance(item, dict):
                visual_roles["supporting"].append(
                    {
                        "role": "supporting_visual",
                        "type": item.get("type", "supporting_element"),
                        "label": item.get("label", item.get("text", "")),
                        "description": item.get(
                            "description",
                            item.get("text", item.get("label", "")),
                        ),
                    }
                )
            elif isinstance(item, str):
                visual_roles["supporting"].append(
                    {
                        "role": "supporting_visual",
                        "type": "supporting_element",
                        "label": item,
                        "description": item,
                    }
                )

        return {
            "scene_index": scene_index,
            "scene_count": scene_count,
            "narration": transcript,
            "visual_idea": visual_idea,
            "headline": headline,
            "composition": composition,
            "visual_hierarchy": {
                "primary": primary_description,
                "secondary": [
                    item.get("label", "")
                    for item in visual_roles["supporting"]
                    if isinstance(item, dict)
                ],
            },
            "visual_roles": visual_roles,
            "reference_style": style.get("style_profile", {}),
            "audio_context": (
                audio_analysis if isinstance(audio_analysis, dict) else {}
            ),
        }

    def direct(
        self,
        transcript: dict[str, Any],
        style: dict[str, Any],
        features: dict[str, bool] | None = None,
        audio_analysis: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate narration-led scene direction."""

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
        enabled = features or {}
        scene_count = len(scenes)

        for index, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                raise TypeError(
                    f"Scene planner returned malformed scene at index {index}."
                )

            for key in (
                "caption",
                "motion",
                "transition",
                "sound_effect",
                "audio",
            ):
                if not isinstance(scene.get(key), dict):
                    scene[key] = {}

            plan = self._get_visual_plan(scene)
            primary = self._get_primary_element(plan)
            supporting = self._get_supporting_elements(plan)

            scene["caption"]["enabled"] = enabled.get(
                "subtitles", True
            )

            # Motion graphics are a core requirement.
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

            scene["visual_direction"] = self._derive_visual_metadata(
                scene=scene,
                plan=plan,
                primary=primary,
                supporting=supporting,
                audio_analysis=audio_analysis,
                style=style,
                scene_index=index,
                scene_count=scene_count,
            )

            icon = self._select_icon(scene)

            if icon:
                scene["icon"] = icon
            else:
                scene.pop("icon", None)

            # External media is never used.
            scene["external_media"] = {
                "required": False,
                "query": "",
                "enabled": False,
            }

            scene["decision_rationale"] = [
                "Scene timing and structure originate from narration analysis.",
                "Visual direction preserves the scene planner's contextual intent.",
                "Reference material is used exclusively for style guidance.",
                "Typography is selected from the local font library.",
                "Local icons are optional supporting elements.",
                "Motion graphics remain enabled for every scene.",
                "External footage, images, and reference media are prohibited.",
                "The supplied voiceover remains the narration source.",
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
