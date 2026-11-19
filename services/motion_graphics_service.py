"""Generate narration-driven HTML motion graphics for HyperFrames."""

from __future__ import annotations

import html
import math
import re
from pathlib import Path
from typing import Any


class MotionGraphicsService:
    """Build safe, category-aware visual components from scene plans."""

    VALID_CATEGORIES = {
        "comparison",
        "sequence",
        "timeline",
        "data_visualization",
        "animated_chart",
        "concept_explanation",
        "conceptual_graphic",
    }

    @staticmethod
    def _safe(value: Any, limit: int = 120) -> str:
        if value is None:
            return ""
        return html.escape(str(value).strip()[:limit])

    @staticmethod
    def _items(scene: dict[str, Any]) -> list[str]:
        plan = scene.get("visual_plan", {})
        if not isinstance(plan, dict):
            plan = {}

        primary = plan.get("primary_element", {})
        if not isinstance(primary, dict):
            primary = {}

        supporting = plan.get("supporting_elements", [])
        values: list[str] = []

        if isinstance(supporting, list):
            for item in supporting:
                if isinstance(item, dict):
                    value = (
                        item.get("label")
                        or item.get("text")
                        or item.get("title")
                    )
                else:
                    value = item

                if isinstance(value, str) and value.strip():
                    values.append(value.strip())

        primary_label = primary.get("label")
        if isinstance(primary_label, str) and primary_label.strip():
            values.insert(0, primary_label.strip())

        if not values:
            transcript = scene.get("transcript", "")
            if isinstance(transcript, str):
                values = re.findall(r"\b[\w'-]{3,}\b", transcript)

        return values[:5] or ["Main concept"]

    @staticmethod
    def _style_variables(
        style: dict[str, Any] | None,
    ) -> tuple[str, str, str]:
        """Return fixed project brand colors; reference colors cannot override them."""

        accent = "#2453D6"
        secondary = "#22C55E"

        typography = {}
        if isinstance(style, dict):
            profile = style.get("style_profile", {})
            if isinstance(profile, dict):
                typography = profile.get("typography", {})

        if not isinstance(typography, dict):
            typography = {}

        weight = str(typography.get("weight", "800"))
        if weight not in {"400", "500", "600", "700", "800", "900"}:
            weight = "800"

        return accent, secondary, weight

    @staticmethod
    def _chart_data(plan: dict[str, Any]) -> list[tuple[str, float]]:
        """Accept charts only when explicit, finite numerical data exists."""

        data = plan.get("data")
        if not isinstance(data, list) or not data:
            return []

        result = []

        for item in data[:8]:
            if not isinstance(item, dict):
                return []

            label = item.get("label")
            value = item.get("value")

            if not isinstance(label, str) or not label.strip():
                return []

            if isinstance(value, bool):
                return []

            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                return []

            if not math.isfinite(number):
                return []

            result.append((label.strip(), number))

        return result

    @staticmethod
    def _icon_markup(scene: dict[str, Any]) -> str:
        """Render an optional project SVG icon using a relative asset path."""

        icon = scene.get("icon")
        if not isinstance(icon, dict):
            return ""

        filename = icon.get("filename")
        if not isinstance(filename, str):
            return ""

        path = Path(filename)

        if (
            path.is_absolute()
            or path.suffix.lower() != ".svg"
            or len(path.parts) != 1
            or path.name != filename
        ):
            return ""

        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.svg", filename):
            return ""

        alt = MotionGraphicsService._safe(
            icon.get("alt", path.stem), 80
        )

        return (
            f'<img class="scene-icon" src="assets/icons/{html.escape(filename, quote=True)}" '
            f'alt="{alt}" aria-hidden="true">'
        )

    @staticmethod
    def _relationship_svg(labels: list[str]) -> str:
        """Create a small scene-specific relationship diagram from narration terms."""
        count = max(1, min(len(labels), 4))
        nodes = []
        for index, label in enumerate(labels[:count]):
            x = 540 if index == 0 else 180 + ((index - 1) * 240)
            y = 180 if index == 0 else 430
            nodes.append(
                f'<g class="semantic-node" transform="translate({x} {y})">'
                f'<circle r="54" fill="#FFFFFF" stroke="#2453D6" stroke-width="6"/>'
                f'<text text-anchor="middle" y="8" fill="#000000" font-size="22">{MotionGraphicsService._safe(label, 18)}</text>'
                '</g>'
            )
        lines = "".join(
            f'<path d="M540 235 L{x} 375" fill="none" stroke="#2453D6" stroke-width="5" stroke-linecap="round"/>'
            for x in [180 + (index * 240) for index in range(max(0, count - 1))]
        )
        return f'<svg class="semantic-diagram" viewBox="0 0 1080 620" role="img" aria-label="Narration relationship diagram">{lines}{"".join(nodes)}</svg>'

    def render_scene(
        self,
        scene: dict[str, Any],
        style: dict[str, Any] | None = None,
    ) -> str:
        """Return category-specific visual markup for one scene."""

        if not isinstance(scene, dict):
            raise TypeError("Scene must be a dictionary.")

        plan = scene.get("visual_plan", {})
        if not isinstance(plan, dict):
            plan = {}

        category = str(
            plan.get("category")
            or plan.get("visual_category")
            or "concept_explanation"
        )

        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Unsupported visual category: {category!r}. "
                "Provide a supported category or a valid composition."
            )

        items = self._items(scene)
        labels = [self._safe(item, 48) for item in items]

        title = self._safe(
            plan.get("headline")
            or scene.get("visual_idea")
            or items[0],
            90,
        )

        if category == "comparison" and len(labels) >= 2:
            content = (
                '<div class="compare-grid">'
                f'<div class="compare-card"><span class="visual-index">01</span><h2>{labels[0]}</h2></div>'
                '<div class="compare-divider">VS</div>'
                f'<div class="compare-card"><span class="visual-index">02</span><h2>{labels[1]}</h2></div>'
                '</div>'
            )

        elif category == "sequence":
            content = '<div class="sequence-list">' + "".join(
                f'<div class="sequence-item"><span>{index:02d}</span><strong>{label}</strong></div>'
                for index, label in enumerate(labels, 1)
            ) + '</div>'

        elif category == "timeline":
            content = '<div class="timeline-track">' + "".join(
                f'<div class="timeline-point"><span>{index:02d}</span><strong>{label}</strong></div>'
                for index, label in enumerate(labels, 1)
            ) + '</div>'

        elif category in {"data_visualization", "animated_chart"}:
            chart_data = self._chart_data(plan)

            if chart_data:
                maximum = max(abs(value) for _, value in chart_data)
                maximum = maximum or 1.0

                bars = "".join(
                    '<div class="chart-column">'
                    f'<div class="chart-bar" style="--bar-height:{abs(value) / maximum * 100:.2f}%"></div>'
                    f'<span>{self._safe(label, 48)}</span>'
                    f'<strong>{value:g}</strong>'
                    '</div>'
                    for label, value in chart_data
                )
                content = (
                    '<div class="chart-panel">'
                    f'<div class="chart-columns">{bars}</div>'
                    '</div>'
                )
            else:
                content = (
                    '<div class="abstract-visual">'
                    '<div class="abstract-orbit orbit-one"></div>'
                    '<div class="abstract-orbit orbit-two"></div>'
                    '<div class="abstract-core"></div>'
                    f'<div class="abstract-label">{title}</div>'
                    '</div>'
                )

        else:
            nodes = "".join(
                f'<div class="concept-node node-{index}">'
                f'<span>{index + 1:02d}</span><strong>{label}</strong>'
                '</div>'
                for index, label in enumerate(labels)
            )
            content = f'<div class="concept-map">{self._relationship_svg(items)}{nodes}</div>'

        accent, secondary, font_weight = self._style_variables(style)
        icon_markup = self._icon_markup(scene)

        return (
            f'<section class="motion-scene category-{self._safe(category, 40)}" '
            f'style="--accent:{accent};--secondary:{secondary};--visual-weight:{font_weight}">'
            f'{icon_markup}'
            f'<div class="visual-heading">{title}</div>'
            f'<div class="visual-content">{content}</div>'
            '</section>'
        )

    @staticmethod
    def animation_script(
        element_id: str,
        start: float,
        duration: float,
        style: dict[str, Any] | None = None,
    ) -> str:
        """Adapt scene animation intensity to measured reference motion."""

        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,80}", element_id):
            raise ValueError("Invalid animation element ID.")

        if not math.isfinite(start) or not math.isfinite(duration):
            raise ValueError("Animation timing must be finite.")

        if start < 0 or duration <= 0:
            raise ValueError("Animation start and duration are invalid.")

        profile = (
            style.get("style_profile", {})
            if isinstance(style, dict)
            else {}
        )
        if not isinstance(profile, dict):
            profile = {}

        motion = profile.get("motion", "unknown")

        motion_settings = {
            "high_motion": (0.36, 90, 1.04),
            "fast_cut_pacing": (0.30, 70, 1.03),
            "moderate_motion": (0.55, 48, 1.02),
            "subtle_motion": (0.80, 20, 1.005),
            "unknown": (0.55, 48, 1.02),
        }

        entrance_base, travel, scale = motion_settings.get(
            motion, motion_settings["unknown"]
        )

        entrance = min(entrance_base, max(0.08, duration * 0.35))
        remaining = max(0.05, duration - entrance)
        selector = f"#{element_id}"

        return (
            f'tl.fromTo("{selector}", '
            f'{{opacity:0, y:{travel}, scale:0.96}}, '
            f'{{opacity:1, y:0, scale:1, duration:{entrance:.3f}, ease:"power3.out"}}, '
            f'{start});'
            f'tl.to("{selector}", '
            f'{{y:{-travel * 0.18:.2f}, scale:{scale}, duration:{remaining:.3f}, ease:"none"}}, '
            f'{start + entrance});'
        )
