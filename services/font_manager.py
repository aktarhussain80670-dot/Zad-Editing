"""Offline font discovery, role-based selection and HyperFrames packaging."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote


class FontManager:
    """Discover local fonts and package selected families for compositions."""

    SUPPORTED_EXTENSIONS = {".ttf", ".otf"}

    ROLE_PREFERENCES = {
        "heading": ("bold", "semibold", "condensed", "sans"),
        "title": ("bold", "display", "sans", "regular"),
        "caption": ("regular", "sans", "roboto", "source"),
        "number": ("mono", "bold", "sans"),
        "label": ("medium", "semibold", "sans", "regular"),
    }

    EXCLUDED_NAME_TERMS = (
        "emoji",
        "flags",
        "symbols",
        "math",
    )

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = (project_root or Path.cwd()).resolve()
        self.font_directory = self.project_root / "assets" / "fonts"
        self._catalog: list[dict[str, Any]] | None = None

    @staticmethod
    def _family_name(path: Path) -> str:
        name = re.sub(r"[^a-zA-Z0-9_-]", "", path.stem)
        digest = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:8]
        return f"Editor_{name or 'LocalFont'}_{digest}"

    @staticmethod
    def _style(filename: str) -> str:
        name = filename.lower()
        if "italic" in name or "oblique" in name:
            return "italic"
        return "normal"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _weight(filename: str) -> str:
        name = filename.lower()

        if "black" in name or "extrabold" in name:
            return "900"
        if "semibold" in name or "demibold" in name:
            return "600"
        if "bold" in name:
            return "700"
        if "medium" in name:
            return "500"
        if "light" in name:
            return "300"

        return "400"

    def discover(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Discover supported font files directly inside assets/fonts."""

        if refresh:
            self._catalog = None

        if self._catalog is not None:
            return [dict(font) for font in self._catalog]

        if not self.font_directory.is_dir():
            raise FileNotFoundError(
                f"Local font directory not found: {self.font_directory}"
            )

        catalog = []

        for path in sorted(self.font_directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            if any(
                term in path.stem.lower()
                for term in self.EXCLUDED_NAME_TERMS
            ):
                continue

            catalog.append(
                {
                    "name": path.stem,
                    "family": self._family_name(path),
                    "path": path.resolve(),
                    "filename": path.name,
                    "format": path.suffix.lower(),
                    "weight": self._weight(path.stem),
                    "style": self._style(path.stem),
                    "license_status": "unverified",
                }
            )

        if not catalog:
            raise RuntimeError("No usable local TTF or OTF fonts were discovered.")

        self._catalog = catalog
        return [dict(font) for font in catalog]

    def select_font(
        self,
        role: str = "caption",
        style_hint: str | None = None,
    ) -> dict[str, Any]:
        """Select a local font using a visual role and optional style hint."""

        if role not in self.ROLE_PREFERENCES:
            raise ValueError(f"Unsupported font role: {role}")

        catalog = self.discover()
        hint = (style_hint or "").lower()
        preferences = self.ROLE_PREFERENCES[role]

        def score(font: dict[str, Any]) -> tuple[int, str]:
            name = font["name"].lower()
            points = sum(
                3 for token in preferences if token in name
            )
            points += sum(
                5 for token in hint.split() if token in name
            )

            if role in {"heading", "title"} and font["weight"] in {"700", "900"}:
                points += 4

            if role == "caption" and font["weight"] == "400":
                points += 4

            return points, name

        return max(catalog, key=score)

    def prepare(
        self,
        composition_directory: Path,
        selections: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """Package selected local fonts and return composition-relative CSS."""

        composition_directory = composition_directory.resolve()
        font_directory = composition_directory / "fonts"
        font_directory.mkdir(parents=True, exist_ok=True)

        selected = selections or {
            role: self.select_font(role)
            for role in self.ROLE_PREFERENCES
        }

        css_rules = []
        role_variables = []

        packaged = {}

        allowed_weights = {"300", "400", "500", "600", "700", "900"}
        allowed_styles = {"normal", "italic"}

        for role, font in selected.items():
            if role not in self.ROLE_PREFERENCES:
                raise ValueError(f"Unsupported font role: {role}")

            if not isinstance(font, dict):
                raise TypeError(f"Font selection for {role} must be a dictionary.")

            source = Path(font["path"]).resolve()
            allowed_directory = self.font_directory.resolve()

            try:
                source.relative_to(allowed_directory)
            except ValueError as exc:
                raise ValueError(
                    f"Font source must be inside assets/fonts: {source}"
                ) from exc

            if not source.is_file():
                raise FileNotFoundError(f"Selected font does not exist: {source}")

            if source.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported font format: {source.name}")

            family = font.get("family")
            weight = str(font.get("weight", ""))
            font_style = font.get("style", "normal")

            if not isinstance(family, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_-]{0,100}", family
            ):
                raise ValueError(f"Invalid CSS font family for role: {role}")

            if weight not in allowed_weights:
                raise ValueError(f"Invalid font weight for role: {role}")

            if font_style not in allowed_styles:
                raise ValueError(f"Invalid font style for role: {role}")

            filename = source.name
            destination = font_directory / filename

            if source != destination.resolve():
                if destination.exists():
                    if (
                        not destination.is_file()
                        or self._sha256(source) != self._sha256(destination)
                    ):
                        raise FileExistsError(
                            f"Existing composition font content differs: {destination}"
                        )
                else:
                    shutil.copy2(source, destination)

            font_format = (
                "opentype"
                if source.suffix.lower() == ".otf"
                else "truetype"
            )

            if filename not in packaged:
                css_rules.append(
                    "@font-face {"
                    f"font-family:'{family}';"
                    f"src:url('fonts/{quote(filename, safe='-_.')}') format('{font_format}');"
                    f"font-style:{font_style};"
                    f"font-weight:{weight};"
                    "font-display:block;"
                    "}"
                )
                packaged[filename] = family

            role_variables.append(
                f"--font-{role}:'{family}',sans-serif;"
            )

        css_rules.append(
            ":root {" + "".join(role_variables) + "}"
        )

        return "\n".join(css_rules)
