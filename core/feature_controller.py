"""Resolve feature gates. A missing optional credential disables its feature."""

from config.settings import Settings


def enabled_features(settings: Settings) -> dict[str, bool]:
    return {
        "subtitles": settings.enable_subtitles,
        "motion_graphics": settings.enable_motion_graphics,
        "transitions": settings.enable_transitions,
        "sound_effects": settings.use_sound_effects,
        "background_music": settings.use_background_music,
        "external_clips": settings.use_external_clips,
        "external_images": settings.enable_external_images and bool(settings.pexels_api_key),
        "preserve_original_voiceover": settings.preserve_original_voiceover,
    }
