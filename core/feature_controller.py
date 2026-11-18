"""Resolve production feature flags for universal motion-graphics editing."""

from config.settings import Settings


def enabled_features(settings: Settings) -> dict[str, bool]:
    """Return enabled features while enforcing the production media policy."""
    return {
        "subtitles": settings.enable_subtitles,
        "motion_graphics": settings.enable_motion_graphics,
        "transitions": settings.enable_transitions,
        "sound_effects": settings.use_sound_effects,
        "background_music": settings.use_background_music,
        "external_clips": False,
        "external_images": False,
        "preserve_original_voiceover": True,
    }
