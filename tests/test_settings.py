from config.settings import Settings


def test_optional_keys_disable_media_features():
    settings = Settings(pexels_api_key="", pixabay_api_key="")
    assert not settings.use_external_clips
    assert not settings.use_sound_effects
    assert not settings.use_background_music


def test_invalid_dimensions_are_rejected():
    try:
        Settings(video_width=0).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid output dimensions accepted")
