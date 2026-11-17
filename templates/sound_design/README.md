Sound layers are mixed with FFmpeg by `services/audio_mixer.py`. Narration is
always the first and duration-governing stream. Background music and effects are
only accepted when their independent feature gates are enabled.
