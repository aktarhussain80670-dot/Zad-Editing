# UNIVERSAL-VIDEO-EDITOR

Copyright © 2026 UNIVERSAL-VIDEO-EDITOR. All rights reserved.

An independent Python editing planner for voiceover-led videos. This repository
also contains the pre-existing Ideavo Next.js application; this Python project
lives at the same repository root without replacing that application.

## Current implementation

Local algorithms analyze sampled video color statistics, aggregate measurable
reference characteristics, transcribe a voiceover with a locally executed
Whisper model, produce deterministic narration-timed scene plans, apply feature
gates, validate timeline continuity, and write a JSON editing guide. No hosted
LLM or AI service is used. Whisper is used only for local transcription.

The current version does **not** yet render final MP4s, inspect typography or
caption animation, retrieve Pexels/Pixabay media, integrate HyperFrames, resume
checkpoints, or mix audio. Those capabilities are not represented as complete.

## Requirements and setup

- Python 3.11+, FFmpeg/FFprobe, and Node.js only if/when adding the planned
  HyperFrames renderer.
- Install Python dependencies: `python -m pip install -r requirements.txt`.
- Install Whisper model weights on first local use; model download is performed
  by Whisper at runtime. No API key is required. Choose `WHISPER_MODEL` based
  on hardware; larger models require more memory and compute.
- Copy `.env.example` to `.env` and export the values in your shell or load them
  with your preferred environment loader. Pexels/Pixabay keys are optional and
  unused by the current planner.

## Inputs and execution

Put one voiceover in `storage/voiceover/` and three or four references in
`storage/reference_video/`, or pass explicit paths:

```sh
python main.py --voiceover storage/voiceover/narration.wav \
  --reference storage/reference_video/ref1.mp4 \
  --reference storage/reference_video/ref2.mp4 \
  --reference storage/reference_video/ref3.mp4
```

The validated guide is written to `storage/editing_guides/editing_guide.json`.
Reference videos are read-only. This command only generates a plan, not a final
video.

## Testing

Install development dependencies with `python -m pip install '.[dev]'`, then run
`python -m pytest -q` and `python -m compileall -q ai config core main.py`.
Tests do not call external media APIs or require credentials.

## GitHub Actions

`validate-project.yml` compiles and tests. `edit-video.yml` currently generates
an editing guide only. Add inputs to the repository (not via workflow_dispatch,
which has no arbitrary MP4 upload control), set `VOICEOVER_PATH` as a repository
variable, and place three or four references under
`storage/reference_video/`. Download the guide as an Actions artifact; artifact
retention is seven days in this workflow. Do not commit private credentials or
large user media.

## Security and licensing

Never commit API credentials or private media. Third-party media must be used
only under provider terms and applicable licenses. The proprietary project
license does not change third-party terms. FFmpeg commands are invoked using
argument arrays (not shell interpolation) by implemented media operations.

## Limitations and next work

Current visual analysis reports sampled brightness, contrast, and saturation;
it does not infer exact fonts, text, or transitions. Rendering, provider
adapters, actual output verification, and safe checkpoint/resume are next
implementation phases. The existing web application remains independent.
