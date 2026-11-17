# UNIVERSAL-VIDEO-EDITOR

Copyright © 2026 UNIVERSAL-VIDEO-EDITOR. All rights reserved.

Independent local Python editing pipeline for voiceover-led videos. This repository also contains the pre-existing Ideavo Next.js application; the Python project is implemented alongside it without modifying the web application.

## Architecture and current capabilities

Python makes editing decisions locally: reference videos are sampled for color, scene-change candidates, motion differences, and a low-confidence caption-region cue. Multi-reference aggregation uses medians, variability, coverage confidence, and agreement thresholds. Local Whisper produces a verbatim transcript and word times. The director uses deterministic rules to create continuous scene timings, captions, motion, transition and media instructions; timeline validation gates guide output. There is no hosted LLM and no AI key configuration.

The rendering path uses the official HyperFrames CLI to lint and render a locally generated HTML composition, then FFmpeg to mix the original narration and encode the final MP4. Output is FFprobe-probed and fully decode-checked by FFmpeg. When external visuals are enabled and a Pexels key is configured, only scenes marked as requiring footage are searched. Missing media or provider errors leave the scene on a generated typographic background.

Pixabay's documented public API offers photo and video search, but does not document music or sound-effect search. The project deliberately does not scrape undocumented endpoints; Pixabay-backed music and SFX retrieval are therefore unavailable. The audio mixer does accept local optional audio tracks when the corresponding feature is enabled.

## Requirements

- Python 3.11+ and pip, with PyTorch compatible with your hardware (a dependency of local Whisper).
- FFmpeg and FFprobe on `PATH`.
- Node.js 22+ and HyperFrames CLI 0.9.59+ (`npm install --global hyperframes@0.9.59`). HyperFrames renders HTML to video with its documented CLI (`hyperframes lint` and `hyperframes render -c ... -o ...`).
- First local Whisper model use downloads the selected model weights. This is local inference after download; no transcription API is called.

Install Python dependencies:

```sh
python -m pip install -r requirements.txt
python -m pip install '.[dev]'
```

Copy `.env.example` to `.env` and export/load its values using your shell or a local dotenv loader. The application reads environment variables; it does not automatically parse `.env`.

## Inputs and run

Put one original voiceover under `storage/voiceover/` and three or four reference videos under `storage/reference_video/`, or pass their paths directly:

```sh
python main.py --voiceover storage/voiceover/narration.wav \
  --reference storage/reference_video/ref1.mp4 \
  --reference storage/reference_video/ref2.mp4 \
  --reference storage/reference_video/ref3.mp4
```

This runs the full pipeline and writes `storage/final_video/narration.mp4`, the guide at `storage/editing_guides/editing_guide.json`, and validation metadata under `storage/metadata/`. The original voiceover remains the only narration. It is duration-governing and is never replaced with reference audio. `--plan-only` creates the guide without rendering. `--no-resume` ignores previous checkpoints. Checkpoints are reused only when all input hashes and relevant settings match and generated files still exist.

Optional locally supplied music/effects can be added without a provider key (as long as the corresponding `ENABLE_BACKGROUND_MUSIC` / `ENABLE_SOUND_EFFECTS` switch is true): use `--music path/to/music.mp3` and repeat `--sound-effect SECONDS=path/to/effect.wav`. Music is ducked under the original voiceover and fades at the edges; effects are placed at the requested timeline offset.

Optional Pexels retrieval uses `PEXELS_API_KEY`; Pixabay media calls use `PIXABAY_API_KEY`. Pexels is for external video only. The configured Pixabay key is not used for audio search because that endpoint is not present in its documented public API. No API key is needed for local algorithms or Whisper.

The default renderer builds original caption/title layouts and word highlighting from Whisper timestamps; style extraction currently has reliable visual statistics but cannot identify exact fonts or decode reference captions. Unsupported observations are marked unknown and use deterministic styling fallbacks. Without Pexels footage, scenes use typography-led graphic backdrops. External media is optional, not required for rendering.

## Tests and CI

Run `python -m pytest -q`, `python -m compileall -q ai config core providers services main.py tests`, `python scripts/check_project_structure.py`, and `python -m ruff check ai config core providers services tests main.py`. API tests are mocked and require no real provider keys. Real media integration requires the runtimes above and real inputs.

GitHub Actions `validate-project.yml` runs compilation, structural checks, tests and Ruff. `edit-video.yml` uses workflow dispatch, installs Python, Node, FFmpeg and HyperFrames, runs the real renderer, then uploads video/guide/metadata artifacts (seven-day retention). `workflow_dispatch` does not upload arbitrary files: provide input media through repository checkout or a separately provisioned storage mechanism. The current workflow expects `VOICEOVER_PATH` as a repository variable and reference media in `storage/reference_video/`. Large/private media should not be committed; artifact retention is limited.

## Security, privacy, and licensing

Credentials are never hardcoded or logged. Media retrieval is HTTPS-only, restricted to provider CDNs, size-limited, and records source/licensing metadata. Request timeouts and bounded retries are used. Never commit API keys or private media. Review Pexels terms and source attribution/license details for intended use. Third-party libraries, fonts and media remain subject to their own licenses; this project does not claim those rights.

The source code is proprietary. See `LICENSE` for restrictions. HyperFrames itself is Apache-2.0; this project does not alter HyperFrames or third-party license terms.

## Troubleshooting

- `FFmpeg not found` / `FFprobe not found`: install FFmpeg and ensure both executables are on `PATH`.
- `HyperFrames CLI is not installed`: install the pinned CLI with npm; Node.js must be >=22. `hyperframes doctor` checks its runtime prerequisites.
- Whisper import/model/device errors: install Python requirements, ensure PyTorch works on your system, and choose a smaller model or `WHISPER_DEVICE=cpu` when needed.
- Pexels missing key / rate limit / no results: the footage stage logs a warning and renders a typography fallback.
- Pixabay audio search: unavailable via documented public API; no undocumented endpoint or scraping fallback is used.
- Resume after a failure: rerun the same command. Changed input bytes/settings invalidate prior checkpoints.

## Known limitations

- Typography and caption detection in reference videos is a heuristic region score, not OCR or exact font recognition. Scene-change candidates may also reflect camera movement.
- No Pixabay background music/SFX provider can be implemented against its published public API; these tracks must currently be supplied locally.
- Pexels search/download and full HyperFrames/FFmpeg end-to-end rendering have not been verified in this sandbox unless specifically reported by the executed checks.
- Automatic local model download may require network access and substantial disk, RAM and CPU/GPU resources.
