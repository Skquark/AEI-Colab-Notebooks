# Changelog

All notable changes to this repository are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `OpenVoice-V2_Colab.ipynb` — first voice-conversion (audio → audio)
  notebook in the suite, wrapping
  [myshell-ai/OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2).
  MIT-licensed instant voice cloning by **MIT + MyShell**, 36.6k★.
  VITS-based tone-color converter that takes a 5-30 s reference clip
  and applies its **timbre** to any source audio, with native
  multilingual support for English, Spanish, French, Chinese, Japanese,
  Korean (via MeloTTS as the base speaker). Six tabs:
  - **Convert** — audio in + ref audio in → audio out. `tau` slider
    controls how strongly the target timbre overrides the source.
  - **TTS + Convert** — type text in any of 7 languages, get it spoken
    in the reference voice.
  - **Style Controls** — explainer on what OpenVoice can and can't do
    (timbre vs. emotion vs. accent)
  - **Batch** — convert every audio file in a directory with one ref
    voice
  - **VRAM** — release loaded V1/V2/MeloTTS models
  - **Help** — multilingual notes, watermarking, comparison with
    RVC/SoVITS
  - **Two model versions in one notebook**: V2 (default, multilingual)
    and V1 (simpler, EN/ZH only)
  - **Every output is watermarked** with a 32-bit string at 16 kbps
    via `wavmark` (default `@MyShell`)
- **New "Voice Conversion" section in the README** — TTS is text→audio;
  voice conversion is audio→audio. They're complementary, so the new
  section lives just below the TTS suite.

### Added (prior in this cycle)
- `Kokoro-82M_Colab.ipynb` — self-contained Colab wrapper around
  [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), the
  Apache-2.0 82M-param TTS model. CPU-friendly, ~330 MB on disk,
  54 voices across 9 languages. Six tabs:
  - **Generate** — text + voice + speed; **Random Quote / Gatsby /
    Frankenstein** one-click fill buttons from upstream Space assets
  - **Stream** — long texts auto-chunked and concatenated; each chunk
    saved individually
  - **Tokens** — G2P-only inspect mode (no audio, no VRAM)
  - **Batch** — one .wav per line, downloaded as a zip
  - **VRAM** — release loaded language pipelines
  - **Help** — pronunciation syntax `[word](/IPA/)`, voice codes,
    language codes, BibTeX citation
  - **Custom pronunciation** via Markdown-link syntax
    `[Kokoro](/kˈOkəɹO/)`
  - **Voice blending**: pass `af_bella,am_michael` to average two voices
  - Sample assets (en.txt, gatsby5k.md, frankenstein5k.md) fetched
    from the upstream Space and cached in `/content/kokoro_samples/`

### Fixed
- README polish: `23 notebook(s)` → `22 notebook(s)` in the Tools legend
  to match the actual file count
- README: `comercialmente` typo (`commercially` × 3 occurrences across the
  Wan 2.2 sub-model license sections)
- README: TTS intro now says "**ten models**" (was "nine") to match
  the new Kokoro-82M entry
- README: stale `Skquark/Pixal3D` link in the model-cards table
  replaced with the real upstream `TencentARC/Pixal3D`
  ([arXiv 2605.10922](https://arxiv.org/abs/2605.10922))
- README: removed redundant trailing "Apache 2.0." in the Kokoro-82M
  section (already stated at the top of the paragraph)

### Added (prior in this cycle)
- `Wan2.2_S2V_Colab.ipynb` — a self-contained Colab wrapper around
  [Tencent / Wan-AI's Wan 2.2 S2V](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B),
  the 14 B MoE audio-driven video generation model. Takes a
  character image + audio clip and produces a cinematic video of
  the character speaking/singing the audio. Trained on both
  speech and singing per the [Wan-S2V paper](https://arxiv.org/abs/2508.18621).
  Two execution paths:
  - **🌐 Cloud API (default)** — uses Alibaba's DashScope API.
    No GPU needed, ~$0.10-0.50/call, works on any Colab runtime.
    Same backend the [official HF Space](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V) uses
    (the cloud runs a distilled model, so it's cheaper than the
    full 14B).
  - **💻 Local inference (heavy)** — clones the official
    [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) repo,
    downloads 27 GB of weights, runs on A100 80GB or H100.
  Inputs: character image + audio (required), text prompt
  (optional), pose video (optional, for pose-driven generation).
  Resolutions: 480P (cheaper) and 720P (higher quality). Audio
  length determines output video length. Step 7 batch reads a
  folder of (image, audio) pairs and submits one cloud job
  per pair.
- `Wan 2.2 S2V` section in `README.md` between the Wan 2.2 Animate
  and Text-to-Speech sections. New TOC entry under `### Video`.
  New row in the Video Models attribution table (with both arXiv
  links: 2508.18621 for the S2V paper and 2503.20314 for the
  Wan 2.2 base paper).
- `tools/qa_check.py` updated to include the new notebook in
  the audit list.

### Added (Wan 2.2 Animate)
- `Wan2.2_Animate_Colab.ipynb` — a self-contained Colab wrapper around
  [Tencent / Wan-AI's Wan 2.2 Animate](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B),
  the 14 B MoE model for **character animation and replacement**.
  Takes two inputs (character image + reference video) and outputs
  a video in either **Move** mode (animate the character with the
  video's motion) or **Mix** mode (replace the character in the
  video with the one in the image).
  Two execution paths in one notebook:
  - **🌐 Cloud API (default)** — uses Alibaba's DashScope API.
    No GPU needed, ~$0.10-0.50/call, works on any Colab runtime.
    Same backend the [official HF Space](https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate) uses.
  - **💻 Local inference (heavy)** — clones the official
    [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) repo,
    downloads 28 GB of weights, runs on A100.
  Two quality levels: `wan-pro` (25 fps, 720p) and `wan-std`
  (15 fps, 720p, cheaper). Step 7 batch reads a folder of
  (image, video) pairs and submits one cloud job per pair.
- `Wan 2.2 Animate` section in `README.md` between the Wan 2.2
  and Text-to-Speech sections. New TOC entry under `### Video`.
  New row in the Video Models attribution table.
- `tools/qa_check.py` updated to include the new notebook in
  the audit list.

### Changed
- `Wan2.2-5B_Colab.ipynb` was replaced by the more general
  `Wan2.2_Colab.ipynb`. Same 9-cell layout, expanded features.
- **Two official variants in one notebook** instead of just the 5B:
  - `5B-TI2V` (default, 10 GB on disk, fits 24 GB GPUs with offload)
  - `14B-A14B` (heavy, 27 GB on disk, needs 40+ GB VRAM, downloads
    on demand when picked in the UI)
- **Enhancement features borrowed from the `ginigen/Wan-2.2-Enhanced`**
  Space UX:
  - **6 preset style buttons** (Cinematic / Animation / Nature /
    Slow Motion / Action / Portrait) that prepend a tuned style
    to the user's prompt with one click
  - **Resolution preset dropdown** — 6 common aspect ratios
    (1280×704, 704×1280, 832×480, 480×832, 1280×720, 720×1280)
    plus Custom
  - **Negative prompt field** — most video models benefit from
    one. Default is tuned to the upstream Space.
  - **Auto-enhance prompt toggle** — when on, prepends a
    `smooth, fluid motion` baseline if the prompt has no motion
    keywords. Off by default.
  - **Smart resolution auto-pick** — when an image is uploaded,
    the resolution dropdown auto-snaps to the closest matching
    preset.
- `tools/qa_check.py` updated to point at the renamed notebook.

### Added
- `Wan2.2-5B_Colab.ipynb` — a self-contained Colab wrapper around
  [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers),
  the 5 B dense Text+Image-to-Video variant of Wan 2.2 (Alibaba).
  Both T2V and I2V in one model. 720p @ 24 fps, up to 5-second
  clips. Apache 2.0. Uses the official `diffusers` main-branch
  port (`WanPipeline` + `AutoencoderKLWan`) so no ComfyUI /
  custom SDK needed. Fits a 24 GB L4 with sequential CPU offload.
  Tabs for Text-to-Video, Image-to-Video, and VRAM. Smart
  resolution auto-calculation from uploaded images (matches
  upstream Space UX). Step 7 batch reads a `.txt` of prompts.
  Drive-cached weights at `AEI_TTS_Cache/huggingface/hub/`.
- `Wan 2.2 5B` section in `README.md` between the Notebook
  Generator and Text-to-Speech sections. New `### Video`
  sub-section in the TOC. New `### Video Models` row in the
  Model Cards & Upstream Attribution table.
- `tools/qa_check.py` updated to include the new notebook in
  the audit list.

### Added (CI)
- **CI workflow** at `.github/workflows/qa.yml` that runs on every
  push to `main` and every PR. Three jobs:
  1. `validate` — runs `tools/validate.py` (fast AST-parse check)
  2. `qa-audit` — runs `tools/qa_check.py` (full polish audit)
  3. `size-report` — posts a per-notebook KB / cell-count table to
     the GitHub Actions job summary
  Uses Python 3.12, no extra dependencies. Cancels in-progress
  runs of the same workflow on the same branch.
- **QA status badge** at the top of `README.md` linking to the
  Actions run.
- **Tools** section in `README.md` documenting `validate.py` +
  `qa_check.py` and how to invoke them.
- **Live cost estimator** in `Hunyuan3D-3_Colab.ipynb` Step 4.
  Heuristic per-call cost based on `generate_type` + `face_count` +
  `enable_pbr`, displayed next to the Generate button and updated
  live as you change inputs. Also tracks session total cost and job
  count in the Status tab, with a Reset button.
- **Optional "More info" block** in `Notebook_Generator.ipynb`.
  New spec fields (`hf_card_url`, `github_url`, `arxiv_url`,
  `citation`) appear in the generated notebook's header as a
  "### More info" section with links + a BibTeX fenced block. The
  Qwen3-TTS preset ships pre-filled with the official Qwen HF card,
  GitHub repo, and a citation stub. The generator's Step 4 form
  has a new "Optional: More info block" accordion for these fields.
- **`Model Cards & Upstream Attribution`** section in `README.md`
  with a 5-row 3D table and a 9-row TTS table, all linking to the
  upstream GitHub repos, Hugging Face model cards, and arXiv
  papers. Consolidates attribution in one place so users can dig
  deeper without hunting.
- `tools/validate.py` — fast AST-parse check on every code cell.
  Exits 0 / 1. Lenient on the 3 pre-existing notebooks that don't
  follow the 9-cell pattern (Pixal3D_Wheel_Builder, TTS_Model_Loader,
  TTS_Voice_Library). For 9-cell notebooks it enforces the exact
  cell ID convention. Wire into CI / pre-commit.
- `tools/qa_check.py` — full polish audit (info tooltips, try/except,
  concurrency, clear_output, demo.load, FileLink in Step 6).
  Excludes the 2 pre-existing Pixal3D notebooks. Print a tidy
  per-notebook table with findings called out.
- Updated `CONTRIBUTING.md` to reference the two new tools and the
  generator workflow ("How to add a new notebook" now starts with
  "use `Notebook_Generator.ipynb`").

### Changed
- `Hunyuan3D-3_Colab.ipynb` Step 4: added session cost tracking
  (the `Status` tab now shows jobs / spend instead of just GPU
  info), and the Generate buttons show per-call cost on the
  result line.

### Added (continued)
- `Hunyuan3D-3_Colab.ipynb` — a self-contained Colab wrapper around
  the [Tencent Cloud Hunyuan 3D Global API v3](https://www.tencentcloud.com/document/product/1665/119114),
  adapted from [exedesign/Hunyuan-3D-v3](https://github.com/exedesign/Hunyuan-3D-v3)
  (which was built for ComfyUI). The flagship 3D model is not yet
  open-weights, but the Global API is available for anyone with a
  paid Tencent Cloud account. This notebook gives you Text-to-3D
  and Image-to-3D in a single 7-step Colab pattern with no GPU
  or weight download. Configurable: PBR materials, face count
  (40K-1.5M), generate type (Normal/LowPoly/Geometry/Sketch),
  polygon type (triangle/quadrilateral). Credentials loaded from
  Colab secrets (`TENCENT_SECRET_ID`, `TENCENT_SECRET_KEY`).
  Outputs cached to Drive. Step 7 batch supports both text-prompt
  lists and image folders. Cost: ~$0.10-0.60 per request,
  prominently bannered in the UI.
- `Hunyuan3D 3.0` section in `README.md` between the Hunyuan3D-2.1
  and Hunyuan3D-2 sections. Cross-references both local-weight
  notebooks for users who'd rather not pay for the API.
- `Notebook_Generator.ipynb` — a meta-tool that scaffolds new
  9-cell AEI Colab notebooks from a spec. Fill in the form in
  Step 4, click **Generate**, and a fully-formed notebook (Drive
  cache, lazy engine wrapper, info= tooltips, demo.load welcome,
  batch try/except, and the same `step1-install` .. `step7-batch`
  pattern used by every other notebook) gets written to
  `/content/<Name>_Colab.ipynb`. Includes 8 curated presets
  (Qwen3-TTS, Higgs-Audio, MisoTTS, Trellis, Step1X-3D, TripoSG,
  Wan 2.1, HunyuanVideo). Modality-aware templates for TTS, 3D,
  and image/video. Step 6 self-tests the engine by building and
  validating the Qwen3 preset. Step 7 batch-generates all 8
  presets in one click.
- `Notebook Generator` section in `README.md` between
  `Hunyuan3D-2` and `Text-to-Speech`. TOC entry added.

### Changed
- Cross-notebook polish pass across all 16 notebooks. Audited for
  `info=` tooltips, try/except coverage, `concurrency_limit`,
  `clear_output()` before launch, `demo.load` welcome. Added 12
  tooltips to Qwen3-TTS (the flagship) and 5 to TTS_Voice_Library;
  added `demo.load` welcome to both Hunyuan3D notebooks. Fixed a
  source-line collapse bug in Hunyuan3D-2 (cell source list was a
  single 16 KB flat string instead of properly newline-terminated
  entries). All authored notebooks are now audit-clean. The
  pre-existing Pixal3D and Pixal3D_Wheel_Builder notebooks were
  intentionally left alone (we only fixed the orphan `except`
  bug in Pixal3D earlier).

### Added
- `Hunyuan3D-2.1_Colab.ipynb` — full self-contained wrapper around
  [Tencent's Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
  (the production successor to 2.0). Brings the 3.3 B shape DiT
  (`hunyuan3d-dit-v2-1`) and the 2 B PBR texture pipeline
  (`hunyuan3d-paintpbr-v2-1`) into a 7-step Colab pattern. PBR pipeline
  generates albedo + metallic + roughness maps. Tabs for
  Shape / Shape+PBR / Texture-existing / VRAM. `custom_rasterizer` and
  `mesh_inpaint_processor` CUDA extensions built from source; graceful
  fallback to disabled PBR when nvcc is missing. RealESRGAN x4plus
  upscaler auto-fetched. `torchvision` compatibility patch from
  upstream `torchvision_fix.py` applied at import time. Drive-cached
  weights at `AEI_TTS_Cache/huggingface/hub/`. 6 example images
  pre-fetched from upstream.
- `Hunyuan3D-2.1` section in `README.md` (positioned as flagship) with
  GPU support matrix and Quick Start. Cross-references the existing
  2.0 notebook for users on smaller GPUs.
- `Hunyuan3D-2_Colab.ipynb` — full self-contained wrapper around
  [Tencent's Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).
  Seven DiT variants (`2mini`, `2mini-turbo`, `2`, `2-turbo`, `2-fast`,
  `2mv`, `2mv-turbo`) + Hunyuan3D-Paint for texturing. Tabs for
  Shape-only / Shape+Texture / Texture-existing / VRAM. Custom
  CUDA extension (`custom_rasterizer`) built from source with
  graceful fallback to disabled Paint when nvcc is missing. Drive-
  cached weights at `AEI_TTS_Cache/huggingface/hub/`. 6 example
  images pre-fetched from upstream repo.
- `Hunyuan3D-2` section in `README.md` with GPU support matrix and
  Quick Start. License: [Tencent Hunyuan Community License](https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE.txt)
  (non-commercial research use).
- `Cube_3D_Colab.ipynb` — combined notebook for Roblox's Cube 3D v0.5
  (text → 3D, ~12 GB) and CubePart (multi-part decomposition, ~10 GB).
  Single Drive-cached cache directory; tabs UI with separate VRAM controls.
  9 cells following the standard 7-step pattern.
- Pre-fetched 5 example .glb inputs (drone, flower, jellyfish_car,
  pirate_chest, wizard) + PNG thumbnails in Step 2, so the CubePart
  tab's Examples gallery is immediately usable without a fresh user mesh.
- `gr.Gallery` of examples in the CubePart tab that loads the picked
  mesh + parts list into the inputs on click (mirrors the upstream HF
  Space UI).
- Clarifying intro markdown at the top of the CubePart tab explaining
  what it does (mesh → parts) and what it doesn't (text → mesh, which
  is not yet open-sourced).
- Step 6 Quick Test now does a Cube 3D → CubePart end-to-end round-trip
  so the user can see the full pipeline in one cell.
- `tqdm` added to the Step 1 pip install (used by the upstream
  diffusion loop in CubePart).
- Cube 3D + CubePart section in `README.md` with GPU support matrix
  and Quick Start
- `EXAMPLE_MESHES` shared between Step 2 (downloader) and Step 4 (Gallery)

### Updated
- README: clarified that CubePart is decomposition-only, not text-to-mesh
- README Quick Start: Step 2 now also pre-fetches example meshes
- README Table of Contents: added Cube 3D + CubePart
- License section: added Cube 3D / CubePart attribution (MIT code,
  OpenRAIL-M weights, no high-risk downstream use)
- 9 new TTS notebooks in the suite:
  - `Qwen3-TTS_Colab.ipynb` — flagship, 3 modes (CustomVoice / VoiceDesign / Base), Apache 2.0
  - `Higgs-Audio_Colab.ipynb` — 4B conversational TTS, 100+ langs, Research/Non-Commercial
  - `MisoTTS_Colab.ipynb` — 8B Sesame CSM, English only, watermarked
  - `Supertonic-3_Colab.ipynb` — 99M ONNX, CPU-only, 31 langs
  - `Dia_Colab.ipynb` — 1.6B multi-speaker dialogue, 21 non-verbal tags
  - `IndicF5_Colab.ipynb` — 400M, 11 Indian languages, gated
  - `MOSS-TTS_Colab.ipynb` — 8B, 31 langs, IPA/Pinyin, `[pause X.Ys]` markers
  - `dots.tts-soar_Colab.ipynb` — 2B continuous-VAE, 48 kHz, Apache 2.0
  - `Fish-S2-Pro_Colab.ipynb` — 5B Dual-AR, 80+ langs, 15K inline emotion tags
- 2 shared infrastructure notebooks:
  - `TTS_Model_Loader.ipynb` — pre-caches all model weights to Google Drive
  - `TTS_Voice_Library.ipynb` — curated reference voice clips with transcripts
- `LICENSE` (MIT) with third-party model attribution
- `CONTRIBUTING.md` with notebook conventions (cell IDs, launch pattern, etc.)
- `CHANGELOG.md` (this file)
- Table of Contents + per-notebook sections + GPU support table in `README.md`
- `.gitignore` for local caches (`pixal3d_wheels/`, `__pycache__`, etc.)

### Changed (TTS suite polish)
- All 9 TTS model notebooks now use the Google Drive cache pattern (was
  in-VM only, lost on session restart). They read weights from
  `/content/drive/MyDrive/AEI_TTS_Cache/huggingface/` (where
  `TTS_Model_Loader` puts them) so the "Run TTS_Model_Loader first" story
  actually works across sessions.
- Cell IDs normalized to `step1-install`, `step2-cache`, `step3-core`,
  `step4-ui`, `step5-keepalive`, `step6-quicktest`, `step7-batch` across
  all 9 TTS notebooks for stable deep-linking
- `Pixal3D_Colab` Step 1: removed leftover `except ImportError:` that
  prevented the cell from parsing
- MisoTTS Step 1: added `ffmpeg` apt-get install (needed for torchaudio
  to load m4a/mp3 reference clips)
- Fish-S2-Pro Step 1: added VRAM check + SystemExit on insufficient GPU
- `Higgs` Step 2: `LoRA` field added to cloning accordion
- All TTS `Step 7` batch loops: added `try/except` around each
  iteration so a single bad line no longer kills the whole batch
- Dia `top_k` parameter was a dead slider (wasn't passed to the model)
- MisoTTS `MIMI_FRAME_SIZE` was reading a non-existent attribute
  (`_audio_tokenizer.frame_size`); replaced with `int(0.08 * SAMPLE_RATE)`
- Fish-S2-Pro `Examples` block had inconsistent tuple arity (1 vs 3 elements)
- Fish-S2-Pro `Gradio` outputs were overwriting error with timing message
- TTS_Voice_Library `voice_dd` dropdown wasn't refreshed when filters changed
- TTS_Voice_Library `count_md` wasn't updated on filter change
- TTS_Model_Loader used non-existent `hf` CLI and dummy login (now uses
  `snapshot_download` Python API)
- TTS_Voice_Library used `client_id` fallback for VCTK (VCTK uses
  `speaker_id` only)
- MisoTTS `eff_temp` was based on toggle, not actual transcribe result
- MisoTTS had no seed handling (now wires through UI/Step 6/Step 7)
- MisoTTS status markdown showed literal `\n` instead of newlines
- Em-dash inconsistency across notebooks (some had literal `\u2014`)
- Inconsistent keep-alive patterns (now all use `requests.get`)

### UX polish
- All TTS notebooks now use a standard launch block with
  `clear_output()`, `concurrency_limit=2`, and a `demo.load` welcome message
- All TTS Gradio UIs now use `info=` tooltips on every slider and ref audio
  (e.g., explains what `top_p=0.95` means)
- Qwen3-TTS button labels differentiated by tab ("Generate (CustomVoice)"
  vs "Generate (VoiceDesign)" vs "Generate (Clone)")
- Qwen3-TTS now has a `gr.Examples` block (was the only notebook without one)
- MisoTTS Examples block now has a hint about the voice continuation flow
- MOSS-TTS Step 2 title cleaned up ("Pre-cache Models" instead of
  "Pre-cache Models and Example Reference Clips")

## [1.0.0] — 2026-05-29

### Added
- Initial release
- `Pixal3D_Colab.ipynb` — image-to-3D with PBR textures, 1536 px resolution
- `Pixal3D_Wheel_Builder.ipynb` — builds CUDA extension wheels for
  `o_voxel`, `cumesh`, `flex_gemm`, `nvdiffrec_render` and uploads to
  GitHub Releases
- GitHub Release assets: `wheels-a100-v1.0`, `wheels-l4-v1.0`,
  `wheels-t4-v1.0` (each ~40 MB, prebuilt for sm80/sm89/sm75)
