# AEI Colab Notebooks

[![QA](https://github.com/Skquark/AEI-Colab-Notebooks/actions/workflows/qa.yml/badge.svg)](https://github.com/Skquark/AEI-Colab-Notebooks/actions/workflows/qa.yml)

Free, self-contained Google Colab notebooks for AI-powered 3D generation, video creation, text-to-speech, and voice conversion. No sign-ups. No tokens. Just works.

See [LICENSE](LICENSE) for terms. [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a notebook. [CHANGELOG.md](CHANGELOG.md) for what's changed. [Model Cards & Upstream Attribution](#model-cards--upstream-attribution) for credits.

## Contents

### 3D
- [Pixal3D — Image to 3D with PBR Textures](#pixal3d--image-to-3d-with-pbr-textures)
- [Pixal3D Wheel Builder](#pixal3d-wheel-builder)
- [Cube 3D + CubePart](#cube-3d--cubepart)
- [Hunyuan3D 3.0 — Tencent Cloud API](#hunyuan3d-30--tencent-cloud-api-wrapper)
- [Hunyuan3D-2.1 — Tencent PBR 3D Pipeline *(flagship)*](#hunyuan3d-21--tencent-pbr-3d-pipeline-flagship)
- [Hunyuan3D-2 — Tencent Image / Text-to-3D](#hunyuan3d-2--tencent-image--text-to-3d)
- [Notebook Generator — scaffold new model notebooks](#notebook-generator--scaffold-new-model-notebooks)

### Text-to-Speech
- [Qwen3-TTS *(flagship)*](#qwen3-tts)
- [Higgs-Audio](#higgs-audio)
- [MisoTTS](#misotts)
- [Supertonic-3](#supertonic-3)
- [Dia](#dia)
- [IndicF5](#indicf5)
- [MOSS-TTS v1.5](#moss-tts-v15)
- [dots.tts-soar](#dotstts-soar)
- [Fish S2 Pro](#fish-s2-pro)
- [VoxCPM2](#voxcpm2)
- [Kokoro-82M](#kokoro-82m)
- [TTS Model Loader](#tts-model-loader)
- [TTS Voice Library](#tts-voice-library)

### Voice Conversion
- [OpenVoice V2](#openvoice-v2)

### Video
- [Wan 2.2 — Text & Image-to-Video](#wan-22--text--image-to-video)
- [Wan 2.2 Animate — Character Animation & Replacement](#wan-22-animate--character-animation--replacement)
- [Wan 2.2 S2V — Audio-Driven Cinematic Video](#wan-22-s2v--audio-driven-cinematic-video)

---

## Pixal3D — Image to 3D with PBR Textures

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Pixal3D_Colab.ipynb)

State-of-the-art image-to-3D generation by **Tencent ARC Lab** (SIGGRAPH 2026). Pixal3D lifts pixel features directly into 3D through back-projection, establishing pixel-to-3D correspondences that produce near-reconstruction-quality geometry with full PBR (Physically Based Rendering) textures — base color, roughness, metallic, and opacity. Built on Microsoft's TRELLIS.2 4B backbone.

### What You Get

- **GLB mesh output** with 4-channel PBR texture atlas (up to 4096×4096)
- **Gradio web UI** — interactive generation with preview renders
- **Batch processing** — process entire directories of images, resume-aware
- **Manual FOV control** — override camera field-of-view when auto-detection gets it wrong
- **HDRI preview renders** — forest, sunset, and courtyard lighting for turntable previews
- **Low-VRAM mode** — runs on L4 (22 GB) and T4 (15 GB) at 1024 px
- **Quality-first defaults** — A100 (40 GB) uses 1536 px resolution with 12 sampling steps

### GPU Support

| GPU | VRAM | Resolution | Mode | Time (approx) |
|-----|------|-----------|------|---------------|
| A100 | 40 GB | 1536 px | Standard | ~60 seconds |
| L4 | 24 GB | 1024 px | Low-VRAM | ~90 seconds |
| T4 | 16 GB | 1024 px | Low-VRAM | ~120 seconds |

The notebook auto-detects which GPU you're on and adjusts settings accordingly. All three GPU types are supported.

### Quick Start

1. Open the notebook in Colab (click the badge above)
2. Connect to a GPU runtime — A100 recommended for best quality
3. Run all cells in order (Runtime → Run all)
4. Upload an image to the Gradio UI (Step 4) or use batch processing (Step 7)

### How It Works

Pixal3D uses a three-stage cascade pipeline, each powered by a diffusion transformer (DiT):

1. **Sparse Structure** — generates a coarse 3D structure from pixel-aligned image features
2. **Shape Refinement** — upsamples to 1024³ voxel resolution with detailed geometry
3. **Texture Generation** — synthesizes full PBR material maps (albedo, roughness, metallic, opacity)

Camera pose is estimated automatically using MoGe-2, and background removal is handled by BiRefNet. Output meshes are simplified, remeshed, and exported as standard GLB files ready for Blender, Unity, Unreal, or any 3D tool.

### Model Credits

| Component | Source | License |
|-----------|--------|---------|
| Pixal3D Pipeline | [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) | MIT |
| TRELLIS.2 Backbone | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | MIT |
| MoGe-2 Camera | [Ruicheng/moge-2-vitl](https://huggingface.co/Ruicheng/moge-2-vitl) | CC BY-NC-SA 4.0 |
| DinoV3 Features | [camenduru/dinov3-vitl16](https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m) | Apache 2.0 |
| BiRefNet Background | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | MIT |

---

## Pixal3D Wheel Builder

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Pixal3D_Wheel_Builder.ipynb)

Builds the CUDA extension wheels that power the Pixal3D notebook. Run once per GPU type — the resulting wheels are uploaded to GitHub Releases and automatically installed by the main notebook.

**Packages built:** `o_voxel` (mesh conversion) · `cumesh` (GPU mesh processing) · `flex_gemm` (sparse convolution) · `nvdiffrec_render` (PBR renderer, optional)

### GPU Wheels

Prebuilt wheels are hosted as GitHub Release assets, one release per GPU architecture:

| GPU | Architecture | Release |
|-----|-------------|---------|
| A100 | sm80 | [`wheels-a100-v1.0`](https://github.com/Skquark/AEI-Colab-Notebooks/releases/tag/wheels-a100-v1.0) |
| L4 | sm89 | [`wheels-l4-v1.0`](https://github.com/Skquark/AEI-Colab-Notebooks/releases/tag/wheels-l4-v1.0) |
| T4 | sm75 | [`wheels-t4-v1.0`](https://github.com/Skquark/AEI-Colab-Notebooks/releases/tag/wheels-t4-v1.0) |

Each release is ~40 MB. The main notebook auto-detects your GPU and downloads the correct one — nothing to configure.

---

## Cube 3D + CubePart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Cube_3D_Colab.ipynb)

Two complementary 3D pipelines from [Roblox's Cube project](https://github.com/Roblox/cube), open-sourced as part of their push toward 3D intelligence:

- **Cube 3D v0.5** — autoregressive text-to-shape. Given a natural-language prompt (and a target bounding box), produces a single `.glb` 3D mesh. VQ-VAE tokenizer + GPT-style shape transformer. ~12 GB of weights.
- **CubePart** — open-vocabulary part-controllable **mesh decomposition** (not text-to-mesh). Given a canonically-aligned `.glb` mesh and a list of part names (e.g. `body, front left wheel, front right wheel, headlights`), produces a colored `trimesh.Scene` with one independent mesh per part. SIGGRAPH 2026 paper. ~10 GB of weights. **Note:** the upstream open-source release is Stage 2 only (decomposition); Stage 1 (text → full mesh latent) is not yet released. To try CubePart, use the pre-fetched example meshes, the output of Cube 3D, or your own `.glb`.

The notebook has tabs for each pipeline and a **VRAM** tab to free the engine you don't need. Both pipelines can run sequentially on a 16 GB GPU; running them simultaneously needs ≥24 GB.

### Quick Start

1. **Pick the GPU runtime** — L4 (24 GB) or A100 (40 GB) recommended. T4 (16 GB) is the minimum for running one pipeline at a time.
2. **Run Steps 1-4 in order.** Step 1 clones the Roblox/cube repo and installs dependencies. Step 2 downloads ~22 GB of weights + 5 pre-built example meshes to your Google Drive. Step 3 defines the inference wrappers. Step 4 launches the Gradio UI.
3. **Type a prompt in the Cube 3D tab** → get a single mesh. Then switch to the **CubePart tab** and either click a thumbnail in the Examples gallery or upload your own mesh.

### GPU Support

| GPU | VRAM | Both engines | Cube 3D only | CubePart only |
|-----|------|--------------|--------------|---------------|
| **A100** | 40 GB | ✅ Comfortable | ✅ | ✅ |
| **L4** | 24 GB | ✅ Tight | ✅ | ✅ |
| **T4** | 16 GB | ❌ (OOM) | ✅ | ✅ |
| **CPU / Mac** | — | ❌ | ❌ (EngineFast is CUDA-only) | ❌ |

---

## Hunyuan3D-2.1 — Tencent PBR 3D Pipeline *(flagship)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Hunyuan3D-2.1_Colab.ipynb)

The **production-ready successor** to Hunyuan3D-2.0, repackaged as a 7-step Colab notebook with Drive cache. The 2.1 release brings two major upgrades: a 3.3 B shape DiT (vs 2.0's 1.1 B) and a 2 B **PBR** (physically-based-rendering) texture pipeline that generates albedo + metallic + roughness maps. The result is a `.glb` mesh that responds to lighting realistically — metallic reflections, subsurface scattering — not just a diffuse map.

- **Shape generation (3.3 B DiT, v2-1)** — flow-matching transformer, ~6.5 GB on disk, ~10 GB VRAM.
- **PBR texture synthesis (2 B Hunyuan3D-Paint-2.1)** — multi-view diffusion that bakes albedo + metallic + roughness into a single UV-mapped texture, ~4 GB on disk, ~21 GB VRAM.
- **Mesh cleanup** — `FloaterRemover`, `DegenerateFaceRemover`, `FaceReducer` keep the output low-poly and printable.
- **Texture an existing mesh** — upload any `.glb / .obj / .ply` + a reference image, get it re-textured with PBR. Useful for re-texturing outputs from Cube 3D, CubePart, etc.

### Quick Start

1. **Pick the GPU runtime** — A100 (40 GB) or L4 (24 GB, tight for full pipeline) recommended. T4 (16 GB) can run the shape pipeline only.
2. **Run Steps 1-4 in order.** Step 1 clones the 2.1 repo, builds the `custom_rasterizer` + `mesh_painter` CUDA extensions, downloads RealESRGAN, and applies the `torchvision` compatibility fix needed by `basicsr`. Step 2 pre-caches weights (~10 GB) + 6 example images. Step 3 defines the lazy-loading engine wrappers. Step 4 launches the Gradio UI.
3. **In the Shape tab**: drop in an image (or pick one from the Examples gallery) and click **Generate Shape**. Then switch to **Shape + PBR Texture** to add a PBR texture. Use the VRAM tab to swap pipelines in and out of GPU memory.

### GPU Support

| GPU | VRAM | Shape only (3.3 B) | Shape + PBR (~29 GB) |
|-----|------|--------------------|-----------------------|
| **A100** | 40 GB | ✅ | ✅ (comfortable) |
| **L4** | 24 GB | ✅ | ⚠️ (tight, may OOM on big meshes) |
| **T4** | 16 GB | ✅ | ❌ (OOM) |
| **CPU / Mac** | — | ❌ (CUDA only) | ❌ |

To drop the PBR component on a 16 GB GPU, edit `COMPONENTS = ['shape-2.1']` in Step 2 (and the Paint tab will be disabled).

### License

Model weights: [Tencent Hunyuan Community License](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE.txt) — non-commercial research use. Code: [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) is MIT-licensed. The RealESRGAN x4 upscaler is BSD-3-Clause.

### Related notebook

`Hunyuan3D-2_Colab.ipynb` is the older 2.0 wrapper — keep it for the lighter `2mini-turbo` (0.6 B), `2-turbo` (1.1 B, fast), and `2mv-turbo` (multi-view) variants that fit on smaller GPUs.

---

## Hunyuan3D 3.0 — Tencent Cloud API Wrapper

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Hunyuan3D-3_Colab.ipynb)

A Colab-friendly wrapper around the [Tencent Cloud Hunyuan 3D Global API v3](https://www.tencentcloud.com/document/product/1665/119114), adapted from [exedesign/Hunyuan-3D-v3](https://github.com/exedesign/Hunyuan-3D-v3) (which was built for ComfyUI). Tencent's flagship 3D model is not yet open-weights — but the **Global API** is available for anyone to call with a paid Tencent Cloud account. This notebook lets you generate `.glb` meshes from **text prompts or images** without any local GPU or weight download.

- **Text-to-3D** and **Image-to-3D** in a single 7-step Colab notebook
- **Configuration**: PBR materials (on/off), face count (40K to 1.5M), generate type (`Normal` / `LowPoly` / `Geometry` / `Sketch`), polygon type (triangle/quadrilateral)
- **No GPU, no weight download** — the heavy work is on Tencent's side; we just poll a job-id
- **Colab-secrets credential loading** — `TENCENT_SECRET_ID` and `TENCENT_SECRET_KEY` are read from the secrets panel, never hard-coded
- **Drive-cached outputs** — downloaded `.glb` files mirror to `/content/drive/MyDrive/AEI_3D_Out/Hunyuan3D-3/`
- **Cost banner** prominently shown in Step 4 — ~$0.10-0.60 per request, billed to your Tencent Cloud account
- **Step 7 batch** supports both text-prompt lists (`.txt` files) and image folders

### Quick Start

1. **Set up a Tencent Cloud international account** at [console.intl.cloud.tencent.com](https://console.intl.cloud.tencent.com/) and create a SecretId / SecretKey under CAM > API Keys.
2. **Activate the Hunyuan 3D service** in the AI Services console.
3. **Add credit** (~$10-20 recommended) so you have headroom for the batch tab.
4. **Add the credentials to Colab secrets** (left sidebar → 🔑 icon): `TENCENT_SECRET_ID` and `TENCENT_SECRET_KEY`.
5. **Open the notebook, run Steps 1-4 in order.** Step 1 installs the SDK and reads the credentials, Step 2 does a connectivity probe, Step 3 defines the client, Step 4 launches the Gradio UI.
6. **Use Text-to-3D or Image-to-3D tab**, watch the progress bar, download the `.glb`.

### Cost

~**$0.10-0.60 per request**, billed to your Tencent Cloud account. The notebook has no built-in cost controls — set a billing alert on the Tencent side before running batches. We recommend adding **$10-20** of credit before running a batch.

### License

API usage is governed by Tencent's terms. Model output is yours to use in your own projects, subject to Tencent's acceptable use policy.

---

## Hunyuan3D-2 — Tencent Image / Text-to-3D

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Hunyuan3D-2_Colab.ipynb)

A self-contained wrapper around [Tencent's Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) — the official Tencent Hugging Face Space turned into a 7-step Colab notebook with a Drive cache. The pipeline turns a single image (or several views) into a clean, optionally textured `.glb` mesh in seconds.

- **Shape generation (DiT flow-matching)** — five model variants covering the 0.6 B mini shape expert through the 1.1 B full model, with and without step-distillation (`-Turbo`) and multi-view (`-mv`).
- **Texture synthesis (Hunyuan3D-Paint)** — paints a high-resolution diffuse texture onto a white mesh using a stable-diffusion-style multi-view pipeline. Optional: requires building a small CUDA extension (`custom_rasterizer`) on Step 1. If the build fails, the rest of the notebook still works; the Paint tab will tell you so.
- **Mesh cleanup** — `FloaterRemover`, `DegenerateFaceRemover`, and `FaceReducer` are wired in so the output is a low-poly, watertight mesh you can drop into a game engine or Three.js.
- **Texture an existing mesh** — there's a dedicated tab where you upload any `.glb / .obj / .ply` mesh + a reference image and Paint handles the rest. Useful for re-texturing outputs from Cube 3D, CubePart, or any other source.

### Quick Start

1. **Pick the GPU runtime** — L4 (24 GB) or A100 (40 GB) recommended. T4 (16 GB) works for `2mini-turbo` and `2-turbo` (shape only).
2. **Run Steps 1-4 in order.** Step 1 clones the Tencent repo and builds the optional Paint CUDA extension. Step 2 pre-caches weights + 6 example images to Google Drive. Step 3 defines the lazy-loading engine wrappers. Step 4 launches the Gradio UI.
3. **In the Shape tab**: pick a model variant in the sidebar, drop in an image (or pick one from the Examples gallery), and click **Generate Shape**. Use the VRAM tab to swap pipelines in and out of GPU memory.

### GPU Support

| GPU | VRAM | `2mini-turbo` (0.6 B) | `2-turbo` (1.1 B) | `2-turbo` + Paint | `2` (1.1 B full) | `2mv-turbo` (1.1 B) |
|-----|------|-----------------------|-------------------|--------------------|------------------|----------------------|
| **A100** | 40 GB | ✅ | ✅ | ✅ | ✅ | ✅ |
| **L4** | 24 GB | ✅ | ✅ | ✅ (tight) | ✅ | ✅ |
| **T4** | 16 GB | ✅ | ✅ | ❌ (OOM) | ❌ (OOM) | ❌ (OOM) |
| **CPU / Mac** | — | ❌ (CUDA only) | ❌ | ❌ | ❌ | ❌ |

Default variant in the UI: **`2mini-turbo`** (smallest, fastest, safest for T4). To use a heavier variant, edit `VARIANTS = ['2mini-turbo']` in Step 2 to add `'2-turbo'` and/or `'2'`, then re-run. Model weights are pulled from `tencent/Hunyuan3D-2`, `tencent/Hunyuan3D-2mini`, and `tencent/Hunyuan3D-2mv` on Hugging Face.

### License

Model weights: [Tencent Hunyuan Community License](https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE.txt) — non-commercial research use. Code: [Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) is MIT-licensed.

---

## Notebook Generator — scaffold new model notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Notebook_Generator.ipynb)

A **meta-tool** for adding new models to the AEI-Colab-Notebooks suite. Fill in a short spec form, click **Generate**, and a fully-formed 9-cell Colab notebook (Drive cache, lazy engine wrapper, info= tooltips, demo.load welcome, batch try/except, and the same `step1-install` .. `step7-batch` pattern used by every other notebook in the repo) gets written to `/content/<Name>_Colab.ipynb`. **Bootstrap a new model notebook in under 2 minutes**, then open it and fill in the model-specific `infer()` call in Step 3.

- **9-cell scaffold** matching every other notebook in the repo (so it passes the same audit)
- **Curated presets** for 8 upcoming models (Qwen3-TTS, Higgs-Audio, MisoTTS, Trellis, Step1X-3D, TripoSG, Wan 2.1, HunyuanVideo)
- **Modality-aware** UI: TTS → audio, 3D → mesh, image/video → file output
- **Built-in validation**: generator's Step 6 runs the Qwen3 preset through `build_notebook` and confirms the output parses
- **Batch mode**: Step 7 emits all 8 presets to `/content` in one click

After generating, the only thing left to do is open the new notebook, replace the `=== TODO ===` markers in Step 3's `load()` and `infer()` methods with the actual model API, and the UI works.

---

## Wan 2.2 — Text & Image-to-Video

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Wan2.2_Colab.ipynb)

A self-contained Colab wrapper around [Tencent / Wan-AI's Wan 2.2](https://github.com/Wan-Video/Wan2.2) video model. **Two officially-released variants in one notebook**, plus a layer of prompt-enhancement features borrowed from the `ginigen/Wan-2.2-Enhanced` Space (preset style buttons, resolution dropdown, negative prompt, auto-enhance, smart resolution).

- **Two variants in one notebook**:
  - **`5B-TI2V`** (default, 10 GB on disk, fits 24 GB GPUs with sequential CPU offload) — fast, recommended for most Colab users
  - **`14B-A14B`** (heavy, 27 GB on disk, needs 40+ GB VRAM) — highest-quality MoE, slow. Downloads on demand when you pick it in the UI.
- **Two modes per variant**: Text-to-Video (T2V) and Image-to-Video (I2V)
- **720p @ 24 fps output**, up to 5 seconds (121 frames) per call
- **Apache 2.0 license** on both variants
- **Official diffusers port** — uses `WanPipeline` + `AutoencoderKLWan` from the `diffusers` main branch (installed from git, not PyPI, because the Wan classes aren't in a stable release yet)
- **Drive-cached weights** — 10 GB (5B) or 27 GB (14B) on first run, then instant
- **Step 7 batch** reads a `.txt` of prompts and generates one `.mp4` per line

### Enhancement features (from `ginigen/Wan-2.2-Enhanced`)

- **6 preset style buttons** (Cinematic / Animation / Nature / Slow Motion / Action / Portrait) that prepend a tuned style to your subject with one click
- **Resolution preset dropdown** — pick from 6 common aspect ratios (1280×704, 704×1280, 832×480, 480×832, 1280×720, 720×1280) or drop to Custom for the sliders
- **Negative prompt field** — most video models benefit from one. Default is tuned to the upstream Space.
- **Auto-enhance prompt toggle** — when on, prepends a `smooth, fluid motion` baseline if your prompt has no motion keywords. Off by default.
- **Smart resolution auto-pick** — when you upload an image, the dropdown auto-snaps to the closest matching preset.

### Why these specific variants

The 8 user-suggested HF Spaces for Wan 2.2 boil down to 3 distinct options:
- **5B TI2V (this notebook, default)** — 10 GB, fits 24 GB GPUs, Apache 2.0, both T2V and I2V
- **14B-A14B MoE (this notebook, opt-in)** — 27 GB, highest quality, 40+ GB VRAM, Apache 2.0, both T2V and I2V
- **Wan 2.2 S2V** (sound-to-video) and **Wan 2.2 Animate** (video-to-video) are different sub-tasks and need separate notebooks (future work)

The community Spaces (Lightning, MoE-Distill, AOTI) all wrap the same model weights with different optimizers — no new model, no new architecture, just a speed/memory tradeoff. We've kept the official diffusers port because it's the most maintainable.

### GPU Support

| GPU | VRAM | 5B (720p / 121 frames) | 14B-A14B (720p / 121 frames) | Notes |
|-----|------|------------------------|------------------------------|-------|
| **A100** | 40 GB | ✅ ~3-5 min | ✅ ~10-30 min | Only GPU that can host the 14B comfortably |
| **L4** | 24 GB | ✅ ~5-8 min | ❌ (OOM) | Recommended minimum for 5B with offload |
| **T4** | 16 GB | ⚠️ ~15 min at 480p | ❌ (OOM) | Drop to 480p; works but slow |

### Quick Start

1. Pick the **L4 (24 GB)** runtime for the 5B variant, or **A100 (40 GB)** for the 14B-A14B. T4 (16 GB) can run the 5B at 480p with offload but it will be slow.
2. Run Steps 1-4 in order — Step 1 installs diffusers from main + all deps, Step 2 caches the 5B weights to Drive, Step 3 defines the lazy-loading pipeline, Step 4 launches the Gradio UI.
3. Open the public Gradio URL, type a prompt (and optionally upload an image), pick a preset style if you want, and click **Generate**. The 14B downloads on demand if you pick it in the UI.

### License

Apache 2.0 — the most permissive of all the Wan 2.2 sub-models. Outputs are yours to use commercially.

---

## Wan 2.2 Animate — Character Animation & Replacement

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Wan2.2_Animate_Colab.ipynb)

A self-contained Colab wrapper around [Tencent / Wan-AI's Wan 2.2 Animate](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) — the **14 B MoE** model for **character animation and replacement**. The model takes **two inputs**: a **character image** (the subject) and a **reference video** (the motion), and produces a video in either **Move** mode (animate the character to mimic the video's motion) or **Mix** mode (replace the character in the video with the one in the image). Apache 2.0.

- **Two execution paths in one notebook**:
  - **🌐 Cloud API (default)** — uses Alibaba's DashScope API. No GPU needed, ~$0.10-0.50/call, works on any Colab runtime. **Same backend the [official HF Space](https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate) uses.**
  - **💻 Local inference (heavy)** — clones the [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) repo, downloads 28 GB of weights, runs preprocessing + inference locally on an A100.
- **Two task modes**:
  - **Move**: animate the character image with the motion from the reference video
  - **Mix**: replace the character in the video with the one in the image
- **Two quality levels**:
  - `wan-pro` — 25 fps, 720p
  - `wan-std` — 15 fps, 720p (cheaper)

### Quick Start (cloud path, recommended)

1. **Sign up for Alibaba Cloud's DashScope** at [dashscope.aliyun.com](https://dashscope.aliyun.com/) and activate the **wan2.2-animate-move** and **wan2.2-animate-mix** services.
2. **Add your API key** to Colab secrets (left sidebar → 🔑) as `DASHSCOPE_API_KEY`.
3. Run Steps 1-4 in order — Step 1 installs requests, Step 2 reads the key, Step 3 defines the client, Step 4 launches the Gradio UI.
4. Open the public URL, upload a character image + a reference video, pick a mode (move or mix) and a quality (pro or std), click **Generate**. ~$0.10-0.50 per request, billed to your DashScope account.

### Quick Start (local path, A100 only)

1. Switch the **Mode** dropdown to **💻 Local** at the top of Step 4.
2. Click **Pre-load local engine** — clones the official repo + downloads 28 GB of weights. This takes 10-30 minutes.
3. Pick the **A100 (40 GB)** runtime. The local path needs 40+ GB of VRAM.
4. Same UI — the local tab does the actual inference on the loaded weights.

### License

Apache 2.0 — the model weights are yours to use commercially. The cloud API is subject to Alibaba Cloud's terms of service.

---

## Wan 2.2 S2V — Audio-Driven Cinematic Video

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Wan2.2_S2V_Colab.ipynb)

A self-contained Colab wrapper around [Tencent / Wan-AI's Wan 2.2 S2V](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) — the **14 B MoE** audio-driven video generation model. Take a **character image** + **audio clip** and produce a cinematic video of the character speaking, singing, or reacting to the audio. Apache 2.0.

- **Two execution paths in one notebook**:
  - **🌐 Cloud API (default)** — uses Alibaba's DashScope API. No GPU needed, ~$0.10-0.50/call, works on any Colab runtime. **Same backend the [official HF Space](https://huggingface.co/spaces/Wan-AI/Wan2.2-S2V) uses** (the cloud runs a distilled model, so it's cheaper than the full 14B).
  - **💻 Local inference (heavy)** — clones the [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) repo, downloads 27 GB of weights, runs locally. Needs **A100 (80 GB)** or H100.
- **Two required inputs**: character image + audio clip
- **Optional advanced inputs**: text prompt (describes the scene), pose video (for pose-driven generation, drives body motion)
- **Two resolution levels**: 480P (cheaper, faster) and 720P (higher quality)
- **Audio length determines video length** — the output video is as long as the input audio
- **Trained for both speech and singing** — the paper specifically benchmarks against Hunyuan-Avatar and Omnihuman on cinematic character animation

### Quick Start (cloud path, recommended)

1. **Sign up for Alibaba Cloud's DashScope** at [dashscope.aliyun.com](https://dashscope.aliyun.com/) and activate the **wan2.2-s2v** service.
2. **Add your API key** to Colab secrets (left sidebar → 🔑) as `DASHSCOPE_API_KEY`.
3. Run Steps 1-4 in order — Step 1 installs requests, Step 2 reads the key, Step 3 defines the client, Step 4 launches the Gradio UI.
4. Open the public URL, upload a character image + an audio clip, pick 480P or 720P, click **Generate**. The video length is determined by the audio length.

### Quick Start (local path, A100 80GB or H100 only)

1. Switch the **Mode** dropdown to **💻 Local** at the top of Step 4.
2. Click **Pre-load local engine** — clones the official repo + downloads 27 GB of weights. This takes 10-30 minutes.
3. Pick the **A100 (80 GB)** runtime. Local S2V needs 80+ GB of VRAM (this is the heaviest of all the Wan 2.2 sub-models).
4. Same UI — the local tab does the actual inference on the loaded weights.

### License

Apache 2.0 — the model weights are yours to use commercially. The cloud API is subject to Alibaba Cloud's terms of service.

---

## Text-to-Speech

A matching lineup of self-contained Colab notebooks for state-of-the-art text-to-speech — **eleven models** covering English, Chinese, French, German, Spanish, Japanese, Korean, 11 Indian languages, 30 languages (VoxCPM2), and 80+ others. Every notebook follows the same 7-step pattern: **Install → Pre-cache → Core functions → Gradio UI → Keep-alive → Quick test → Batch synthesis**. No token or sign-up needed for any notebook except `IndicF5` (HF-gated). See [OpenVoice V2](#openvoice-v2) below for **voice conversion** (audio → audio with a cloned voice), which complements the TTS suite.

### Quick Start

1. **Run `TTS_Model_Loader.ipynb` first** to pre-download model weights to Google Drive (~40–45 GB for the full suite, resumable, per-notebook toggles)
2. **Run `TTS_Voice_Library.ipynb`** to grab a curated set of reference voice clips with transcripts (used by the voice-cloning tabs)
3. Open any of the model notebooks below — they auto-load from the Drive cache and skip the weight-download step

### GPU Support

| GPU | VRAM | Notebooks that fit | Notes |
|-----|------|--------------------|-------|
| **A100** | 40 GB | All 11 | Recommended flagship target |
| **L4** | 24 GB | All 11 (MisoTTS, Fish S2 Pro need bf16 + careful memory) | Best price/perf for cloning |
| **T4** | 16 GB | 9 of 11 (excludes MisoTTS, Fish S2 Pro) | Qwen3 auto-falls back to 0.6B variant |
| **CPU** | — | Supertonic-3, Kokoro-82M | 99M ONNX / 82M StyleTTS — runs anywhere |

### Qwen3-TTS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Qwen3-TTS_Colab.ipynb)

*Flagship* — Apache 2.0 TTS from Alibaba in three modes:

- **CustomVoice** — 9 premium voices (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee) covering Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **VoiceDesign** — describe a new voice in natural language ("Male, 17, tenor, gaining confidence…") and the model generates a matching voice
- **Base (Clone)** — 3-second reference clip + transcript → clone any voice

Auto-picks the 0.6B variant for T4 (16 GB) and the 1.7B variant for L4/A100.

### Higgs-Audio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Higgs-Audio_Colab.ipynb)

`bosonai/higgs-audio-v3-tts-4b` (via the `multimodalart/higgs-audio-v3-tts-4b-transformers` port) — conversational/expressive 4B TTS in 100+ languages with zero-shot voice cloning. The notebook auto-transcribes your reference clip with Moonshine ASR on CPU (no extra setup), so voice cloning is one-click. Pure 🤗 Transformers — no SGLang server required. *Research & Non-Commercial.*

### MisoTTS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/MisoTTS_Colab.ipynb)

`MisoLabs/MisoTTS` (8B) — Sesame-style CSM architecture for highly emotive conversational speech. Voice continuation from prompt audio, Mimi codec, **SilentCipher watermark** built in. English only. **L4/A100 only** (T4 too small for the 8B backbone).

### Supertonic-3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Supertonic-3_Colab.ipynb)

`Supertone/supertonic-3` (99M, ONNX) — 31 languages, 10 voices, 44.1 kHz, with expression tags (`<laugh>`, `<breath>`, `<sigh>`). **CPU-only** — no GPU needed, runs on the free Colab tier. MIT code, OpenRAIL-M model weights.

### Dia

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Dia_Colab.ipynb)

`nari-labs/Dia-1.6B-0626` (1.6B) — ultra-realistic multi-speaker dialogue with **21 non-verbal tags** (`(laughs)`, `(coughs)`, `(sighs)`, `(gasps)`, etc.) and 5–10s voice cloning. English only. Apache 2.0.

### IndicF5

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/IndicF5_Colab.ipynb)

`ai4bharat/IndicF5` (400M) — polyglot TTS for **11 Indian languages** (Hindi, Tamil, Bengali, Marathi, Punjabi, Kannada, Malayalam, Telugu, Odia, Gujarati, Assamese) with cross-language zero-shot voice cloning. **Gated — needs `HF_TOKEN` in Colab secrets** (see Step 1). MIT.

### MOSS-TTS v1.5

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/MOSS-TTS_Colab.ipynb)

`OpenMOSS-Team/MOSS-TTS-v1.5` (8B) — 31 languages, voice cloning, **duration control** (specify target token count), **IPA / Pinyin input** for pronunciation, inline `[pause X.Ys]` markers, and code-switching. Apache 2.0.

### dots.tts-soar

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/dots.tts-soar_Colab.ipynb)

`rednote-hilab/dots.tts-soar` (2B) — **fully continuous** autoregressive TTS (no discrete tokens anywhere in the pipeline) with **48 kHz studio-quality output** and a frozen AudioVAE + BigVGAN-style decoder. Achieves the **best SIM on the 24-language MiniMax multilingual benchmark** (83.9 average). Continuation cloning (with transcript) and x-vector-only cloning (without). Apache 2.0.

### Fish S2 Pro

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Fish-S2-Pro_Colab.ipynb)

`fishaudio/s2-pro` (5B: 4B slow + 400M fast Dual-AR) — 80+ languages, **15,000+ free-form inline emotion/prosody tags** (`[whisper]`, `[low voice in small room]`, `[with strong accent]`, `[pitch up]` …), native multi-speaker `<|speaker:N|>` tokens, multi-turn dialogue. RL-aligned with GRPO. **L4/A100 only** (T4 OOMs). *Fish Audio Research License — non-commercial.*

### Kokoro-82M

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Kokoro-82M_Colab.ipynb)

`hexgrad/Kokoro-82M` (82M) — Apache 2.0, **runs comfortably on CPU** (GPU optional), 54 voices across 9 languages (English 🇺🇸🇬🇧, Spanish 🇪🇸, French 🇫🇷, Hindi 🇮🇳, Italian 🇮🇹, Brazilian Portuguese 🇧🇷, Japanese 🇯🇵, Mandarin 🇨🇳). Six tabs:

- **Generate** — text + voice + speed; **Random Quote / Gatsby / Frankenstein** one-click fill buttons
- **Stream** — long texts auto-chunked and concatenated; each chunk saved individually
- **Tokens** — G2P-only inspect mode (no audio generated, no VRAM used)
- **Batch** — one .wav per line, downloaded as a zip
- **VRAM** — release loaded language pipelines
- **Help** — pronunciation syntax `[word](/IPA/)`, voice codes, language codes, citation

**Custom pronunciation** via Markdown-link syntax: `[Kokoro](/kˈOkəɹO/)` → the grapheme `Kokoro` is replaced with the IPA inside the parens. **Voice blending**: pass `af_bella,am_michael` to average two voices.

### VoxCPM2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/VoxCPM2_Colab.ipynb)

`openbmb/VoxCPM2` (2B) — **tokenizer-free** TTS from [OpenBMB](https://github.com/OpenBMB/VoxCPM) (Tsinghua / ModelBest inc), built on a [MiniCPM-4](https://huggingface.co/openbmb/MiniCPM4-0.5B) backbone. Models speech in a **continuous latent space** (no discrete tokens), enabling four flagship capabilities: **plain TTS** with content-aware prosody, **Voice Design** from a text description alone, **Controllable Cloning** from a short reference clip, and **Ultimate Cloning** that preserves every vocal nuance. **30 languages** natively + 9 Chinese dialects, **48 kHz studio-quality** output, ~8 GB VRAM. Apache-2.0 — free for commercial use. 27.3k★.

Eight tabs:

- **TTS** — plain text → speech, voice inferred from content. Works in all 30 languages.
- **Voice Design** — type a description in parens (e.g. `(A young woman, gentle)Hello!`) → model creates a matching voice from scratch, no ref audio needed.
- **Voice Clone** — short ref clip (5-30 s) → clone the timbre. V2 only.
- **Ultimate Clone** — ref audio + transcript + prompt audio → preserve every vocal nuance. V2 only.
- **Streaming** — long texts streamed chunk-by-chunk (RTF ~0.3 on RTX 4090).
- **Batch** — one .wav per line, downloaded as a zip.
- **VRAM** — free loaded model.
- **Help** — 30 languages, 4 modes, tuning knobs, benchmarks, citation.

**Two model versions in one notebook**:
- **VoxCPM2** (default, 2B, 30 langs, 48 kHz, ~8 GB VRAM) — recommended
- **VoxCPM-0.5B** (legacy, 0.5B, ZH/EN only, 16 kHz, ~5 GB VRAM) — for T4/edge devices

**Optional text normalization** (WeTextProcessing) for natural handling of numbers and abbreviations. **Optional ZipEnhancer denoiser** for noisy reference audio. Achieves the **highest average SIM** of any open-source TTS on the MiniMax Multilingual benchmark across 20+ languages.

## Voice Conversion

A single dedicated voice-conversion notebook, since it's a fundamentally different task from TTS: take an **input audio** (any language, any speaker) and a short **reference audio** of the target voice, and output the input audio *as if it were spoken in the target voice*. Used for dubbing, character voicing, multilingual content, anonymization, and creative audio workflows.

### OpenVoice V2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/OpenVoice-V2_Colab.ipynb)

`myshell-ai/OpenVoiceV2` — instant voice cloning by **MIT + MyShell**. VITS-based tone-color converter that takes a 5–30 s reference clip and applies its **timbre** to any source audio, with **native multilingual support for English, Spanish, French, Chinese, Japanese, Korean** (via MeloTTS as the base speaker). 36.6k★, MIT licensed — **free for commercial use**. Six tabs:

- **Convert** — audio in + ref audio in → audio out. `tau` slider controls how strongly the target timbre overrides the source.
- **TTS + Convert** — type text in any of 7 languages, get it spoken in the reference voice.
- **Style Controls** — explainer on what OpenVoice can and can't do (timbre vs. emotion vs. accent)
- **Batch** — convert every audio file in a directory with one ref voice
- **VRAM** — release loaded V1/V2/MeloTTS models
- **Help** — multilingual notes, watermarking, comparison with RVC/SoVITS

**Two model versions in one notebook**:
- **V2** (default) — multilingual, better quality, ~140 MB
- **V1** — simpler, EN/ZH only, ~150 MB

**Every output is watermarked** with a 32-bit string at 16 kbps via [`wavmark`](https://github.com/wavmark/wavmark) (default `@MyShell`). Disable by setting the Watermark field to an empty string. The watermark survives MP3 compression at 128 kbps.

> "OpenVoice can accurately clone the reference tone color and generate speech in multiple languages and accents." — [arXiv 2312.01479](https://arxiv.org/abs/2312.01479)

### TTS Model Loader

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/TTS_Model_Loader.ipynb)

Pre-downloads all the model weights, code packages, and reference voice clips used by every notebook in the TTS suite. Run **once**, then every other notebook starts instantly without re-downloading.

- **Resumable** — interrupted downloads pick up where they left off
- **Per-notebook toggles** — pick the models you actually need
- **Drive-cached** at `/content/drive/MyDrive/AEI_TTS_Cache/` so the cache survives Colab session resets
- **Total storage**: ~35–40 GB for the full suite

**Run this first**, before any model notebook.

### TTS Voice Library

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/TTS_Voice_Library.ipynb)

Curated reference voice clips with transcripts, ready to drop into the voice-cloning tabs of any cloning-capable notebook (Higgs, MOSS, Qwen3-Base, dots.tts, Fish, MisoTTS, IndicF5).

- **12+ voices** across English, Chinese, Kannada, Marathi, Punjabi, Tamil, plus 2-speaker dialogue
- **Public-domain / CC-BY / MIT** licensed material only — every entry has a transcript for the cloning tabs that require one
- **Optional streaming extras**: LJSpeech (Unlicense / public domain) and VCTK (CC-BY-4.0, multi-speaker English)
- **Gradio UI** with filter by language, gender, license, and free-text search
- **Drive mirror** at `/content/drive/MyDrive/AEI_TTS_Cache/voices/`

### Notebook Overview

| Notebook | Model | Size | Languages | License |
|----------|-------|------|-----------|---------|
| **Qwen3-TTS** *(flagship)* | Qwen3-TTS-12Hz | 0.6B / 1.7B | 10 | Apache 2.0 |
| **Higgs-Audio** | Higgs-Audio v3 | 4B | 100+ | Research / Non-Commercial |
| **MisoTTS** | MisoTTS 8B | 8B | English | Other (see card) |
| **Supertonic-3** | Supertonic-3 | 99M (ONNX) | 31 | MIT code · OpenRAIL-M model |
| **Dia** | Dia 1.6B | 1.6B | English | Apache 2.0 |
| **IndicF5** | IndicF5 | 400M | 11 Indian | MIT (gated) |
| **MOSS-TTS v1.5** | MOSS-TTS v1.5 | 8B | 31 | Apache 2.0 |
| **dots.tts-soar** | dots.tts-soar | 2B | 24+ | Apache 2.0 |
| **Fish S2 Pro** | Fish S2 Pro | 4B + 400M (Dual-AR) | 80+ | Fish Audio Research (non-commercial) |
| **VoxCPM2** | VoxCPM2 | 2B | 30 (+ 9 ZH dialects) | Apache 2.0 |
| **Kokoro-82M** | Kokoro-82M | 82M | 9 | Apache 2.0 |

---

## Tools

Two Python scripts in `tools/` keep the notebooks consistent:

- **`tools/validate.py`** — fast AST-parse check on every code cell. Exits 0/1. Designed to be wired into CI.
- **`tools/qa_check.py`** — full polish audit (info= tooltips, try/except coverage, `concurrency_limit`, `clear_output()`, `demo.load` welcome, `FileLink` in Step 6). Excludes the 2 pre-existing Pixal3D notebooks.

Both run from the repo root with no dependencies:

```bash
python3 tools/validate.py    # OK: all 24 notebook(s) parse cleanly.
python3 tools/qa_check.py    # OK: all authored notebooks pass the polish audit.
```

A GitHub Actions workflow at `.github/workflows/qa.yml` runs both on every push to `main` and every PR. See the [QA badge](#aei-colab-notebooks) at the top of this README for live status. See [CONTRIBUTING.md](./CONTRIBUTING.md) for the recommended pre-submit checklist.

---

## Why This Exists

Most Colab AI notebooks require paid tokens, subscription services, or lengthy manual compilation of CUDA dependencies. This project precompiles everything into ready-to-install wheels and hosts them on GitHub — so you can open a notebook, run all cells, and get results immediately, on any supported GPU, completely free.

The 3D pipeline (Pixal3D) was the first notebook in the series. The TTS suite, video generators, and voice-conversion tools have all been added since, with the same "open-and-run" philosophy across every modality.

---

## Model Cards & Upstream Attribution

Every model in this repo is the work of its respective authors. We just wrap them in self-contained Colab notebooks. For deeper documentation, training details, or commercial-license questions, please visit the upstream links below.

### Video Models

| Notebook | Upstream repo | Hugging Face | Paper |
|----------|---------------|--------------|-------|
| Wan 2.2 (5B + 14B) | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) · [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | [arXiv 2503.20314](https://arxiv.org/abs/2503.20314) |
| Wan 2.2 Animate | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | [Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) | [arXiv 2503.20314](https://arxiv.org/abs/2503.20314) |
| Wan 2.2 S2V | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) | [arXiv 2508.18621](https://arxiv.org/abs/2508.18621) · [arXiv 2503.20314](https://arxiv.org/abs/2503.20314) |

### 3D Models

| Notebook | Upstream repo | Hugging Face | Paper |
|----------|---------------|--------------|-------|
| Pixal3D | [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) | [TencentARC/Pixal3D](https://huggingface.co/spaces/TencentARC/Pixal3D) | [arXiv 2605.10922](https://arxiv.org/abs/2605.10922) |
| Cube 3D + CubePart | [Roblox/cube](https://github.com/Roblox/cube) | [tencent/cubepart](https://huggingface.co/Roblox/cubepart) | — |
| Hunyuan3D-2 | [Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | [tencent/Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) | [arXiv 2501.12202](https://arxiv.org/abs/2501.12202) |
| Hunyuan3D-2.1 | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) | [arXiv 2506.15442](https://arxiv.org/abs/2506.15442) |
| Hunyuan3D 3.0 (API) | [Tencent Cloud](https://www.tencentcloud.com/document/product/1665/119114) (Cloud API v3) | — | — |

### TTS Models

| Notebook | Upstream repo | Hugging Face | Paper |
|----------|---------------|--------------|-------|
| Qwen3-TTS | (Alibaba) | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | — |
| Higgs-Audio | [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio) | [bosonai/higgs-audio-v2-tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) | — |
| MisoTTS | (Miso Labs) | [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS) | — |
| Supertonic-3 | [Supertone/supertonic](https://github.com/Supertone/supertonic) | [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) | — |
| Dia | [nari-labs/dia](https://github.com/nari-labs/dia) | [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) | — |
| IndicF5 | [AI4Bharat/IndicF5](https://github.com/AI4Bharat/IndicF5) | [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) | — |
| MOSS-TTS v1.5 | [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) | [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) | — |
| dots.tts-soar | [rednote-hilab/dots.tts](https://github.com/rednote-hilab/dots.tts) | [rednote-hilab/dots.tts-soar](https://huggingface.co/rednote-hilab/dots.tts-soar) | — |
| Fish S2 Pro | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | — |
| VoxCPM2 | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | [arXiv 2509.24650](https://arxiv.org/abs/2509.24650) |
| Kokoro-82M | [hexgrad/kokoro](https://github.com/hexgrad/kokoro) | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | [arXiv 2306.07691](https://arxiv.org/abs/2306.07691) (StyleTTS 2) · [arXiv 2203.02395](https://arxiv.org/abs/2203.02395) (iSTFTNet) |

### Voice Conversion Models

| Notebook | Upstream repo | Hugging Face | Paper |
|----------|---------------|--------------|-------|
| OpenVoice V2 | [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) | [myshell-ai/OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2) | [arXiv 2312.01479](https://arxiv.org/abs/2312.01479) |

### Reference voices

- [TTS_Voice_Library.ipynb](./TTS_Voice_Library.ipynb) — curated set of CC0 / CC-BY / public-domain reference clips with transcripts, for the voice-cloning tabs. All clips are from upstream sources; see the notebook for per-clip attribution.

---

## License

Notebooks in this repository are provided for educational and personal use. Individual notebooks use third-party models under their respective licenses — check each model's page for commercial use terms.

- **Pixal3D pipeline**: MIT
- **TTS models** (per notebook):
  - Apache 2.0: Qwen3-TTS, Dia, MOSS-TTS, dots.tts-soar, **VoxCPM2**, Kokoro-82M, Supertonic-3 *code* (model weights are OpenRAIL-M)
  - MIT: IndicF5, **OpenVoice V2**
  - Research / Non-Commercial: Higgs-Audio, Fish S2 Pro
  - Other (see model card): MisoTTS
- **Cube 3D + CubePart** (the `Cube_3D_Colab.ipynb`):
  - Code: MIT (the [Roblox/cube](https://github.com/Roblox/cube) repo)
  - Weights: [OpenRAIL-M](https://huggingface.co/Roblox/cubepart#license) — no high-risk downstream use (e.g. no facial recognition, no biometric ID, no surveillance)

Generated 3D assets are not moderated by Roblox safety systems. Use of the model weights is subject to the Cube repo's license terms; users are solely responsible for the outputs they generate.
