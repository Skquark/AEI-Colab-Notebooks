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
- [TripoSplat — Image to 3D Gaussians (TripoAI/VAST-AI)](#triposplat--image-to-3d-gaussians-mit)
- [NoPoSplat — 2-3 Photos → 3DGS, pose-free (MIT, ICLR 2025)](#noposplat--2-3-photos--3dgs-pose-free-mit-iclr-2025)
- [Wild Gaussian Splatting — Video / Image Folder → 3DGS (MASt3R + INRIA)](#wild-gaussian-splatting--video--image-folder--3dgs-cc-by-nc-sa--inria)
- [MapAnything — Universal 3DGS-from-Images (Meta, Apache 2.0)](#mapanything--universal-3dgs-from-images-meta-apache-20)
- [Pi3X — Video-Native 3DGS, Permutation-Equivariant (BSD-3 + CC BY-NC-4.0)](#pi3x--video-native-3dgs-permutation-equivariant-bsd-3--cc-by-nc-40)
- [TextureMapPrep — Seamless PBR Maps for Game Assets](#texturemapprep--seamless-pbr-maps-for-game-assets)
- [SplatTransform — 3DGS post-processor (PlayCanvas, MIT)](#splattransform--3dgs-post-processor-playcanvas-mit)
- [SkinTokens — Mesh to Rig with TokenRig (VAST-AI, MIT)](#skintokens--mesh-to-rig-with-tokenrig-vast-ai-mit)
- [SuGaR — Surface-Aligned 3DGS to Mesh (INRIA, non-commercial)](#sugar--surface-aligned-3dgs-to-mesh-inria-non-commercial)
- [GauStudio — 3DGS to Mesh via TSDF (MIT + INRIA mixed)](#gaustudio--3dgs-to-mesh-via-tsdf-mit--inria-mixed)
- [Asset Library Browser — browse, tag, preview, export your 200+ assets](#asset-library-browser)
- [Game engine integration — Three.js / WebGPU loader for the grounded assets](#game-engine-integration--loading-the-assets-into-threejs--webgpu)
- [Mesh Optimizer — post-process for game-ready assets](#mesh-optimizer--post-process-for-game-ready-assets)
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
- [Audio Post-Processor — trim / normalize / denoise / convert](#audio-post-processor--post-process-for-podcast-music-and-broadcast)

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

## TripoSplat — Image to 3D Gaussians (MIT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/TripoSplat_Colab.ipynb)

[TripoAI / VAST-AI-Research TripoSplat](https://www.tripo3d.ai/research/triposplat) ([arXiv 2605.16355](https://arxiv.org/abs/2605.16355), [HF VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat), [code](https://github.com/VAST-AI-Research/TripoSplat), **MIT** — model + code). Converts a **single 2D image** into a 3D Gaussian Splat scene that can be rendered in real-time by a 3DGS viewer. The first **fully-commercial-OK** image-to-3D model in the suite (Hunyuan3D weights are non-commercial).

### Architecture (3.78 GB total)

- **Image encoders**: DINOv3 ViT-H/16+ (1.68 GB) + Flux2 VAE encoder (336 MB) → fused 8192-token conditioning
- **DiT flow model**: 24 blocks × 1024-dim × 16-head flow-matching transformer with RMSNorm-QK, RoPE, and share-mod (741 MB fp16)
- **Octree + Gaussian decoder**: produces 32k → 262k Gaussians (multiple of 32) with per-Gaussian position, scale, rotation, opacity, and SH degree-0 color (576 MB)
- **Background removal**: BiRefNet Swin-L (444 MB) — auto-removes BG if no alpha channel
- **Sampler**: 20-step Euler flow matching with classifier-free guidance (default 3.0, shift 3.0)
- **Total**: 3.78 GB across 5 safetensors

### Output formats (2 — native 3DGS only)

| Format | Type | Viewer | Use case |
| --- | --- | --- | --- |
| **`.ply`** | 3DGS standard | [Antimatter15](https://antimatter15.com/splat/), [gsplat.tech](https://gsplat.tech/) | 3DGS renderers, gaussian-splat research |
| **`.splat`** | 32-byte packed | [Antimatter15](https://antimatter15.com/splat/) | Web 3DGS viewers, smallest of the native formats |

**Mesh outputs have been intentionally removed from this notebook.** TripoSplat's 3DGS-derived mesh exports (via Poisson/alpha_shape/ball_pivoting reconstruction of the Gaussian cloud) were consistently low-quality — holes, missing surfaces, no proper UVs, and slow to compute. For game-ready textured meshes, use **[Pixal3D](#pixal3d--image-to-3d-with-pbr-textures)** instead. For high-quality 3DGS-to-mesh research pipelines, see [SuGaR](#sugar--surface-aligned-3dgs-to-mesh-inria-non-commercial) (sharp, 2-3 hrs/scene) and [GauStudio](#gaustudio--3dgs-to-mesh-via-tsdf-mit--inria-mixed) (smooth, ~10 min/scene).

### Workflow

The notebook is now focused on **producing 3DGS for real-time viewing** (the highest-quality output TripoSplat gives). Workflow for a 200+ image library:

- **Step 1 — install + Drive mount** (recommended). Set `CONNECT_GOOGLE_DRIVE = True` (default) so all subsequent outputs are mirrored to `/content/drive/MyDrive/AEI_3D_Out/TripoSplat/`. Outputs are otherwise lost on disconnect.
- **Step 6 — single image**: set `QUICK_INPUT_IMAGE` to a specific path (or leave blank for auto-pick from `/content`). Outputs are named after the image stem — `hero.png` → `hero.ply` + `hero.splat`. ~30 s on L4.
- **Step 7 — batch**: choose `BATCH_INPUT_MODE = 'folder'` (point at a folder of images) or `'txt list'`. Toggle `BATCH_RECURSIVE` for subfolders — the slug is prefixed with the parent folder name (`characters_hero`) so files from different subfolders don't collide. Outputs are flat in `batch_dir/`.
  - **Quality defaults** (balanced, suitable for 200+ batches on T4): `steps=30`, `num_gaussians=262144`, `guidance=3.0`, `shift=3.0`. Speed presets in the cell markdown (steps=20/131k = 1 min/img; steps=12/65k = 30 s/img preview).
  - **Crash safety**: per-item Drive mirror (each completed model is safe immediately, not at end), per-asset progress log (`_batch_progress.jsonl`), and `BATCH_RESUME_FOLDER` lets you re-run on a crashed batch to pick up where you left off.
- **Step 8 — One-Shot Game-Ready Prep (SplatTransform Lite)**: the recommended next step. Compresses every PLY in the latest batch folder to game-friendly formats (`.sog`, `.splat`, `.spz`, `.glb`), generates 3 quality tiers (full / standard / background) and 2 colliders (`.hull.glb` + `.collision.glb`). Auto-discovers the latest batch if you don't set `INPUT_BATCH_DIR`. Idempotent install of `splat-transform` on first run. See "Game-ready workflow" below for the full pipeline.

The previous post-processing step (clean, fill holes, UV unwrap, smooth) and the game-asset export pipeline (`.glb`/`.obj`/`.fbx`/`.stl`/`.ply`/`.3mf` from `_mesh.ply`) have been **removed** since they produced unusable meshes. The Pixal3D notebook has a new `Step 8 — Post-Process existing GLBs` (added as part of this refactor) that handles the same game-asset post-processing pipeline for any `.glb` source — including ones you get from Pixal3D or a 3rd-party mesher like Kiri Engine.

### Grounding (drop on the ground for placement)

**All STEP 8 outputs are grounded by default** so they "just work" in your engine. A `_ground_splat()` helper runs before compression and does:

1. **Translation to origin in XZ** (median of all point X, Z positions)
2. **PCA in the XZ plane, snapped to nearest 90°** (the principal horizontal axis becomes +X, ±90° rounded)
3. **Translation to Y=0** (the lowest point becomes the ground)
4. Records the transform in each asset's `_meta.json`:
   ```json
   "grounding": {
     "applied": true,
     "translation": {"x": 0.5, "y": -0.2, "z": -0.3},
     "rotation_y_degrees": 90,
     "size_units": {"height": 1.8, "width": 0.5, "depth": 0.4},
     "size_class": "prop"
   }
   ```
5. **Derives `size_class`** from the final bbox height: `< 0.5m` → `small_prop`, `0.5-2m` → `prop` (people, weapons), `2-10m` → `tree_or_vehicle`, `> 10m` → `building`. Used by Asset_Library_Browser for filtering and procedural placement.

The result: every compressed asset comes out with the bottom on Y=0, centered in XZ, and facing one of 4 cardinal directions. Three.js, your WebGPU engine, or any game engine can drop them on the ground with `splat.position.y = 0` and they'll just work. The transform is **idempotent and reversible** — recorded in meta.json so you can apply, undo, or override per-asset.

Set `APPLY_GROUNDING = False` to disable (e.g., for assets already in a grid layout).

### Game-ready workflow (200+ image library)

The recommended end-to-end pipeline for converting a 200+ image library into game-ready assets:

1. **TripoSPlat STEP 7 (batch)** — 200× `.ply` + `.splat` files. ~30-60 sec per image at quality defaults. Per-item Drive mirror so a T4 disconnect doesn't lose work.
2. **TripoSPlat STEP 8 (SplatTransform Lite)** — for each PLY, produces:
   - 3 visual tiers: `<slug>_full.sog` (~17 MB), `<slug>_standard.sog` (~4-5 MB), `<slug>_background.sog` (~1-2 MB)
   - 2 colliders: `<slug>_hull.glb` (~10-50 KB convex), `<slug>_collision.glb` (~1-3 MB voxel)
   - `<slug>_meta.json` with grounding, tier sizes, `size_class`
   - All **grounded by default** so they drop on Y=0 in your engine
   - For 200 assets: ~30-40 min total
3. **Asset_Library_Browser** — set `LIBRARY_DIR` to the `game_ready/` folder. Auto-recognizes all 13+ formats (3DGS compressed, voxel collision, KHR_gaussian_splatting GLB, meshes, images). Tag assets as hero/standard/background for the LOD system.
4. **For hero assets (5-10)**: also run [GauStudio_Colab](#gaustudio--3dgs-to-mesh-via-tsdf-mit--inria-mixed) to get a high-fidelity textured mesh as a fallback for engines that don't support 3DGS yet.

For the 1500+ library, run batches in sessions of 200-300 with the new `BATCH_RESUME_FOLDER` param, then STEP 8 against each batch's output.

### Quick Start

> Requires **Colab Runtime Version 2026.01** (ships torch 2.9.0+cu126). L4 GPU (22 GB) recommended; T4 (16 GB) is tight — use `num_gaussians ≤ 65536` and `steps ≤ 15`. First run: ~3-5 min to download weights, ~30-60 s per generation on L4.

### GPU Support

| GPU | VRAM | Gaussians | Steps | Time (approx) | Notes |
|-----|------|-----------|-------|---------------|-------|
| A100 | 40 GB | 262144 | 20 | ~30 s | Best quality. Default settings. |
| L4 | 22 GB | 131072 | 20 | ~60 s | Recommended. Default 131k, push to 262k if VRAM allows. |
| T4 | 16 GB | 65536 | 12 | ~90 s | Low-VRAM mode. Use `num_gaussians ≤ 65k` and `steps ≤ 12`. |

The notebook auto-detects GPU and warns if VRAM is below 20 GB.

### Why this complements Pixal3D

- **Output**: Gaussians (real-time rendering, perfect view interpolation) vs textured PBR mesh
- **Speed**: 30-60 s per image (TripoSplat) vs 60-120 s (Pixal3D, includes PBR bake)
- **License**: MIT (commercial-OK) vs Pixal3D SIGGRAPH 2026 (research)
- **Use TripoSplat for**: real-time 3DGS previews, AR/VR demos, fast iteration, low-LOD game assets (mesh)
- **Use Pixal3D for**: game-ready textured PBR assets, shippable meshes, print-ready
- **Bridge them**: TripoSplat's 3DGS PLY → [SuGaR_Colab](#sugar--surface-aligned-3dgs-to-mesh-inria-non-commercial) for high-quality mesh OR upload to [GauStudio_Colab](#gaustudio--3dgs-to-mesh-via-tsdf-mit--inria-mixed) for fast mesh. TripoSplat's `.ply` also works as 3DGS fallback for game engines that support 3DGS rendering.

### License

[MIT](https://huggingface.co/VAST-AI/TripoSplat) — model + code commercial-OK.

---

## NoPoSplat — 2-3 Photos → 3DGS, pose-free (MIT, ICLR 2025)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/NoPoSplat_Colab.ipynb)

**[NoPoSplat](https://noposplat.github.io/)** (Botao Ye et al., NVIDIA + ETH Zurich, ICLR 2025 Oral) is a feed-forward 3DGS reconstruction model: **upload 2-3 unposed, uncalibrated photos of a scene → get a complete 3D Gaussian Splat scene back in ~10 seconds**. No COLMAP, no bundle adjustment, no per-scene optimization. The closest open-source equivalent to a Luma-style "upload photos, get 3DGS" workflow.

The model is a ViT-L encoder (built on the MASt3R backbone) with four heads: a 3D point head (DPT), a Gaussian parameter head (DPT-GS), a second-view head, and a **Unified Gaussian Adapter** that fuses both views' predictions into a single 3DGS scene in a canonical frame. Critically, NoPoSplat is **pose-free**: it does not require camera extrinsics or intrinsics as input.

### When to use NoPoSplat vs TripoSplat
* **TripoSplat** — single image → 3DGS via a learned generative prior. Best for imagined/photographic scenes where you only have one photo. ~30-60s.
* **NoPoSplat (this notebook)** — 2-3 unposed photos → real 3DGS via direct reconstruction. Best for capturing real places/objects with a phone. ~10-20s.
* **WildGaussianSplatting** (planned) — video → 3DGS via MASt3R pose + per-scene 3DGS optimization. Best for high-fidelity capture from video. 5-15 min.

### What you get
* **Input:** 2 or 3 images (`.jpg` / `.png`) of the same scene from different viewpoints
* **Output:** a real 3DGS scene (`.ply`, ~200-500 MB for 30k-100k Gaussians) + predicted camera poses
* **Runtime:** ~10 seconds for 2 views at 256×256, ~20-30 seconds for 2 views at 512×512 (T4)
* **VRAM:** ~6-10 GB on T4 (model is ~1.1B parameters at bf16)

### Pipeline
```
NoPoSplat (this notebook)  →  .ply  →  SplatTransform_Colab STEP 3  →  SOG/SPLAT/SPZ/GLB
                                                ↓
                                  Asset_Library_Browser_Colab
                                                ↓
                                  Three.js / WebGPU game engine
```

### Checkpoints (all MIT, hosted on 🤗 botaoye/NoPoSplat)
* `mixRe10kDl3dv_512x512.ckpt` — SOTA, 2 views at 512² (2.3 GB) — **default**
* `mixRe10kDl3dv.ckpt` — 2 views at 256², faster
* `re10k_3views.ckpt` — 3 views at 256²
* `re10k.ckpt` — 2 views at 256², RealEstate10K only
* `acid.ckpt` — 2 views at 256², ACID dataset

### License
* **Notebook + code:** MIT (upstream [cvg/NoPoSplat](https://github.com/cvg/NoPoSplat))
* **MASt3R backbone weights (CC BY-NC-SA 4.0):** the ViT-L weights NoPoSplat's encoder was fine-tuned from. Consumed for inference only. If you need a fully-commercial pipeline, see the [WildGaussianSplatting notebook](#) (same backbone, different downstream path).
* **Your output (.ply, camera poses):** yours, no restriction.

### Related notebooks
* **TripoSPlat** — image → 3DGS, MIT, single image
* **WildGaussianSplatting** (planned) — video → 3DGS with per-scene optimization, non-commercial
* **SplatTransform** — post-process 3DGS for game engines (SOG/SPLAT/SPZ/GLB)
* **Asset_Library_Browser** — browse, tag, and ship your rigged library to a game engine

---

## Wild Gaussian Splatting — Video / Image Folder → 3DGS (CC BY-NC-SA + INRIA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/WildGaussianSplatting_Colab.ipynb)

> **⚠️ License notice:** This notebook is for **research and evaluation only**. The 3DGS
> training uses the original [INRIA Gaussian-Splatting License](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)
> (non-commercial, requires written consent from Inria for commercial use). The
> MASt3R backbone is **CC BY-NC-SA 4.0** by Naver. Your output `.ply` / `.mp4` is
> yours, but do not use the trained scenes commercially without obtaining appropriate
> licenses. See the full license breakdown in the notebook header.

**[Wild Gaussian Splatting](https://github.com/nerlfield/wild-gaussian-splatting)**
(Daniel Kovalenko, reface.ai) is the closest open-source equivalent to a **Luma Labs
Capture / Genie** workflow: upload a casual video or a folder of overlapping photos,
get a polished 3DGS scene back. It chains [MASt3R](https://github.com/naver/mast3r)
(Naver, ICLR 2025) for pose + point cloud estimation with the original
[INRIA 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) training loop,
producing a real 3DGS scene (not just novel-view video) in 5-15 minutes on a T4.

### How it works

1. **Frame extraction** (if you uploaded a video) → save to a folder
2. **MASt3R alignment** → predict camera poses, intrinsics, depth, and per-image point clouds jointly
3. **COLMAP-format export** → write `cameras.txt`, `images.txt`, `points3D.ply` so the 3DGS loader can read them
4. **3DGS training** → 3,000-30,000 iterations of the original INRIA densify/prune loop
5. **Render an orbit video** → MP4 of the final scene
6. **Output:** `point_cloud.ply` (real 3DGS) + `cameras.json` + `renders.mp4` + per-frame PNGs

### When to use this vs NoPoSplat
* **NoPoSplat** — 2-3 unposed photos → real 3DGS in ~10 seconds. MIT-licensed notebook; output is yours. No INRIA / non-commercial baggage.
* **Wild Gaussian Splatting (this notebook)** — video / image folder → real 3DGS in 5-15 min with proper per-scene optimization. Higher fidelity than NoPoSplat but inherits non-commercial licenses. Best for hero / portfolio assets.
* **TripoSPlat** — single image → 3DGS via learned generative prior. Best for imagined / single-photo scenes. MIT, fully commercial-OK.

### Pipeline
```
Wild Gaussian Splatting (this notebook)  →  .ply + .mp4
                                                ↓
                                  SplatTransform_Colab STEP 3  →  SOG/SPLAT/SPZ/GLB
                                                ↓
                                  Asset_Library_Browser_Colab
                                                ↓
                                  Three.js / WebGPU game engine
```

### Requirements
* **GPU:** NVIDIA, ≥ 12 GB VRAM (T4 15 GB works for ≤15 frames; L4/A100 needed for longer clips)
* **RAM:** ≥ 12 GB
* **Disk:** ≈ 10 GB free (PyTorch + CUDA + 2.75 GB MASt3R + 2.29 GB DUSt3R + working space)
* **ffmpeg** (`apt-get install ffmpeg`) — required for video frame extraction and MP4 render
* **Time on first run:** 8-12 min (PyTorch + diff-gaussian-rasterization + simple-knn compile + checkpoints)
* **Time on subsequent runs:** 2-3 min (everything cached in your Drive)
* **Per-scene runtime:** 5-15 min for ≤15 frames, ~1 min per additional 5 frames

### Related notebooks
* **NoPoSplat** — 2-3 photos → 3DGS in ~10s (MIT, faster but lower fidelity)
* **TripoSPlat** — single image → 3DGS (MIT, generative prior)
* **MapAnything** — universal 3DGS from any 1+ images, Apache 2.0
* **Pi3X** — video-native 3DGS with permutation-equivariance, BSD-3 + CC BY-NC-4.0
* **SplatTransform** — post-process 3DGS for game engines (SOG/SPLAT/SPZ/GLB)
* **Asset_Library_Browser** — browse, tag, and ship your rigged library to a game engine

---

## MapAnything — Universal 3DGS-from-Images (Meta, Apache 2.0)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/MapAnything_Colab.ipynb)

**[MapAnything](https://map-anything.github.io/)** is Meta's universal 3D reconstruction framework (3DV 2026). One feed-forward transformer handles **12+ 3D reconstruction tasks** in a single forward pass: image-only SfM, image+pose, image+intrinsics+depth, registration, depth completion, monocular metric depth, and more. Output is a metric 3D point cloud + camera poses which we feed into `gsplat` to produce a real 3DGS scene.

**The standout property is universality**: give it 2 images, 20 images, an image + known depth from a depth sensor, an image + known camera poses from COLMAP, or a video without any calibration at all — and it always produces a metric 3D reconstruction.

### How it differs from our other 3DGS notebooks

| Notebook | Inputs | Approach | License |
|---|---|---|---|
| **TripoSplat** | 1 image | Generative prior, feed-forward | MIT |
| **NoPoSplat** | 2-3 unposed photos | Pose-free, feed-forward | MIT (+ MASt3R backbone CC BY-NC-SA) |
| **Wild GS** | Video / image folder | MASt3R pose + 3DGS optimization | CC BY-NC-SA + INRIA non-commercial |
| **MapAnything (this)** | Any 1+ images; optional poses/intrinsics/depth | Universal feed-forward transformer, then gsplat | **Apache 2.0** (commercial-OK) |
| **Pi3X** | Video frames (any order, any count) | Permutation-equivariant feed-forward, then gsplat | BSD-3 code + CC BY-NC-4.0 weights (non-commercial) |

### Pipeline
```
images (and optional poses / depth / intrinsics)
       ↓
MapAnything (universal transformer, 1 forward pass)
       ↓
metric point cloud + camera poses + intrinsics + depth
       ↓
write COLMAP-format (cameras.txt, images.txt, points3D.ply)
       ↓
gsplat (1-3 min training on T4)
       ↓
3DGS .ply (SOG/SPLAT/SPZ after SplatTransform_Colab STEP 3)
```

### Requirements
* **GPU:** NVIDIA, ≥ 6 GB VRAM (T4 15 GB works with `minibatch_size=1`)
* **RAM:** ≥ 12 GB
* **Disk:** ≈ 8 GB free (PyTorch + CUDA + 4.47 GB MapAnything + 1.1 GB DINOv2-giant + working space)
* **Time on first run:** 8-12 min (PyTorch + uniception + MapAnything + DINOv2-giant)
* **Time on subsequent runs:** 2-3 min (everything cached in your Drive)
* **Per-scene runtime:** ~20-60 seconds for 10 images at 518² on T4 (one forward pass)

### License
* **Notebook + code + weights:** **Apache 2.0** — fully commercial-OK.
* **Source:** [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) (Apache 2.0); weights from [🤗&nbsp;facebook/map-anything-apache](https://huggingface.co/facebook/map-anything-apache) (Apache 2.0 variant — Meta also publishes a CC-BY-NC variant at `facebook/map-anything` which we deliberately do not use)

### Related notebooks
* **NoPoSplat** — 2-3 photos → 3DGS in ~10s (MIT, but slower for >3 images)
* **WildGaussianSplatting** — video → 3DGS with per-scene optimization (non-commercial)
* **Pi3X** — video-native 3DGS with permutation-equivariance (non-commercial weights)
* **SplatTransform** — post-process 3DGS for game engines (SOG/SPLAT/SPZ/GLB)
* **Asset_Library_Browser** — browse, tag, and ship your library to a game engine

---

## Pi3X — Video-Native 3DGS, Permutation-Equivariant (BSD-3 + CC BY-NC-4.0)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Pi3X_Colab.ipynb)

> **⚠️ License notice:** **Code = BSD 3-Clause** (commercial-OK). **Weights = CC BY-NC-4.0** (strictly non-commercial, per the official model card). Use this notebook for research, personal projects, and non-commercial work. For commercial 3DGS-from-video, use **MapAnything** (Apache 2.0) instead.

**[π³ / Pi3X](https://yyfz.github.io/pi3/)** is a feed-forward neural network for visual geometry reconstruction from Shanghai AI Lab + ZJU (ICLR 2026). Unlike every other 3DGS-from-images tool in this suite, **π³ is permutation-equivariant** — there is no fixed reference view, input order is irrelevant, the model handles long videos without drift, and reconstruction quality stays stable on sequences of 10, 50, 200+ frames.

**Pi3X** is the engineering update of π³: smoother Convolutional Head, optional camera/pose/depth conditioning, continuous confidence prediction, and approximate metric scale.

### How it differs from our other 3DGS notebooks
* **MapAnything** — Apache 2.0, universal transformer, any number of images. Commercial-OK.
* **WildGaussianSplatting** — non-commercial, MASt3R pose + INRIA 3DGS optimization, 5-15 min.
* **Pi3X (this)** — **BSD-3 code + CC BY-NC-4.0 weights**, permutation-equivariant feed-forward for video, ~5-30 s for 100 frames. Ideal for **long phone videos** where other feed-forward methods drift.

### Pipeline
```
video frames (or image folder) — any number, any order
       ↓
Pi3X (π³) — permutation-equivariant feed-forward transformer, 1 forward pass
       ↓
metric point cloud + per-frame camera poses + intrinsics + depth
       ↓
write COLMAP-format (cameras.txt, images.txt, points3D.ply)
       ↓
gsplat (1-3 min training on T4)
       ↓
3DGS .ply (SOG/SPLAT/SPZ after SplatTransform_Colab STEP 3)
```

### Requirements
* **GPU:** NVIDIA, ≥ 6 GB VRAM (T4 15 GB works; A100/L4 recommended for 100+ frames)
* **RAM:** ≥ 12 GB
* **Disk:** ≈ 8 GB free (PyTorch + CUDA + ~5 GB Pi3X + working space)
* **Time on first run:** 5-8 min (PyTorch + π³ + safetensors download)
* **Time on subsequent runs:** 1-2 min (everything cached in your Drive)
* **Per-video runtime:** ~5-30 seconds for 100 frames at 224×224 on T4 (one forward pass)

### Key parameters
* **`use_multimodal=False`** (default) — disables conditioning on pose/intrinsics/depth and frees ~2 GB GPU. Recommended for unordered video frames.
* **`edge_mask=True`** — masks out edge artifacts using normal/depth consistency, improving 3DGS quality.
* **`use_amp=True`** — bf16 on Ampere+, fp16 fallback, ~2x speedup.
* **`max_frames=64`** — Pi3X is permutation-equivariant, so all frames are weighted equally; the model handles 200+ frames gracefully.

### License
* **Code:** BSD 3-Clause (commercial-OK) — upstream [yyfz/Pi3](https://github.com/yyfz/Pi3)
* **Weights:** CC BY-NC-4.0 (strictly non-commercial) — [🤗&nbsp;yyfz233/Pi3X](https://huggingface.co/yyfz233/Pi3X)
* **Your output (.ply, camera poses, renders):** yours, no restriction.

### Related notebooks
* **MapAnything** — universal 3DGS-from-images, Apache 2.0 (commercial-OK alternative)
* **WildGaussianSplatting** — video → 3DGS with per-scene optimization, CC BY-NC-SA + INRIA
* **NoPoSplat** — 2-3 photos → 3DGS in ~10s, MIT
* **TextureMapPrep** — generate the 6 PBR maps (albedo+normal+depth+height+rough+metal) from an albedo
* **SplatTransform** — post-process 3DGS for game engines (SOG/SPLAT/SPZ/GLB)
* **Asset_Library_Browser** — browse, tag, and ship your library to a game engine

---

## TextureMapPrep — Seamless PBR Maps for Game Assets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/TextureMapPrep_Colab.ipynb)

**[Marigold](https://marigoldcomputervision.github.io/) + [LOTUS](https://lotus3d.github.io/) +
[OpenCV](https://opencv.org/) + [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)**
all wired into a single batch pipeline for converting one or many seamless **albedo**
PNG textures into the full **6-map PBR set** your game engine expects:

* **albedo** (base color, sRGB)
* **normal** (tangent-space, OpenGL convention — flip Y in Unity)
* **depth** (linear, 16-bit PNG)
* **height** (for displacement, derived from normal via Poisson)
* **roughness** (R channel, linear — from Marigold appearance IID)
* **metallic** (R channel, linear — from Marigold appearance IID)

…and then optionally upscales **every map** with Real-ESRGAN (2x or 4x) while
preserving the seamless wraparound.

## Why these models

* **[Marigold v1.1](https://huggingface.co/prs-eth/marigold-depth-v1-1)** (PRS-ETH,
  CVPR 2024 Oral, Best Paper Candidate) — Apache-2.0 code, RAIL++-M weights.
  One unified `diffusers` API for depth + surface normals + intrinsic image
  decomposition (albedo + roughness + metallic). Already in the diffusers core.
* **[LOTUS v1](https://lotus3d.github.io/)** (EnVision Research, ICLR 2025) —
  Apache-2.0 code + weights. The depth + normal model the
  [PBRFusion4DepthDemo-InstNorm](https://huggingface.co/spaces/NightRaven109/PBRFusion4DepthDemo-InstNorm)
  space uses. Comparable quality to Marigold, different architecture.
  Single-step inference = ~5x faster than standard diffusion-based methods.
* **[FLUX 2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) +
  [NightRaven109/kleinalbedo4B5ksteps](https://huggingface.co/NightRaven109/kleinalbedo4B5ksteps)**
  (Apache-2.0) — the optional Albedo path from
  [PBRFusion4AlbedoKlein](https://huggingface.co/spaces/NightRaven109/PBRFusion4AlbedoKlein).
  Generates a clean, shadowless albedo from the input. ~8 GB. Off by default.
* **[OpenCV](https://opencv.org/)** (Apache-2.0) — classic normal-from-albedo
  (Sobel gradient), normal-from-depth (cross product on 3D points),
  height-from-normal (Poisson reconstruction via FFT), tile-wrap for seamless
  textures, all the basic PBR image ops.
* **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** (BSD-3, 35.9k stars,
  on PyPI as `realesrgan`) — the standard 2x/4x texture super-resolution. Uses
  `BORDER_WRAP` for seamless upscaling.

## How it differs from our other 3D/texture notebooks

| Notebook | Purpose |
|---|---|
| **Mesh_Optimizer** | post-process 3D meshes (decimate, repair, UV-unwrap, ground) |
| **SplatTransform** | 3DGS asset post-processor (SOG/SPLAT/SPZ) |
| **SkinTokens** | generate rigged characters from meshes |
| **Asset_Library_Browser** | browse, tag, and ship your 200+ assets |
| **TextureMapPrep (this)** | generate PBR map sets from albedo images |

This notebook slots in **before** a textured mesh enters `Mesh_Optimizer` /
`Pixal3D_Colab` / your game engine — it's the "I have an albedo, give me
the other 5 maps" stage.

## Pipeline diagram

```
seamless albedo.png  (one or many)
       ↓
[1] Marigold Depth       →  depth.png (16-bit)
[1] Marigold Normals     →  normal_map (used directly)
[1] Marigold IID         →  albedo_intrinsic, roughness, metallic
[1] (optional) FLUX Albedo  →  cleaner albedo_shadowless.png
       ↓
[2] OpenCV fallbacks     →  normal-from-albedo (if no normals)
                            normal-from-depth (verification)
                            height-from-normal (Poisson)
                            displacement.png
       ↓
[3] Real-ESRGAN 2x/4x    →  every map upscaled
                            (BORDER_WRAP to preserve seamless)
       ↓
/content/PBR_Out/<slug>/
  ├── albedo.png
  ├── albedo_shadowless.png  (if FLUX path)
  ├── normal.png
  ├── depth.png
  ├── height.png
  ├── displacement.png
  ├── roughness.png
  ├── metallic.png
  └── 2x/  or  4x/  (if upscaled)
```

## Requirements
* **GPU:** NVIDIA, ≥ 8 GB VRAM (T4 15 GB works for all paths except FLUX 2-klein
  which needs ~10 GB; FLUX is disabled by default on T4)
* **RAM:** ≥ 12 GB
* **Disk:** ≈ 5 GB free (PyTorch + diffusers + Marigold ~1.4 GB + LOTUS ~1.4 GB +
  Real-ESRGAN ~64 MB; FLUX 2-klein adds ~8 GB if enabled)
* **Time on first run:** 8-12 min (PyTorch + diffusers + all models)
* **Time on subsequent runs:** 1-2 min (everything cached in your Drive)
* **Per-texture runtime:** ~3-10 s for Marigold paths, ~20-40 s for FLUX path,
  ~1-2 s per upsample

## Where it fits in our pipeline
```
TextureMapPrep (this notebook)
   →  6 PBR maps per texture
   →  Mesh_Optimizer (if you want a quick preview mesh)
   →  Pixal3D_Colab / Hunyuan3D (full 3D from your PBR maps)
   →  Asset_Library_Browser
   →  Three.js / WebGPU game engine
```

> **⚠️ License note:** Marigold code is **Apache-2.0**, Marigold model weights are
> **RAIL++-M** (research + commercial w/ safety conditions). LOTUS is **Apache-2.0**
> (code + weights). FLUX 2-klein-4B is **Apache-2.0**. Real-ESRGAN is **BSD-3**.
> OpenCV is **Apache-2.0**. Most output maps are yours to use freely; the
> Marigold weights carry RAIL++-M conditions (mainly about safety classifiers).

---

## SplatTransform — 3DGS post-processor (PlayCanvas, MIT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/SplatTransform_Colab.ipynb)

The **missing piece** for shipping 3DGS assets to a game engine. Takes the raw 3DGS `.ply` output from [TripoSplat](#triposplat--image-to-3d-gaussians-mit) (~150-250 MB each) and converts it to game-engine-friendly formats with **~10× compression** via [PlayCanvas `splat-transform`](https://github.com/playcanvas/splat-transform) (MIT, commercial-OK).

**What it produces:**

| Format | Engine | Size vs PLY | Use for |
|--------|--------|--------------|---------|
| **`.sog`** | PlayCanvas | ~10% (90% smaller) | Production PlayCanvas scenes |
| **`.spz`** | Niantic Scaniverse | ~5% | Mobile AR, smallest files |
| **`.glb` + `KHR_gaussian_splatting`** | Three.js, PlayCanvas, Babylon.js (glTF 2.0 standard) | ~20% | Future-proof, glTF-native |
| **`.ply`** | Universal (Antimatter15, gsplat.tech, LumaAI) | 100% (lossless) | Universal fallback |
| **`.collision.glb`** | Any glTF 2.0 importer | n/a | Runtime physics (NOT a visual mesh) |
| **`lod-meta.json` + multiple `.sog`** | PlayCanvas viewer | depends on LODs | Streaming progressive loading |

**What it does NOT do:** generate a textured visual mesh from 3DGS. The only mesh `splat-transform` produces is a **voxel collision mesh** (for runtime physics). For visual mesh extraction from 3DGS, use [SuGaR_Colab](#sugar--surface-aligned-3dgs-to-mesh-inria-non-commercial) or [GauStudio_Colab](#gaustudio--3dgs-to-mesh-via-tsdf-mit--inria-mixed). For game-ready textured meshes from a single image, use [Pixal3D_Colab](#pixal3d--image-to-3d-with-pbr-textures).

**What it does:**

- **Step 1** — install Node 22+ and `splat-transform` from npm
- **Step 2** — Drive cache + GPU check (WebGPU optional for most steps; SOG SH k-means benefits from GPU)
- **Step 2b (Engine Preset)** — optional toggle, default OFF. When ON, biases STEP 3's output formats + SH strip defaults for a WebGPU/PlayCanvas engine consumer (SOG-only by default, DC-only SH, grounding off for props). See "Engine Preset" below for the prop-vs-environment distinction.
- **Step 3** — batch convert a folder of TripoSplat PLYs into all 4 game formats (SOG, SPZ, GLB, PLY) with size + compression-ratio reports. **New batch-hardening params:** `PARALLEL_WORKERS` (default 2) runs K conversions concurrently via `ThreadPoolExecutor`; `RESUME` (default True) + `FORCE_REDO` skip sources whose outputs already exist (survives Colab disconnects); `VALIDATE_OUTPUTS` re-opens each SOG to confirm it parses; `EMIT_MANIFEST` (default True) writes `<output_dir>/manifest.json` with per-source paths/sizes/ratios/splat-counts/assetClass + a top-level summary.
- **Step 4** — decimate (reduce Gaussian count for web previews / low-LOD) and/or strip SH bands (drop higher-frequency color)
- **Step 5** — **density chain** (per-tier full-model SOGs, NOT streaming). N full SOGs of decreasing Gaussian counts. Good for object props (engine swaps in/out based on distance).
- **Step 5b** — **streamed-SOG octree** (native `lod-meta.json`, for environment captures). Generates N decimated SOGs in a temp dir, then bundles them via `splat-transform -l 0..N lodN.sog out/lod-meta.json -C 512 -X 16`. The engine progressively streams chunks from low to high LOD. Tunable `LOD_COUNTS` + `LOD_CHUNK_COUNT` + `LOD_CHUNK_EXTENT`. Optional pre-grounding (`APPLY_GROUNDING=True` for environments; keep OFF for props to avoid double-application).
- **Step 6** — generate **3 types of colliders** for runtime physics:
  - **Voxel collision mesh** (`.collision.glb` — marching cubes from splat cloud, GPU-only)
  - **Convex hull** (`.hull.glb` — trimesh convex_hull on subsampled splat positions, ~10-50 KB, perfect for distant background)
  - **Concave hull / alpha shape** (`.concave.glb` — trimesh alpha_shape, follows surface concavities, ~50-500 KB, good for characters/weapons)
  - Optional **re-grounding** (`RE_GROUND_ON_IMPORT = False`): re-applies the same `_ground_splat()` transform from TripoSplat STEP 8 to input PLYs. Use when feeding PLYs from other sources (GauStudio, hand-made) that need to be grounded to match. Off by default because TripoSplat users will have already-grounded PLYs.
  - Optional GPU rasterized turntable previews (`.webp`)
- **Step 7** — final Drive mirror of all export folders + a README explaining each folder
- **Step 8** — keep-alive
- **Step 9** — help / format reference / engine compatibility / known issues

**Tip:** For the 80% case (compress every PLY, generate the 3 quality tiers, generate hull + voxel colliders), use **TripoSplat STEP 8 (SplatTransform Lite)** instead — it's the same operations in one cell with auto-discovery of the latest batch. This notebook is the right place for advanced cases: streamed LOD chains, WebGPU turntable previews, alpha-shape concave colliders, custom decimation curves for noisy splats.

**Engine Preset (Step 2b):** if your downstream consumer is a **WebGPU Gaussian-splat game engine** that loads bundled `.sog` (SOG v2) directly, auto-orients object-scale assets at load time, and renders only DC + degree-1 SH, enable `ENGINE_PRESET = True` and pick `ASSET_CLASS = 'prop'` or `'environment'`. The preset biases every output choice toward the engine's needs, while leaving each individual STEP 3 toggle still overrideable. **The critical caveat:** don't pre-ground props for auto-orienting engines (the engine applies its own 180° flip + recenter, double-application breaks positioning). For environment captures (loaded in world coords), use STEP 5b's streamed-SOG octree + turn pre-grounding ON.

**STEP 5 vs STEP 5b — when to use which:**

- **STEP 5 (density chain):** N full-model SOGs of decreasing Gaussian counts. Engine swaps between them based on camera distance. Good for **props** (object-scale assets). Not for progressive streaming.
- **STEP 5b (streamed-SOG octree):** one `lod-meta.json` + spatial SOG chunks. Engine progressively streams chunks from low to high LOD as the camera explores. Good for **environment captures** (terrain, city blocks, big rooms). Built via splat-transform's native `-l <level>` tagging + `lod-meta.json` output.

**STEP 6 collision note:** if your engine bakes its own collision at scene-import time (Unreal, Unity DOTS physics, custom Havok/PhysX), skip STEP 6 entirely — leave all 3 `GENERATE_*` toggles False.

**STEP 3 batch hardening:** `PARALLEL_WORKERS` (1-8, default 2) for concurrent conversions on thousands-of-files batches; `RESUME` skips sources whose outputs already exist (survives Colab disconnects); `EMIT_MANIFEST` writes `<output_dir>/manifest.json` with per-source metadata + `assetClass` (auto-inferred from splat count vs `ASSET_CLASS_THRESHOLD`) so the engine's upload step can automate asset records.


**License:** MIT (PlayCanvas Ltd.). Commercial-OK, no copyleft.

**Compute:** Node 22+ runtime (Colab needs `apt-get install nodejs`; the install script uses NodeSource for 22 LTS). GPU optional — most steps work on CPU; SOG SH k-means benefits from GPU (5-10× faster). L4 / T4 both fine. First run: ~2-3 min install. Subsequent: instant per file.

**200+ library workflow step:** TripoSplat (200× PLY = 30 GB) → **this notebook** (200× SOG = 3 GB, saves 27 GB) → Asset_Library_Browser for browsing.

---

## SkinTokens — Mesh to Rig with TokenRig (VAST-AI, MIT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/SkinTokens_Colab.ipynb)

Automated **skeleton generation + skinning weight prediction** for any 3D mesh, via a unified autoregressive model over learned *SkinTokens*. The official Gradio demo is at [🤗&nbsp;VAST-AI/SkinTokens](https://huggingface.co/spaces/VAST-AI/SkinTokens); this notebook packages the same model for **free Google Colab** with Drive cache, batch mode, low-VRAM toggle, and a Colab-side file picker.

**Why this matters for game dev:** A rigged character is the prerequisite for animation. Until now, rigging either required hours of manual weight painting in Blender, or one of three lossy decoupling methods (RigNet, Puppeteer, UniRig). SkinTokens unifies skeleton + skinning into a single autoregressive sequence via learned discrete *skin tokens*, achieving **98%–133%** better skinning accuracy and **17%–22%** better bone prediction than state-of-the-art baselines.

### Method in three stages
1. **Learn SkinTokens** — an FSQ-CVAE compresses sparse skinning weights into a 64,000-entry codebook
2. **Unified autoregressive modeling** — a Qwen3-0.6B Transformer generates the skeleton + SkinTokens as one interleaved sequence
3. **RL refinement via GRPO** — 4 geometric/semantic rewards fine-tune for out-of-distribution assets

### What you get
* **Input:** `.obj` / `.fbx` / `.glb` (one or many at once)
* **Output:** a rigged `.glb` with predicted **skeleton + per-vertex skinning weights**
* **Checkpoints:** ≈ 1.6 GB (GRPO-refined autoregressive model + FSQ-CVAE), cached on Drive
* **bpy_server sidecar:** Blender Python runs as a long-lived subprocess on port 59876 (matches the official demo's pattern) so the 30-60 s bpy import only happens once per session

### Quick Start
1. Open the notebook in Colab with a **T4 GPU** runtime (15 GB VRAM, 25 GB RAM recommended)
2. Run cells 1-3 in order: install → imports → core helpers (≈ 8-12 min on first run, 2-3 min with Drive cache)
3. Run cell 4 for the **Gradio UI** (interactive multi-mesh, full sliders) **or** cell 7 for a **single-mesh Colab picker** **or** cell 8 for a **folder batch**
4. The keep-alive cell 6 keeps the runtime alive for 12 h so the Gradio UI stays reachable

### Sampling parameters
All match the official demo's defaults: `top_k=5`, `top_p=0.95`, `temperature=1.0`, `repetition_penalty=2.0`, `num_beams=10`, `max_length=2048`. Plus three pipeline toggles:
* **Use existing skeleton** — if your mesh already has a rig, predict skinning only (faster, cleaner)
* **Preserve original texture & scale** — transfer the predicted rig back onto the un-preprocessed mesh
* **Voxel skin post-processing** — apply a voxel-based mask to the predicted skin weights (slower)

And two Colab / T4 toggles:
* **Low VRAM (CPU offload Qwen3)** — drops peak VRAM from ~14 GB to ~6-8 GB. Recommended for T4. ~1.3x slowdown.
* **Mirror outputs to Google Drive** — copy each rigged `.glb` to `AEI_3D_Out/SkinTokens/`.

### Hardware
* **GPU:** NVIDIA, ≥ 14 GB VRAM recommended. T4 (15 GB) works with Low VRAM enabled.
* **RAM:** ≥ 25 GB (Blender shared object + FSQ-CVAE encoder/decoder)
* **Disk:** ≈ 8 GB free (PyTorch + CUDA + bpy wheel + checkpoints)
* **First run:** 8-12 min (PyTorch + flash-attn + bpy) + ≈ 3 min (checkpoint download)
* **Subsequent runs:** 2-3 min (everything cached in Drive)

### License
* **Notebook + checkpoints:** MIT (upstream [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens))
* **bpy wheel:** GPL-3.0 (Blender Foundation), shipped unmodified via the public HF Space
* **Successor lineage:** SkinTokens (2026) → [UniRig (SIGGRAPH '25)](https://github.com/VAST-AI-Research/UniRig) → [CharacterGen](https://charactergen.github.io) (image → rigged 3D)

### Related notebooks
* **TripoSplat** — image → 3D Gaussians (also VAST-AI)
* **Hunyuan3D-2.1** — image → textured mesh
* **SplatTransform** — post-process 3DGS for game engines
* **Mesh Optimizer** — decimate, repair, UV-unwrap, quantize game-ready meshes
* **Asset_Library_Browser** — browse, tag, and ship your rigged library to a game engine

---

## SuGaR — Surface-Aligned 3DGS to Mesh (INRIA, non-commercial)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/SuGaR_Colab.ipynb)

**⚠️ License warning:** [SuGaR](https://github.com/Anttwo/SuGaR) ([Guédon & Lepetit, CVPR 2024](https://arxiv.org/abs/2311.12775)) uses the **INRIA Gaussian-Splatting License** — free for research/evaluation, **NOT for commercial use** without explicit INRIA consent. This notebook is for **personal-asset evaluation only**. For commercial assets, use Kiri Engine, Polycam, or another commercial alternative.

**What it does:** takes the 3DGS PLY from TripoSplat (or any 3DGS scene), adds surface-alignment constraints, re-trains the Gaussians to lie on a coherent surface, and extracts a **textured mesh** that follows the actual surface much more accurately than naive point-cloud reconstruction.

**Why use this instead of TripoSplat's default mesh export?** TripoSplat's `*_mesh.ply` / `.glb` / `.fbx` come from sampling 1 point per Gaussian and running Poisson reconstruction — fast (~30s) but lossy. SuGaR takes 2-3 hours per scene on L4 and gives you significantly better mesh quality, with proper UVs and texture maps included.

**Pipeline:** TripoSplat (image → 3DGS PLY) → Step 2 bridge (PLY → COLMAP scene) → Step 3 optional 3DGS re-train → Step 4 SuGaR surface-align + mesh extract → Step 5 game-asset transform → Step 6 Drive mirror.

**Compute:** L4 22 GB recommended. T4 16 GB will OOM. CPU-only not viable (CUDA rasterizer). First run: ~15 min install + ~2-3 hrs compute. For 200+ scenes, this is realistic only for **5-10 hero assets** — for the long tail, use TripoSplat's default mesh or the 3DGS PLY directly in a 3DGS viewer.

**Required citation:** Guédon & Lepetit, "SuGaR: Surface-Aligned Gaussian Splatting for 3D Mesh Reconstruction", CVPR 2024.

---

## GauStudio — 3DGS to Mesh via TSDF (MIT + INRIA mixed)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/GauStudio_Colab.ipynb)

**License notice:** [GauStudio](https://github.com/GAP-LAB-CUHK-SZ/gaustudio) (Chongjie Ye et al., GAP Lab @ CUHK-Shenzhen, [arXiv 2403.19632](https://arxiv.org/abs/2403.19632)) is **MIT-licensed for the main framework**, but the bundled `gaustudio-diff-gaussian-rasterization` is a derivative of INRIA's rasterizer and inherits its **non-commercial** restriction. Same situation as SuGaR.

**What it does:** Takes a trained 3DGS scene (or a TripoSplat PLY) and extracts a clean mesh by **rendering depth maps from the 3DGS cameras + fusing them with TSDF** (truncated signed distance function) via [vdbfusion](https://github.com/nicolamattina/vdbfusion). A fundamentally different algorithm family from TripoSplat's Poisson or SuGaR's density-field.

**GauStudio vs SuGaR — when to use which:**

| | GauStudio (this notebook) | SuGaR |
|--|--------------------------|-------|
| Time per scene | ~5-10 min | ~2-3 hrs |
| VRAM | T4 15 GB OK | L4 22 GB required |
| Mesh quality (single-view) | Modest | Modest (similar) |
| Mesh quality (multi-view) | Good | Best |
| Algorithm | TSDF fusion of depth maps | Density-field + Poisson |
| Topological style | Smoother, cleaner | Sharper, more detail |
| Best for | Quick sweeps, low-LOD game assets, T4 | Hero assets, CAD-like outputs, L4/A100 |

**Why include both?** Different use cases. For the 200+ image library:
- **GauStudio for the long tail** (50-100 subjects in a few hours on T4) — good enough for low-LOD game assets, smoothing style actually preferred
- **SuGaR for the 5-10 hero assets** (10-30 hrs on L4) — sharpest mesh quality for close-up game objects

**Pipeline:** TripoSplat (image → 3DGS PLY) → Step 3 bridge (PLY → `cameras.json`) → Step 5 `gs-extract-mesh` CLI → Step 6 game-asset transform + `.glb` → Step 8 Drive.

**Compute:** T4 15 GB works (GauStudio's minimum is 6 GB). L4 / A100 give headroom. First run: ~10 min install + ~5-10 min extract. **Skips texturing** (mvs-texturing C++ build is brittle on Colab) — texture in Blender or the target game engine.

**Required citation:** Ye, Danelljan, Yu, Ke, "GauStudio: A Modular Framework for 3D Gaussian Splatting", CVPR 2024.

---

## Asset Library Browser

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Asset_Library_Browser_Colab.ipynb)

After running TripoSplat + GauStudio / SuGaR + Mesh Optimizer on your 200+ images, you have a folder full of assets but **no way to browse them, pick the best ones, or ship them to a game engine**. This notebook fills that gap — a Gradio UI that scans your asset library, lets you preview / tag / favorite / filter, render thumbnails, and export to Unity AssetBundle-style folders, Godot .tres, or a static HTML portfolio.

**What it does:**

- **Step 1** — install deps (gradio 5.49.1, trimesh, open3d, Pillow, no model weights needed). CPU-only.
- **Step 2** — scan a library folder, recognize formats (3DGS raw PLY / packed `.splat` / compressed `.sog` `.spz` `.ksplat` `.lcc` / `KHR_gaussian_splatting` GLB via header magic; mesh GLB/OBJ/FBX/PLY/STL/3MF; voxel `-collision.glb`; PlayCanvas `lod-meta.json`; image PNG/JPG/WEBP; text, JSON, ZIP), build a metadata sidecar JSON (`_library_meta.json`) with tags / favorites / notes per asset. **Auto-reads `<slug>_meta.json` sidecars** (written by TripoSPlat STEP 8 / SplatTransform STEP 6 / Mesh_Optimizer) and merges `size_class` + `grounding` + `tiers` + `colliders` into the per-asset record. Idempotent — re-runs preserve user metadata.
- **Step 3** — the main Gradio UI: gallery with thumbnails, filter sidebar (**tag, format, size_class, grounded-only, favorite, untagged-only, has-colliders, modified-in-last-N-days, search**), sort by any field (name / size / size_class / grounding / format / mtime), group by format / size_class / grounding / folder. Click any asset to preview (3D mesh + collision GLB in inline `<model-viewer>`; 3DGS gets a per-format metadata card with viewer link — PlayCanvas for `.sog` + `KHR_gaussian_splatting` GLB, Scaniverse for `.spz`, LumaAI for `.lcc`, Antimatter15 fallback; VIDEO inline `<video>`; AUDIO inline `<audio>`; HDR + TEXT + META inline content viewer; `lod-meta.json` shows the JSON contents). Asset detail panel shows: green `size_class` badge, blue `grounded {deg}°` badge, **Modified** + **Added to library** dates, **Variants: N** badge, **Views: N** badge (increments every click), **Variant** dropdown to switch between sibling files. Tag editor, favorite toggle, **tier cycle button (`T` key)** to walk size_class through `small_prop → prop → tree_or_vehicle → building`. **Fullscreen preview** button (hides gallery + filters, shows large preview). **Comparison modal** (`D` key): side-by-side A/B previews with diff-highlighted stats table. **Keyboard nav**: `←`/`→` previous/next, `P`/`N` aliases, `Space` toggle favorite, `T` cycle tier, `D` compare, `F` focus search, `?` open help modal. **Batch actions**: pick multiple assets via `gr.CheckboxGroup`, then add tag / set tier / remove from library meta / **rename files** (prepend/append/replace modes; updates disk + META). **Bulk ZIP export**: 2 buttons (export CHECKED to ZIP, export all currently-filtered to ZIP) — packs files + `<slug>_meta.json` sidecars + `aei_manifest_<ts>.csv` into `/content/aei_library_export_<ts>.zip`. **Saved filter presets** persisted to `_library_meta.json["presets"]` — save current filter under a name, reload from dropdown. **Toasts**: `gr.Warning` when filter matches zero, `gr.Info` when filter shows subset. All state persists in the sidecar JSON.
- **Step 4** — batch render thumbnails (256×256 PNGs via trimesh's offscreen renderer; 3DGS gets a placeholder with the extension as a label). Idempotent.
- **Step 5** — export: Unity AssetBundle-style folder (`Assets/AEI_Library/Meshes/` + thumbnails + README), Godot folder (mesh files + README), static HTML portfolio (self-contained `index.html` with inline `<model-viewer>` per asset, deploy to GitHub Pages / Netlify), CSV manifest for inventory.
- **Step 6** — stats dashboard: total assets, format breakdown, **size_class breakdown**, **grounding status**, top tags, biggest files, missing/orphaned file report.
- **Step 7** — keep-alive (Gradio runs forever otherwise).
- **Step 8** — help / format reference / known issues.

**Compute:** CPU-only. The browser is a UI, not a model. No GPU required, no model weights to download. First run: ~3 min install. Subsequent: instant. Can run alongside other notebooks in the same session (read-only on the library folder).

**What it doesn't do:** 3DGS files don't get a thumbnail (no easy in-notebook 3DGS renderer) and can't be previewed in `<model-viewer>`. Workaround: run GauStudio on the 3DGS PLY → get a GLB → come back and preview the GLB. This is a fundamental limitation of all current web-based 3D viewers (Antimatter15, gsplat.js, LumaAI all require JS-based 3DGS renderers that aren't droppable into Gradio).

**Recommended workflow for the 200+ library:**

1. Run TripoSplat batch → 200 raw 3DGS PLYs
2. Run GauStudio batch on a subset → 200 GLB meshes
3. Run Mesh Optimizer on the GLBs → polished meshes with UVs
4. **Open this notebook**, set `LIBRARY_DIR` to your output folder, run STEP 2 to scan, STEP 4 to thumbnail, STEP 3 to browse / tag / favorite
5. STEP 5 to export the curated subset to Unity / Godot / a static HTML portfolio

---

## Game engine integration — loading the assets into Three.js / WebGPU

Once you have a `game_ready/` folder full of grounded splats, meshes, and colliders, you need to actually **use** them in your game engine. This is the missing piece that turns 200+ raw files into a working asset library.

The output of [TripoSPlat STEP 8](#triposplat--image-to-3d-gaussians-mit) (or [SplatTransform_Colab](#splattransform--3dgs-post-processor-playcanvas-mit)) looks like this per asset `<slug>`:

```
game_ready/
├── <slug>_full.sog          # 17 MB - hero tier (no decimate, 3 SH bands)
├── <slug>_standard.sog       # 4-5 MB - default LOD (25% Gaussians, 2 SH)
├── <slug>_background.sog     # 1-2 MB - mass-placement (6%, 0 SH)
├── <slug>_hull.glb           # 10-50 KB - convex collider
├── <slug>_collision.glb      # 1-3 MB - voxel marching-cubes collider
├── <slug>_meta.json          # sizes, transform, size_class
└── <slug>_grounded.ply      # intermediate (can ignore at runtime)
```

All `.sog` and collider `.glb` files are **already grounded** (bottom on Y=0, centered in XZ, axis-aligned). You can drop them in your scene with `splat.position.y = 0` and they work.

### TypeScript loader (Three.js + gsplat.js + WebGPU)

```typescript
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SplatLoader } from '@playcanvas/gsplat';  // or your 3DGS loader

interface AssetMeta {
  slug: string;
  source_ply: string;
  generated_at: string;
  tiers: Record<string, { path: string; size_bytes: number }>;
  colliders: Record<string, { path: string; size_bytes: number }>;
  grounding: {
    applied: boolean;
    translation: { x: number; y: number; z: number };
    rotation_y_radians: number;
    rotation_y_degrees: number;
    bounds_after: { min: number[]; max: number[] };
    size_units: { height: number; width: number; depth: number };
    size_class: 'small_prop' | 'prop' | 'tree_or_vehicle' | 'building';
  };
}

interface LoadedAsset {
  slug: string;
  meta: AssetMeta;
  splats: { full: THREE.Object3D; standard: THREE.Object3D; background: THREE.Object3D };
  hull: THREE.Object3D;       // invisible, for raycasting
  collision: THREE.Object3D;   // invisible, for physics
}

/**
 * Loads a single grounded 3DGS asset from a game_ready/ folder.
 * All outputs are pre-grounded; this just wires them up.
 */
export async function loadGroundedAsset(
  folder: string,    // e.g. '/assets/game_ready/'
  slug: string,      // e.g. 'hero_knight'
  scene: THREE.Scene
): Promise<LoadedAsset> {
  // 1. Load the meta.json sidecar (sizes, transform, size_class)
  const metaRes = await fetch(`${folder}/${slug}_meta.json`);
  const meta: AssetMeta = await metaRes.json();

  // 2. Load the 3 visual tiers (3DGS SOG files)
  const splatLoader = new SplatLoader();
  const [full, standard, background] = await Promise.all([
    splatLoader.loadAsync(`${folder}/${slug}_full.sog`),
    splatLoader.loadAsync(`${folder}/${slug}_standard.sog`),
    splatLoader.loadAsync(`${folder}/${slug}_background.sog`),
  ]);

  // 3. Load both collider GLBs (hull + voxel)
  const gltfLoader = new GLTFLoader();
  const [hullGltf, collisionGltf] = await Promise.all([
    gltfLoader.loadAsync(`${folder}/${slug}_hull.glb`),
    gltfLoader.loadAsync(`${folder}/${slug}_collision.glb`),
  ]);

  // 4. Apply the recorded grounding transform to the visual splats
  //    (Note: the .sog files are ALREADY pre-transformed by TripoSPlat
  //    STEP 8 / SplatTransform, so the translation/rotation values below
  //    are just for reference / undo. Setting position.y = 0 is a no-op
  //    if the asset was grounded at the source.)
  const g = meta.grounding;
  [full, standard, background].forEach((s) => {
    s.position.set(0, 0, 0);                                  // already at origin
    s.rotation.y = g.rotation_y_radians;                       // apply recorded rotation
  });

  // 5. Configure colliders: invisible, used only for raycasting
  hullGltf.scene.visible = false;
  collisionGltf.scene.visible = false;
  hullGltf.scene.name = `${slug}_hull`;
  collisionGltf.scene.name = `${slug}_collision`;

  // 6. Add everything to the scene
  scene.add(full, standard, background, hullGltf.scene, collisionGltf.scene);

  return { slug, meta, splats: { full, standard, background }, hull: hullGltf.scene, collision: collisionGltf.scene };
}

/**
 * Pick the right LOD tier based on the camera distance to the asset.
 * Background tier for distant (fastest, smallest), full tier for hero.
 */
export function selectLOD(asset: LoadedAsset, distanceToCamera: number): THREE.Object3D {
  if (distanceToCamera < 20) {
    return asset.splats.full;             // 17 MB - close-up, hero quality
  } else if (distanceToCamera < 100) {
    return asset.splats.standard;        // 4-5 MB - medium range
  } else {
    return asset.splats.background;      // 1-2 MB - distant, mass-placement
  }
}

/**
 * Raycast against the cheaper hull collider (10-50 KB) for picking.
 * Fall back to the accurate voxel collider for interactive objects.
 */
export function setupPicking(
  asset: LoadedAsset,
  raycaster: THREE.Raycaster,
  scene: THREE.Scene
): void {
  // Make hull visible to raycaster but invisible to camera
  asset.hull.visible = false;
  asset.hull.traverse((c) => (c as any).raycast = THREE.Mesh.prototype.raycast);
  scene.add(asset.hull);

  raycaster.intersectObjects([asset.hull], true);
}
```

### Usage example

```typescript
// 1. Load the asset
const hero = await loadGroundedAsset('/assets/game_ready/', 'hero_knight', scene);
console.log(`Loaded ${hero.slug} (${hero.meta.grounding.size_class}, ${hero.meta.grounding.size_units.height.toFixed(2)}m tall)`);

// 2. Drop on the ground (it's already grounded, so y=0 is fine)
hero.splats.full.position.set(10, 0, 5);

// 3. Per-frame LOD: pick tier based on camera distance
function updateLOD(camera: THREE.Camera) {
  const dist = camera.position.distanceTo(hero.splats.full.position);
  const currentVisible = [hero.splats.full, hero.splats.standard, hero.splats.background]
    .find((s) => s.visible);
  const desired = selectLOD(hero, dist);
  if (currentVisible !== desired) {
    currentVisible.visible = false;
    desired.visible = true;
  }
}

// 4. Picking
canvas.addEventListener('click', (e) => {
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects([hero.hull], true);
  if (hits.length > 0) {
    console.log(`Clicked ${hero.slug} at ${hits[0].point.toArray()}`);
  }
});
```

### Tiering constants (for procedural placement)

The `size_class` field in meta.json enables smart placement grids:

| Size class | Height | Recommended grid spacing | Use for |
|---|---|---|---|
| `small_prop` | < 0.5m | 0.5m | Debris, grass, rocks, small pickups |
| `prop` | 0.5-2m | 1-2m | People, weapons, furniture |
| `tree_or_vehicle` | 2-10m | 3-5m | Trees, cars, large structures |
| `building` | > 10m | 10-30m | Buildings, landmarks |

```typescript
const GRID_SPACING: Record<AssetMeta['grounding']['size_class'], number> = {
  small_prop: 0.5,
  prop: 2,
  tree_or_vehicle: 4,
  building: 15,
};

function placeAsset(asset: LoadedAsset, x: number, z: number) {
  const tier = asset.meta.grounding.size_class;
  const spacing = GRID_SPACING[tier];
  // Snap to grid
  const snappedX = Math.round(x / spacing) * spacing;
  const snappedZ = Math.round(z / spacing) * spacing;
  asset.splats.standard.position.set(snappedX, 0, snappedZ);
}
```

### Notes

- **Three.js version**: r150+ has WebGPU renderer (`THREE.WebGPURenderer`). gsplat.js works with both WebGL and WebGPU.
- **Browser compatibility**: SOG is supported in all major browsers via the PlayCanvas Engine. `KHR_gaussian_splatting` GLB is a future-proof standard but currently only PlayCanvas + Three.js (via plugins) support it. SPZ/KSPLAT are great for mobile AR.
- **First-frame timing**: 3DGS files are large (1-17 MB). For 200+ assets, lazy-load on demand; don't preload everything.
- **Memory budget**: 200 assets × 4-5 MB standard tier ≈ 1 GB of GPU memory. Background tier drops this to ~300 MB. Use LOD aggressively.

---

## Production pipeline — choosing the right tool for your 200+ image library

For converting 200+ images into a game-asset library, here's the honest decision tree based on what each tool actually delivers in 2026:

| Your input | Best tool | Why | Time per scene | Quality |
|------------|-----------|-----|----------------|---------|
| **Single image, shippable textured mesh** | **[Pixal3D](#pixal3d--image-to-3d-with-pbr-textures)** | PBR-textured GLB out of the box, no post-processing | 60-90 s | **Best (textures!)** |
| **Single image, real-time 3DGS preview** | **[TripoSplat](#triposplat--image-to-3d-gaussians-mit)** | Real-time 3DGS rendering, no mesh needed | 30 s | Best visual quality (3DGS) |
| **Single image, low-LOD game asset** | TripoSplat `.ply` + [GauStudio](#gaustudio--3dgs-to-mesh-via-tsdf-mit--inria-mixed) | Fast TSDF recon, OK for low-poly LOD | 30 s + 10 min | Medium (smooth) |
| **Single image, 5-10 hero assets** | [Pixal3D](#pixal3d--image-to-3d-with-pbr-textures) (then [Mesh Optimizer](#mesh-optimizer--post-process-for-game-ready-assets) post) | Best quality + clean post-process for game engines | 90 s + 5 min | **Highest** |
| **Single image, 3DGS→high-quality mesh research** | [SuGaR](#sugar--surface-aligned-3dgs-to-mesh-inria-non-commercial) | Surface-aligned recon, sharpest mesh from 3DGS | 2-3 hrs | Very high (research/eval only — INRIA non-commercial) |
| **Production / commercial, fast iteration** | **[Kiri Engine 3DGS-to-Mesh](https://www.kiriengine.app/blog/what-is-3dgs-to-mesh)** | Off-the-shelf, paid, supports TripoSplat output | ~5 min | High (commercial-OK) |
| **Production / commercial, single-image** | **[Polycam](https://poly.cam/) / [LumaGen](https://lumalabs.ai/gen) / [Meshy](https://www.meshy.ai/)** | Industry-standard, proprietary models, paid | ~2 min | Highest (commercial-OK) |

**Practical recommendation for your 200+ library (3DGS-first pipeline):**

1. Run **[TripoSplat STEP 7 batch](#triposplat--image-to-3d-gaussians-mit)** on all 200 images at quality defaults (steps=30, 262k Gaussians, ~30-60 s/image, 1.5-3 hrs total). Per-item Drive mirror so a T4 disconnect doesn't lose work.
2. Run **[TripoSplat STEP 8 (SplatTransform Lite)](#triposplat--image-to-3d-gaussians-mit)** to compress every PLY to game-ready formats (SOG/SPLAT/SPZ/GLB), generate 3 quality tiers (full/standard/background), and produce 2 colliders (hull + voxel). All **grounded by default** so assets drop on Y=0 in your engine. ~30-40 min for 200 assets.
3. Open the **[Asset Library Browser](#asset-library-browser)** with `LIBRARY_DIR` set to the `game_ready/` folder. Tag assets as hero/standard/background based on `size_class` from meta.json.
4. For the 5-10 hero assets where you also want a textured mesh fallback, run **[GauStudio](#gaustudio--3dgs-to-mesh-via-tsdf-mit--inria-mixed)** (~10 min/asset) or **[Pixal3D](#pixal3d--image-to-3d-with-pbr-textures)** (60-90 s/asset, better quality but research-only license).
5. For any commercial shipping with engines that don't support 3DGS, use **[Kiri Engine](https://www.kiriengine.app/blog/what-is-3dgs-to-mesh)** to convert the TripoSplat PLYs to commercial-OK meshes.

**Alternative (textured-mesh-first pipeline):** if your engines don't support 3DGS, swap step 1 for [Pixal3D STEP 7 batch](#pixal3d--image-to-3d-with-pbr-textures) and skip the STEP 8 3DGS compression (use [Pixal3D STEP 8 post-process](#pixal3d--image-to-3d-with-pbr-textures) for the hero assets instead).

---

## Mesh Optimizer — post-process for game-ready assets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Mesh_Optimizer_Colab.ipynb)

The companion to all our 3D generation notebooks: takes the **raw, often-broken mesh output** from Pixal3D, Hunyuan3D, Cube 3D, or any other 3D pipeline, and turns it into a **clean, game-ready asset** ready for Unity / Unreal / Godot / Three.js / 3D printing.

**Stack**: [`trimesh`](https://github.com/mikedh/trimesh) (3.6k★, MIT) for I/O + repair + smoothing · [`pyfqmr`](https://github.com/Kramer84/pyfqmr) (MIT, 100 KB) for fast quadric decimation (sp4cerat's gold-standard algorithm) · [`pymeshlab`](https://pymeshlab.readthedocs.io) (MIT) for advanced MeshLab filters (UV unwrap, hole filling) · [`Open3D`](https://www.open3d.org) (MIT) for point-cloud ops + alignment. **All CPU-only. No GPU required.**

Nine pipeline stages (all optional, all tunable):

1. **Load** — STL, PLY, OBJ, GLB, GLTF, 3MF, OFF, COLLADA via trimesh auto-detect
2. **Ground** — translate so the bottom sits on Y=0, center in XZ, optionally axis-align via PCA in the XZ plane (snapped to 90°). Writes a `<slug>_meta.json` sidecar with the transform + size_class — same format as TripoSPlat STEP 8's meta.json so the Asset_Library_Browser can read either source consistently.
3. **Clean** — merge duplicate vertices, remove degenerate faces, fix normals
4. **Fill holes** — pymeshlab `close_holes` (configurable max hole size)
5. **Decimate** — quadric edge-collapse to target face count (lossy or lossless)
6. **Smooth** — Laplacian / Taubin (volume-preserving) / Humphrey (shrinkage-reducing) / HC
7. **Remesh** — isotropic uniform remeshing via pymeshlab
8. **Recompute normals** — for proper shading after smoothing
9. **UV Unwrap** — for textured meshes (skips if already unwrapped)

Plus five export formats: `.glb` (Unity/Unreal/Three.js), `.obj + .mtl` (Blender/Maya), `.stl` (3D print), `.ply` (Meshlab/CloudCompare), `.3mf` (Windows 3D Builder).

Seven tabs:

- **Quick Optimize** — 4 one-click presets: Game-Ready (50% decimate + Taubin smooth + UV), Print-Ready (quad remesh + UV), Low-Poly (10% decimate + Humphrey smooth), Lossless (clean only)
- **Custom Pipeline** — full control over every stage, accordion-grouped UI
- **Inspect** — face/vertex counts, watertight check, manifold check, volume, area, bbox
- **Batch** — apply any preset to every mesh in a directory, outputs a zip
- **Compare** — before/after stats side-by-side with delta percentages
- **Transform & Normalize** — scale, recenter, flip axes, normalize to target extent, snap to grid, re-orient Y-up/Z-up/X-up. Useful before feeding meshes into other AI tools (Cube/Hunyuan3D) or for fitting into a unit cube
- **Help** — when-to-use table, format cheatsheet, citation

> "pyfqmr's quadric edge-collapse is the only decimation algorithm that gives Blender-quality results in Python." — see the [sp4cerat/Fast-Quadric-Mesh-Simplification](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification) paper for the underlying math.

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

1. **Run `TTS_Model_Loader.ipynb` first** to pre-download model weights to Google Drive (~50–70 GB for the full TTS suite; most users only need 1-2 models, so typical real usage is 5-20 GB; per-notebook toggles, resumable)
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

> **numpy 2.x compatibility patch (Colab Runtime 2026.01)**: Colab Runtime 2026.01 ships numpy 2.0.x where the string ufuncs (`_center`, `_ljust`, `_rjust`, `_zfill`, `_strip_*`, `_lstrip_*`, `_rstrip_*`, `_partition*`, `_rpartition*`, `_slice`, `_expandtabs*`, `_replace`, `is*`, `find`/`index`, `startswith`/`endswith`, `str_len`, etc.) are not yet exposed in `numpy._core.umath`. The first import of `numpy._core.strings` (triggered by `import torchaudio`, `import librosa`, `import soundfile`, MOSS-TTS' `AutoProcessor`, or any access to `np.strings`) then fails with `ImportError: cannot import name '_center' from 'numpy._core.umath'`. MOSS-TTS pulls all of these as transitive deps, so a fresh runtime is unusable without intervention. STEP 1 of the notebook applies a small monkeypatch right after `import numpy` that injects pure-Python ufuncs into `numpy._core.umath` for every name `numpy._core.strings` needs. Real implementations for the 19 most-called names (`_center`, `_ljust`, `_rjust`, `_zfill`, the strip/lstrip/rstrip family, and the boolean `is*` predicates); passthrough stubs for the rest. MOSS-TTS itself never actually calls any of these on string arrays during inference — the patch is only needed so the `from numpy._core.umath import ...` line in `numpy/_core/strings.py` succeeds at import time. The patch is a no-op on newer numpy versions (numpy 2.0.2+ already has all the ufuncs), so the same notebook runs unchanged on current and future runtimes.

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
- **Total storage**: ~50–70 GB for the full TTS suite (most users only need 1-2 models, so typical real usage is 5-20 GB). Toggle off the models you don't need in the loader UI.

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
| **OpenVoice V2** | OpenVoiceV2 | — | 7 (cross-lingual VC) | MIT |
| **Mesh Optimizer** | trimesh + pyfqmr + pymeshlab + Open3D | — | — | MIT (all deps) |
| **Audio Post-Processor** | pydub + imageio-ffmpeg + soundfile + librosa + pyloudnorm + noisereduce + resemble-enhance | — | — | MIT/BSD/ISC (all deps) |

---

## Audio Post-Processor — post-process for podcast, music, and broadcast

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Audio_PostProcessor_Colab.ipynb)

The companion to every TTS / VC / voice-cloning pipeline: takes the **raw audio output** from Qwen3-TTS, VoxCPM2, Higgs, MisoTTS, MOSS, dots.tts, Fish, Kokoro, OpenVoice V2, or any other source, and turns it into a **clean, ready-to-publish** audio file: trimmed, denoised, loudness-normalized (LUFS), and in the format you need (WAV / MP3 / FLAC / OGG / Opus / M4A / AAC / AIFF).

**Stack**: [`pydub`](https://github.com/jiaaro/pydub) (MIT) for format conversion + silence detect · [`imageio-ffmpeg`](https://github.com/imageio/imageio-ffmpeg) (BSD-4) for the static ffmpeg binary · [`soundfile`](https://github.com/bastibe/python-soundfile) (BSD-3, libsndfile) for fast WAV/FLAC/OGG/AIFF round-trips · [`librosa`](https://github.com/librosa/librosa) (ISC) for DSP analysis · [`pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm) (MIT) for ITU-R BS.1770-4 LUFS / LKFS broadcast-standard normalization · [`noisereduce`](https://github.com/timsainb/noisereduce) (MIT) for spectral-gating noise reduction · [`pedalboard`](https://github.com/spotify/pedalboard) (GPL-3.0, by Spotify Audio Intelligence Lab) for studio-quality VST-style effects (compressor, limiter, reverb, delay, chorus, phaser, distortion, pitch shift, bitcrush, resample, MP3 simulation) · [`resemble-enhance`](https://github.com/resemble-ai/resemble-enhance) (MIT, by Resemble AI) for AI speech denoising + bandwidth extension (44.1 kHz PyTorch model, **optional GPU acceleration**, CPU supported). **Most tabs CPU-only; AI Enhance benefits from GPU but works on CPU.**

> **License notes**:
> - `pedalboard` is GPL-3.0. This notebook is MIT. The `pedalboard` library is used as a `pip install`-ed dependency, not as a linked/embedded library, so the GPL's copyleft provisions don't extend to the notebook code. You can use the notebook and the resulting audio commercially; the `pedalboard` source itself remains under GPL-3.0 (relevant only if you redistribute `pedalboard`'s source or a modified version of it).
> - `resemble-enhance` is MIT. The AI speech model (`Enhancer` + `Denoiser`) is also MIT. ~500 MB of model weights are downloaded on first use and cached locally.

### Eight pipeline stages (all optional, all tunable)

1. **Load** — auto-detect format from extension. 8 input formats.
2. **Auto-trim silence** — remove leading/trailing silence from TTS output using pydub silence detection (configurable threshold in dBFS and minimum silence length in ms). Applied *first* so all subsequent steps operate only on the speech region. Default off; the **TTS Polish** preset enables it.
3. **High-pass filter** — cut rumble & mic handling noise (configurable cutoff, default 0=off).
4. **Denoise** — two backends: spectral gate (`noisereduce`, fast, no model) or Resemble AI denoiser (slow, better for speech). Tunable strength for the spectral gate.
5. **AI Enhance (Resemble Enhance)** — optional 44.1 kHz PyTorch model that does denoise + bandwidth extension + artifact reduction in one pass. 4 hyperparameters exposed.
6. **LUFS normalize** — broadcast-standard ITU-R BS.1770-4 loudness target (default -16 LUFS for podcast).
7. **Peak limit** — guard against clipping (default -1 dBFS).
8. **Export** — 8 output formats with configurable bitrate & sample rate.

Plus a full **Effects Chain** powered by `pedalboard` (Spotify Audio Intelligence Lab, GPL-3.0): **Gain**, **Compressor**, **Limiter**, **HighpassFilter**, **LowpassFilter**, **LadderFilter** (Moog-style 4-mode), **Reverb**, **Delay**, **Chorus**, **Phaser**, **Distortion**, **Clipping**, **PitchShift**, **Bitcrush**, **Resample**, **MP3Compressor** — every parameter exposed in the UI.

### 6 quick-process presets

| Preset | Best for | Output LUFS | Peak | Pipeline highlights |
| --- | --- | --- | --- | --- |
| **Podcast** | Two-speaker voice | -16 LUFS | -1 dBFS | Denoise 0.5 |
| **Music** | Music (preserves bass) | -14 LUFS | -1 dBFS | Denoise 0.3 |
| **Speech** | Broadcast-ready voice | -23 LUFS | -2 dBFS | Denoise 0.7, HPF 100 Hz |
| **Broadcast** | EBU R128 / ATSC A/85 | -23 LUFS | -1 dBFS | Denoise 0.4 |
| **TTS Polish** | TTS output (auto-trim leading/trailing silence, level-normalize) | -16 LUFS | -1 dBFS | Silence-trim (-40 dBFS, min 500 ms) + HPF 80 Hz, no AI |
| **Studio Polish** *(AI)* | Thin / compressed TTS or noisy speech | -16 LUFS | -1 dBFS | Resemble Enhance (44.1 kHz) |

Plus 8 export formats: `.wav` (lossless), `.flac` (lossless compressed), `.aiff` (lossless), `.mp3` (universal), `.ogg` (open-source), `.opus` (streaming), `.m4a` (Apple), `.aac` (raw ADTS).

### Ten tabs

- **Quick Process** — 6 one-click presets: Podcast, Music, Speech, Broadcast, **TTS Polish** (auto-trim silence + LUFS, best for TTS output), **Studio Polish (AI)** (Resemble Enhance, slowest but best quality)
- **Trim & Split** — trim to time range, split by silence, split into N-second chunks, detect silence ranges
- **Normalize** — peak / LUFS (ITU-R BS.1770-4) / no-op re-encode
- **Effects Chain** — 16 pedalboard effects in 4 accordion groups (Filters / Dynamics / Time & Space / Distortion & Pitch / Lo-Fi), 4 chain presets (Podcast Voice, Vocal Cleaning, Music Mastering, Lo-Fi Tape), or build a custom chain with every parameter exposed
- **Format Convert** — 8 formats with configurable bitrate (32-320 kbps) and sample rate (8 kHz - 96 kHz), mono/stereo
- **AI Enhance** — Resemble Enhance (Resemble AI, MIT): 44.1 kHz AI speech model that does **denoise + bandwidth extension + artifact reduction** in one pass. 4 hyperparameters exposed (CFM solver Midpoint/RK4/Euler, NFE 1-128, prior temperature tau, denoiser strength lambd, optional 2-pass denoise-first). Before/after audio + stats. GPU auto-detected, falls back to CPU.
  - **Single file** — upload one file, get enhanced output with before/after comparison
  - **Model source accordion** — `run_dir` text field lets you point at a custom fine-tuned checkpoint (folder must contain `hparams.yaml` + `ds/G/default/mp_rank_00_model_states.pt`); leave `<default>` to use the upstream Resemble AI weights (~713 MB, auto-downloaded, cached in `HF_HOME`).
  - **Device dropdown** — `auto` (default, GPU if available), `cuda` (force GPU), `cpu` (force CPU).
- **Denoise** — two backends:
  - **Spectral gate** (`noisereduce`): fast, no model download, good for steady hum/hiss. Strength slider 0-1.
  - **Resemble denoiser** (44.1 kHz AI model, Resemble AI MIT): same engine as the AI Enhance tab, denoise-only path, no bandwidth extension. Slow (1-5s/min on GPU, 10-60s on CPU) but better for speech. Shares the **run_dir** and **device** settings with the AI Enhance tab. Resamples to 44.1 kHz mono internally.
  - Backend radio + collapsible Resemble options group (visible only when Resemble is selected)
- **Batch** — unified directory processing. Pick a **mode** via radio:
  - **Quick Process preset** — apply any of the 6 presets to every audio file in a directory, download as a .zip
  - **AI Enhance** — process every audio file in a folder with Resemble Enhance. Pattern filter, recursive subdirectory walk, per-file progress, OK/Failed/Total/Time summary, subdirectory-preserving output. Shares the model `run_dir` and `device` settings from the AI Enhance tab.
  - Per-file failures are caught and logged; one bad file will not abort the batch.
- **Compare** — before/after waveform + stats (peak, RMS, LUFS, duration) with delta
- **Help** — when-to-use, format cheatsheet, LUFS targets, citation

### Loudness targets cheat-sheet

| Target LUFS | Use case |
| --- | --- |
| -14 LUFS | Spotify, YouTube, Apple Music (music default) |
| -16 LUFS | Podcast, audiobook default |
| -18 LUFS | AES streaming recommendation |
| -23 LUFS | EBU R128 / ATSC A/85 broadcast standard |
| -24 LUFS | ATSC A/85 (digital TV, US) |
| -27 LUFS | Cinematic / film dialog reference |

---

## Tools

Two Python scripts in `tools/` keep the notebooks consistent:

- **`tools/validate.py`** — fast AST-parse check on every code cell. Exits 0/1. Designed to be wired into CI.
- **`tools/qa_check.py`** — full polish audit (info= tooltips, try/except coverage, `concurrency_limit`, `clear_output()`, `demo.load` welcome, `FileLink` in Step 6). Excludes the 2 pre-existing Pixal3D notebooks.

Both run from the repo root with no dependencies:

```bash
python3 tools/validate.py    # OK: all 26 notebook(s) parse cleanly.
python3 tools/qa_check.py    # OK: all authored notebooks pass the polish audit.
```

A GitHub Actions workflow at `.github/workflows/qa.yml` runs both on every push to `main` and every PR. See the [QA badge](#aei-colab-notebooks) at the top of this README for live status. See [CONTRIBUTING.md](./CONTRIBUTING.md) for the recommended pre-submit checklist.

---

## Why This Exists

Most Colab AI notebooks require paid tokens, subscription services, or lengthy manual compilation of CUDA dependencies. This project precompiles everything into ready-to-install wheels and hosts them on GitHub — so you can open a notebook, run all cells, and get results immediately, on any supported GPU, completely free.

The 3D pipeline (Pixal3D) was the first notebook in the series. The TTS suite, video generators, voice-conversion tools, and post-processing helpers (the **Mesh Optimizer** for transform/normalize/repair of any generated 3D asset, and the **Audio Post-Processor** for trim/normalize/denoise/convert of any generated audio) have all been added since, with the same "open-and-run" philosophy across every modality.

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
| **TripoSplat** | [VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat) | [arXiv 2605.16355](https://arxiv.org/abs/2605.16355) |
| Mesh Optimizer | [mikedh/trimesh](https://github.com/mikedh/trimesh) · [Kramer84/pyfqmr](https://github.com/Kramer84/pyfqmr-Fast-quadric-Mesh-Reduction) · [cnr-isti-vclab/PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) · [isl-org/Open3D](https://github.com/isl-org/Open3D) | — | — |
| Audio Post-Processor | [jiaaro/pydub](https://github.com/jiaaro/pydub) · [imageio/imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) · [bastibe/python-soundfile](https://github.com/bastibe/python-soundfile) · [librosa/librosa](https://github.com/librosa/librosa) · [csteinmetz1/pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) · [timsainb/noisereduce](https://github.com/timsainb/noisereduce) · [spotify/pedalboard](https://github.com/spotify/pedalboard) · [resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) | — | — |

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
- **Mesh Optimizer** (the `Mesh_Optimizer_Colab.ipynb`):
  - All four deps are MIT: [trimesh](https://github.com/mikedh/trimesh), [pyfqmr](https://github.com/Kramer84/pyfqmr-Fast-quadric-Mesh-Reduction), [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) (Python wrapper, MIT — the underlying MeshLab C++ engine is GPL), [Open3D](https://github.com/isl-org/Open3D). Suitable for commercial use of generated/optimized meshes.
- **Audio Post-Processor** (the `Audio_PostProcessor_Colab.ipynb`):
  - Seven permissive deps: [pydub](https://github.com/jiaaro/pydub) (MIT), [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) (BSD-4), [python-soundfile](https://github.com/bastibe/python-soundfile) (BSD-3, libsndfile is LGPL), [librosa](https://github.com/librosa/librosa) (ISC), [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) (MIT), [noisereduce](https://github.com/timsainb/noisereduce) (MIT), [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) (MIT, by Resemble AI; model weights also MIT).
  - Plus [pedalboard](https://github.com/spotify/pedalboard) (GPL-3.0) for the Effects Chain tab. Used as a `pip install`-ed dependency, not linked/embedded, so the GPL doesn't extend to the notebook's MIT code. Commercial use of processed audio is fine.

Generated 3D assets are not moderated by Roblox safety systems. Use of the model weights is subject to the Cube repo's license terms; users are solely responsible for the outputs they generate.
