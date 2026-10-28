# AEI Colab Notebooks

[![QA](https://github.com/Skquark/AEI-Colab-Notebooks/actions/workflows/qa.yml/badge.svg)](https://github.com/Skquark/AEI-Colab-Notebooks/actions/workflows/qa.yml)

Free, self-contained Google Colab notebooks for AI-powered 3D generation, video creation, text-to-speech, and voice conversion. No sign-ups. No tokens. Just works.

See [LICENSE](LICENSE) for terms. [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a notebook. [CHANGELOG.md](CHANGELOG.md) for what's changed. [Model Cards & Upstream Attribution](#model-cards--upstream-attribution) for credits.

## Contents

### 3D
- [Pixal3D — Image to 3D with PBR Textures](#pixal3d--image-to-3d-with-pbr-textures)
- [Pixal3D Wheel Builder](#pixal3d-wheel-builder)
- [TRELLIS.2 — Microsoft Image to PBR 3D (4B O-Voxel, MIT)](#trellis2--microsoft-image-to-pbr-3d-4b-o-voxel-mit)
- [Cube 3D + CubePart](#cube-3d--cubepart)
- [Hunyuan3D 3.0 — Tencent Cloud API](#hunyuan3d-30--tencent-cloud-api-wrapper)
- [Hunyuan3D-2.1 — Tencent PBR 3D Pipeline *(flagship)*](#hunyuan3d-21--tencent-pbr-3d-pipeline-flagship)
- [Hunyuan3D-2 — Tencent Image / Text-to-3D](#hunyuan3d-2--tencent-image--text-to-3d)
- [TripoSplat — Image to 3D Gaussians (TripoAI/VAST-AI)](#triposplat--image-to-3d-gaussians-mit)
- [NoPoSplat — 2-3 Photos → 3DGS, pose-free (MIT, ICLR 2025)](#noposplat--2-3-photos--3dgs-pose-free-mit-iclr-2025)
- [Wild Gaussian Splatting — Video / Image Folder → 3DGS (MASt3R + INRIA)](#wild-gaussian-splatting--video--image-folder--3dgs-cc-by-nc-sa--inria)
- [MapAnything — Universal 3DGS-from-Images (Meta, Apache 2.0)](#mapanything--universal-3dgs-from-images-meta-apache-20)
- [Depth Anything 3 — Any-View Metric Depth, Pose & 3DGS (ByteDance, Apache 2.0 / CC BY-NC 4.0)](#depth-anything-3--any-view-metric-depth-pose--3dgs-bytedance-apache-20--cc-by-nc-40)
- [Pi3X — Video-Native 3DGS, Permutation-Equivariant (BSD-3 + CC BY-NC-4.0)](#pi3x--video-native-3dgs-permutation-equivariant-bsd-3--cc-by-nc-40)
- [HY-World 2.0 — WorldMirror 2.0 Multi-Modal World Reconstruction (Tencent-Hunyuan, ⚠️ territory-excluded)](#hy-world-20--worldmirror-20-multi-modal-world-reconstruction-tencent-hunyuan--territory-excluded)
- [GaussianGPT — Autoregressive 3D Gaussian Scene Generation (ECCV 2026, MIT)](#gaussiangpt--autoregressive-3d-gaussian-scene-generation-eccv-2026-mit)
- [InfiniSplat — Single-Image 3D Gaussian Reconstruction (SIGGRAPH Asia 2026, Apache 2.0)](#infinisplat--single-image-3d-gaussian-reconstruction-siggraph-asia-2026-apache-20)
- [LHM / LHM++ — Large Animatable 3DGS Human Avatar (1–8 Photos → Animatable Character, Apache 2.0)](#lhm--lhm--large-animatable-3dgs-human-avatar-18-photos--animatable-character-apache-20)
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
- [Ming 3.0 TTS — 44.1 kHz Studio Speech, Voice Cloning, 482 Voices & Podcast Studio (MIT)](#ming-30-tts--441-khz-studio-speech-voice-cloning-482-voices--podcast-studio-mit)
- [Chatterbox — Zero-Shot Voice Cloning, Emotion Exaggeration & Turbo Tags (MIT)](#chatterbox--zero-shot-voice-cloning-emotion-exaggeration--turbo-tags-mit)
- [Breeze TTS 2 — Voice Design, Clone & Direction (EN/ZH)](#breeze-tts-2--voice-design-clone--direction)
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
- [Seed-VC — Zero-Shot Speech & 44.1 kHz Singing Voice Conversion](#seed-vc--zero-shot-speech--441-khz-singing-voice-conversion)

### Music & Foley
- [MiniMax-Music3 — Lyrics + Caption-to-Music (Qwen3 + DiT, ⚠️ ≥24 GB VRAM)](#minimax-music3--lyrics--caption-to-music)
- [ACE-Step — Ultra-Fast Full-Song Generation, Lyric Editing & Repainting (3.5B, Apache 2.0)](#ace-step--ultra-fast-full-song-generation-lyric-editing--repainting-35b-apache-20)
- [MMAudio — Synchronized Video-to-Audio Foley & Sound Effects (MIT)](#mmaudio--synchronized-video-to-audio-foley--sound-effects-mit)

### Video
- [LTX-Video 2.5 — ComfyUI Audio/Video Generation](#ltx-video-25--comfyui-audiovideo-generation)
- [MiniMax-H3 — Video + Audio Generation (33B, INT4, ⚠️ territory-excluded)](#minimax-h3--video--audio-generation-33b-int4--territory-excluded)
- [MiniMax-H3 — ComfyUI Video + Audio Generation](#minimax-h3--comfyui-video--audio-generation)
- [Wan 2.2 — Text & Image-to-Video](#wan-22--text--image-to-video)
- [Wan 2.2 Animate — Character Animation & Replacement](#wan-22-animate--character-animation--replacement)
- [Wan 2.2 S2V — Audio-Driven Cinematic Video](#wan-22-s2v--audio-driven-cinematic-video)
- [Wan-VACE — All-in-One Video Creation, Inpainting & Reference-to-Video](#wan-vace--all-in-one-video-creation-inpainting--reference-to-video)
- [LongCat-Video-Avatar 1.5 — Audio-Driven Avatar Video Studio](#longcat-video-avatar-15--audio-driven-avatar-video-studio)
- [InfiniteTalk — Sparse-Frame Audio-Driven Video Generation & Dubbing](#infinitetalk--sparse-frame-audio-driven-video-generation--dubbing)
- [LivePortrait — Fast Portrait Animation, Facial Reenactment & Expression Editor](#liveportrait--fast-portrait-animation-facial-reenactment--expression-editor)
- [FramePack — 60-Second Constant-VRAM Next-Frame Video Generation](#framepack--60-second-constant-vram-next-frame-video-generation)
- [SeedVR2 — 1-Step Diffusion Video Super-Resolution & Restoration (1080p/4K)](#seedvr2--1-step-diffusion-video-super-resolution--restoration-1080p4k)

---

## Pixal3D — Image to 3D with PBR Textures

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Pixal3D_Colab.ipynb)

State-of-the-art image-to-3D generation by **Tencent ARC Lab** (SIGGRAPH 2026). Pixal3D lifts pixel features directly into 3D through back-projection, establishing pixel-to-3D correspondences that produce near-reconstruction-quality geometry with full PBR (Physically Based Rendering) textures — base color, roughness, metallic, and opacity. Built on Microsoft's TRELLIS.2 4B backbone.

### What You Get

- **Single-Image & Multi-View to 3D**: Supports both **Single-Image to 3D** (`Pixal3DImageTo3DPipeline`) and **Multi-View Images to 3D** (`Pixal3DMVImageTo3DPipeline`, 2–4+ views with optional `transforms.json` camera extrinsics and feature averaging)
- **GLB mesh output** with 4-channel PBR texture atlas (up to 4096×4096)
- **Gradio web UI** — tabbed interactive generation (Single-Image & Multi-View tabs) with preview renders
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

## TRELLIS.2 — Microsoft Image to PBR 3D (4B O-Voxel, MIT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/TRELLIS-2_Colab.ipynb)

Self-contained Colab studio for **Microsoft's** [`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) ([GitHub](https://github.com/microsoft/TRELLIS.2)) — the #1 most-downloaded open-weight 3D generative model on Hugging Face. TRELLIS.2 introduces a field-free **O-Voxel (Omni-Voxel)** representation that handles open surfaces, non-manifold geometry, and internal structures natively while synthesizing full **4-channel PBR textures** (Base Color, Metallic, Roughness, Opacity) with `<100ms` CUDA mesh extraction (`cumesh` + `o_voxel`).

- **Instant Prebuilt CUDA Wheel Stack**: Reuses our precompiled GitHub Release wheels (`o_voxel`, `cumesh`, `flex_gemm`, `nvdiffrec_render`) for **T4 (`sm75`)**, **L4 (`sm89`)**, and **A100 (`sm80`)** — zero CUDA compilation wait time.
- **Two Workflows in Step 4 Gradio UI**:
  1. **Image-to-3D O-Voxel PBR Generation** (`Trellis2ImageTo3DPipeline`): Supports `512`, `1024_cascade` (default), and `1536_cascade` resolutions, exporting PBR `.glb`, `.obj`, native `.vxz` O-Voxel grids, and turntable `.mp4` previews.
  2. **Untextured Mesh PBR Texturing** (`Trellis2TexturingPipeline`): Upload any existing `.glb` / `.obj` / `.ply` mesh + reference image to bake 1K–4K PBR textures.
- **Ungated & Low-VRAM Ready**: Automatic ungated fallbacks for `DinoV3FeatureExtractor` (`camenduru/dinov3-vitl16-pretrain-lvd1689m`) and `BiRefNet` (`ZhengPeng7/BiRefNet`) plus `low_vram` sub-model offloading (`keep_model_loaded=False`) for 16 GB T4 and 24 GB L4 GPUs. **MIT License (commercial-OK).**

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

- **Shape generation (DiT flow-matching)** — all **9 official model variants** covering the 0.6 B mini shape expert (`2mini-turbo`, `2mini-fast`, `2mini`), 1.1 B multi-view (`2mv-turbo`, `2mv-fast`, `2mv`), and 1.1 B full model (`2-turbo`, `2-fast`, `2`), plus **FlashVDM Turbo VAE** (`hunyuan3d-vae-v2-0-turbo` / `hunyuan3d-vae-v2-mini-turbo` via `enable_flashvdm()`) for up to 32× faster shape decoding.
- **Texture synthesis (`hunyuan3d-paint-v2-0-turbo` & `hunyuan3d-paint-v2-0` + `hunyuan3d-delight-v2-0`)** — paints a high-resolution diffuse texture onto a white mesh using a multi-view diffusion pipeline (defaulting to the fast `hunyuan3d-paint-v2-0-turbo` variant with automatic `hunyuan3d-delight-v2-0` image delighting). Optional: requires building a small CUDA extension (`custom_rasterizer`) on Step 1. If the build fails, the rest of the notebook still works; the Paint tab will tell you so.
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

Unified pose-free feed-forward 3DGS reconstruction notebook combining **[YoNoSplat](https://github.com/cvg/YoNoSplat)** (Botao Ye et al., ETH Zurich CVG, **ICLR 2026** official successor — **2 to 32 arbitrary unposed, uncalibrated views** with a Pi3 backbone and intrinsic prediction) and **[NoPoSplat](https://noposplat.github.io/)** (ICLR 2025 Oral — **2 to 3 unposed views** at up to 512×512). **Upload 2–32 uncalibrated photos or a video clip of a scene → get a complete 3D Gaussian Splat scene back in seconds**. No COLMAP, no bundle adjustment, no per-scene optimization.

### Two Engines in One Notebook
- **YoNoSplat (ICLR 2026, default)** — scales feed-forward pose-free 3DGS from 2 views up to **32 arbitrary views** (`dl3dv_224x224_ctx2to32.ckpt` & `re10k_224x224_ctx2to32.ckpt` from [`botaoye/YoNoSplat`](https://huggingface.co/botaoye/YoNoSplat)), predicting camera poses, intrinsics (`Free_Pose`), scale head, and 3D Gaussians jointly.
- **NoPoSplat (ICLR 2025 Oral)** — ViT-L MASt3R-based encoder with Unified Gaussian Adapter for high-resolution 2–3 view capture (`mixRe10kDl3dv_512x512.ckpt`, `re10k_3views.ckpt`, `re10k.ckpt`, `acid.ckpt` from [`botaoye/NoPoSplat`](https://huggingface.co/botaoye/NoPoSplat)).

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
* **Notebook + code:** **Apache 2.0** (upstream [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything), auto-updated via `git pull --ff-only` to include the Aug 2026 commits that eliminate the redundant 1.1 GB `torch.hub` DINOv2-giant download)
* **Checkpoint Variant Selector:** defaults to [🤗&nbsp;facebook/map-anything-apache](https://huggingface.co/facebook/map-anything-apache) (**Apache 2.0**, fully commercial-OK), with an optional toggle in Steps 1–7 to switch to [🤗&nbsp;facebook/map-anything](https://huggingface.co/facebook/map-anything) (**CC-BY-NC 4.0**, higher-accuracy research checkpoint).

### Related notebooks
* **NoPoSplat** — 2-3 photos → 3DGS in ~10s (MIT, but slower for >3 images)
* **WildGaussianSplatting** — video → 3DGS with per-scene optimization (non-commercial)
* **Pi3X** — video-native 3DGS with permutation-equivariance (non-commercial weights)
* **SplatTransform** — post-process 3DGS for game engines (SOG/SPLAT/SPZ/GLB)
* **Asset_Library_Browser** — browse, tag, and ship your library to a game engine

---

## Depth Anything 3 — Any-View Metric Depth, Pose & 3DGS (ByteDance, Apache 2.0 / CC BY-NC 4.0)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Depth-Anything-3_Colab.ipynb)

A Colab port of **ByteDance Seed's** [`ByteDance-Seed/Depth-Anything-3`](https://github.com/ByteDance-Seed/Depth-Anything-3) ([`depth-anything/DA3NESTED-GIANT-LARGE-1.1`](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) & [`depth-anything/DA3METRIC-LARGE`](https://huggingface.co/depth-anything/DA3METRIC-LARGE)). Depth Anything 3 unifies monocular metric depth estimation, multi-view camera pose estimation, 3D point-cloud reconstruction, and feed-forward **3D Gaussian Splatting (`.ply`)** into a single transformer backbone using cross-view self-attention.

- **5 Selectable Checkpoints**:
  - **`depth-anything/DA3NESTED-GIANT-LARGE-1.1` (1.4B Flagship)**: Nested Any-View Metric Depth + Camera Poses + direct 3DGS `.ply` & trajectory video rendering (`infer_gs=True`).
  - **`depth-anything/DA3METRIC-LARGE` (0.35B, Apache 2.0)** & **`depth-anything/DA3-LARGE-1.1` / `DA3-BASE` (Apache 2.0)**: Commercial-OK metric and multi-view depth + pose models.
- **Dual-Tab Gradio Studio**:
  - **Tab 1 — Single / Multi-View Images → Depth, `.glb` Point Cloud & 3DGS `.ply`**: Exports colorized depth maps, 16-bit millimeter depth PNGs, raw `.npz` arrays, interactive 3D point cloud + camera frustum `.glb` scenes, and 3DGS `.ply` splats.
  - **Tab 2 — Video → Multi-View 3DGS & Trajectory Render**: Extracts keyframes from any video clip and reconstructs a unified 3DGS scene with novel-view trajectory rendering (`extend`, `interp`, `circle`, `wander`, `dolly_zoom`).

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
* **`edge_mask=True`** (with tunable `edge_rtol=0.03`, `edge_normal_tol_deg=5.0`, `conf_threshold=0.1`) — masks out depth-discontinuity floaters using upstream `pi3.utils.geometry.depth_normal_edge` consistency filtering before writing COLMAP and `.glb` outputs.
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

## HY-World 2.0 — WorldMirror 2.0 Multi-Modal World Reconstruction (Tencent-Hunyuan, ⚠️ territory-excluded)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/HY-World-2.0_Colab.ipynb)

A Colab port of the
[Tencent-Hunyuan/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0)
**WorldMirror 2.0** world-reconstruction model.  Upload a handful of images
(or a short video) and the model predicts a full 3D scene in one forward pass:

* 3D scene point cloud with per-view colors
* 3DGS `.ply` (opens in [SuperSplat](https://supersplat.xyz) / PlayCanvas / gsplat.js)
* GLB mesh (triangulated **or** colored point cloud)
* Per-view depth maps (`.png` + raw metric `.npy`)
* Per-view surface normals
* Camera poses (OpenCV c2w + K matrices in `camera_poses.json`)
* Two Rerun `.rrd` recordings (full 3D + splat cloud) for desktop viewing
* In-browser 3D + splat visualisation via [`gradio_rerun`](https://pypi.org/project/gradio_rerun/)

### How it differs from our other 3DGS-from-X notebooks

| Notebook | Inputs | Speed | Pose-free | Splat output | License |
|---|---|---|---|---|---|
| **HY-World 2.0** | images **or** video | ~10 s | yes | yes (native) | ⚠️ Tencent (EU/UK/KR excluded) |
| **MapAnything** | 1-100 images | ~10 s | yes | yes (via colmap→gsplat) | Apache 2.0 |
| **NoPoSplat** | 2-3 photos | ~10 s | yes | yes (native) | MIT |
| **Pi3X** | 5-60 video frames | ~15 s | yes | yes (via colmap→gsplat) | BSD-3 code / CC BY-NC-4.0 weights |
| **WildGaussianSplatting** | image folder / video | 5-15 min | no (SfM first) | yes (per-scene train) | CC BY-NC-SA + INRIA |
| **TripoSplat** | text / image | ~10 s | yes | yes (native) | MIT |

WorldMirror is the only model in the suite that **also outputs a complete
3D mesh, normals, and depth maps** in a single forward pass.  Trade-off:
the 5 GB weight download + the territory-excluding license.

### ⚠️ License — Territory Restriction + MAU Cap

This notebook bundles the **HY-WorldMirror-2.0** weights, released under
the **Tencent HY-WORLD 2.0 Community License Agreement** (see
`License.txt` in the model repo):

* **Territory**: this license **does not apply in the European Union, the
  United Kingdom, or South Korea**.  If you are located in any of those
  jurisdictions, you are **not licensed** to use these weights and should
  not run this notebook there.  Continuing past the header cell is your
  acceptance of the license.
* **Commercial use cap**: if your product or service exceeds **1 M
  MAU**, you must request a separate commercial license from
  `hunyuan3d@tencent.com`.
* **Output ownership**: Tencent claims no rights in the outputs you
  generate.  You are solely responsible for outputs and their subsequent
  uses.
* **The vendored `hyworldmirror/` Python package** is the official code
  from [Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)
  and is **Apache-2.0 licensed**.  Only the **weights** carry the
  restricted license above.

This is the **most restrictive license of any notebook in the suite**.
MOSS-TTS (non-commercial) is comparable in restrictiveness; everything
else is MIT / Apache-2.0 / BSD-3 / CC-BY-NC-*.

### Pipeline

1. **STEP 1** — installs PyTorch 2.4.0 + cu124 (the only torch that has a
   pre-built `gsplat==1.5.3` wheel), `pip install`s gsplat, gradio 6.12.0,
   gradio_rerun 0.32.2, the inference deps, then `git clone`s the Space
   repo + the upstream Tencent repo.  Patches
   `hyworldmirror/models/layers/attention.py` to drop flash-attn and use
   PyTorch SDPA on bf16 instead (flash-attn wheels require torch 2.8 and
   conflict with gsplat's torch 2.4 requirement).  Pre-downloads the
   5.05 GB safetensors into the Drive cache.

2. **STEP 2** — Drive cache prologue, version checks, `load_worldmirror()`
   singleton loader, and the verbatim §3(d) Notice text.

3. **STEP 3** — `extract_video_frames()`, `process_uploaded_files()`,
   `render_depth_colormap()` / `render_normal_colormap()`, and
   `save_gs_ply_aes()` (3DGS binary PLY matching the upstream format
   that SuperSplat / PlayCanvas / gsplat.js all open).  Also loads the
   briaai/RMBG-1.4 ONNX for sky segmentation.

4. **STEP 4** — the full Gradio UI: file upload, frame gallery, frame
   selector, 7 reconstruction options (mesh / point cloud, sky filtering,
   confidence percentile, max-image-side, max-gaussians cap), per-view
   depth + normal galleries with prev/next navigation, 2 Rerun viewers
   (3D scene + Gaussian splat cloud), and 4 download buttons (GLB + PLY +
   RRD + zipped folder).  Defaults to 2-concurrent queue.

5. **STEP 6** — single-image quick test path (Colab file picker → 1-view
   reconstruction → zipped download via `FileLink`).

6. **STEP 7** — batch processor for a folder of scene sub-folders.  Writes
   a JSONL progress log so you can resume after a Colab disconnect.

### Requirements

* **GPU**: 24 GB VRAM recommended (A100, L4, RTX 4090).  T4 (16 GB) may
  OOM during inference; drop `Max image side` to 322 in that case.
* **Disk**: ~5.2 GB for the weights (cached on Drive after first run) +
  ~2 GB for gsplat + gradio_rerun + vendored hyworldmirror/.
* **RAM**: 12 GB Colab RAM is sufficient; the model loads weights via
  `safetensors` (memory-mapped, no full load).
* **First-run setup**: ~8 min (5 GB weight download + torch re-install +
  gsplat wheel).
* **Subsequent runs**: ~30 s (reuses Drive cache).

### Outputs per scene

```
scene_<selector>.glb          # mesh (or point cloud) you can open anywhere
gaussians.ply                 # 3DGS splats → SuperSplat / PlayCanvas
depth_NNN.png + depth_NNN.npy # per-view depth (turbo colormap + raw metric)
normal_NNN.png                # per-view normals
camera_poses.json             # c2w + K per frame
reconstruction.rrd            # Rerun recording (3D + cameras + normals)
gaussians.rrd                 # Rerun recording (Gaussian splat cloud)
```

### Related notebooks

* **MapAnything** — universal 3DGS-from-images, Apache 2.0 (commercial-OK alternative)
* **WildGaussianSplatting** — video → 3DGS with per-scene optimization
* **NoPoSplat** — 2-3 photos → 3DGS in ~10s, MIT
* **Pi3X** — video-native 3DGS, BSD-3 / CC BY-NC-4.0
* **SplatTransform** — post-process the 3DGS for game engines
* **TextureMapPrep** — generate the 6 PBR maps from an albedo

---

## GaussianGPT — Autoregressive 3D Gaussian Scene Generation (ECCV 2026, MIT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/GaussianGPT_Colab.ipynb)

A Colab port of [GaussianGPT](https://github.com/nicolasvonluetzow/GaussianGPT) (ECCV 2026, Nicolas von Lützow et al.). GaussianGPT generates **3D Gaussian scenes completely autoregressively** via next-token prediction — the same paradigm as GPT-style text models, but applied to 3D Gaussian splats.

### How it differs from our other 3DGS notebooks

| | GaussianGPT | MapAnything | NoPoSplat | HY-World 2.0 |
|--|-------------|-------------|-----------|--------------|
| **Input** | None (unconditional) | 1-100 images | 2-3 photos | Images or video |
| **Method** | Autoregressive GPT | Feed-forward | Feed-forward | Feed-forward |
| **Output** | 3DGS .ply + GIF | 3DGS .ply + GLB | 3DGS .ply | 3DGS .ply + GLB + depth |
| **Speed** | ~5-15s per scene | ~10s | ~10s | ~10s |
| **License** | MIT | Apache 2.0 | MIT | ⚠️ Tencent (EU/UK/KR excluded) |
| **Use case** | Generate new scenes from scratch | Reconstruct from images | Reconstruct from photos | Reconstruct from multi-view |

GaussianGPT is the only notebook in the suite that **generates new scenes without any input image** — it's a generative model, not a reconstruction model. It also supports completion (keep part of a scene, fill in the rest) and outpainting (extend scenes spatially).

### Two model variants

| Variant | Grid | GPT size | Checkpoints | Use case |
|---|---|---|---|---|
| **Scene-level (VFront)** | 20³ voxels | GPT-2-medium (24L, 1024d, ~350M) | ~5.5 GB | Indoor room scenes (3D-FRONT) |
| **Object-level (PhotoShape)** | 32³ voxels | GPT-2-small (12L, 768d, ~100M) | ~3.1 GB | Individual objects |

### Pipeline

1. **STEP 1** — installs torch 2.9.0+cu128, MinkowskiEngine (pre-built wheel from alpsaur fork v0.5.8, no compilation needed), gsplat + pytorch3d from source (~10-15 min each), remaining pip deps. First run: ~25-35 min.
2. **STEP 2** — downloads checkpoints from `kaldir.vc.cit.tum.de/gaussiangpt/` to Drive cache. ~5.5 GB (scene-level) or ~3.1 GB (object-level).
3. **STEP 3** — imports + lazy model loader + render/PLY helpers.
4. **STEP 4** — Gradio UI: model variant selector, temperature/top-k/top-p/num_samples/seed, render options, .ply + .gif download buttons.
5. **STEP 6** — single-sample quick test with FileLink.
6. **STEP 7** — batch generation with JSONL progress log.

### Requirements

* **GPU**: L4 (22 GB) or A100 (40 GB) recommended. T4 (15 GB) may be tight.
* **Disk**: ~5.5 GB for scene-level checkpoints (cached on Drive after first run).
* **First-run setup**: ~25-35 min (gsplat + pytorch3d compilation).
* **Subsequent runs**: ~30s (reuses Drive cache).

### Technical notes

- **MinkowskiEngine**: Uses the [alpsaur fork](https://github.com/alpsaur/MinkowskiEngine) v0.5.8 pre-built wheel (9.6 MB, cp312, torch 2.9.x+cu128). No compilation needed — the biggest dependency risk is eliminated.
- **Flash Attention**: NOT required. The code has a built-in SDPA fallback for non-Hopper GPUs (L4 = sm_89, not Hopper).
- **torch 2.9.0+cu128**: Matches the MinkowskiEngine wheel's build version. Force-reinstalled over whatever Colab ships.

### Outputs

```
sample_0000.pt          # Gaussian payload (means, sh0, opacities, scales, quats)
sample_0000.gif         # Orbit render (120 frames @ 24 fps)
sample_0000.ply         # INRIA-style 3DGS .ply (SuperSplat / PlayCanvas compatible)
```

### License

MIT License (Copyright © 2026 Nicolas von Lützow). Fully compatible with our suite.

### Related notebooks

* **MapAnything_Colab** — universal 3DGS-from-images, Apache 2.0
* **NoPoSplat_Colab** — 2-3 photos → 3DGS in ~10s, MIT
* **HY-World-2.0_Colab** — multi-view → 3DGS + GLB + depth + normals
* **SplatTransform_Colab** — 3DGS format converter + voxel collision mesh
* **TripoSplat_Colab** — text/image → 3DGS

---

## InfiniSplat — Single-Image 3D Gaussian Reconstruction (SIGGRAPH Asia 2026, Apache 2.0)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/InfiniSplat_Colab.ipynb)

A Colab port of [zju3dv/InfiniSplat](https://github.com/zju3dv/InfiniSplat) supporting both **InfiniSplat v2** ([`Edisoneh/InfiniSplat-v2`](https://huggingface.co/Edisoneh/InfiniSplat-v2) — **Multi-View 2–32+ unposed/posed images & calibrated 360° equirectangular panorama $\rightarrow$ 3DGS**) and **InfiniSplat v1** ([`PLUS-WAVE/InfiniSplat`](https://huggingface.co/PLUS-WAVE/InfiniSplat), SIGGRAPH Asia 2026 Journal Track — **Single RGB image + optional LiDAR depth $\rightarrow$ 3DGS**).

### Two Versions & Three Modes in One Notebook

| Version / Mode | Input | Checkpoint | Use case |
|---|---|---|---|
| **InfiniSplat v2 — Multi-View** *(default)* | 2–32+ images (unposed or posed) | `Edisoneh/InfiniSplat-v2` (`model_v2.safetensors`) | Feed-forward multi-view 3DGS reconstruction |
| **InfiniSplat v2 — 360° Panorama** | 1 equirectangular 360° panorama | `model_v2.safetensors` + `equi2pers_grid.pt` | Full 360° indoor/outdoor room & scene 3DGS |
| **InfiniSplat v1 — Single RGB / LiDAR** | 1 RGB image (+ optional depth) | `PLUS-WAVE/InfiniSplat` (`pretrained_model.pth`) | Monocular or depth-guided single-image 3DGS |

### Pipeline

1. **STEP 1** — installs torch 2.9.1+cu128 + pip deps (no custom CUDA builds!), clones repo, downloads checkpoints (~3 GB). First run: ~5-10 min.
2. **STEP 2** — imports + lazy model loader + PLY/video export helpers.
3. **STEP 3** — Gradio UI: image upload, mode selector, focal length override, floater filter toggle, render video toggle, .ply + .mp4 downloads.
4. **STEP 4** — keep alive.
5. **STEP 5** — single-image quick test with FileLink.
6. **STEP 6** — batch processing with JSONL progress log.
7. **STEP 7** — help / format reference / pipeline overview.

### Requirements

* **GPU**: T4 (15 GB) is sufficient. L4 / A100 are faster but not required.
* **Disk**: ~3 GB for checkpoints (cached on Drive after first run).
* **First-run setup**: ~5-10 min (pip installs + checkpoint download).
* **Subsequent runs**: ~15s (model load from cache).
* **No custom CUDA builds** — all deps are pre-built pip wheels.

### Outputs

```
scene.ply     # INRIA-style 3DGS (SuperSplat / PlayCanvas compatible)
orbit.mp4     # Novel-view orbit video (60 frames @ 10 fps, optional)
```

### License

Apache License 2.0. Fully compatible with our suite.

### Limitations

- **Occluded regions are hallucinated.** The model cannot see behind objects — it infers 3D from a single 2D image.
- **Best for indoor scenes.** Trained on HyperSim (synthetic indoor scenes). Outdoor scenes may produce artifacts.
- **Depth-guided mode requires a depth file.** The `lidar` mode needs a matching `.npy` / `.npz` / `.h5` / `.exr` depth file.

### Related notebooks

- **MapAnything_Colab** — universal 3DGS-from-images, Apache 2.0
- **NoPoSplat_Colab** — 2-3 photos → 3DGS, MIT
- **TripoSplat_Colab** — text/image → 3DGS, MIT
- **GaussianGPT_Colab** — generative 3DGS (no input), MIT
- **SplatTransform_Colab** — 3DGS format converter

---

## LHM / LHM++ — Large Animatable 3DGS Human Avatar (1–8 Photos → Animatable Character, Apache 2.0)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/LHM_Colab.ipynb)

Self-contained Colab studio for **Alibaba XR Lab's** [`aigc3d/LHM-plusplus`](https://github.com/aigc3d/LHM-plusplus) ([`3DAIGC/LHMPP-700M`](https://huggingface.co/3DAIGC/LHMPP-700M)) and [`aigc3d/LHM`](https://github.com/aigc3d/LHM) ([`3DAIGC/LHM-1B`](https://huggingface.co/3DAIGC/LHM-1B), ICCV 2025 Highlight). Reconstructs a complete, **animatable 3D Gaussian Splatting (`.ply`) human avatar** from **1 to 8 uncalibrated character photos** in seconds and animates it with SMPL-X 3D body + hand + facial motion sequences.

- **Supports Both LHM++ (Multi-View 1–8 Photos) & LHM (Single-View)**:
  - **`3DAIGC/LHMPP-700M` (Default)**: Feed-forward 1–8 view character reconstruction with optional neural refinement (`neural_render`) or direct Gaussian splat rendering (`gs_render`).
  - **`3DAIGC/LHM-1B` / `LHM-500M-HF` / `LHM-MINI`**: Single-photo animatable 3DGS human reconstruction that fits comfortably on T4 (16 GB) and L4 (24 GB).
- **Outputs**: Canonical T-pose / posed **3D Gaussian Splatting `.ply`**, interactive **3D `.glb` preview**, and **animated character `.mp4` video** driven by built-in or custom SMPL-X motion sequences. **Apache 2.0 License.**

---

## MiniMax-Music3 — Lyrics + Caption-to-Music

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/MiniMax-Music3_Colab.ipynb)

A Colab port of [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) — a 11B parameter lyrics- and caption-conditioned music generation model. Produces **44.1 kHz stereo WAV** files up to several minutes long, with structure-aware singing, instrumentation, and arrangement from a single (caption, lyrics) pair.

### How it differs from our other audio notebooks

| | MiniMax-Music3 | Higgs-Audio | MisoTTS | Qwen3-TTS |
|--|----------------|-------------|---------|-----------|
| **Output** | Music (with vocals) | Speech + sound | Speech | Speech |
| **Model** | 11B modular | 3B | 0.6B | 1.7B |
| **VRAM** | ~24 GB (bf16) | ~6 GB | ~3 GB | ~4 GB |
| **License** | MIT | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Use for** | Songs, jingles, BGM | Dialogue, narration | TTS | TTS |

### Architecture

```
caption + tagged lyrics
        ↓
  Qwen3-8B LM + RVQ depth decoder  →  text tokens
                                        ↓
  2.4B flow-matching DiT (30 steps, 200-frame windows, 100-frame overlap)
                                        ↓
  Flow-VAE vocoder  →  44.1 kHz stereo WAV
```

Three modular stages wired by `diffusers.ModularPipeline`: (1) **condition encoder** (Qwen3-8B + RVQ) turns caption + tagged lyrics into discrete codes, (2) **denoise** stage runs a 2.4B flow-matching DiT in 200-frame windows with 100-frame overlap, (3) **vocoder** (Flow-VAE) decodes codes to stereo audio.

### ⚠️ License — Territory Restriction

MiniMax-Music3 is MIT-licensed, but the model weights are distributed under the [MiniMax Music Generation Model License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-Music3) which **excludes EU, UK, South Korea, AND USA** (same territory restriction as MiniMax-H3). Not gated — anyone with a HF account can download.

### ⚠️ Hardware — ≥24 GB VRAM required

Unlike most notebooks in this suite, Music3 **will not run on T4 (16 GB)**. The `ModularPipeline` does not yet expose `enable_sequential_cpu_offload()` (as of diffusers main, August 2026), so the entire 11B pipeline must be loaded into VRAM in bf16. Recommended GPUs:

| GPU | VRAM | Works? | Notes |
|-----|------|--------|-------|
| A100 (Colab Pro) | 40 GB | Yes | Best. ~2 min/song at 30 steps. |
| L4 (Colab free / Pro) | 22 GB | Marginal | Will OOM during first song. Use A100 for batch. |
| T4 (Colab free) | 16 GB | No | Insufficient. Use MiniMax-H3 for audio instead. |

### Pipeline

1. **STEP 1** — installs torch 2.7+cu128, diffusers from main (for `ModularPipeline` support), torchaudio, scipy, spaces stub. First run: ~5-10 min.
2. **STEP 2** — downloads ~27 GB of Music3 weights to Drive cache: tokenizer, Qwen3-8B LM (4 shards), condition encoder, RVQ depth decoder, 2.4B DiT (2 shards), Flow-VAE vocoder, scheduler. First run: ~20-40 min depending on bandwidth.
3. **STEP 3** — imports, lazy pipe loader (`_load_pipe()` loads on first call), `generate_song(prompt, lyrics, duration, steps, seed, output_dir)` returns path + audio ndarray.
4. **STEP 4** — Gradio UI with caption + lyrics text boxes, duration/steps/seed sliders, randomize checkbox.
5. **STEP 5** — keep alive.
6. **STEP 6** — quick test (single song).
7. **STEP 7** — batch generation from JSON scene list with hash-based skip + resume log.
8. **STEP 9** — ffmpeg-based muxer: combine a generated song with a video file (from `LTX-2-5_Colab`, `MiniMax-H3_Colab`, etc.) to produce a music video. Three modes: **Replace** (swap audio), **Mix** (blend with `MIX_VOLUME` slider), **Concat** (loop video to song length).
9. **STEP 8** — tail the latest log / errors (debug aid).

### Output

```
output_<slug>_d<dur>s_s<steps>_seed<seed>_<ts>.wav   # 44.1 kHz stereo, float32
```

### Music Video Workflow (Music3 + LTX-2.5)

Generate songs in this notebook, generate video clips in `LTX-2-5_ComfyUI_Colab`, then return here to run STEP 9 to mux them:

```python
# In Music3 STEP 9, set:
VIDEO_PATH = '/content/drive/MyDrive/AEI_3D_Out/LTX-Video-2.5/scene_001.mp4'
SONG_PATH  = '/content/drive/MyDrive/AEI_3D_Out/MiniMax-Music3/song_lofi_d60s_s30_seed42.wav'
MUX_MODE   = 'Replace'   # or 'Mix' or 'Concat'
```

### Requirements

* **GPU**: A100 (40 GB) recommended. L4 (22 GB) is marginal for first song.
* **Disk**: ~27 GB for Music3 weights (cached on Drive).
* **First-run**: ~30-60 min total (download + install + lazy pipe init).
* **Subsequent**: ~30-60 s (load from Drive + lazy init).
* **Inference**: ~2-15 min per song depending on duration + steps (A100).

### Related notebooks

- **MiniMax-H3_Colab** — joint audio+video (synchronized, 32 kHz). Use if you want a single model that generates both.
- **LTX-2-5_ComfyUI_Colab** — video-only, muxes with Music3 output via STEP 9.
- **Higgs-Audio_Colab** — speech + sound effects, not songs.

---

## ACE-Step — Ultra-Fast Full-Song Generation, Lyric Editing & Repainting (3.5B, Apache 2.0)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/ACE-Step_Colab.ipynb)

Self-contained Colab studio for **StepFun + ACE Studio's** [`ACE-Step/ACE-Step-v1-3.5B`](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) ([GitHub](https://github.com/ace-step/ACE-Step)) — a 3.5B diffusion-based music foundation model (`music_dcae_f8c8` + `ace_step_transformer` + `umt5-base` + `music_vocoder`) that generates up to **4 minutes (240s) of 48 kHz stereo music** in ~20 seconds on A100 and under 60 seconds on a free **T4 (16 GB)** GPU.

- **4-Tab Music Production Studio**:
  1. **Text & Lyrics to Song (`text2music`)**: Genre/instrument tags + structured `[verse]`/`[chorus]`/`[bridge]` lyrics across 19 languages with quick genre presets and APG/CFG/Euler/Heun/PingPong schedulers.
  2. **Retake & Audio2Audio Variations (`retake` / `audio2audio`)**: Generate controlled variations or remix an uploaded reference song (`ref_audio_strength`).
  3. **Repaint & Localized Lyric Editing (`repaint` / `edit`)**: Regenerate a specific time window (`repaint_start` to `repaint_end`) or edit lyrics in place without changing the rest of the vocal/instrumental arrangement.
  4. **Song Outpainting (`extend`)**: Prepend an intro (`left_extend_length`) or append an outro (`right_extend_length`) to any track.
- **Cross-Notebook Integration**: Saves outputs to both `/content/drive/MyDrive/AEI_3D_Out/ACE-Step/` and `/content/drive/MyDrive/AEI_TTS_Out/ACE-Step/` for instant discovery by `Seed-VC`, `InfiniteTalk`, and `LongCat-Video-Avatar`. **Apache 2.0 License.**

---

## MMAudio — Synchronized Video-to-Audio Foley & Sound Effects (MIT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/MMAudio_Colab.ipynb)

Self-contained Colab studio for **Sony AI + UIUC's** [`hkchengrex/MMAudio`](https://github.com/hkchengrex/MMAudio) (`large_44k_v2`, 1.03B multimodal joint flow-matching transformer with 25 fps **Synchformer** visual-temporal alignment and **BigVGANv2** 44.1 kHz vocoder, CVPR 2025). Synthesizes **frame-accurate 44.1 kHz sound effects, footsteps, impacts, and environmental foley** directly from silent videos or text prompts in seconds on **T4 (16 GB)**, **L4 (24 GB)**, or **A100**.

- **3-Tab Foley Studio**:
  1. **Video-to-Audio Synchronized Foley**: Upload any silent video (`.mp4`/`.mov`/`.webm`) + optional text prompt and negative prompt to synthesize frame-accurate audio and automatically mux a finished `.mp4` (`make_video`) alongside standalone `.wav` export.
  2. **Text-to-Audio Sound Effects & Ambience**: Visual-free 44.1 kHz SFX and ambient soundbed generation from text prompts alone.
  3. **Multi-Variation Foley Explorer**: Generates up to 4 candidate soundtracks with distinct seeds in parallel for side-by-side auditioning.
- **License**: **MIT License** (commercial-OK).

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

* **[Marigold v1.1](https://huggingface.co/prs-eth/marigold-depth-v1-1), [Marigold High-Res v1.1 (`marigold-depth-hr-v1-1`)](https://huggingface.co/prs-eth/marigold-depth-hr-v1-1) & [Marigold v2.0](https://huggingface.co/huawei-bayerlab/marigold-v2-0)** (PRS-ETH / Huawei Bayer Lab) — Apache-2.0 code, RAIL++-M weights. Unified `diffusers` API for high-resolution depth (`processing_resolution=1024` in `marigold-hr`), surface normals, and intrinsic image decomposition (albedo + roughness + metallic).
* **[LOTUS v1](https://lotus3d.github.io/) & [Lotus-2](https://huggingface.co/jingheya/Lotus-2)** (EnVision Research, ICLR 2025 / 2026) — Apache-2.0 code + weights. Single-step deterministic dense geometry prediction (~5× faster than multi-step diffusion methods) with sharper fine-detail preservation in Lotus-2.
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
- **Step 3** — batch convert a folder of 3DGS inputs (`.ply`, `.compressed.ply`, Niantic `.spz` v2–v4, `.sog` v2 bundled/unbundled `meta.json`, `.ksplat`, `.splat`) into game formats (`.sog` v2 bundled & unbundled, `.spz` v3/v4, `.compressed.ply`, `.glb`, `.ply`, standalone `.html` viewer) with size + compression-ratio reports, **2DGS / mip-antialiasing metadata inspection & preservation** (`SPLAT_RENDER_MODE`), and optional **360° equirectangular panorama rendering** (`--projection equirect`). **Batch-hardening params:** `PARALLEL_WORKERS` (default 2) runs K conversions concurrently via `ThreadPoolExecutor`; `RESUME` (default True) + `FORCE_REDO` skip sources whose outputs already exist (survives Colab disconnects); `VALIDATE_OUTPUTS` re-opens each SOG to confirm it parses; `EMIT_MANIFEST` (default True) writes `<output_dir>/manifest.json` with per-source paths/sizes/ratios/splat-counts/assetClass + a top-level summary.
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
All match the official demo's defaults: `top_k=5`, `top_p=0.95`, `temperature=1.0`, `repetition_penalty=2.0`, `num_beams=10`, `max_length=2048`, plus checkpoint selection (`grpo_1400.ckpt` RL-aligned default vs `last.ckpt`), k-NN (`k=8`) post-sampling skin weight refinement, and skeleton export controls (`group_per_vertex`, `connect_tail_to_unique_child`, `add_root`). Plus three pipeline toggles:
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

## LTX-Video 2.5 — ComfyUI Audio/Video Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/LTX-2-5_ComfyUI_Colab.ipynb)

High-performance, joint **Audio-Video (AV)** generation pipeline powered by **Lightricks' LTX-2.5 (22B)** inside a headless ComfyUI backend on Google Colab. Supports photorealistic text-to-video, image-to-video (first frame and last frame conditioning), two-stage distilled 2x spatial upscaling, and the full suite of **6 official LTX-2.5 IC-LoRAs** (`Layout-To-Render`, `Ingredients`, `Refine-Details`, `Alpha-Gen`, `SDR-To-HDR`, and `Pixel-Spatial-Upscaler`).

- **Joint AV Synthesis**: Produces synchronized 24 fps video with matching ambient audio in a single denoise pass
- **Official LTX-2.5 IC-LoRA Suite**: Select and strength-tune any of Lightricks' 6 official IC-LoRAs across Step 2, Step 4 Gradio UI, Step 7 JSON batch, and Step 9 ReDetail upscaler
- **Gradio Web UI (Step 4)**: Interactive generation with text-to-video, image upload conditioning, canvas presets, seed controls, and integrated video player with audio
- **Two-Stage Distilled Pipeline**: 1-stage fast generation (8 steps) or 2-stage spatial upscale with LTXVLatentUpsampler and 3-step refiner
- **Batch Processing (Step 7)**: Sequential scene generation from JSON prompt files with resume capability
- **Video Upscaler (Step 9)**: Dedicated latent video upscaler with IC-LoRA selector and VHS audio passthrough
- **Hardware Support**: Runs on L4 (24 GB) and A100 (40 GB) with GGUF / int8-convrot quantization

---

## MiniMax-H3 — Video + Audio Generation (33B, INT4, ⚠️ territory-excluded)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/MiniMax-H3_Colab.ipynb)

A Colab port of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — a 33B parameter video generation model that produces **video with synchronized audio** (ambience, foley, speech). Supports text-to-video, image-to-video (first/last frame), and reference-based generation.

### How it differs from our other video notebooks

| | MiniMax-H3 | Wan 2.2 | Wan 2.2 Animate |
|--|------------|---------|-----------------|
| **Output** | Video + **audio** | Video only | Video only |
| **Model** | 33B DiT | 14B DiT | 14B DiT |
| **VRAM** | ~18.5 GB (INT4) | ~12 GB | ~14 GB |
| **License** | ⚠️ MiniMax (excludes USA/EU/UK/KR) | Apache 2.0 | Apache 2.0 |
| **Use for** | Video + synced audio | Text/image → video | Character animation |

MiniMax-H3 is the only notebook in the suite that generates **audio alongside video** via a split deployment: remote conditioner (HF Space) for text encoding + local INT4-quantized denoiser for video+audio generation.

### Architecture

```
prompt + images → [remote conditioner HF Space] → prompt_embeds
                                                    ↓
              INT4 DiT (15.5 GB) → denoise → VAEs → video.mp4 + audio
```

1. **Remote conditioner** (`multimodalart/qwen3vl-conditioner` HF Space): encodes prompt + optional keyframes via `gradio_client`. Saves 62 GB download + 14 GB VRAM.
2. **Local denoiser** (Colab GPU): 33B transformer INT4-quantized with torchao (~15.5 GB VRAM). VAEs loaded sequentially after denoising (~10.3 GB). Peak VRAM: ~18.5 GB.

### ⚠️ License — Territory Restriction

MiniMax H3 Community License excludes **EU, UK, South Korea, AND USA**. More restrictive than HY-World 2.0 (which excluded EU/UK/KR but not USA). Not gated — anyone can download.

### Pipeline

1. **STEP 1** — installs torch 2.11.0+cu128, diffusers from PR commit, torchao, transformers 5.8.0, av, stubs `spaces` module. First run: ~10-15 min.
2. **STEP 2** — downloads FL2VA transformer (61.7 GB) + VAEs (10.3 GB) to Drive cache. First run: ~30-60 min.
3. **STEP 3** — imports, remote conditioner, lazy model loader with torchao INT4 quantization.
4. **STEP 4** — Gradio UI: prompt + optional first/last frame, canvas selector, duration/steps/seed, prompt rewrite toggle.
5. **STEP 5** — keep alive.
6. **STEP 6** — quick test (single video).
7. **STEP 7** — batch generation from prompt list.

### Requirements

* **GPU**: L4 (22 GB) or A100 (40 GB). T4 is too small for INT4.
* **Disk**: ~72 GB for FL2VA weights (cached on Drive).
* **First-run**: ~45-75 min total (download + install + quantization).
* **Subsequent**: ~5-10 min (load from Drive + quantize).
* **Inference**: ~2-10 min per video depending on canvas/frames/steps.
* **Network**: Requires access to `multimodalart/qwen3vl-conditioner` HF Space.

### Outputs

```
output.mp4    # Video + synchronized stereo audio (32 kHz)
```

---

## MiniMax-H3 — ComfyUI Video + Audio Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/MiniMax-H3_ComfyUI_Colab.ipynb)

Headless ComfyUI workflow for **MiniMax-H3 (33B)** video generation with `--disable-pinned-memory` optimization. Generates cinematic video with native synchronized soundscapes from text prompts or first/last frame keyframe pairs.

- **Memory-Optimized Backend**: Disables host memory pinning, enabling 33B model execution within standard Colab system RAM limits
- **Gradio Interface (Step 4)**: Complete UI with 16 aspect ratio canvas presets, turbo LoRA controls (4-step and 8-step distillation), and live polling
- **Attention Kernel Customization**: Supports `sage`, `sol`, `sage+sol`, and chunked SwiGLU FFN on 16 GB GPUs (T4 low-memory mode)
- **Batch & Smoke-test**: Step 7 JSON batch generator and Step 8.5 lightweight diagnostic smoke-test

---

## Wan 2.2 — Text & Image-to-Video

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Wan2.2_Colab.ipynb)

A self-contained Colab wrapper around [Tencent / Wan-AI's Wan 2.2](https://github.com/Wan-Video/Wan2.2) video model. **Three officially-released variants in one notebook** (`5B-TI2V`, `14B-I2V`, and `14B-T2V`), **4-Step / 8-Step Lightning LoRA acceleration** ([`lightx2v/Wan2.2-Distill-Loras`](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)), `torchao` INT8 quantization + Diffusers block-level group offloading (`enable_group_offload`), plus prompt-enhancement features from the `ginigen/Wan-2.2-Enhanced` Space.

- **Three variants in one notebook**:
  - **`5B-TI2V`** (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`, default, 10 GB on disk) — fast unified Text-to-Video & Image-to-Video, fits comfortably on L4 (24 GB) and T4 (16 GB at 480p)
  - **`14B-I2V`** (`Wan-AI/Wan2.2-I2V-A14B-Diffusers`, 27 GB on disk) — flagship MoE Image-to-Video; runs on L4 (24 GB) with `torchao` INT8 + block-level group offload or natively on A100 (40 GB)
  - **`14B-T2V`** (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`, 27 GB on disk) — flagship MoE Text-to-Video; runs on L4 (24 GB) with `torchao` INT8 + block-level group offload or natively on A100 (40 GB)
- **⚡ 4-Step & 8-Step Lightning LoRA Toggle**: Optional high-noise + low-noise dual-expert rank-64 LoRAs from [`lightx2v/Wan2.2-Distill-Loras`](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras) (`guidance_scale=1.0`) for up to **~7× faster generation**
- **720p @ 24 fps output**, up to 5 seconds (121 frames) per call
- **Apache 2.0 license** on all variants
- **Official diffusers port** — uses `WanPipeline`, `WanImageToVideoPipeline`, and `AutoencoderKLWan` with `enable_group_offload(offload_type="block_level", use_stream=True)`
- **Drive-cached weights** — cached to Google Drive on first run, then instant
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

A self-contained Colab wrapper around **Wan-Animate-2** ([`Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers) & [`Wan-AI/Wan2.2-Animate-2-14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Diffusers)) and **Wan 2.2 Animate v1** ([`Wan-AI/Wan2.2-Animate-14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B-Diffusers)) — Wan-AI's **14B** models for **character animation and replacement**. Takes a **character image** (the subject) and a **reference video** (the motion), and animates or replaces the character. Apache 2.0.

- **Two execution paths in one notebook**:
  - **💻 Local GPU Inference (Diffusers + TorchAO INT8 + Block-Level Group Offload)** — runs locally on **L4 (24 GB)**, **A100 (40 GB)**, or **T4 (15 GB)** using:
    - **`Wan2.2-Animate-2-14B-Distilled` (10-Step Fast Default)** — end-to-end character animation in **10 Euler steps** with `guidance_scale=1.0` (no CFG overhead) and **no external DWPose preprocessor required**
    - **`Wan2.2-Animate-2-14B` (40-Step Standard)** — full-step Animate-2 quality
    - **`Wan2.2-Animate-14B` (Classic v1)** — Move (animate character image with reference motion) and Mix (replace character in video) modes
  - **🌐 Cloud API (DashScope)** — optional Alibaba DashScope API (`wan2.2-animate-move` / `wan2.2-animate-mix`) requiring no local GPU VRAM (~$0.10–0.50/call).

### Quick Start (Local GPU Path — L4 / A100 / T4)

1. Run Steps 1–4 in order — Step 1 installs `diffusers` from main + `torchao`, Step 2 configures the persistent Google Drive cache (`AEI_3D_Cache/Wan2.2-Animate`), Step 3 defines `WanAnimateLocalEngine` (with `Int8WeightOnlyConfig` + `enable_group_offload`), Step 4 launches the Gradio UI.
2. Select **💻 Local GPU**, choose **`Wan2.2-Animate-2-14B-Distilled (10-step Fast)`**, upload a character portrait + driving motion video, and click **Generate**.

### Quick Start (Cloud API Path)

1. Sign up for Alibaba Cloud's DashScope at [dashscope.aliyun.com](https://dashscope.aliyun.com/) and add `DASHSCOPE_API_KEY` to Colab secrets (🔑).
2. Switch the **Mode** dropdown to **🌐 Cloud API (DashScope)** in Step 4, upload a character image + reference video, pick Move or Mix, and click **Generate**.

### License

Apache 2.0 — the model weights are yours to use commercially. The cloud API is subject to Alibaba Cloud's terms of service.

---

## Wan 2.2 S2V — Audio-Driven Cinematic Video

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Wan2.2_S2V_Colab.ipynb)

A self-contained Colab wrapper around [Tencent / Wan-AI's Wan 2.2 S2V](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) — the **14 B MoE** audio-driven video generation model. Take a **character image** + **audio clip** and produce a cinematic video of the character speaking, singing, or reacting to the audio. Apache 2.0.

- **Two execution paths in one notebook**:
  - **💻 Local Single-GPU Inference (L4 24GB / A100 40GB)** — clones [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2), caches `Wan-AI/Wan2.2-S2V-14B` to Google Drive, and runs `WanS2V` locally on a single Colab GPU using CPU initialization, `t5_cpu=True`, `convert_model_dtype=True`, `offload_model=True`, `torchao` `Int8WeightOnlyConfig()` DiT quantization, PyTorch SDPA fallback, and optional **4-Step Lightning LoRA** ([`lightx2v/Wan2.2-Distill-Loras`](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras)) acceleration.
  - **🌐 Cloud API (DashScope)** — uses Alibaba's DashScope API (`wan2.2-s2v`). No GPU needed, ~$0.10-0.50/call, works on any Colab runtime.
- **Two required inputs**: character image + audio clip
- **Optional advanced inputs**: text prompt (describes the scene), pose video (for pose-driven generation, drives body motion)
- **Two resolution levels**: 480P (faster, fits L4 24 GB) and 720P (higher quality)
- **Audio length determines video length** — the output video is as long as the input audio
- **Trained for both speech and singing** — benchmarks against Hunyuan-Avatar and OmniHuman on cinematic character animation

### Quick Start (Local GPU Path — L4 24GB / A100 40GB)

1. Run Steps 1–4 in order — Step 1 clones `Wan-Video/Wan2.2` with single-GPU CPU-init & SDPA patches and installs `torchao`, Step 2 caches weights to Google Drive (`AEI_3D_Cache/Wan2.2-S2V`), Step 3 defines `WanS2VLocalEngine`, Step 4 launches the Gradio UI.
2. In Step 4, select **💻 Local GPU**, enable **4-Step Lightning LoRA** and **INT8 Quantization**, upload a character image + audio clip, and click **Generate**.

### Quick Start (Cloud API Path)

1. Sign up for Alibaba Cloud's DashScope at [dashscope.aliyun.com](https://dashscope.aliyun.com/) and add `DASHSCOPE_API_KEY` to Colab secrets (🔑).
2. Switch the **Mode** dropdown to **🌐 Cloud API** in Step 4, upload a character image + audio clip, pick 480P or 720P, and click **Generate**.

### License

Apache 2.0 — the model weights are yours to use commercially. The cloud API is subject to Alibaba Cloud's terms of service.

---

## Wan-VACE — All-in-One Video Creation, Inpainting & Reference-to-Video

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Wan-VACE_Colab.ipynb)

Self-contained Colab studio for **Wan-AI's** [`Wan-AI/Wan2.1-VACE-1.3B-diffusers`](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers) and [`Wan-AI/Wan2.1-VACE-14B-diffusers`](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers) (**Video All-in-one Creation and Editing**) via native `diffusers.WanVACEPipeline`. Unifies Text-to-Video, Multi-Subject Reference-to-Video, Video-to-Video, Masked Video Inpainting, First/Last Keyframe interpolation, and Spatial Video Outpainting in a single model.

- **Two Model Variants**:
  - **`1.3B-VACE` (Default — T4 16 GB & L4 24 GB)**: Fast, lightweight VACE model requiring only ~8–12 GB VRAM.
  - **`14B-VACE` (Flagship — L4 24 GB & A100 40 GB)**: Auto-enables `torchao` `Int8WeightOnlyConfig()` + `enable_group_offload(offload_type="block_level", use_stream=True)` on `<32 GB` GPUs.
- **3-Tab Gradio Studio**:
  1. **🖼️ Reference-to-Video (R2V) & Text-to-Video**: Upload 1–3 subject/character/object images (`reference_images`) + text prompt to synthesize new scenes preserving subject identity.
  2. **🎭 Video-to-Video & Masked Video Inpainting (Swap / Move-Anything)**: Replace or edit regions inside an existing video using bounding-box masks (`x1, y1, x2, y2`), uploaded mask videos/images, or reference images.
  3. **🎞️ First/Last Keyframe (`FLF2V`) & Spatial Outpainting**: Automatically constructs VACE neutral-gray `(128, 128, 128)` conditioning frames and binary `L`-mode masks from start/end keyframes or `top`/`bottom`/`left`/`right` outpainting margins. **Apache 2.0 License.**

---

## LongCat-Video-Avatar 1.5 — Audio-Driven Avatar Video Studio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/LongCat-Video-Avatar-1.5_Colab.ipynb)

Self-contained Colab studio for **Meituan LongCat Team's** [`meituan-longcat/LongCat-Video-Avatar`](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar) (`LongCat-Video-Avatar 1.5`, 13.6B DiT backbone) based on the [`victor/LongCat-Video-Avatar-1.5`](https://huggingface.co/spaces/victor/LongCat-Video-Avatar-1.5) Hugging Face Space. Generates expressive, lip-synced 480p and 720p avatar videos at 25 fps with coupled Text+Audio Classifier-Free Guidance (`text_guidance_scale=4.0`, `audio_guidance_scale=4.0`), Identity-Preserving Reference Sink long-video continuation (`ref_img_index=10`, `mask_frame_range=3`), and **DBCache** dynamic block caching.

### Three Generation Modes

1. **Single-Avatar Image + Audio to Video (`AI2V` + Long-Video `AVC`)**:
   - Animate any reference portrait or character image with speech or singing audio.
   - Multi-segment **Audio-Video Continuation (`AVC`)** seamlessly chains 93-frame (`3.72s`) segments using a latent Reference Sink (`ref_img_index=10`) with 13 overlapping frames and masked boundary transitions (`mask_frame_range=3`) to prevent identity or background drift across long clips.
2. **Audio + Text Only (`AT2V` — No Reference Image Required)**:
   - Synthesizes a complete talking character and scene from scratch using only a driving audio track and a descriptive text prompt.
3. **Dual-Speaker Multi-Person Conversation (`Multi-AI2V`)**:
   - Drives two characters in a single scene using two independent audio tracks in either **Sequential Turn-Taking (`add`)** or **Simultaneous / Duet (`para`)** mode with automatic left/right half splitting or custom pixel bounding boxes (`y_min, x_min, y_max, x_max`).

### Colab Engineering Highlights

- **Zero-Spike Streaming INT8 Quantizer (`Step 3`)**: Streams the 30.8 GB BF16 Single-Avatar (or 34.0 GB Multi-Avatar) safetensors shards tensor-by-tensor directly into `torchao` INT8 weight-only linear layers on CPU and saves a pre-quantized `~15 GB` `.pt` cache to Google Drive — avoiding the `>52 GB` host RAM spike that otherwise crashes Colab runtimes.
- **Stage-Separated Execution (`L4 24GB` & `A100 40GB`)**: Runs Stage 1 (`Kim_Vocal_2.onnx` vocal separation + `chinese-wav2vec2-base` 12-layer audio features), Stage 2 (`umt5-xxl` text encoding), Stage 3 (INT8 DiT denoising with DBCache), and Stage 4 (Wan2.1 3D causal VAE decode + FFmpeg AAC muxing) sequentially with explicit CPU/GPU handoffs so peak VRAM stays within **22.4 GB on L4**.
- **Gradio Studio UI (`Step 4`), Direct Cell (`Step 6`), and JSON Batch Queue (`Step 7`)**: Full 3-tab Gradio interface with built-in single and dual-speaker examples, direct form-based single/AVC generation, resume-safe JSON batch processing, and a **📂 Cross-Notebook Drive Audio Picker** that discovers recent `.wav`/`.mp3` files from your AEI TTS (`Qwen3-TTS`, `Breeze-TTS-2`, `VoxCPM2`), `Audio_PostProcessor`, and `MiniMax-Music3` Google Drive folders (also accessible via `AUDIO_PATH = "LATEST_DRIVE_AUDIO"` in Step 6).

### GPU Support

| GPU | VRAM | Precision Mode | Resolution | Notes |
|-----|------|----------------|------------|-------|
| **A100** | 40 GB / 80 GB | `bf16` or `int8` (auto) | 480p / 720p | Native BF16 or fast INT8 + DBCache |
| **L4** | 24 GB | `int8` (auto) | 480p (recommended) / 720p | Uses Zero-Spike INT8 DiT (~15 GB) + stage-separated offload |
| **T4** | 16 GB | ❌ | — | Insufficient VRAM for the 13.6B DiT + activations |

### License

MIT ([`meituan-longcat/LongCat-Video`](https://github.com/meituan-longcat/LongCat-Video)).

---

## InfiniteTalk — Sparse-Frame Audio-Driven Video Generation & Dubbing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/InfiniteTalk_Colab.ipynb)

Self-contained Colab studio for **MeiGen-AI's** [`MeiGen-AI/InfiniteTalk`](https://huggingface.co/MeiGen-AI/InfiniteTalk) ([GitHub](https://github.com/MeiGen-AI/InfiniteTalk) · [arXiv 2508.14033](https://arxiv.org/abs/2508.14033)) — a **14B sparse-frame audio-driven video generation and video dubbing** framework built on `Wan2.1-I2V-14B-480P`. Unlike mouth-inpainting lip-sync models, InfiniteTalk conditions on sparse keyframes to synchronize **lip movements, facial expressions, head rotation, and upper-body posture** simultaneously while preserving camera trajectory in video-to-video (`V2V`) dubbing.

### Three Workflows Out of the Box

1. **🖼️ Single-Person Image-to-Video (`I2V`)**:
   - Animate any portrait or character image with speech or singing audio of arbitrary length (`streaming` mode across overlapping 81-frame chunks with `motion_frame=9`).
   - Includes a dedicated **Step 7 Pan/Zoom Conditioning Video Helper** that converts static portraits into subtle cosine-eased pan/zoom `.mp4` inputs for `>60s` generations without color drift.
2. **🎞️ Sparse-Frame Video Dubbing (`V2V`)**:
   - Supply an existing `.mp4` video and a new audio track to re-dub lip movements, head pose, and facial expressions while preserving original camera motion and scene dynamics.
   - Optional automatic **PySceneDetect (`split_video_lib`)** shot-boundary splitting and Lab color-transfer correction (`color_correction_strength`).
3. **👥 Multi-Speaker Conversation (`Multi-I2V` / `Multi-V2V`)**:
   - Drive two speakers in the same image or video using two separate audio tracks in either **Sequential Turn-Taking (`add`)** or **Parallel / Duet (`para`)** mode with spatial reference-attention routing (`get_attn_map_with_target`) and automatic left/right or custom bounding-box overlays.

### Fast 8-Step FusionX LoRA Default & Smart Selective Downloader

- **⚡ `7.5x` Faster Inference (`fast_8step_lora` Default)**:
  - Uses MeiGen-AI's official pre-merged **INT8/FP8 + FusionX LoRA** checkpoints (`infinitetalk_single_int8_lora.safetensors` / `infinitetalk_multi_int8_lora.safetensors`) with `sampling_steps=8`, `shift=2.0`, `text_guide_scale=1.0`, `audio_guide_scale=2.0` — requiring only **16 DiT forward passes per 81-frame chunk instead of 120** (the 40-step standard mode runs 3 passes per step for dual CFG, whereas `text_guide_scale=1.0` triggers a 2-pass fast path).
  - Switchable via a single dropdown to `standard_40step` (`steps=40`, `shift=3.0/5.0`, `text_guide_scale=5.0`, `audio_guide_scale=4.0`).
- **💾 Selective `~30.5 GB` Weight Download (vs `~258 GB` Full Repos)**:
  - Downloads only the required `Wan2.1-I2V-14B-480P` components (`Wan2.1_VAE.pth`, CLIP ViT-H-14, tokenizers = `5.11 GB`), `TencentGameMate/chinese-wav2vec2-base` (`0.36 GB`), `t5_fp8.safetensors` (`6.42 GB`), and the selected pre-quantized InfiniteTalk DiT (`18.60 GB`), skipping `~228 GB` of redundant unquantized checkpoints.
- **🛠️ 11 Built-In Single-GPU Colab Patches**:
  - Automatic PyTorch 2.x SDPA fallback when `flash_attn` / `xfuser` are absent, `quant="int8"` T5 fallback to `t5_fp8.safetensors`, removal of 240 per-step `torch.cuda.empty_cache()` stalls, removal of `@lru_cache` GPU RoPE leak (`21.47 GB` saved in multi-speaker mode), `AutoWrappedQLinear` persistent VRAM budgeting (`~6.5 GB` pinned on L4 24GB, full resident on A100 40GB+), automatic DiT offload before 3D VAE decoding, and automatic short-audio (`<3.24s`) silence padding + exact frame trimming.
- **🎛️ Gradio Studio UI (`Step 8`), JSON Batch Queue (`Step 9`), & Cross-Notebook Drive Audio Picker**:
  - Includes both interactive Colab form cells (`Steps 5–7`) and a 2-tab **Gradio Studio UI (`Step 8`)** with a **📂 Cross-Notebook Drive Audio Picker** that scans your `AEI_TTS_Out`, `AEI_3D_Out`, and `AEI_TTS_Cache/voices` Google Drive folders for one-click TTS/Music $\rightarrow$ Video chaining (or via `INPUT_SOURCE = "latest_drive_tts_audio"` in Step 5), plus a hash-keyed, resume-safe **JSON Batch Scene Processor (`Step 9`)**.

### GPU Support

| GPU | VRAM | DiT Residency | 81-Frame Chunk (`fast_8step_lora`) | Notes |
|-----|------|---------------|------------------------------------|-------|
| **A100** | 40 GB / 80 GB | Full quantized DiT in VRAM (`num_persistent_param_in_dit=None`) | ~35–50 sec | Supports 480p and 720p natively; `--fp8` supported on H100/Ada |
| **L4** | 24 GB | `6B` params pinned (`~6.5 GB`) + `AutoWrappedQLinear` streaming | ~90–130 sec | Supports `int8` (and `fp8` on Ada CC 8.9) at 480p |
| **T4** | 16 GB | Full layer-by-layer streaming (`num_persistent_param_in_dit=0`) | ~4–6 min | 480p `clip` / short `streaming` only (`int8`) |

### License

Apache 2.0 ([`MeiGen-AI/InfiniteTalk`](https://github.com/MeiGen-AI/InfiniteTalk)).

---

## LivePortrait — Fast Portrait Animation, Facial Reenactment & Expression Editor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/LivePortrait_Colab.ipynb)

Self-contained Colab studio for **Kuaishou Technology's** [`KwaiVGI/LivePortrait`](https://github.com/KwaiVGI/LivePortrait) ([Hugging Face](https://huggingface.co/KwaiVGI/LivePortrait)) — ultra-fast implicit-keypoint portrait animation, video-to-video facial reenactment, and interactive single-image expression editing running in **4–6 GB VRAM** on **T4 (16 GB)**, **L4**, or **A100**.

- **3 Workflows in One Studio**:
  1. **🖼️ Image-to-Video Portrait Animation (`I2V`)**: Animate any portrait image using a driving facial video or built-in `.pkl` motion template (`talking.pkl`, `laugh.pkl`, `wink.pkl`, `shy.pkl`) with stitching, eye/lip retargeting, and motion region (`all`, `exp`, `pose`, `lip`, `eyes`) controls.
  2. **🎞️ Video-to-Video Facial Reenactment & Lip Silencing (`V2V`)**: Transfer facial expressions onto an existing portrait video with Kalman temporal smoothing, or run **V2V Lip-Silencing** to create closed-mouth idle avatar loops.
  3. **🎛️ Interactive Portrait Expression Editor (Avatar Prep Tool)**: Fine-tune static portrait expressions with live sliders for **Head Pitch / Yaw / Roll**, **Lip-Open Ratio (`0.0` = gently close an open mouth before feeding into `InfiniteTalk` or `LongCat-Avatar`!)**, **Smile**, **Wink**, **Eyebrow**, and **Gaze X/Y**. **MIT License.**

---

## FramePack — 60-Second Constant-VRAM Next-Frame Video Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/FramePack_Colab.ipynb)

Self-contained Colab studio for **Lvmin Zhang (`lllyasviel`) & Maneesh Agrawala's** [`lllyasviel/FramePack`](https://github.com/lllyasviel/FramePack) ([`FramePack_F1_I2V_HY_20250503`](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503) & [`FramePackI2V_HY`](https://huggingface.co/lllyasviel/FramePackI2V_HY)) — $O(1)$ constant-VRAM next-frame-section video diffusion that compresses historical context frames by importance so a 13B HunyuanVideo backbone can generate **1-second to 60-second (1,800-frame at 30 FPS) videos** on a single **T4 (16 GB)**, **L4 (24 GB)**, or **A100 (40 GB)** GPU.

- **Both Official Models Included**:
  - **`FramePack-F1` (Default — `lllyasviel/FramePack_F1_I2V_HY_20250503`)**: Forward-only temporal generation for sharper motion and higher dynamics.
  - **`FramePack Original` (`lllyasviel/FramePackI2V_HY`)**: Bidirectional inverted anti-drifting sampling that anchors endpoints early.
- **Selective Drive Cache & `DynamicSwapInstaller`**: Downloads only the required encoders/VAE + packed 13B transformer (skipping 26 GB of unused base HunyuanVideo shards), with TeaCache speedup and tunable `gpu_memory_preservation`. **Apache 2.0 License.**

---

## SeedVR2 — 1-Step Diffusion Video Super-Resolution & Restoration (1080p/4K)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/SeedVR2_Colab.ipynb)

Self-contained Colab studio for **ByteDance Seed's** [`ByteDance-Seed/SeedVR2-3B`](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) & [`ByteDance-Seed/SeedVR2-7B`](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) ([`numz/SeedVR2_comfyUI`](https://huggingface.co/numz/SeedVR2_comfyUI), ICLR 2026) — **one-step diffusion video and image super-resolution** viashifted-window adversarial post-training. Restores and upscales 480p AI videos and talking-avatar clips to crisp **1080p / 4K** in a **single diffusion step (`1 NFE`)**.

- **📂 Cross-Notebook Drive Video Picker + Lossless Audio Remux**:
  - Automatically scans your Google Drive outputs from **`InfiniteTalk`**, **`LongCat-Video-Avatar-1.5`**, **`Wan2.2`**, **`Wan-VACE`**, **`LTX-Video-2.5`**, **`MiniMax-H3`**, **`FramePack`**, and **`LivePortrait`** so you can 1-click upscale 480p avatar videos to 1080p/4K while automatically remuxing the original audio track via `ffmpeg`.
- **Single-GPU BlockSwap + Tiled Causal Video VAE**:
  - Supports 6 official FP8/FP16 checkpoints (`3B`, `7B`, `7B-Sharp`) with automatic `blocks_to_swap`, temporal chunk overlap, and Lab perceptual color transfer (`--color_correction lab`) on **T4 (16 GB)**, **L4 (24 GB)**, and **A100 (40 GB)**. **Apache 2.0 License.**

---

## Text-to-Speech

A matching lineup of self-contained Colab notebooks for state-of-the-art text-to-speech — **twelve models** covering English, Chinese, French, German, Spanish, Japanese, Korean, 11 Indian languages, 30 languages (VoxCPM2), and 80+ others. Every notebook follows the same 7-step pattern: **Install → Pre-cache → Core functions → Gradio UI → Keep-alive → Quick test → Batch synthesis**. No token or sign-up needed for any notebook except `IndicF5` (HF-gated). See [OpenVoice V2](#openvoice-v2) below for **voice conversion** (audio → audio with a cloned voice), which complements the TTS suite.

### Quick Start

1. **Run `TTS_Model_Loader.ipynb` first** to pre-download model weights to Google Drive (~50–70 GB for the full TTS suite; most users only need 1-2 models, so typical real usage is 5-20 GB; per-notebook toggles, resumable)
2. **Run `TTS_Voice_Library.ipynb`** to grab a curated set of reference voice clips with transcripts (used by the voice-cloning tabs)
3. Open any of the model notebooks below — they auto-load from the Drive cache and skip the weight-download step

### GPU Support

| GPU | VRAM | Notebooks that fit | Notes |
|-----|------|--------------------|-------|
| **A100** | 40 GB | All 12 | Recommended flagship target |
| **L4** | 24 GB | All 12 (MisoTTS, Fish S2 Pro need bf16 + careful memory) | Best price/perf for cloning |
| **T4** | 16 GB | 10 of 12 (excludes MisoTTS, Fish S2 Pro) | Qwen3 auto-falls back to 0.6B variant; Breeze TTS 2 fits in ~7.7 GiB |
| **CPU** | — | Supertonic-3, Kokoro-82M | 99M ONNX / 82M StyleTTS — runs anywhere |

### Qwen3-TTS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Qwen3-TTS_Colab.ipynb)

*Flagship* — Apache 2.0 TTS from Alibaba in three modes:

- **CustomVoice** — 9 premium voices (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee) covering Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **VoiceDesign** — describe a new voice in natural language ("Male, 17, tenor, gaining confidence…") and the model generates a matching voice
- **Base (Clone)** — 3-second reference clip + transcript → clone any voice

Auto-picks the 0.6B variant for T4 (16 GB) and the 1.7B variant for L4/A100.

### Ming 3.0 TTS — 44.1 kHz Studio Speech, Voice Cloning, 482 Voices & Podcast Studio (MIT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Ming-3-TTS_Colab.ipynb)

[`inclusionAI/Ming-3.0-tts-0.5B`](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B) (Dense 0.5B, ~4 GB VRAM) & [`inclusionAI/Ming-3.0-tts-tiny`](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny) (MoE 7.9B total / 1.3B active, ~19 GB VRAM) ([GitHub](https://github.com/inclusionAI/Ming-3.0-tts) · [Space](https://huggingface.co/spaces/hugging-apps/ming-3-tts-05b)) — **inclusionAI (Ant Group Bailing Team)'s** 44.1 kHz speech synthesis family (**MIT License**), ranked **#1 among open-weight models on Controlled Voice** (Arena Elo 1069) and **#1 overall on Pronunciation Robustness** (93.2% overall, 96.5% shorthand normalization) on the Artificial Analysis TTS Leaderboards:

- **Tab 1 — Zero-Shot Voice Cloning & 14 Paralinguistic Tags**: Clone any 3–10s reference clip + transcript across **7 languages** (`Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Spanish`) with 1-click insertion buttons for all 14 paralinguistic tags (`[Laughter]`, `[Sigh]`, `[Breathing]`, `[Cough]`, `[Uhm]`, `[Surprise-ah]`, `[Surprise-oh]`, etc.) and Drive Voice Library integration.
- **Tab 2 — Emotion, Dialect & Prosody Attribute Control**: Clone speaker timbre with explicit **7-emotion control** (`高兴/happy`, `悲伤/sad`, `愤怒/angry`, `恐惧/fearful`, `厌恶/disgusted`, `惊讶/surprised`, `中性/neutral`), **Cantonese / Sichuanese dialects**, or **Speed / Pitch / Volume (`BASIC`)** controls.
- **Tab 3 — Voice Design & Freeform Delivery Instructions**: Create new voices from natural-language persona descriptions (`VOICE_DESIGN`) or steer delivery style (`FREEFORM`) with zero reference audio.
- **Tab 4 — 482 Built-in Premium Voices (`IP`)**: 5 English voices (`US_James`, `US_Stella`, `US_Emma`, `UK_George`, `UK_Charlotte`), 48 Chinese base voices (`灵小柔`, `灵楷`, `灵小安`, etc.), and 429 Chinese emotion-specific variants.
- **Tab 5 — Two-Speaker Podcast Studio**: Native multi-speaker `speaker_1:` / `speaker_2:` dialogue generation on `Ming-3.0-tts-tiny` (L4/A100) plus automatic turn-by-turn stitched multi-speaker synthesis on `Ming-3.0-tts-0.5B` (works on 16 GB T4!). Saves outputs to `/content/drive/MyDrive/AEI_TTS_Out/Ming-3-TTS/` for automatic discovery in `InfiniteTalk`, `LongCat-Avatar`, and `Seed-VC`.

### Chatterbox — Zero-Shot Voice Cloning, Emotion Exaggeration & Turbo Tags (MIT)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Chatterbox_Colab.ipynb)

[`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) & [`ResembleAI/chatterbox-turbo`](https://huggingface.co/ResembleAI/chatterbox-turbo) ([GitHub](https://github.com/resemble-ai/chatterbox)) — **Resemble AI's** flagship open-source TTS and voice-cloning family (**MIT License**, commercial-OK, fits comfortably in **6–10 GB VRAM on T4**). Combines 4 engines in a 4-tab studio:

- **Tab 1 — Zero-Shot TTS (`ChatterboxTurboTTS` & `ChatterboxTTS`)**: Clone any voice from a 5–20s clip with first-of-its-kind **Emotion Exaggeration (`exaggeration=0.0..2.0`)** and **`cfg_weight`** pacing controls, plus 1-click paralinguistic tag buttons (`[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`, `[gasp]`, `[clear throat]`) in Turbo mode.
- **Tab 2 — Multilingual TTS (`ChatterboxMultilingualTTS`)**: Zero-shot voice cloning across **23 languages** (`ar`, `da`, `de`, `el`, `en`, `es`, `fi`, `fr`, `he`, `hi`, `it`, `ja`, `ko`, `ms`, `nl`, `no`, `pl`, `pt`, `ru`, `sv`, `sw`, `tr`, `zh`).
- **Tab 3 — Zero-Shot Voice Conversion (`ChatterboxVC`)**: Direct speech-to-speech timbre conversion into any target reference voice.
- **Tab 4 — Multi-Speaker Dialogue & Audiobook Studio**: Parse `[Speaker 1]: ...` scripts with up to 3 cloned voices into a stitched WAV in `/content/drive/MyDrive/AEI_TTS_Out/Chatterbox/` (auto-discovered by `InfiniteTalk` and `LongCat-Avatar`).

### Breeze TTS 2 — Voice Design, Clone & Direction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Breeze-TTS-2_Colab.ipynb)

`BreezeBlue/Breeze-TTS-2` (3B) — bilingual (EN/ZH) TTS ranked **#1 among open-weight models** on the Artificial Analysis TTS leaderboard. Three modes:

- **Voice Design** — create a voice from a natural-language description, no reference audio needed (CFG 4.0 recommended)
- **Voice Clone** — clone a speaker from clean reference audio + its exact transcript (auto-transcribed with CPU Whisper in the UI)
- **Voice Direction** — clone a reference voice while steering tone, emotion, and pace via an instruction ("Speak slowly with a restrained, serious tone.")

Inline vocal events in the text: `(laugh)`, `(sigh)`, `(cough)`, `(clears throat)` (EN); `[笑]`, `[叹气]`, `[咳嗽]`, `[清嗓子]` (ZH). Architecture: Qwen3-1B backbone + 12-layer depth decoder (fills 16 RVQ codebooks) + Kyutai Mimi codec → 24 kHz mono. **≈7.7 GiB VRAM — works on T4.**

**Audiobook chapter mode (STEP 7)**: point it at a plain-text file (paragraphs separated by blank lines), and it splits long paragraphs at sentence boundaries, generates each chunk with the same voice/seed, inserts 0.35 s pauses between paragraphs, and concatenates into **one full-chapter WAV** (+ individual per-paragraph WAVs). Resume-safe via a hash-keyed JSONL log.

⚠️ **License**: code is Apache 2.0, but **weights & outputs are research / non-commercial only** (RESONIA, INC. — commercial use requires written authorization). Fine for personal/indie audiobooks; commercial audiobooks need a license from contact@breeze.blue.

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

`nari-labs/Dia2-2B` (2B flagship), `nari-labs/Dia2-1B` (1B fast), and `nari-labs/Dia-1.6B-0626` (1.6B v1 via native 🤗 `transformers` `DiaForConditionalGeneration`) — ultra-realistic streaming multi-speaker `[S1]`/`[S2]` dialogue TTS with **21 non-verbal tags** (`(laughs)`, `(coughs)`, `(sighs)`, `(gasps)`, etc.) and dual-speaker prefix audio voice cloning (`prefix_speaker_1`, `prefix_speaker_2`). English only. Apache 2.0.

### IndicF5

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/IndicF5_Colab.ipynb)

Unified Indian-language TTS notebook supporting both **`bodhan-ai/indic-speak`** (AI4Bharat's ungated 3.3B Llama-3.2 + Vocos Speech Foundation Model covering **all 22 scheduled Indian languages + English** with 8 built-in voices and 5 emotion/style tags `NEUTRAL`, `HAPPY`, `SAD`, `ANGER`, `FEAR`) and **`ai4bharat/IndicF5`** (400M polyglot zero-shot voice cloning across 11 Indian languages; gated on HF). MIT / Apache 2.0.

### MOSS-TTS v1.5

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/MOSS-TTS_Colab.ipynb)

4-tab **OpenMOSS Audio Studio** covering:
- **`OpenMOSS-Team/MOSS-TTS-v1.5` (8B)** — 31 languages, zero-shot voice cloning, **duration control** (target token count), **IPA / Pinyin input**, inline `[pause X.Ys]` markers, and code-switching.
- **`OpenMOSS-Team/MOSS-VoiceGenerator` (1.7B)** — text-described voice design from natural-language persona instructions (with a 1-click button to send designed voices directly into Tab 1 for 31-language cloning).
- **`OpenMOSS-Team/MOSS-TTSD-v1.0` (8B)** — multi-speaker `[S1]`/`[S2]` conversational dialogue synthesis with dual-speaker reference continuation.
- **`OpenMOSS-Team/MOSS-SoundEffect` (8B)** — text-to-SFX and environmental ambience synthesis with duration control. Apache 2.0.

> **numpy 2.x compatibility patch (Colab Runtime 2026.01)**: Colab Runtime 2026.01 ships numpy 2.0.x where the string ufuncs (`_center`, `_ljust`, `_rjust`, `_zfill`, `_strip_*`, `_lstrip_*`, `_rstrip_*`, `_partition*`, `_rpartition*`, `_slice`, `_expandtabs*`, `_replace`, `is*`, `find`/`index`, `startswith`/`endswith`, `str_len`, etc.) are not yet exposed in `numpy._core.umath`. The first import of `numpy._core.strings` (triggered by `import torchaudio`, `import librosa`, `import soundfile`, MOSS-TTS' `AutoProcessor`, or any access to `np.strings`) then fails with `ImportError: cannot import name '_center' from 'numpy._core.umath'`. MOSS-TTS pulls all of these as transitive deps, so a fresh runtime is unusable without intervention. STEP 1 of the notebook applies a small monkeypatch right after `import numpy` that injects pure-Python ufuncs into `numpy._core.umath` for every name `numpy._core.strings` needs. Real implementations for the 19 most-called names (`_center`, `_ljust`, `_rjust`, `_zfill`, the strip/lstrip/rstrip family, and the boolean `is*` predicates); passthrough stubs for the rest. MOSS-TTS itself never actually calls any of these on string arrays during inference — the patch is only needed so the `from numpy._core.umath import ...` line in `numpy/_core/strings.py` succeeds at import time. The patch is a no-op on newer numpy versions (numpy 2.0.2+ already has all the ufuncs), so the same notebook runs unchanged on current and future runtimes.

### dots.tts-soar

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/dots.tts-soar_Colab.ipynb)

`rednote-hilab/dots.tts-soar` (2B, 16-step ODE) & `rednote-hilab/dots.tts-mf` (2B, **4-step MeanFlow distilled variant** for ~4× faster inference with fused CFG) — **fully continuous** autoregressive TTS (no discrete tokens anywhere in the pipeline) with **48 kHz studio-quality output** and a frozen AudioVAE + BigVGAN-style decoder. Achieves the **best SIM on the 24-language MiniMax multilingual benchmark** (83.9 average). Continuation cloning (with transcript) and x-vector-only cloning (without). Apache 2.0.

### Fish S2 Pro

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Fish-S2-Pro_Colab.ipynb)

`fishaudio/s2-pro` (5B: 4B slow + 400M fast Dual-AR) — 80+ languages, **15,000+ free-form inline emotion/prosody tags** (`[whisper]`, `[low voice in small room]`, `[with strong accent]`, `[pitch up]` …), native multi-speaker `<|speaker:N|>` tokens, multi-turn dialogue. RL-aligned with GRPO. **L4/A100 only** (T4 OOMs). *Fish Audio Research License — non-commercial.*

### Kokoro-82M

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Kokoro-82M_Colab.ipynb)

`hexgrad/Kokoro-82M` (v1.0, 54 voices) & `hexgrad/Kokoro-82M-v1.1-zh` (v1.1-zh, **103 voices** including `af_maple`, `af_sol`, `bf_vale`, and 100 Mandarin Chinese voices `zf_001`–`zf_099` / `zm_009`–`zm_100` with `misaki[en,ja,zh]` and inline English `en_callable` fallback) — 82M parameters, Apache 2.0, **runs comfortably on CPU** (GPU optional), across 9 languages (English 🇺🇸🇬🇧, Spanish 🇪🇸, French 🇫🇷, Hindi 🇮🇳, Italian 🇮🇹, Brazilian Portuguese 🇧🇷, Japanese 🇯🇵, Mandarin 🇨🇳). Six tabs:

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

### Seed-VC — Zero-Shot Speech & 44.1 kHz Singing Voice Conversion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Seed-VC_Colab.ipynb)

[`Plachta/Seed-VC`](https://huggingface.co/Plachta/Seed-VC) ([GitHub](https://github.com/Plachtaa/seed-vc)) — **zero-shot speech and singing voice conversion** using a flow-matching Diffusion Transformer (`DiT`), **CAM++** speaker timbre embeddings, **Whisper** linguistic features, **RMVPE** pitch extraction, and **NVIDIA BigVGAN v2** vocoding (`4–8 GB VRAM`, runs fast on **T4 16 GB**).

- **Tab 1 — Speech Voice Conversion (22.05 kHz)**: Convert any spoken recording into a target speaker's voice from a 5–30s reference clip with zero training and sliding-window crossfade stitching for long recordings.
- **Tab 2 — Singing Voice Conversion (44.1 kHz + F0 Conditioning + Pitch Shift)**: Convert any song or vocal stem into a target person's singing voice while preserving melody, vibrato, and expression, with automatic median pitch matching (`auto_f0_adjust`) and semitone transposition (`-12` to `+12` semitones) — ideal for creating custom singing tracks to animate in `InfiniteTalk` or `LongCat-Avatar`!
- **Tab 3 — Batch Folder Voice Conversion**: Convert entire directories of audio files into a downloadable ZIP archive.

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
| **Chatterbox** | Chatterbox + Turbo + Multilingual + VC | 350M / 500M | 23 | MIT |
| **Breeze TTS 2** | Breeze-TTS-2 | 3B (Qwen3-1B + Mimi) | 2 (EN/ZH) | Code Apache 2.0 · Weights Non-Commercial |
| **Higgs-Audio** | Higgs-Audio v3 | 4B | 100+ | Research / Non-Commercial |
| **MisoTTS** | MisoTTS 8B | 8B | English | Other (see card) |
| **Supertonic-3** | Supertonic-3 | 99M (ONNX) | 31 | MIT code · OpenRAIL-M model |
| **Dia** | Dia2-2B / Dia2-1B / Dia-1.6B | 1B / 1.6B / 2B | English | Apache 2.0 |
| **IndicF5** | Indic-Speak / IndicF5 | 3.3B / 400M | 22 Indian + EN | Apache 2.0 / MIT |
| **MOSS-TTS v1.5** | MOSS-TTS v1.5 + VoiceGenerator + TTSD + SFX | 1.7B / 8B | 31 | Apache 2.0 |
| **dots.tts-soar** | dots.tts-soar / dots.tts-mf | 2B | 24+ | Apache 2.0 |
| **Fish S2 Pro** | Fish S2 Pro | 4B + 400M (Dual-AR) | 80+ | Fish Audio Research (non-commercial) |
| **VoxCPM2** | VoxCPM2 | 2B | 30 (+ 9 ZH dialects) | Apache 2.0 |
| **Kokoro-82M** | Kokoro-82M v1.0 & v1.1-zh | 82M | 9 (103 voices) | Apache 2.0 |
| **OpenVoice V2** | OpenVoiceV2 | — | 7 (cross-lingual VC) | MIT |
| **Seed-VC** | Seed-VC Speech & 44.1kHz Singing | ~200M DiT | Multilingual | GPL-3.0 |
| **ACE-Step** | ACE-Step-v1-3.5B | 3.5B | 19 (Music & Lyrics) | Apache 2.0 |
| **MMAudio** | MMAudio large_44k_v2 | 1.03B | Video/Text-to-Foley | MIT |
| **Mesh Optimizer** | trimesh + pyfqmr + pymeshlab + Open3D | — | — | MIT (all deps) |
| **Audio Post-Processor** | pydub + imageio-ffmpeg + soundfile + librosa + pyloudnorm + noisereduce + resemble-enhance + Kim_Vocal_2 / HDemucs + FlashSR | — | — | MIT/BSD/ISC (all deps) |

---

## Audio Post-Processor — post-process for podcast, music, and broadcast

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Skquark/AEI-Colab-Notebooks/blob/main/Audio_PostProcessor_Colab.ipynb)

The companion to every TTS / VC / voice-cloning pipeline: takes the **raw audio output** from Qwen3-TTS, VoxCPM2, Higgs, MisoTTS, MOSS, dots.tts, Fish, Kokoro, OpenVoice V2, or any other source, and turns it into a **clean, ready-to-publish** audio file: trimmed, denoised, loudness-normalized (LUFS), and in the format you need (WAV / MP3 / FLAC / OGG / Opus / M4A / AAC / AIFF).

**Stack**: [`pydub`](https://github.com/jiaaro/pydub) (MIT) for format conversion + silence detect · [`imageio-ffmpeg`](https://github.com/imageio/imageio-ffmpeg) (BSD-4) for the static ffmpeg binary · [`soundfile`](https://github.com/bastibe/python-soundfile) (BSD-3, libsndfile) for fast WAV/FLAC/OGG/AIFF round-trips · [`librosa`](https://github.com/librosa/librosa) (ISC) for DSP analysis · [`pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm) (MIT) for ITU-R BS.1770-4 LUFS / LKFS broadcast-standard normalization · [`noisereduce`](https://github.com/timsainb/noisereduce) (MIT) for spectral-gating noise reduction · [`pedalboard`](https://github.com/spotify/pedalboard) (GPL-3.0, by Spotify Audio Intelligence Lab) for studio-quality VST-style effects (compressor, limiter, reverb, delay, chorus, phaser, distortion, pitch shift, bitcrush, resample, MP3 simulation) · [`resemble-enhance`](https://github.com/resemble-ai/resemble-enhance) (MIT, by Resemble AI) for AI speech denoising + bandwidth extension (44.1 kHz PyTorch model, **optional GPU acceleration**, CPU supported). **Most tabs CPU-only; AI Enhance benefits from GPU but works on CPU.**

> **License notes**:
> - `pedalboard` is GPL-3.0. This notebook is MIT. The `pedalboard` library is used as a `pip install`-ed dependency, not as a linked/embedded library, so the GPL's copyleft provisions don't extend to the notebook code. You can use the notebook and the resulting audio commercially; the `pedalboard` source itself remains under GPL-3.0 (relevant only if you redistribute `pedalboard`'s source or a modified version of it).
> - `resemble-enhance` is MIT. The AI speech model (`Enhancer` + `Denoiser`) is also MIT. ~500 MB of model weights are downloaded on first use and cached locally.

### Ten pipeline stages (all optional, all tunable)

1. **Load** — auto-detect format from extension. 8 input formats.
2. **Vocal / Stem Separation (`isolate_vocals`)** — strip background music, room reverb, and accompaniment from speech or singing clips using `Kim_Vocal_2.onnx` (`audio-separator`), `torchaudio` Hybrid Demucs (`HDEMUCS_HIGH_MUSDB_PLUS`), or spectral voice isolation before feeding clips into TTS voice-cloning or `InfiniteTalk` / `LongCat-Avatar`.
3. **Auto-trim silence** — remove leading/trailing silence from TTS output using pydub silence detection (configurable threshold in dBFS and minimum silence length in ms). Applied *first* so all subsequent steps operate only on the speech region. Default off; the **TTS Polish** and **Vocal Clone Prep** presets enable it.
4. **High-pass filter** — cut rumble & mic handling noise (configurable cutoff, default 0=off).
5. **Denoise** — two backends: spectral gate (`noisereduce`, fast, no model) or Resemble AI denoiser (slow, better for speech). Tunable strength for the spectral gate.
6. **AI Enhance (Resemble Enhance)** — optional 44.1 kHz PyTorch model that does denoise + bandwidth extension + artifact reduction in one pass. 4 hyperparameters exposed.
7. **48 kHz Audio Super-Resolution (`FlashSR`)** — ultra-fast (200–400× real-time) neural bandwidth extension from 16/24 kHz TTS outputs to crisp 48 kHz broadcast audio via `YatharthS/FlashSR`.
8. **LUFS normalize** — broadcast-standard ITU-R BS.1770-4 loudness target (default -16 LUFS for podcast).
9. **Peak limit** — guard against clipping (default -1 dBFS).
10. **Export** — 8 output formats with configurable bitrate & sample rate.

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
python3 tools/validate.py    # OK: all 58 notebook(s) parse cleanly.
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
| LTX-Video 2.5 (ComfyUI + IC-LoRAs) | [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) · [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) · [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) | [arXiv 2501.00103](https://arxiv.org/abs/2501.00103) |
| MiniMax-H3 (Direct & ComfyUI) | [MiniMax-AI](https://github.com/MiniMax-AI) | — | — |
| Wan 2.2 (5B + 14B T2V/I2V + Lightning LoRA) | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) · [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) · [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) · [lightx2v/Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras) | [arXiv 2503.20314](https://arxiv.org/abs/2503.20314) |
| Wan 2.2 Animate (Animate-2 Distilled & v1) | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | [Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers) · [Wan-AI/Wan2.2-Animate-2-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Diffusers) · [Wan-AI/Wan2.2-Animate-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B-Diffusers) | [arXiv 2503.20314](https://arxiv.org/abs/2503.20314) |
| Wan 2.2 S2V (Local Single-GPU + Cloud) | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) | [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) · [lightx2v/Wan2.2-Distill-Loras](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras) | [arXiv 2508.18621](https://arxiv.org/abs/2508.18621) · [arXiv 2503.20314](https://arxiv.org/abs/2503.20314) |
| Wan-VACE (1.3B & 14B) | [ali-vilab/VACE](https://github.com/ali-vilab/VACE) | [Wan-AI/Wan2.1-VACE-1.3B-diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers) · [Wan-AI/Wan2.1-VACE-14B-diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers) | [arXiv 2503.07598](https://arxiv.org/abs/2503.07598) |
| LongCat-Video-Avatar 1.5 | [meituan-longcat/LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) | [meituan-longcat/LongCat-Video-Avatar](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar) · [spaces/victor/LongCat-Video-Avatar-1.5](https://huggingface.co/spaces/victor/LongCat-Video-Avatar-1.5) | [arXiv 2510.22200](https://arxiv.org/abs/2510.22200) |
| InfiniteTalk | [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) | [MeiGen-AI/InfiniteTalk](https://huggingface.co/MeiGen-AI/InfiniteTalk) · [Wan-AI/Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) | [arXiv 2508.14033](https://arxiv.org/abs/2508.14033) |
| LivePortrait | [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) | [KwaiVGI/LivePortrait](https://huggingface.co/KwaiVGI/LivePortrait) | [arXiv 2407.03168](https://arxiv.org/abs/2407.03168) |
| FramePack (F1 & Original) | [lllyasviel/FramePack](https://github.com/lllyasviel/FramePack) | [lllyasviel/FramePack_F1_I2V_HY_20250503](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503) · [lllyasviel/FramePackI2V_HY](https://huggingface.co/lllyasviel/FramePackI2V_HY) | [arXiv 2504.12626](https://arxiv.org/abs/2504.12626) |
| SeedVR2 (3B & 7B Video Upscaler) | [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) · [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) | [ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) · [ByteDance-Seed/SeedVR2-7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) · [numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI) | [arXiv 2506.05301](https://arxiv.org/abs/2506.05301) |

### 3D Models

| Notebook | Upstream repo | Hugging Face | Paper |
|----------|---------------|--------------|-------|
| Pixal3D (Single-Image & Multi-View) | [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) | [arXiv 2605.10922](https://arxiv.org/abs/2605.10922) |
| TRELLIS.2 (4B O-Voxel PBR) | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | — |
| Cube 3D + CubePart | [Roblox/cube](https://github.com/Roblox/cube) | [Roblox/cubepart](https://huggingface.co/Roblox/cubepart) | — |
| Hunyuan3D-2 (+ FlashVDM & Paint-Turbo) | [Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | [tencent/Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) · [tencent/Hunyuan3D-2mini](https://huggingface.co/tencent/Hunyuan3D-2mini) · [tencent/Hunyuan3D-2mv](https://huggingface.co/tencent/Hunyuan3D-2mv) | [arXiv 2501.12202](https://arxiv.org/abs/2501.12202) |
| Hunyuan3D-2.1 | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) | [arXiv 2506.15442](https://arxiv.org/abs/2506.15442) |
| Hunyuan3D 3.0 (API) | [Tencent Cloud](https://www.tencentcloud.com/document/product/1665/119114) (Cloud API v3) | — | — |
| **TripoSplat** | [VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat) | [arXiv 2605.16355](https://arxiv.org/abs/2605.16355) |
| NoPoSplat + YoNoSplat | [cvg/YoNoSplat](https://github.com/cvg/YoNoSplat) · [cvg/NoPoSplat](https://github.com/cvg/NoPoSplat) | [botaoye/YoNoSplat](https://huggingface.co/botaoye/YoNoSplat) · [botaoye/NoPoSplat](https://huggingface.co/botaoye/NoPoSplat) | [arXiv 2410.24207](https://arxiv.org/abs/2410.24207) |
| Wild Gaussian Splatting | [naver/mast3r](https://github.com/naver/mast3r) · [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | — | [arXiv 2406.09756](https://arxiv.org/abs/2406.09756) |
| MapAnything | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | [facebook/map-anything-apache](https://huggingface.co/facebook/map-anything-apache) · [facebook/map-anything](https://huggingface.co/facebook/map-anything) | — |
| Depth Anything 3 (DA3-Nested & Metric) | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | [depth-anything/DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) · [depth-anything/DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE) | [arXiv 2511.10647](https://arxiv.org/abs/2511.10647) |
| Pi3X | [yyfz/Pi3](https://github.com/yyfz/Pi3) | [yyfz233/Pi3X](https://huggingface.co/yyfz233/Pi3X) | — |
| HY-World 2.0 (WorldMirror 2.0) | [Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) | [tencent/HY-World-2.0](https://huggingface.co/tencent/HY-World-2.0) | — |
| GaussianGPT | [nicolasvonluetzow/GaussianGPT](https://github.com/nicolasvonluetzow/GaussianGPT) | — | — |
| InfiniSplat (v2 Multi-View/360° & v1) | [zju3dv/InfiniSplat](https://github.com/zju3dv/InfiniSplat) | [Edisoneh/InfiniSplat-v2](https://huggingface.co/Edisoneh/InfiniSplat-v2) · [PLUS-WAVE/InfiniSplat](https://huggingface.co/PLUS-WAVE/InfiniSplat) | — |
| LHM / LHM++ (Animatable 3DGS Human) | [aigc3d/LHM-plusplus](https://github.com/aigc3d/LHM-plusplus) · [aigc3d/LHM](https://github.com/aigc3d/LHM) | [3DAIGC/LHMPP-700M](https://huggingface.co/3DAIGC/LHMPP-700M) · [3DAIGC/LHM-1B](https://huggingface.co/3DAIGC/LHM-1B) | [arXiv 2503.10625](https://arxiv.org/abs/2503.10625) |
| TextureMapPrep | [prs-eth/Marigold](https://github.com/prs-eth/Marigold) · [EnVision-Research/Lotus](https://github.com/EnVision-Research/Lotus) | [prs-eth/marigold-depth-hr-v1-1](https://huggingface.co/prs-eth/marigold-depth-hr-v1-1) · [jingheya/Lotus-2](https://huggingface.co/jingheya/Lotus-2) · [huawei-bayerlab/marigold-v2-0](https://huggingface.co/huawei-bayerlab/marigold-v2-0) | — |
| SkinTokens (TokenRig) | [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) | [VAST-AI/SkinTokens](https://huggingface.co/VAST-AI/SkinTokens) | — |
| SuGaR | [Anttwo/SuGaR](https://github.com/Anttwo/SuGaR) | — | [arXiv 2311.12775](https://arxiv.org/abs/2311.12775) |
| GauStudio | [GAP-LAB-CUHK-SZ/gaustudio](https://github.com/GAP-LAB-CUHK-SZ/gaustudio) | — | [arXiv 2403.19632](https://arxiv.org/abs/2403.19632) |
| SplatTransform (v3.10+) | [playcanvas/splat-transform](https://github.com/playcanvas/splat-transform) | — | — |
| Mesh Optimizer | [mikedh/trimesh](https://github.com/mikedh/trimesh) · [Kramer84/pyfqmr](https://github.com/Kramer84/pyfqmr-Fast-quadric-Mesh-Reduction) · [cnr-isti-vclab/PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) · [isl-org/Open3D](https://github.com/isl-org/Open3D) | — | — |
| Audio Post-Processor | [jiaaro/pydub](https://github.com/jiaaro/pydub) · [imageio/imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) · [bastibe/python-soundfile](https://github.com/bastibe/python-soundfile) · [librosa/librosa](https://github.com/librosa/librosa) · [csteinmetz1/pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) · [timsainb/noisereduce](https://github.com/timsainb/noisereduce) · [spotify/pedalboard](https://github.com/spotify/pedalboard) · [resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) · [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | [YatharthS/FlashSR](https://huggingface.co/YatharthS/FlashSR) | — |

### TTS Models

| Notebook | Upstream repo | Hugging Face | Paper |
|----------|---------------|--------------|-------|
| Qwen3-TTS | (Alibaba) | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | — |
| Ming 3.0 TTS (`0.5B` Dense & `tiny` MoE) | [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) | [inclusionAI/Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B) · [inclusionAI/Ming-3.0-tts-tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny) | — |
| Chatterbox (TTS, Turbo, Multilingual, VC) | [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) · [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) | — |
| Breeze TTS 2 | [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts) | [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) | — |
| Higgs-Audio | [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio) | [bosonai/higgs-audio-v2-tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) | — |
| MisoTTS | (Miso Labs) | [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS) | — |
| Supertonic-3 | [Supertone/supertonic](https://github.com/Supertone/supertonic) | [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) | — |
| Dia (Dia2-2B / Dia2-1B / Dia-1.6B) | [nari-labs/dia](https://github.com/nari-labs/dia) | [nari-labs/Dia2-2B](https://huggingface.co/nari-labs/Dia2-2B) · [nari-labs/Dia2-1B](https://huggingface.co/nari-labs/Dia2-1B) · [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) | — |
| IndicF5 + Indic-Speak | [AI4Bharat/IndicF5](https://github.com/AI4Bharat/IndicF5) | [bodhan-ai/indic-speak](https://huggingface.co/bodhan-ai/indic-speak) · [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) | — |
| MOSS-TTS Studio (v1.5 + VoiceGenerator + TTSD + SFX) | [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) | [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) · [OpenMOSS-Team/MOSS-VoiceGenerator](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator) · [OpenMOSS-Team/MOSS-TTSD-v1.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0) · [OpenMOSS-Team/MOSS-SoundEffect](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect) | — |
| dots.tts-soar + dots.tts-mf | [rednote-hilab/dots.tts](https://github.com/rednote-hilab/dots.tts) | [rednote-hilab/dots.tts-soar](https://huggingface.co/rednote-hilab/dots.tts-soar) · [rednote-hilab/dots.tts-mf](https://huggingface.co/rednote-hilab/dots.tts-mf) | — |
| Fish S2 Pro | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) | [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | — |
| VoxCPM2 | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | [arXiv 2509.24650](https://arxiv.org/abs/2509.24650) |
| Kokoro-82M (v1.0 & v1.1-zh) | [hexgrad/kokoro](https://github.com/hexgrad/kokoro) | [hexgrad/Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh) · [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | [arXiv 2306.07691](https://arxiv.org/abs/2306.07691) (StyleTTS 2) · [arXiv 2203.02395](https://arxiv.org/abs/2203.02395) (iSTFTNet) |

### Music & Foley Models

| Notebook | Upstream repo | Hugging Face | Paper |
|----------|---------------|--------------|-------|
| MiniMax-Music3 | [MiniMax-AI](https://github.com/MiniMax-AI) | — | — |
| ACE-Step (3.5B Music Studio) | [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step) | [ACE-Step/ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) | [arXiv 2506.00045](https://arxiv.org/abs/2506.00045) |
| MMAudio (Synchronized Foley) | [hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio) | [hkchengrex/MMAudio](https://huggingface.co/hkchengrex/MMAudio) | [arXiv 2412.15322](https://arxiv.org/abs/2412.15322) |

### Voice Conversion Models

| Notebook | Upstream repo | Hugging Face | Paper |
|----------|---------------|--------------|-------|
| OpenVoice V2 | [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) | [myshell-ai/OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2) | [arXiv 2312.01479](https://arxiv.org/abs/2312.01479) |
| Seed-VC (Speech & 44.1 kHz Singing) | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC) | [arXiv 2411.09943](https://arxiv.org/abs/2411.09943) |

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
