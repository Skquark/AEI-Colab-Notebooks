# AEI Colab Notebooks

Free, self-contained Google Colab notebooks for AI-powered 3D generation, image editing, and video creation. No sign-ups. No tokens. Just works.

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

## Coming Soon

More notebooks will be added over time. Planned:

- Image editing and generation
- Video generation (text-to-video, image-to-video)
- 3D asset texturing and material editing
- Model quantization and GGUF variants

---

## Why This Exists

Most Colab AI notebooks require paid tokens, subscription services, or lengthy manual compilation of CUDA dependencies. This project precompiles everything into ready-to-install wheels and hosts them on GitHub — so you can open a notebook, run all cells, and get results immediately, on any supported GPU, completely free.

---

## License

Notebooks in this repository are provided for educational and personal use. Individual notebooks use third-party models under their respective licenses — check each model's page for commercial use terms. The Pixal3D pipeline is MIT licensed.
