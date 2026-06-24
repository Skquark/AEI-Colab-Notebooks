# Changelog

All notable changes to this repository are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Polish pass: MapAnything + Pi3X (v2)

Full review of both new 3DGS-from-images notebooks against the upstream APIs
and the model cards. Adds the missing parameters, fixes the wrong defaults,
and adds info= tooltips to all form controls.

**MapAnything v2 (was 29 tips → now 64):**
- **All upstream `model.infer()` params exposed:**
  `ignore_depth_scale_inputs` and `ignore_pose_scale_inputs` were missing
  (now added); `stride` (1-32, subsample every Nth image) was missing
  (critical for long videos). `apply_confidence_mask` checkbox exposed.
  `min_init_opacity` slider (0-1) wired into `write_colmap` to filter
  `non_ambiguous_mask` → init_opacity for gsplat.
- **`save_glb()` function added** mirroring upstream
  `demo_inference_on_colmap_outputs.py --save_glb`. Trimesh PointCloud
  export with colors from `img_no_norm`. Subsampled to 200k points to keep
  file size <50 MB. Toggle in UI + form cells. Mirrored to Drive when enabled.
- **Advanced "Input Ignore Flags" sub-accordion** in the Gradio UI for
  the 5 `ignore_*_inputs` flags. Hidden by default, expanded for power users.
- **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** set before any torch
  import (Meta's recommended memory optimization for large view counts).
- **4 viewers in `_splat_viewer_html`** (was 2): supersplat.dev,
  playcanvas.com/viewer, gsplat.tech, plus the GLB link when enabled.
- **Separate `gr.File` for the .glb** so users can download it independently.
- **Gradio height: 900 → 1100** to fit the extra accordion without scrolling.
- **info= tooltips on all 35 new form controls** in STEPs 4/6/7.

**Pi3X v2 (was 26 tips → now 41):**
- **Pi3X `metric` scaling factor applied** to the point cloud in
  `write_colmap` (was being silently ignored). Multiplies points by
  `results['metric'][0]` to convert from scale-invariant to true metric.
  Without this, the entire scene was in arbitrary units.
- **`interval` slider (1-60)** wired into `load_images_as_tensor(..., interval=N)`.
  Upstream default for video is 10 (not 1). For 1 minute of 30 fps video at
  1 fps = 1800 frames, `interval=10` = 180 frames (manageable).
- **`conf_threshold` slider (0-0.95)** with sigmoid on raw confidence logits,
  wired into `write_colmap` to filter low-confidence points. `conf_mask`
  added to results dict.
- **`max_points` slider (100k-5M)** subsamples `points3D.ply` to keep gsplat
  happy. Long videos can produce 10M+ points; gsplat prefers ≤2M.
- **`conditions_path` Textbox** for Pi3X multimodal prior injection
  (`.npz` with pose/intrinsics/depth). Loaded with `np.load()` and passed
  as `model(imgs, conditions=...)`. Only used when `use_multimodal=True`.
- **`save_glb()` function added** mirroring upstream `example_mm.py --save_path`.
  Single-image color (PIL resize) for the point cloud, subsampled to 200k pts.
- **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** set before any torch
  import (long-video memory optimization).
- **Edge mask applied to point cloud** in `write_colmap` (was being computed
  in `infer()` but never actually used for the COLMAP output).
- **4 viewers in `_splat_viewer_html`** (was 2): supersplat.dev,
  playcanvas.com/viewer, gsplat.tech, plus the GLB link.
- **Separate `gr.File` for the .glb** so users can download it independently.
- **Gradio height: 900 → 1100** to fit the extra controls.
- **info= tooltips on all 15 new form controls** in STEPs 4/6/7.

**Total improvements:**
- 4 new Gradio controls + 1 Textbox in MapAnything UI
- 5 new Gradio controls + 1 Textbox in Pi3X UI
- 35 new info= tooltips in MapAnything (29 → 64)
- 15 new info= tooltips in Pi3X (26 → 41)
- 2 new utility functions (`save_glb`) in both notebooks
- 2 new upstream param groups (`ignore_*_scale_inputs`, `conditions.npz`)
- 0 new QA findings (only the pre-existing MOSS-TTS HF cache warning remains)

### Polish pass: TextureMapPrep v2 (post-review fixes)

Full review of the new `TextureMapPrep_Colab.ipynb` against the upstream
`diffusers` Marigold pipeline source. Found and fixed several issues:

- **CRITICAL FIX: Marigold IID `prediction` is `np.ndarray`, NOT a dict.**
  The v1 code did `iid['albedo']` / `iid['roughness']` / `iid['metallicity']` which
  would have raised `KeyError` at runtime. The v1 docs confused `prediction` with
  the output of `pipe.image_processor.visualize_intrinsics(...)` (which IS a
  list of dicts, but visualized). Fixed by:
  1. Reading `target_names` from `pipe.target_properties` (the model knows its
     own target order: `[albedo, roughness, metallicity]` for appearance,
     `[albedo, shading, residual]` for lighting).
  2. Slicing `out.prediction` of shape `(N*T, H, W, 3)` by index, returning
     a proper `dict[str, HxWx3 float32]`.
- **Added `ensembling_kwargs` (`reduction` radio button).** 'median' (default)
  or 'mean' for depth/IID ensembles. The v1 code had `reduction` only on
  the depth path. Now all 3 Marigold paths (depth/normals/IID) expose it.
- **Added `output_uncertainty` toggle** for depth (with the upstream
  requirement that `ensemble_size >= 3`). Saves `uncertainty_depth.png`
  showing per-pixel predictive uncertainty.
- **Added `resample_method` (bilinear/bicubic/nearest/nearest-exact/area)**
  radio for all 3 Marigold paths. Default is 'bicubic' for best quality.
- **Added `use_input_albedo_as_is` toggle.** When OFF (default), saves
  Marigold IID's intrinsic albedo (linear -> sRGB) as `albedo.png` (removes
  baked lighting, gives a cleaner PBR albedo). When ON, saves the input
  image verbatim. The intrinsic albedo is ALWAYS saved as
  `albedo_intrinsic.png` for comparison.
- **Added `seed` parameter** for reproducibility across all 3 Marigold
  pipelines (LOTUS already had it).
- **Added `seed` to the metadata.json** so each run is reproducible.
- **Normal-to-uint8 conversion now uses `np.clip`** to avoid values outside
  [0, 255] from float drift.
- **Height-from-normal uses FFT-based Poisson** (skimage was imported but
  unused; replaced with the same approach used in MapAnything/Pi3X).
- **Reduced vertical density of STEPS for IID appearance vs lighting** so
  the IID slider block is cleaner.

- **9-cell standard pattern** (matches `tools/validate.py`).
- **94 info= tooltips** on all form controls (was 57 in v1).
- **15 try/except** blocks.
- All Y's (Drive cache, concurrency_limit, clear_output, demo.load welcome).
- **FileLink in Step 6** (output folder + meta.json).

### TextureMapPrep — New notebook: Seamless PBR Maps for Game Assets
New notebook at `TextureMapPrep_Colab.ipynb`. Unified batch pipeline for
converting seamless albedo PNGs into the full **6-map PBR set** (albedo,
normal, depth, height, roughness, metallic) plus optional Real-ESRGAN 2x/4x
upscale. Inspired by the
[NightRaven109/PBRFusion](https://huggingface.co/spaces/NightRaven109/PBRFusion4AlbedoKlein)
HF spaces (LOTUS depth + FLUX Albedo LoRA + CCSR upscale) but unified through
the **Marigold** diffusers API instead of three separate frameworks.

**Framework choice (Marigold-first):**
The PBRFusion spaces use 3 separate models assembled ad-hoc. Marigold (PRS-ETH,
CVPR 2024 Oral) gives us depth + surface normals + intrinsic image
decomposition (albedo + roughness + metallic) through **one unified diffusers
API**. Code is Apache-2.0, model weights are RAIL++-M (commercial w/ safety
conditions). Already in the diffusers core. This makes the notebook ~3 GB
total + much simpler glue code. LOTUS (Apache-2.0) and FLUX 2-klein (Apache-2.0)
are kept as alternative / opt-in paths.

**Model inventory:**
* **Marigold v1.1** (`prs-eth/marigold-{depth,normals,iid-appearance,iid-lighting}-v1-1`)
  - Apache-2.0 code, RAIL++-M weights. The main path.
* **LOTUS v1** (`jingheya/lotus-{depth-g-v2-1-disparity,normal-g-v1-1}`)
  - Apache-2.0 (code+weights). Alternative depth+normals; same architecture
  the PBRFusion4DepthDemo-InstNorm space uses. Single-step diffusion.
* **FLUX 2-klein-4B** + Albedo LoRA - Apache-2.0. Optional clean-albedo path
  (off by default; needs ~8 GB VRAM). Same Albedo LoRA as PBRFusion4AlbedoKlein.
* **OpenCV** (Apache-2.0) - Sobel normal-from-albedo, cross-product
  normal-from-depth, FFT-based Poisson height-from-normal, tile-wrap for seamless
  handling, all the basic PBR image ops.
* **Real-ESRGAN** (BSD-3, 35.9k stars, on PyPI) - 2x / 4x texture upscale
  with BORDER_WRAP for seamless preservation.
* **scikit-image** (BSD-3) - Poisson reconstruction utilities (inpaint_biharmonic fallback).

- **9-cell standard pattern** (matches `tools/validate.py`): `view-in-github`,
  `header`, `step1-install`, `step2-cache`, `step3-core`, `step4-ui`,
  `step5-keepalive`, `step6-quicktest`, `step7-batch`.
- **STEP 1 - install + clone.** torch 2.5.1+cu121, diffusers>=0.30 (Marigold
  pipelines landed in 0.28), `realesrgan` from PyPI, git-clone
  `EnVision-Research/Lotus`. Pre-downloads all 4 Marigold checkpoints
  + 4 LOTUS checkpoints to Drive cache.
- **STEP 2 - imports + lazy model cache.** Imports `MarigoldDepthPipeline`,
  `MarigoldNormalsPipeline`, `MarigoldIntrinsicsPipeline` from diffusers,
  `LotusGPipeline`/`LotusDPipeline` from the cloned repo, cv2, PIL,
  `inpaint_biharmonic`, `RealESRGANer`, RRDBNet. Defines:
  - `get_marigold(task)` / `get_lotus(task, mode)` / `get_flux_albedo()` /
    `get_realesrgan(scale)` - lazy model cache, local-snapshot aware.
  - `infer_depth_marigold(img, ...)` / `infer_normals_marigold(img, ...)` /
    `infer_iid_marigold(img, model, ...)` - Marigold inference wrappers
    with seamless wrap + crop.
  - `infer_depth_lotus(img, ...)` / `infer_normals_lotus(img, ...)` - LOTUS
    inference wrappers.
  - `infer_albedo_flux(img, ...)` - FLUX 2-klein + Albedo LoRA + Reinhard
    color match.
  - `seamless_wrap(img, border_px)` - cv2 `BORDER_WRAP` wrap-then-crop.
  - `normal_from_albedo(albedo, strength)` - OpenCV Sobel gradient.
  - `normal_from_depth(depth, fx, fy, cx, cy, strength)` - OpenCV 3D
    point-cloud cross product.
  - `height_from_normal(normal)` - FFT-based Poisson reconstruction.
  - `displacement_from_height(height, scale)` - scale + offset.
  - `upscale_image(img, scale, seamless_border)` - Real-ESRGAN with wrap.
- **STEP 3 - core helpers.** `run_full_pbr(image_path, output_dir, **kwargs)`
  writes a 6-PBR-map set per texture. `run_batch(input_dir, output_dir, **kwargs)`
  processes a folder (skips already-done via `meta.json`). Plus
  `normal_to_opengl()` / `normal_to_directx()` and `make_seamless_check()`.
- **STEP 4 - Gradio UI.** All params exposed: depth method (marigold/lotus/skip),
  steps, ensemble, resolution; normal method (marigold/lotus/opencv-sobel/opencv-depth),
  steps, ensemble, resolution, strength, convention (opengl/directx);
  IID model (appearance/lighting), steps, ensemble, resolution;
  FLUX albedo (toggle + width/height/steps/LoRA weight/color match);
  height from normal (toggle + displacement scale); upscale factor (1/2/4);
  seamless border (0-256). 4 viewers: albedo, normal, depth, output folder.
- **STEP 5 - keep-alive** (12 h, with model-state summary).
- **STEP 6 - Colab single-texture picker** (20 form params + 1 button,
  FileLink for meta.json + per-texture folder).
- **STEP 7 - Colab batch** (17 form params + 1 button, same workflow).
- **57 info= tooltips** on all form controls. 14 try/except. Drive cache
  set BEFORE any `huggingface_hub` import. `clear_output()` before
  `demo.launch`. `default_concurrency_limit=2`. `demo.load` welcome.
- **License summary in every cell's docstring** (Apache-2.0 + RAIL++-M +
  BSD-3 + OpenCV + FLUX-2.0). All weights are pre-cached to Drive.

### Pi3X — New notebook: Video-Native 3DGS, Permutation-Equivariant (ICLR 2026)
New notebook at `Pi3X_Colab.ipynb`. **[π³ / Pi3X](https://yyfz.github.io/pi3/)**
is a feed-forward neural network for visual geometry reconstruction from
Shanghai AI Lab + ZJU (ICLR 2026). Unlike every other 3DGS-from-images tool in
this suite, **π³ is permutation-equivariant** — there is no fixed reference
view, input order is irrelevant, the model handles long videos without drift,
and reconstruction quality stays stable on sequences of 10, 50, 200+ frames.

**License posture:** code is BSD-3-Clause (commercial-OK), upstream
[yyfz/Pi3](https://github.com/yyfz/Pi3). **Weights are CC BY-NC-4.0** (strictly
non-commercial, per the official model card). Notebook shipped with prominent
warning. Your output `.ply` files are yours.

- **9-cell standard pattern** (matches `tools/validate.py`): `view-in-github`,
  `header`, `step1-install`, `step2-cache`, `step3-core`, `step4-ui`,
  `step5-keepalive`, `step6-quicktest`, `step7-batch`.
- **STEP 1 — install + repo + checkpoint.** torch 2.5.1+cu121 + torchvision
  0.20.1 + numpy 1.26.4 (pinned to match upstream), `git clone --depth=1` the
  BSD-3 upstream, install inference deps, download `yyfz233/Pi3X` model
  (≈ 5 GB safetensors) to Drive cache. No DINOv2 download — Pi3X uses its own
  ViT encoder.
- **STEP 2 — imports + lazy model cache.** `Pi3X.from_pretrained(...)` cached
  per `(use_multimodal,)` so subsequent Gradio clicks are instant. Defines
  `infer()` (frames → results dict with `points`, `local_points`, `rays`,
  `conf`, `camera_poses`, `metric`), `write_colmap()` (Pi3X → COLMAP-format
  using `recover_intrinsic_from_rays_d` from upstream `pi3.utils.geometry`),
  `train_gsplat()` (gsplat 1.3.0 simple_trainer, ~1-3 min on T4),
  `_mirror_to_drive()` (same-path-skip), and `run_full_pipeline()`.
- **STEP 3 — core helpers.** `run_single_scene()` (auto-detects video vs
  folder), `run_batch()` (skips already-done scenes).
- **STEP 4 — Gradio UI.** Multi-file upload (any 1+ images, any order) OR video
  upload. Exposed params: `use_multimodal` (default OFF, frees ~2 GB GPU),
  `use_amp` (bf16/fp16), `edge_mask` (depth-normal edge filter), `video_fps`
  (0.5-5), `max_frames` (4-200, Pi3X is permutation-equivariant so all frames
  are weighted equally), `do_drive_save` toggle. Generates `.ply` + preview
  pane with file links to supersplat.dev / playcanvas.com/viewer.
- **STEP 5 — keep-alive** (12 h, with repeated license reminder).
- **STEP 6 — Colab single-scene picker** (8 form params + 1 button).
- **STEP 7 — Colab batch** (8 form params + 1 button).
- **26 info= tooltips** on all form controls. 13 try/except blocks. Drive
  cache set BEFORE any `huggingface_hub` import. `clear_output()` before
  `demo.launch`. `default_concurrency_limit=2`. `demo.load` welcome.

### MapAnything — New notebook: Universal 3DGS-from-Images (Meta, Apache 2.0)
New notebook at `MapAnything_Colab.ipynb`. **[MapAnything](https://map-anything.github.io/)**
is Meta's universal 3D reconstruction framework (3DV 2026). One feed-forward
transformer handles **12+ 3D reconstruction tasks** in a single forward pass:
image-only SfM, image+pose, image+intrinsics+depth, registration, depth
completion, monocular metric depth, and more. Output is a metric 3D point
cloud + camera poses which we feed into `gsplat` to produce a real 3DGS scene.

**The standout property is universality**: give it 2 images, 20 images, an
image + known depth from a depth sensor, an image + known camera poses from
COLMAP, or a video without any calibration at all — and it always produces a
metric 3D reconstruction.

**License posture:** **fully commercial-OK**. Code is Apache 2.0 (upstream
[facebookresearch/map-anything](https://github.com/facebookresearch/map-anything)).
Weights are from [🤗&nbsp;facebook/map-anything-apache](https://huggingface.co/facebook/map-anything-apache)
— the **Apache 2.0** variant. (Meta also publishes a CC-BY-NC variant at
`facebook/map-anything` which we deliberately do not use.) Output `.ply`
files are yours.

- **9-cell standard pattern** (matches `tools/validate.py`): `view-in-github`,
  `header`, `step1-install`, `step2-cache`, `step3-core`, `step4-ui`,
  `step5-keepalive`, `step6-quicktest`, `step7-batch`.
- **STEP 1 — install + repo + checkpoint.** torch 2.4.1+cu121 + torchvision
  0.19.1 (pinned — matches upstream `pyproject.toml`), `git clone --depth=1`
  the Apache 2.0 upstream, install `uniception==0.1.7` + lightweight inference
  deps, download `map-anything-apache/model.safetensors` (4.47 GB) to Drive
  cache. DINOv2-giant (~1.1 GB) is fetched on first `from_pretrained()`.
- **STEP 2 — imports + lazy model cache.** `MapAnything.from_pretrained(...)`
  cached per `(amp_dtype, minibatch_size)`. Defines `infer()` (uses upstream
  `mapanything.utils.image.load_images`), `write_colmap()` (MapAnything
  outputs → COLMAP-format with PINHOLE camera model, 1-indexed cameras.txt +
  images.txt + points3D.ply), `train_gsplat()` (gsplat 1.3.0 simple_trainer),
  `_mirror_to_drive()` (same-path-skip), and `run_full_pipeline()`.
- **STEP 3 — core helpers.** `run_single_scene()` (auto-detects video vs
  folder), `run_batch()` (skips already-done scenes).
- **STEP 4 — Gradio UI.** Multi-file upload (any 1+ images) OR video upload.
  Exposed params: `gsplat_iterations` (1k-30k), `apply_mask`, `mask_edges`,
  `use_multiview_confidence` (cross-view depth consistency), `confidence_percentile`
  (0-50), `amp_dtype` (bf16/fp16/fp32), `minibatch_size` (1-8), `video_fps`
  (0.5-5), `max_frames` (4-64), `do_drive_save` toggle. Generates `.ply` +
  preview pane with file links to supersplat.dev / playcanvas.com/viewer.
- **STEP 5 — keep-alive** (12 h, with repeated license reminder).
- **STEP 6 — Colab single-scene picker** (8 form params + 1 button).
- **STEP 7 — Colab batch** (8 form params + 1 button).
- **29 info= tooltips** on all form controls. 9 try/except blocks. Drive
  cache set BEFORE any `huggingface_hub` import. `clear_output()` before
  `demo.launch`. `default_concurrency_limit=2`. `demo.load` welcome.

### Polish pass: NoPoSplat + Wild Gaussian Splatting (v2)

Full review + polish of both new notebooks based on the upstream source code
and the model cards. Adds the missing parameters, fixes the wrong defaults,
removes the invalid params, and adds info= tooltips to all form controls.

**NoPoSplat v2 (was 3 tips → now 27):**
- **OPACITY_MAPPING params + global_step slider:** `cfg.opacity_mapping`
  has `initial`, `final`, `warm_up` — the `warm_up` controls how the opacity
  mapping transitions from initial to final as the encoder's `global_step`
  increases. Now exposed as `global_step` and `opacity_warm_up` sliders in
  the UI and the form cells.
- **Encoder cache key fixed:** previously `(ckpt, size, max_views)` — now
  `(ckpt, size, max_views, global_step, opacity_initial, opacity_final,
  opacity_warm_up)` so different opacity settings return a fresh encoder.
- **GaussianAdapterCfg sliders:** `sh_degree` (0-4), `gaussian_scale_min`
  (0.001-1.0), `gaussian_scale_max` (1.0-50.0) — the upstream exposes these
  but the v1 notebook hardcoded them. Now the user can tune them in
  an "Advanced encoder config" accordion.
- **Separate 2-view vs 3-view checkpoints in dropdown** with `2V — ` and
  `3V — ` prefixes. The `_do_run` handler validates that the right number
  of images was uploaded (2 vs 3) and returns a clear error message if not.
- **Removed base64 from `_splat_viewer_html`:** previously encoded the
  entire `.ply` (200-500 MB) as a base64 string in the HTML — wasteful and
  slow. Now just a small panel with download links to supersplat.dev and
  playcanvas.com/viewer.
- **info= tooltips on all STEP 6/7 form params:** 19 new tooltips
  describing image size, global_step, opacity_warm_up, sh_degree, scale
  range, and drive-mirror behavior.

**Wild Gaussian Splatting v2 (was 4 tips → now 47):**
- **Fixed wrong MASt3R defaults:** v1 used `niter1=500, niter2=200` (from the
  2024 dust3r notebook) but the actual `sparse_scene_optimizer` defaults
  are `niter1=300, niter2=300`. The lr values were also wrong (`lr2=0.014` →
  `0.01`).
- **Removed invalid params:** `min_conf_thr=1.5` and `mask_images=True`
  are not in MASt3R's `sparse_global_alignment` signature. They were
  silently dropped before, now they're not passed at all.
- **Added MASt3R params to the UI:** `subsample` (8), `kinematic_mode`
  (hclust-ward / mst / hclust-complete / hclust-average / hclust-single),
  `opt_pp` (refine principal point), `opt_depth` (refine per-pixel depth),
  `loss_dust3r_w` (0.01).
- **Added 3DGS training params to the UI:** `sh_degree` (0-3 dropdown),
  `antialiasing` (Mip-NeRF 360 filter), `white_background` (for object
  captures on white backdrops), `opacity_reset_interval` (3000 default).
- **Fixed orbit render:** v1 used a hand-rolled camera path with
  OpenGL/OpenCV coordinate confusion. v2 uses the upstream's
  `generate_ellipse_path_from_camera_infos` (copied inline from the
  nerlfield/gaussian-splatting fork) which generates a proper ellipse
  through the training-view centroid using the COLMAP poses.
- **num_orbit_frames slider** (30-240, default 120) for the orbit MP4.
- **info= tooltips on all STEP 6/7 form params:** 30 new tooltips.

**Total improvements:**
- 8 new Gradio sliders/checkboxes/dropdowns in NoPoSplat UI
- 11 new Gradio sliders/checkboxes/dropdowns in WildGS UI
- 49 new info= tooltips across both notebooks (3+4 → 27+47)
- 0 new QA findings (only the pre-existing MOSS-TTS HF cache warning remains)

### Wild Gaussian Splatting — New notebook: Video / Image Folder → 3DGS (CC BY-NC-SA + INRIA)
New notebook at `WildGaussianSplatting_Colab.ipynb`. The closest open-source
equivalent to a Luma Labs Capture / Genie workflow: upload a casual video or a
folder of overlapping photos, get a polished 3DGS scene back in 5-15 minutes on a T4.

**License posture:** the wrapper is Apache-2.0 (per upstream README). The 3DGS
training is the original [INRIA implementation](https://github.com/graphdeco-inria/gaussian-splatting)
licensed under the **INRIA non-commercial research license** (commercial use requires
written consent from Inria). The MASt3R backbone (used for pose + point cloud
estimation) is **CC BY-NC-SA 4.0** by Naver. Same situation as our SuGaR/GauStudio
notebooks — ship with prominent warning. Your output `.ply` / `.mp4` is yours.

- **9-cell standard pattern** (matches `tools/validate.py`).
- **STEP 1 — install + repo + checkpoints + 3DGS compile.** `apt-get install
  ffmpeg` first, then torch 2.4.1+cu121, `git clone --recurse-submodules` the
  upstream (3 submodules: dust3r, mast3r, gaussian-splatting), install top-level
  + DUSt3R/MASt3R requirements (skips the broken `sql` line), build the RoPE
  CUDA kernel (~1-2 min), `pip install -e` the two 3DGS submodules
  (`diff-gaussian-rasterization` + `simple-knn`, ~3-5 min compile), download
  MASt3R (2.75 GB) + DUSt3R (2.29 GB) checkpoints to Drive cache.
- **STEP 2 — imports + lazy MASt3R model + shared pipeline.** Uses
  `mast3r.model.load_model` + `mast3r.cloud_opt.sparse_ga.sparse_global_alignment`.
  Defines `run_mast3r()` (image folder → COLMAP-format dataset),
  `run_3dgs()` (subprocess wrapper around the upstream `train.py` with all the
  lr/densify args), and `run_full_pipeline()` (chains both with a single call).
  Custom COLMAP-format writer (cameras.txt, images.txt, points3D.ply) replaces
  the real COLMAP binary which Colab doesn't have.
- **STEP 3 — core helpers.** `extract_video_frames()` (ffmpeg), `run_single_scene()`,
  `run_batch()` (auto-detects videos vs subdirs).
- **STEP 4 — Gradio UI.** `gr.Video` upload + multi-file image upload,
  iterations slider (3k-30k), image size dropdown (256/512), video fps slider,
  Drive-mirror toggle. Generates `.ply` + orbit `.mp4` + preview pane.
- **STEP 5 — keep-alive** (12 h, with repeated license reminder).
- **STEP 6 — Colab single-scene picker** (5 form params + 1 button).
- **STEP 7 — Colab batch** (5 form params + 1 button).
- **All 4 sliders/checkboxes have `info=` tooltips** (UI polish). 10 try/except
  blocks. Drive cache set BEFORE any downloads.

### NoPoSplat — New notebook: 2-3 Photos → 3DGS, pose-free (MIT, ICLR 2025)
New notebook at `NoPoSplat_Colab.ipynb`. The closest open-source equivalent
to a Luma-style "upload photos, get 3DGS" workflow. Feed-forward model:
2-3 unposed, uncalibrated images in → complete 3DGS scene out in ~10s on T4.

- **9-cell standard pattern** (matches `tools/validate.py`): `view-in-github`,
  `header`, `step1-install`, `step2-cache`, `step3-core`, `step4-ui`,
  `step5-keepalive`, `step6-quicktest`, `step7-batch`.
- **STEP 1 — install + repo + checkpoint.** torch 2.4.1+cu121 + torchvision
  0.19.1 (pinned — newer breaks `torchvision.transforms.functional` imports
  the upstream uses), `git clone --depth=1` the MIT upstream, install
  requirements (skips the broken `sql` line), optional RoPE CUDA kernel build
  (~1-2 min), download `mixRe10kDl3dv_512x512.ckpt` (2.3 GB) to Drive cache.
- **STEP 2 — imports + lazy encoder cache.** Uses `get_encoder(cfg)` and
  `checkpoint_filter_fn` from the upstream. Encoder cached per (ckpt, size,
  max_views) so subsequent Gradio clicks are instant. Defines
  `_load_image()` (BICUBIC resize, pad to 32-multiple) + `_build_context()`
  that builds the `BatchedViews` dict the encoder expects.
- **STEP 3 — core helpers.** `run_single_pair()` writes one `.ply`;
  `run_batch()` processes a folder of images into all C(N,2) pairs;
  `_mirror_to_drive()` with the same-path-skip pattern.
- **STEP 4 — Gradio UI.** Multi-file upload (2-3 images), checkpoint dropdown
  (5 options, auto-downloads on first use), image size slider (128-512),
  Drive-mirror toggle, preview pane with file links to
  supersplat.dev / playcanvas.com/viewer.
- **STEP 5 — keep-alive** (12 h, matches other Colab notebooks).
- **STEP 6 — Colab-side single-pair picker** (12 form params + 1 button).
- **STEP 7 — Colab-side folder batch** (12 form params + 1 button).
- **Custom `_gaussians_to_ply()` writer** that emits a binary 3DGS PLY
  (positions + scales + rotations + SH DC + opacities) that any 3DGS viewer
  (SuperSplat, splat-transform, gsplat.js, PlayCanvas) can read.
- **Pipeline diagram** in `header`: `NoPoSplat → SplatTransform STEP 3 →
  Asset_Library_Browser → Three.js / WebGPU game engine`. NoPoSplat's output
  feeds straight into our existing game-ready pipeline.
- **License posture:** the notebook is MIT (upstream MIT). The MASt3R
  backbone weights are CC BY-NC-SA 4.0 (we only consume them for inference,
  never train, never redistribute). Output `.ply` files are fully yours.
- **All 3 sliders/checkboxes have `info=` tooltips** (UI polish). 9
  try/except blocks. Drive cache set BEFORE any `huggingface_hub` import.

### SkinTokens — New notebook: Mesh to Rig with TokenRig (VAST-AI, MIT)
New notebook at `SkinTokens_Colab.ipynb`. The [official HF Space](https://huggingface.co/spaces/VAST-AI/SkinTokens)
ships a polished Gradio demo, but it's ZeroGPU-only and not runnable in free Colab.
This notebook packages the same MIT-licensed model for free Colab with Drive cache,
batch mode, low-VRAM toggle, and a Colab-side file picker.

- **9-cell standard pattern** (matches `tools/validate.py`): `view-in-github`,
  `header`, `step1-install`, `step2-cache`, `step3-core`, `step4-ui`,
  `step5-keepalive`, `step6-quicktest`, `step7-batch`.
- **STEP 1 — install + repo + checkpoints + bpy_server sidecar.** Installs
  torch 2.7.0+cu128, flash-attn matching wheel (cu12torch2.7 cxx11abiTRUE
  cp312 — matches the Space's prebuilt), then bpy 4.5.4rc0 from the public
  HF Space (PyPI has no cp312 bpy wheel — only cp311 / cp313). Clones the
  upstream `VAST-AI-Research/SkinTokens` repo, downloads both checkpoints
  (GRPO-refined TokenRig + FSQ-CVAE SkinTokens, ≈ 1.6 GB), and boots a
  long-lived `bpy_server.py` subprocess on port 59876 (the same
  Bottle+Tornado HTTP pattern the official demo uses). All cached in
  `AEI_3D_Cache/SkinTokens/` on Drive.
- **STEP 2 — imports + lazy model cache.** `src.*` modules loaded once. Model
  cached on first call so subsequent Gradio clicks are instant. Defines
  `run_rig()` (the shared pipeline) and `_mirror_to_drive()` (with the
  same-path-skip that TripoSPlat / SplatTransform use).
- **STEP 3 — core helpers.** `run_single()`, `run_batch()` (skips
  already-done outputs on re-run), and `_mirror_to_drive()`.
- **STEP 4 — Gradio UI** mirroring the official demo: multi-file upload,
  sampling sliders (`top_k`, `top_p`, `temperature`, `repetition_penalty`,
  `num_beams`, `max_length`), pipeline toggles (`use_skeleton`,
  `use_transfer`, `use_postprocess`), Colab toggles (`low_vram`,
  `do_drive_save`), Run button, status textbox, downloadable .glb files.
  `demo.load` welcome. `clear_output()` before launch.
- **STEP 5 — keep-alive** (12 h, matches other Colab notebooks in this repo).
- **STEP 6 — Colab-side single-mesh picker** with all 12 form params + 1
  button. The same `run_rig()` call STEP 4 uses under the hood.
- **STEP 7 — Colab-side folder batch** with all 12 form params + 1 button.
  Recursively processes `.obj` / `.fbx` / `.glb` in `INPUT_DIR`, preserves
  directory layout in `OUTPUT_DIR`, output always `.glb`.
- **Low-VRAM mode:** CPU-offload the Qwen3 backbone before inference. Drops
  peak VRAM from ~14 GB to ~6-8 GB on T4. ~1.3x slowdown.
- **All 11 sliders/checkboxes** have `info=` tooltips (UI polish). 19
  try/except blocks. Drive cache set BEFORE any `huggingface_hub` import.

### SplatTransform — Engine Preset + Streamed-SOG Octree + Manifest (4 commits)
For a new downstream consumer: a WebGPU Gaussian-splat game engine that
ingests bundled PlayCanvas SOG v2, auto-orients object-scale assets at
load time, and whose shader evaluates only DC + degree-1 SH.

- **Engine Preset (commit ad541f4, default OFF):** new `step2b-engine-preset`
  cell + `step2a-engine-preset-doc` markdown. When ON, biases STEP 3's
  output formats + SH strip defaults:
  - `WRITE_SOG=True`, `WRITE_SPZ/GLB/PLY=False` (still individually toggleable)
  - `STRIP_SH_BANDS=3` (DC-only — engine's shader max)
  - `RE_GROUND_ON_IMPORT` set from `ASSET_CLASS` (False for props, True
    for environments)
  - When OFF: existing defaults unchanged (no regression for current users).
  - Prop-vs-environment distinction documented in the markdown cell:
    **don't pre-ground props for auto-orienting engines** (double-applies).
- **STEP 5 relabel (commit 9a66662):** title now reads "Density Chain
  (per-tier full-model SOGs, NOT streaming)" with markdown clarifying the
  distinction from streamed octree mode. Behavior unchanged.
- **STEP 5b — Streamed-SOG Octree (commit 9a66662):** new cell that builds
  native `lod-meta.json` per asset via splat-transform's `-l <level>` tagging
  + `lod-meta.json` output (PlayCanvas streamed format). Generates N
  decimated SOGs in a temp dir, then bundles them:
  ```
  splat-transform lod0.sog -l 0 lod1.sog -l 1 ... out/lod-meta.json -C 512 -X 16
  ```
  Tunable params: `LOD_COUNTS` (e.g. 200k/100k/50k/20k), `LOD_CHUNK_COUNT`
  (default 512), `LOD_CHUNK_EXTENT` (default 16m). Pre-grounding opt-in via
  `APPLY_GROUNDING` (recommended for environments, NOT for props). Resume +
  FORCE_REDO. Mirror to Drive.
- **STEP 6 note (commit 9a66662):** added markdown block clarifying that
  engines with built-in collision baking (Unreal, Unity DOTS, custom
  Havok/PhysX) can skip this step entirely.
- **STEP 3 batch hardening (commit 65266a1):**
  - `PARALLEL_WORKERS` (1-8, default 2): `ThreadPoolExecutor` runs K
    conversions concurrently for thousands-of-files batches.
  - `RESUME` (default True) + `FORCE_REDO`: skip source PLY if all enabled
    format outputs already exist with non-zero size. Survives Colab
    disconnects.
  - `VALIDATE_OUTPUTS` (default True): re-opens each produced SOG and
    confirms the file parses + is non-trivial.
  - `EMIT_MANIFEST` (default True): writes `<output_dir>/manifest.json`
    with per-source metadata — paths, sizes, compression ratios, source
    splat counts, `assetClass` inferred from splat count vs
    `ASSET_CLASS_THRESHOLD` (default 1M), errors per source, top-level
    summary (total_sources / succeeded / failed).
  - Per-file error log printed at end (first 20 errors shown + count of
    remaining).
  - Cell count 10 → 13. Try blocks 16 → 26, except handlers 17 → 27.

### Asset_Library_Browser — premium UI Tier 1-3 (3 commits)
- **Tier 1 (commit e3e48c9):** Fullscreen preview toggle (`gr.Group(visible=...)`),
  keyboard nav (←/→/Space/P/N/T/F/? via JS keydown listeners), smart filters
  (untagged_only, has_colliders_only, recent_days slider), sort + group by
  any field, batch actions (`gr.CheckboxGroup` + add-tag / set-tier /
  remove-from-meta buttons).
- **Tier 2 (commit b96af41):** Format extensions. `classify()` now recognizes
  `.splat` (3DGS_PACKED), `.usdz` / `.vrm` / `.dae` / `.x3d` / `.mtl` (MESH),
  `.svg` / `.bmp` / `.tiff` / `.gif` (IMAGE), `.hdr` / `.exr` (HDR),
  `.mp4` / `.webm` / `.mov` / `.mkv` (VIDEO), `.wav` / `.mp3` / `.ogg` /
  `.flac` / `.m4a` (AUDIO), `.csv` / `.tsv` / `.md` / `.log` (TEXT), `.yml`
  / `.yaml` (META), `.tar` / `.gz` / `.7z` / `.rar` (ARCHIVE). Preview HTML
  gained branches for VIDEO (`<video>`), AUDIO (`<audio>`), HDR, TEXT, META
  (inline content viewer). Variant switcher: dropdown appears when an asset
  has siblings (same slug, other extensions). Tier cycle button (`T` key)
  walks size_class through `(none) → small_prop → prop → tree_or_vehicle →
  building → (none)`.
- **Tier 3 (commit f57a597):** Reactive count display in toolbar (updates
  with filter), `gr.Warning` toast when filter matches zero, `gr.Info`
  toast when filter matches subset, rich empty-state preview, full help
  modal (`gr.Column(visible=False, elem_id='help-modal')` with `Esc` /
  close button), per-asset detail panel shows `Modified` + `Added to
  library` dates + `Variants: N` badge.
- Tooltip count 9 → 19, try blocks 8 → 14, except handlers 8 → 14.

### Asset_Library_Browser — premium UI Tier 4 (commit 56c0de1)
- **Saved filter presets** (persisted to `_library_meta.json["presets"]`):
  - `preset_dd` dropdown lists existing preset names
  - `preset_save_btn` saves the current filter state (9 fields) under the
    textbox name
  - `preset_load_btn` restores a saved preset into all 9 filter widgets
    + re-runs the gallery + count display
  - `preset_delete_btn` removes the selected preset
- **Per-asset view count** (increments in `on_select`, persisted to META):
  detail panel shows `Views: N` badge when N > 0. Useful for finding
  which assets you actually look at most.
- **Rename batch action**: `gr.Radio` (prepend / append / replace) +
  `gr.Textbox` (text to use, or `old|new` for replace). Updates
  filenames on disk + rewrites META. Re-run STEP 2 to refresh scan.
- Tooltip count 19 → 23, try blocks 14 → 15, except handlers 14 → 15.

### Asset_Library_Browser — premium UI Tier 5.1 (commit c1c1d37)
- **Comparison modal** (side-by-side A/B previews):
  - `compare_btn` opens a modal with the current asset as A
  - `compare_b_dd` dropdown picks the B asset (lists all other assets)
  - Two `gr.HTML` columns render A and B previews side-by-side
  - Diff-highlighted stats table shows path / format / size / size_class /
    grounded / favorite / views / tags / modified
  - Rows where A ≠ B get an orange `#fff3e0` background
  - Closes via `Esc` button or `compare_close_btn`
  - Keyboard shortcut: `D` opens compare with the current selection
- Tooltip count 23 → 24, try blocks 15 → 16, except handlers 15 → 16.

### Asset_Library_Browser — premium UI Tier 5.2 (commit 6eda5f6)
- **Bulk ZIP export**:
  - `export_selected_btn` packs the assets checked in the batch
    CheckboxGroup into a single ZIP
  - `export_filtered_btn` packs all currently-filtered assets (no need
    to check them individually)
  - `export_file` (`gr.File`) shows the resulting ZIP for download
  - Each export includes:
    - The selected asset files (preserving relative paths)
    - `<slug>_meta.json` sidecars (when present) for splat assets
    - `aei_manifest_<timestamp>.csv` with per-asset metadata
  - Status HTML shows file count, ZIP size, uncompressed size, sidecar
    count, manifest flag, skipped count
  - Default output: `/content/aei_library_export_<timestamp>.zip`
- Tooltip count 24 → 25, try blocks 16 → 18, except handlers 16 → 18.

### Asset_Library_Browser — tiering + size_class + grounding filters
- **STEP 2 (scan)** now reads `<slug>_meta.json` sidecars (written by
  TripoSPlat STEP 8 or SplatTransform STEP 6) for each asset file. If
  present, merges `size_class`, `grounding`, `tiers`, and `colliders`
  fields into the per-asset record. The library meta.json gets richer
  per-asset records without re-scanning the splat generation step.
- **STEP 3 (Gradio UI)** gained two new filters:
  - `size_dd` (Dropdown) — filter by `small_prop` / `prop` /
    `tree_or_vehicle` / `building` (auto-populated from existing
    `size_class` values in the library)
  - `grounded_cb` (Checkbox) — show only assets that had the grounding
    transform applied
  - Detail panel shows two new badges: green `size_class` badge + blue
    `grounded {deg}°` badge next to the file info
  - `on_save` / `on_fav` updated to thread the new filters so the
    gallery doesn't reset to unfiltered on edit
- **STEP 6 (Stats dashboard)** added two new sections:
  - Size class breakdown (count per class, with "no sidecar" count
    showing which assets need to be re-processed)
  - Grounding status (X grounded of Y with metadata, plus a count
    of assets with no grounding data at all)
- For the 1500+ library: filter to "show me only `tree_or_vehicle`
  assets that are grounded", or "show me ungrounded assets that need
  re-running STEP 8", or "show me all hero/`prop` assets for review".

### Game engine integration (Three.js / WebGPU loader)
- New README section "Game engine integration — loading the assets into
  Three.js / WebGPU" with:
  - File layout of what TripoSPlat STEP 8 / SplatTransform STEP 6
    produces per asset (3 visual tiers + 2 colliders + meta.json)
  - TypeScript loader class with full type annotations:
    `loadGroundedAsset()`, `selectLOD()`, `setupPicking()`
  - Usage example showing per-frame LOD update + click picking
  - Tiering constants table mapping `size_class` to grid spacing
  - Notes on Three.js version, browser compat, memory budget
- TypeScript with `.d.ts` style type hints since the user is building
  a WebGPU engine. Snippets are complete enough to copy-paste.

### Mesh_Optimizer — stage_ground() (parallels splat grounding)
- New `stage_ground()` function (trimesh-based, mirrors the splat
  helper): translates mesh so the bottom sits on Y=0, centers in XZ,
  optionally axis-aligns via PCA in the XZ plane (snapped to 90 deg).
  Returns `(modified_mesh, transform_dict)` with the same shape as
  TripoSPlat STEP 8's `_ground_splat()` output.
- Pipeline integration: `optimize_mesh()` now runs 'ground' as the
  first stage (right after load, before clean). All 4 presets have
  `'ground': True` as the first stage by default.
- Meta.json sidecar: after successful ground, `optimize_mesh()` writes
  a `<slug>_meta.json` with the transform + size_class + timestamp.
  Same format as TripoSPlat STEP 8's meta.json so the Asset_Library_Browser
  can read either source consistently.
- Two new params: `ground_center_xz` (default True),
  `ground_axis_align` (default True). Both can be disabled for assets
  already in a grid layout or for organic shapes.
- This makes the splat and mesh pipelines consistent: every asset
  in the user's library (splats OR meshes) goes through the same
  grounding treatment and gets the same meta.json sidecar format.

### TripoSPlat STEP 8 — One-Shot Game-Ready Prep (SplatTransform Lite)
- New STEP 8 in `TripoSplat_Colab.ipynb`. After a STEP 7 batch finishes, this
  cell auto-discovers the latest batch folder (or honors `INPUT_BATCH_DIR`),
  installs Node 22+ + `@playcanvas/splat-transform` idempotently, and produces
  per-asset:
  - 3 visual quality tiers: `<slug>_full.sog` (no decimate, 3 SH bands, ~17 MB),
    `<slug>_standard.sog` (25% Gaussians, 2 SH bands, ~4-5 MB, default LOD),
    `<slug>_background.sog` (6% Gaussians, 0 SH bands, ~1-2 MB, mass-placement)
  - 2 collider meshes: `<slug>_hull.glb` (convex hull, ~10-50 KB) and
    `<slug>_collision.glb` (voxel marching-cubes, ~1-3 MB)
  - `<slug>_meta.json` sidecar with sizes, paths, `size_class` (small_prop /
    prop / tree_or_vehicle / building derived from final bbox height), and the
    grounding transform
  - `.sog` + `.splat` compression by default; `.spz` + `.glb+KHR_gaussian_splatting`
    opt-in
- **Grounding helper** (`_ground_splat()`): PCA-based axis alignment in the XZ
  plane (snapped to nearest 90°) + translation so the bottom of the splat sits
  on Y=0 + XZ centroid centering. Applied to the PLY before compression, so
  the resulting `.sog`/`.splat`/`.spz`/`.glb` files all come out **already
  grounded** — drop them in your engine with `splat.position.y = 0` and they
  work. Default `APPLY_GROUNDING = True`. Original PLY in STEP 7 output is
  untouched (so you can re-run with grounding off to compare).
- **SplatTransform STEP 6** in `SplatTransform_Colab.ipynb` gained 3 collider
  types (was just voxel marching-cubes): voxel `.collision.glb`, convex hull
  `.hull.glb` (always run), and concave hull / alpha-shape `.concave.glb`
  (opt-in). Also gained `RE_GROUND_ON_IMPORT` toggle (off by default) to
  re-ground PLYs from non-TripoSPlat sources.
- Both cells share the same `_ground_splat()` helper (copy-pasted, ~50 lines).
  Trade-off: keeping each notebook self-contained for Colab (no cross-notebook
  imports). 14 new Gradio `info=` tooltips across the new params; 1 new
  try/except block for the per-asset grounding flow.

### TripoSPlat STEP 7 — crash safety + quality defaults
- Defaults changed for the balanced 200+ library case:
  `BATCH_STEPS: 15 → 30`, `BATCH_NUM_GAUSSIANS: 131072 → 262144`. Cell markdown
  documents speed presets (steps=20/131k = 1 min/img; steps=12/65k = 30 s/img).
- New `BATCH_RESUME_FOLDER` param: paste the path to a crashed batch and the
  cell picks up where it left off (per-asset progress log
  `_batch_progress.jsonl`).
- New `BATCH_RESUME` (default `True`): skip assets with existing `.ply`.
- Per-item Drive mirror (was: at end). Each completed model is safe
  immediately on a T4 disconnect.
- New `BATCH_VERBOSE` (default `True`): per-item progress + ETA.

### TripoSPlat STEP 1 — Google Drive mount
- New `CONNECT_GOOGLE_DRIVE` checkbox (default `True`) at the top of STEP 1.
  When checked, mounts Drive and pre-creates
  `/content/drive/MyDrive/AEI_3D_Out/TripoSplat/` so all batch outputs are
  mirrored automatically. Fixes the common "lost my batch because Drive
  wasn't mounted" pain point for fresh Colab sessions.

### TripoSPlat STEP 4 (Gradio UI) — display fix
- STEP 4 was a single 8001-char source list entry which Colab's renderer
  displayed as one giant line. Split into 158 proper line-by-line entries
  (matches the other steps in this notebook).

### Added
- `Audio_PostProcessor_Colab.ipynb` — new **AI Enhance** tab (6th of 10),
  powered by [`resemble-enhance`](https://github.com/resemble-ai/resemble-enhance)
  (Resemble AI, MIT). 44.1 kHz PyTorch speech model that does **denoise +
  bandwidth extension + artifact reduction** in one pass. This is the only
  tool in the suite that can restore high-frequency content lost in thin
  / compressed TTS output, and is a strict superset of the spectral-gate
  `noisereduce` approach used in the Denoise tab. 4 hyperparameters exposed:
  - **CFM ODE solver**: Midpoint (default) / RK4 / Euler
  - **CFM NFE** (function evaluations): 1-128, default 64. Higher = better
    quality, slower.
  - **CFM prior temperature (tau)**: 0-1, default 0.5. Lower = closer to
    training distribution.
  - **Denoiser strength (lambd)**: 0-1, default 0.5. 0 = no denoise, 1 =
    aggressive.
  - **denoise_first** checkbox: runs the standalone denoiser first, then
    enhances (2-pass, slowest, highest quality for very noisy input).
  Outputs BEFORE/AFTER audio players, peak/RMS/LUFS stats, full log, and a
  download link. GPU auto-detected (1-5s per minute of audio on GPU, 10-60s
  on CPU). 32 new Gradio components in the AI Enhance tab.
- `Audio_PostProcessor_Colab.ipynb` — new **Studio Polish** preset (5th in
  Quick Process), wrapping the AI Enhance pipeline: HPF 80 Hz + Resemble
  Enhance (default params) + LUFS -16 + peak -1. Best for thin / compressed
  TTS output or noisy speech.
- `Audio_PostProcessor_Colab.ipynb` — `process_audio()` now supports a
  `resemble_enhance` step in the pipeline dict, sandwiched between highpass
  and denoise. The denoise step can still run after the AI enhance (as a
  spectral post-polish) or be skipped.
- `Audio_PostProcessor_Colab.ipynb` — `resemble-enhance` added to STEP 1
  installs (resilient to install failure — AI Enhance tab is disabled if
  the import fails). Lazy import in STEP 3 with 8-tuple `_ensure()` return.
- `Audio_PostProcessor_Colab.ipynb` — **Denoise tab** now supports two
  backends via a radio button:
  - **Spectral gate** (`noisereduce`, default, fast, no model, good for
    steady hum/hiss)
  - **Resemble denoiser** (44.1 kHz AI model from `resemble-enhance`, same
    engine as the AI Enhance tab but denoise-only — no bandwidth extension).
    Slow (1-5s/min on GPU, 10-60s on CPU) but better for speech.
  Backend-specific options (`run_dir`, `device`) live in a hidden `gr.Group`
  that shows when Resemble is selected, so the default UI is uncluttered.
  Shares the same `run_dir` and `device` settings as the AI Enhance tab.
  Resamples to 44.1 kHz mono internally (model native). `ui_denoise()` now
  dispatches based on the radio choice. Model status Markdown refreshed on
  click. 3 new `dn_*` Gradio components (`dn_backend`, `dn_run_dir`,
  `dn_device`). Tab description + Help tab + welcome message updated.
- `Audio_PostProcessor_Colab.ipynb` — `res_enhance()` and `res_denoise()`
  now accept `run_dir` (custom checkpoint folder) and `device` (override
  the auto-detected device). The `run_dir` hook is the only extensibility
  point in `resemble-enhance`; it accepts a folder containing
  `hparams.yaml` + `ds/G/default/mp_rank_00_model_states.pt`. By setting
  `HF_HOME` to Google Drive in STEP 1, the 713 MB default checkpoint
  persists across Colab sessions — no re-download. Custom fine-tunes can
  be loaded by pointing `run_dir` at their folder.
- `Audio_PostProcessor_Colab.ipynb` — **AI Enhance tab** now has two
  sub-modes:
  - **Single file** (original): upload → get enhanced output
  - **Batch (directory)**: process every audio file in a folder with one
    click. Pattern filter (`*.wav` / `*.flac` / etc.), recursive subdirectory
    walk, per-file progress, OK/Failed/Total/Time summary,
    subdirectory-preserving output. Per-file failures are caught and logged;
    one bad file does not abort the batch. Backed by new
    `batch_res_enhance()` in STEP 3 (reuses `res_enhance()` internally).
  - **Model source accordion**: `run_dir` text field for custom checkpoints,
    `device` dropdown (`auto` / `cuda` / `cpu`). Both are shared between
    Single file and Batch sub-modes.
  30 new Gradio components (`bae_*` prefix) added to the Batch sub-tab.
  12 new info= tooltips. Cell size from 108 KB to 122 KB.
- `Audio_PostProcessor_Colab.ipynb` — new **Effects Chain** tab (4th of 10),
  powered by [`pedalboard`](https://github.com/spotify/pedalboard) (Spotify
  Audio Intelligence Lab, GPL-3.0). 16 studio-quality audio effects exposed
  in the UI across 5 accordion groups:
  - **Filters**: HighpassFilter, LowpassFilter, LadderFilter (Moog-style
    4-mode: LPF12/LPF24/BPF12/HPF12, with cutoff + resonance)
  - **Dynamics**: Gain, Compressor (threshold/ratio/attack/release),
    Limiter (threshold/release)
  - **Time & Space**: Reverb (room/damping/wet/dry/width),
    Delay (time/feedback/mix), Chorus (rate/depth/mix), Phaser
    (rate/depth/frequency/feedback/mix)
  - **Distortion & Pitch**: Distortion (drive dB), Clipping (threshold),
    PitchShift (semitones ±24)
  - **Lo-Fi**: Bitcrush (bit depth 1-16), Resample (target SR 1-44.1 kHz),
    MP3Compressor (VBR quality 0-10)
  4 chain presets ship out of the box:
  - **Podcast Voice** — HPF 80 Hz + Compressor -20 dB / 3:1 + Gain +3 dB + Limiter -1 dB
  - **Vocal Cleaning** — HPF 100 Hz + Compressor -22 dB / 4:1 + Limiter -1 dB
  - **Music Mastering** — Compressor -15 dB / 2.5:1 + Gain +2 dB + Limiter -0.5 dB
  - **Lo-Fi Tape** — HPF + LPF 8 kHz + Bitcrush 10-bit + Chorus + Reverb
  All 16 effects also available individually via the sliders — every
  parameter exposed. The 4 chain presets and the custom-slider mode share
  the same `apply_effects_chain()` core, which builds a `pedalboard.Pedalboard`
  list and runs it. `pedalboard` is GPL-3.0; used as a `pip install`-ed
  dependency, so the notebook code remains MIT and commercial use of the
  processed audio is fine. UI grew from 18 to 37 info= tooltips, cell size
  from 62 KB to 82 KB.
- `Audio_PostProcessor_Colab.ipynb` — the second **post-processing** notebook
  in the suite (after the Mesh Optimizer), wrapping a curated audio
  processing stack: [`pydub`](https://github.com/jiaaro/pydub) (MIT) for
  format conversion + silence detect, [`imageio-ffmpeg`](https://github.com/imageio/imageio-ffmpeg)
  (BSD-4) for the static ffmpeg binary, [`soundfile`](https://github.com/bastibe/python-soundfile)
  (BSD-3, libsndfile) for fast WAV/FLAC/OGG/AIFF round-trips,
  [`librosa`](https://github.com/librosa/librosa) (ISC) for DSP analysis,
  [`pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm) (MIT) for
  ITU-R BS.1770-4 LUFS / LKFS broadcast-standard normalization, and
  [`noisereduce`](https://github.com/timsainb/noisereduce) (MIT) for
  spectral-gating noise reduction. All CPU-only, no GPU required.

  Takes the **raw audio output** from any TTS / VC / voice-cloning
  pipeline (Qwen3-TTS, VoxCPM2, Higgs, MisoTTS, MOSS, dots.tts, Fish,
  Kokoro, OpenVoice V2, etc.) and turns it into a clean, ready-to-publish
  audio file. Ten tabs:
  - **Quick Process** — 5 one-click presets: Podcast (-16 LUFS), Music
    (-14 LUFS, preserves bass), Speech (-23 LUFS, broadcast-ready),
    Broadcast (strict EBU R128 / ATSC A/85), **Studio Polish (AI)**
  - **Trim & Split** — trim to time range, split by silence, split into
    N-second chunks, detect silence ranges
  - **Normalize** — peak / LUFS (ITU-R BS.1770-4) / no-op re-encode
  - **Effects Chain** — 16 pedalboard VST-style effects with 4 chain presets
  - **Format Convert** — 8 formats (WAV/MP3/FLAC/OGG/Opus/M4A/AAC/AIFF)
    with bitrate (32-320 kbps) and sample rate (8 kHz - 96 kHz)
  - **AI Enhance** — Resemble Enhance 44.1 kHz AI model (denoise + bandwidth
    extension + artifact reduction). GPU auto-detected.
  - **Denoise** — spectral gate with strength slider
  - **Batch** — apply any preset to every audio in a directory
  - **Compare** — before/after waveform + stats (peak, RMS, LUFS, duration)
  - **Help** — when-to-use, format cheatsheet, LUFS targets, citation
  - **8 export formats**: WAV (lossless), FLAC (lossless), AIFF (lossless),
    MP3 (universal), OGG Vorbis, Opus (streaming), M4A/AAC (Apple)
  - **Loudness normalization**: ITU-R BS.1770-4 LUFS / LKFS, broadcast
    standard, supports any target from -30 to -6 LUFS
- `Mesh_Optimizer_Colab.ipynb` — new **Transform & Normalize** tab (7th tab).
  Scale (uniform or per-axis), recenter to origin, flip X/Y/Z axes, normalize
  to a target bbox extent (e.g. 1.0 for unit cube, 2.0 for [-1,1]), snap
  vertices to a grid step, and re-orient between Y-up (Blender/Unity/glTF),
  Z-up (Blender default/3ds Max), and X-up conventions. Useful before feeding
  meshes into other AI tools (Cube/Hunyuan3D) or for fitting into a known
  bounding box. 18 new Gradio components, 1 new `ui_transform()` handler,
  README tab list updated.
- `Mesh_Optimizer_Colab.ipynb` — first **post-processing** notebook in
  the suite, wrapping a curated stack: [`trimesh`](https://github.com/mikedh/trimesh)
  (3.6k★, MIT) for I/O + repair + smoothing, [`pyfqmr`](https://github.com/Kramer84/pyfqmr-Fast-quadric-Mesh-Reduction)
  (MIT, 100 KB wheel) for fast quadric edge-collapse decimation (the
  sp4cerat algorithm — the only one in Python that gives Blender-quality
  results), [`pymeshlab`](https://pymeshlab.readthedocs.io) (MIT) for
  advanced MeshLab filters (UV unwrap, hole filling), and [`Open3D`](https://www.open3d.org)
  (MIT) for point-cloud ops + alignment. All CPU-only, no GPU required.

  Takes the **raw, often-broken mesh output** from Pixal3D, Hunyuan3D,
  Cube 3D, or any other 3D pipeline, and turns it into a clean, game-ready
  asset. (See the new **Transform & Normalize** tab documented at the top
  of this [Unreleased] section.) Original six tabs:
  - **Quick Optimize** — 4 one-click presets: Game-Ready (50% decimate
    + Taubin smooth + UV), Print-Ready (quad remesh + UV), Low-Poly
    (10% decimate + Humphrey smooth), Lossless (clean only)
  - **Custom Pipeline** — full control over every stage, accordion-grouped
  - **Inspect** — face/vertex counts, watertight check, manifold check,
    volume, area, bbox
  - **Batch** — apply any preset to every mesh in a directory
  - **Compare** — before/after stats side-by-side with delta percentages
  - **Help** — when-to-use table, format cheatsheet, citation
  - **5 export formats**: `.glb` (Unity/Unreal/Three.js), `.obj + .mtl`
    (Blender/Maya), `.stl` (3D print), `.ply` (Meshlab/CloudCompare), `.3mf`
  - **8 input formats**: STL, PLY, OBJ, GLB, GLTF, 3MF, OFF, COLLADA
- `Audio_PostProcessor_Colab.ipynb` — new **TTS Polish** preset (6th in
  Quick Process): auto-trim leading/trailing silence (pydub silence
  detection, -40 dBFS threshold, 500 ms minimum silence) + HPF 80 Hz +
  LUFS -16 + peak -1. CPU-only, no AI, best for TTS output that has
  padding or inconsistent levels. The new silence-trim step runs
  *first* in the pipeline so all subsequent stages (highpass, denoise,
  LUFS) only operate on the speech region.
- `Audio_PostProcessor_Colab.ipynb` — new `_trim_silence()` helper
  function in STEP 3 using pydub's `silence.detect_nonsilent`, with
  50ms head/tail padding to avoid clipping the first/last syllable.
  Returns `(trimmed_samples, lead_seconds, trail_seconds)`.
- `Audio_PostProcessor_Colab.ipynb` — `process_audio()` extended to
  support the new `trim_silence` / `trim_silence_thresh` /
  `trim_silence_min_ms` pipeline fields. Applied as stage [0] before
  highpass; all other stages renumbered accordingly.
- `Audio_PostProcessor_Colab.ipynb` — **unified Batch tab**. Previously
  the Batch tab (Quick Process preset) and the AI Enhance > Batch
  (directory) sub-tab were separate. Now consolidated into a single
  Batch tab with a mode radio: **Quick Process preset** (applies any
  of the 6 presets to every file, downloads a .zip) or **AI Enhance**
  (Resemble Enhance on the whole folder, per-file progress, OK/Failed/
  Total/Time summary). Common widgets (input dir, output dir, pattern,
  recursive) are shared; preset/AI-specific widgets are grouped and
  shown/hidden via `_toggle_mode` on radio change. Both `ui_batch` and
  `ui_ai_enhance_batch` handlers are preserved internally and
  dispatched from a thin `_unified_batch` wrapper. The AI Enhance tab
  is now Single-file-only (Batch mode moved to the unified tab).
  Tooltip count: 70 → 71. Tab count: 10 main + 2 AI Enhance sub-tabs
  → 10 main + 1 AI Enhance sub-tab.
- `Audio_PostProcessor_Colab.ipynb` — Step 6 (Quick Test) and Step 7
  (Batch Processing) `@param` dropdowns updated to include
  `tts_polish` and `studio_polish` in addition to the original 4
  presets (now 6 options).
- `Audio_PostProcessor_Colab.ipynb` — welcome message and Help tab
  updated to mention the 6th preset (TTS Polish) and the unified
  Batch tab.
- `MOSS-TTS_Colab.ipynb` — STEP 1 **numpy 2.x compatibility patch**:
  Colab Runtime 2026.01 ships numpy 2.0.x where the string ufuncs
  (`_center`, `_ljust`, `_rjust`, `_zfill`, `_strip_*`, `_lstrip_*`,
  `_rstrip_*`, `_partition*`, `_rpartition*`, `_slice`,
  `_expandtabs*`, `_replace`, `is*`, `find`/`index`,
  `startswith`/`endswith`, `str_len`, etc.) are not yet exposed in
  `numpy._core.umath`. The first import of `numpy._core.strings`
  (triggered by `import torchaudio`, `import librosa`, `import
  soundfile`, MOSS-TTS' `AutoProcessor`, or any access to
  `np.strings`) then fails with `ImportError: cannot import name
  '_center' from 'numpy._core.umath'`. MOSS-TTS pulls all of these
  as transitive deps, so a fresh runtime is unusable. The patch
  injects pure-Python ufuncs into `numpy._core.umath` for every
  name `numpy/_core/strings.py` needs — real implementations for
  the 19 most-called names (using Python `str` methods, not
  `np.char`, to avoid the chicken-and-egg: importing `np.char`
  itself triggers the very import we are trying to fix), and
  passthrough stubs for the rest. The patch is applied right after
  `import numpy` and before any `pip install` / `import
  MOSS-TTS` / `import torchaudio`. MOSS-TTS itself never actually
  calls any of these on string arrays during inference — the
  patch is only needed so `from numpy._core.umath import ...` in
  `numpy/_core/strings.py` succeeds at import time. The patch is a
  no-op on newer numpy versions (numpy 2.0.2+ already has all the
  ufuncs), so the same notebook runs unchanged on current and
  future runtimes. See the README "MOSS-TTS v1.5" section for the
  full explanation and motivation.
- `TripoSplat_Colab.ipynb` — new 3D model: image-to-3D Gaussians
  by [TripoAI / VAST-AI-Research](https://www.tripo3d.ai/research/triposplat)
  ([arXiv 2605.16355](https://arxiv.org/abs/2605.16355),
  [HF VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat),
  [code](https://github.com/VAST-AI-Research/TripoSplat),
  **MIT — commercial-OK**). 8B-param-equiv image-to-3DGS via
  DINOv3 ViT-H/16+ + Flux2 VAE encoder → 24-block, 1024-dim
  flow-matching DiT → Octree + Gaussian decoder (32k → 262k
  Gaussians, multiple of 32). Background removal via BiRefNet
  Swin-L. 3.78 GB total weights across 5 safetensors. 6 export
  formats in one click:
    - **Native 3DGS**: `.ply` (3DGS standard) and `.splat` (32-byte
      packed, for web viewers like Antimatter15)
    - **Reconstructed mesh**: `.glb` (binary glTF, open3d), `.obj`
      (Wavefront text), `.fbx` (custom pure-Python ASCII FBX 7.4
      writer, Y-up or Z-up, works in Blender / Unity / Maya / Godot)
    - **Mesh-as-PLY** for hand-off to `Mesh_Optimizer_Colab.ipynb`
  Mesh reconstruction uses Poisson surface reconstruction (open3d,
  ~5-15s on CPU) with optional alpha-shape fallback. Filter by
  opacity, subsample to 100k-300k points, KNN normals, decimation
  to ≤300k faces. The custom FBX writer is a ~200-line pure-Python
  implementation of the FBX 7.4 ASCII spec (no animations, no
  bones, no cameras — just meshes). Better than the upstream HF
  Space's default behavior: includes 6 export formats (Space only
  exports .ply/.splat), exposes every sampling parameter (Space
  hardcodes 20 steps, 3.0 cfg, 262k Gaussians), has Drive-cached
  outputs at `/content/drive/MyDrive/AEI_3D_Out/TripoSplat/`, and
  has the MOSS-TTS-style numpy 2.x umath patch applied defensively.
  9-cell Pixal3D pattern: STEP 1 install + numpy patch, STEP 2
  Drive cache, STEP 3 pipeline + 6-format export, STEP 4 single-page
  Gradio UI with 15 tooltips and 11 try/except blocks, STEP 5
  keep-alive, STEP 6 quick test (12 steps, 65k Gaussians), STEP 7
  batch from a .txt list. QA-check passes (Y Y Y Y Y Y). Requires
  Colab Runtime 2026.01 (torch 2.9.0+cu126). L4 GPU (22 GB)
  recommended; T4 (16 GB) is tight — use `num_gaussians ≤ 65536`
  and `steps ≤ 15`. See the README "TripoSplat" section for the
  full architecture breakdown, the 6-format export table, and the
  complement-to-Hunyuan3D-2.1 positioning.
- `TripoSplat_Colab.ipynb` — Step 6 (QuickTest) and Step 7 (Batch)
  workflow enhancements for converting 200+ images into a game-asset
  library:
    - **Step 6** adds `QUICK_INPUT_IMAGE` (explicit path; blank = auto-pick
      from `/content`), `QUICK_MESH_METHOD` (poisson / alpha_shape),
      `QUICK_MAX_POINTS` (50k-500k slider), `QUICK_OUTPUT_FORMAT` (all /
      3DGS only / mesh only / native+LOD only), `QUICK_SAVE_TO_DRIVE`
      (mirror outputs to `/content/drive/MyDrive/AEI_3D_Out/TripoSplat/`).
      Outputs are named after the image stem: `hero.png` →
      `hero.ply`, `hero.glb`, `hero_LOD0..2.glb`, `hero.fbx`, etc.
    - **Step 7** adds `BATCH_INPUT_MODE` (folder / txt list),
      `BATCH_INPUT_FOLDER` (default `/content/triposplat_runs/inputs`),
      `BATCH_RECURSIVE` (scan subfolders), `BATCH_MESH_METHOD`,
      `BATCH_MAX_POINTS`, `BATCH_OUTPUT_FORMAT`, `BATCH_MESH_DEPTH`
      (now a slider, default 10), `BATCH_DO_DRIVE_SAVE` (mirror entire
      batch folder to Drive). Each subject gets its own subfolder so
      LOD/GLB/FBX files from different images don't collide. Recursive
      mode uses `parent_stem` prefix for unique slugs across folders.
    - Default `num_gaussians` bumped 65k → 131k (Step 6) and 65k → 131k
      (Step 7) to match the Gradio UI's default for better quality.
    - Bumped `mesh_depth` 9 → 10 default in both cells.
    - LOD chain default expanded to 3 levels (`1.0,0.5,0.25`) for richer
      asset libraries.
- `TripoSplat_Colab.ipynb` — new **STEP 8** for standalone post-processing
  of `*_mesh.ply` files. Reuses the Mesh Optimizer stage functions
  (clean, fill holes, UV unwrap, smooth, export) as pure in-notebook
  helpers — no need to load `Mesh_Optimizer_Colab.ipynb` separately. Toggles
  for each stage with sliders for `max_hole_size` and `smooth_iterations`.
  Also added **`BATCH_POST_PROCESS` flag in Step 7** so the post-processing
  pipeline runs inline on every batch item. Output is `<slug>_game.glb` /
  `<slug>_game.fbx` / `<slug>_game.obj` (etc.) alongside the raw
  TripoSplat exports, with vertex colors preserved, UVs generated, and
  holes closed. Reverted Step 7 to a **flat output folder** (no per-item
  subdirs) — file naming uses `<index>_<slug>` prefix to avoid collisions.
  Installs `pymeshlab` and `pyfqmr` in Step 1 (the Mesh Optimizer deps
  for the new post-processing stages).
- `TripoSplat_Colab.ipynb` — memory cleanup between export stages to
  avoid late OOM kills on T4: explicit `del pcd` after mesh recon,
  `del verts/tris/colors_arr/sg` after FBX write, `del lod_meshes`
  after LOD chain. Each `del` frees 5-15 MB of numpy/open3d buffers
  before the next allocation. Added progress updates during the
  smoothing-groups compute and FBX write (which is the slowest single
  step at 5-10s for 100k faces).
- `SuGaR_Colab.ipynb` — new notebook (8 cells) for **Surface-Aligned
  Gaussian Splatting to Mesh**. Takes a TripoSplat-generated 3DGS PLY
  and a preprocessed image, builds a COLMAP-compatible scene with one
  estimated camera, then runs SuGaR's surface-alignment + mesh
  extraction. Outputs a textured `.obj` + texture atlas `.png` +
  refined 3DGS `.ply` + `.glb` for game engines. **License warning:**
  SuGaR uses the **INRIA Gaussian-Splatting License** (custom
  non-commercial research license) — this notebook is included for
  personal-asset research/evaluation only. For commercial use, get
  INRIA's permission or use Kiri Engine / Polycam instead.
  Compute: L4 22 GB recommended (~2-3 hrs per scene), T4 16 GB will
  OOM, A100 40 GB gives ~40% speedup. Realistic for 5-10 hero
  assets in a 200+ library, not the whole library. See the
  "Production pipeline" section in the README for the full
  decision tree.
- `GauStudio_Colab.ipynb` — new notebook (10 step cells) for
  **3DGS-to-mesh via TSDF fusion**. Faster, lower-VRAM alternative to
  SuGaR for the long tail of a 200+ image library. T4 15 GB works
  (SuGaR needed L4 22 GB); ~5-10 min per scene (SuGaR was 2-3 hrs).
  Mesh is smoother/cleaner than SuGaR but with less geometric
  detail — better trade-off for low-LOD game assets, worse for
  hero-asset close-ups. **License notice:** mixed MIT (main
  framework) + INRIA non-commercial (rasterizer submodule), same
  as SuGaR. Includes a TripoSplat-PLY bridge (single-image input)
  AND a multi-view scene path (3+ overlapping cameras = best
  quality). Skips   texturing (mvs-texturing C++ build is brittle on
  Colab) — output is untextured mesh; texture in Blender or the
  game engine. Required citation: Ye et al., "GauStudio: A Modular
  Framework for 3D Gaussian Splatting", CVPR 2024.
- `Asset_Library_Browser_Colab.ipynb` — new notebook (8 step
  cells) for **browsing, tagging, previewing, and exporting the
  200+ asset library** that TripoSplat + GauStudio/SuGaR +
  Mesh Optimizer produce. CPU-only, no model weights, no GPU
  required. Scans a library folder, recognizes 12+ asset formats
  (3DGS PLY/SPLAT, mesh GLB/OBJ/FBX/PLY/STL/3MF, image
  PNG/JPG/WEBP, text, JSON, ZIP), builds a metadata sidecar
  JSON. Gradio UI: gallery with thumbnails, filter by
  tag/format/favorite/search, click-to-preview (3D mesh in inline
  `<model-viewer>`, 3DGS gets a metadata card + link to
  Antimatter15 viewer), tag editor, favorite toggle. Exports to
  Unity AssetBundle-style folder, Godot .tres + mesh files,
  self-contained static HTML portfolio, CSV manifest. Stats
  dashboard with format breakdown + top tags + missing/orphan
  report. **What it doesn't do:** 3DGS files don't get
  thumbnails or inline previews (no easy in-notebook 3DGS
  renderer; workaround is to run GauStudio first to get a GLB).
  This is a fundamental limitation of all current web-based 3DGS
  viewers.
- `TripoSplat_Colab.ipynb` — **major refactor**. Stripped all
  mesh export code (`gaussians_to_pointcloud`,
  `reconstruct_mesh` with 3 methods, `_apply_transform_to_mesh`,
  `_generate_lod_chain`, `_lod_to_glb`, `write_fbx_ascii`,
  `_compute_smoothing_groups`, `_smoothing_groups_to_fbx_mask`,
  `export_all_formats`, `postprocess_for_game`, all the
  `_*_post_*` helpers, and Step 8's standalone post-process).
  That's ~1500 lines / 47 KB of mesh code removed (86% reduction
  in STEP 3). The notebook now outputs **only `.ply` (3DGS
  standard) and `.splat` (web viewer packed)** — both are the
  high-quality 3DGS outputs the user actually wants. For
  game-ready textured meshes, the user is directed to
  `Pixal3D_Colab.ipynb`. For 3DGS-to-mesh research pipelines,
  the user is directed to `SuGaR_Colab.ipynb` and
  `GauStudio_Colab.ipynb`. The mesh-reconstruction quality
  problems (holes, missing surfaces, no UVs, T4 OOM crashes) are
  gone with the deletion. The Step 6 / Step 7 workflow ergonomics
  (auto-pick image, `QUICK_INPUT_IMAGE`, `BATCH_INPUT_MODE`,
  `BATCH_RECURSIVE`, file naming by image stem, `BATCH_DO_DRIVE_SAVE`)
  are preserved. STEP 3 now ~7.5 KB, STEP 4 UI now ~7.5 KB,
  STEP 6 ~4.5 KB, STEP 7 ~7 KB (was 55/15/8/13 KB respectively).
- `Pixal3D_Colab.ipynb` — ported all the workflow ergonomics
  from the stripped TripoSplat notebook. **Added 2 new cells**
  (now 11 total): Step 8 (post-process existing GLBs via the
  same Mesh Optimizer stages: clean, fill holes, UV unwrap,
  smooth, multi-format export) and Step 9 (help / format
  reference / pipeline overview / Pixal3D vs TripoSplat decision
  tree). Step 6 (single image) gained auto-pick from /content
  and per-stage advanced sampling sliders (SS_GUIDANCE,
  SS_RESCALE, SS_RESCALE_T, SHAPE_GUIDANCE, SHAPE_RESCALE,
  SHAPE_RESCALE_T, TEX_GUIDANCE, TEX_RESCALE, TEX_RESCALE_T) so
  the same code as TripoSplat's batch can be tuned per-stage.
  Step 7 (batch) gained `BATCH_INPUT_MODE` ('folder' vs 'txt
  list'), `BATCH_RECURSIVE`, `MAX_ITEMS`, `BATCH_DO_DRIVE_SAVE`,
  per-image SLUG (image stem with parent prefix in recursive
  mode for collision-free naming), and the Drive mirror goes to
  a separate folder so the canonical output stays clean. Step 5
  (keep-alive) upgraded to the standard 30-min Colab JS pattern
  (was a 5-min Python thread). Cell IDs standardized to
  `step{N}-{slug}`. qa_check now shows 22 tooltips (was 22),
  17 try/17 except (was 9/9) — the new STEP 8 post-process adds 8
  more try/except blocks for safe per-stage error handling.
- `SplatTransform_Colab.ipynb` — new notebook (9 cells) wrapping
  [PlayCanvas `splat-transform`](https://github.com/playcanvas/splat-transform)
  (MIT, commercial-OK) for **3DGS format conversion + compression**.
  Takes TripoSplat's raw 3DGS `.ply` outputs (~150-250 MB each)
  and converts to 4 game-engine-friendly formats with ~10×
  compression: `.sog` (PlayCanvas native, GPU SH k-means),
  `.spz` (Niantic, smallest cross-platform, mobile-friendly),
  `.glb` + `KHR_gaussian_splatting` (the new glTF 2.0 standard
  extension, future-proof), and `.ply` (lossless fallback). Plus
  decimation (reduce Gaussian count for web previews), SH-band
  stripping (drop color detail for faster loads), LOD chains
  (streamed SOG for PlayCanvas progressive loading), voxel
  collision meshes (`.collision.glb` for runtime physics — NOT
  visuals), and GPU rasterized turntable previews (`.webp`).
  Installs Node 22+ via NodeSource (Colab's default nodejs is too
  old) + `npm i -g @playcanvas/splat-transform`. All 4 outputs
  toggleable per run; per-format compression-ratio reports. Step
  7 does a final Drive mirror with a generated README explaining
  each folder. **The missing piece in the 200+ image library
  workflow:** TripoSplat 200× PLY = 30 GB → SplatTransform 200×
  SOG = 3 GB (saves 27 GB of Drive space). **Important caveat
  documented everywhere:** splat-transform is NOT a mesh-from-3DGS
  generator; the only mesh it produces is a voxel collision mesh.
  For visual mesh extraction, the user is still directed to
  SuGaR / GauStudio / Pixal3D.

### Added (prior in this cycle)
- `VoxCPM2_Colab.ipynb` — self-contained Colab wrapper around
  [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), a
  **tokenizer-free** 2B-param TTS from OpenBMB (Tsinghua / ModelBest
  inc). Models speech in a continuous latent space, enabling four
  flagship capabilities: plain TTS, **Voice Design** (description in
  parens, no ref audio), **Controllable Cloning** (ref audio
  short-clip), and **Ultimate Cloning** (ref + transcript + prompt for
  max fidelity). 30 languages + 9 Chinese dialects, 48 kHz output,
  ~8 GB VRAM, Apache-2.0 (commercial-OK). 27.3k★ on GitHub.
  Eight tabs:
  - **TTS** — plain text → speech, voice inferred from content
  - **Voice Design** — `(description)text` syntax
  - **Voice Clone** — ref audio (5-30s) → cloned voice
  - **Ultimate Clone** — ref + transcript + prompt for max sim
  - **Streaming** — long texts chunked and concatenated
  - **Batch** — one .wav per line as a zip
  - **VRAM** — free loaded model
  - **Help** — 30 languages, 4 modes, tuning knobs, benchmarks
  - **Two model versions**: VoxCPM2 (default, 2B, 30 langs, 48 kHz)
    and VoxCPM-0.5B (legacy, 0.5B, ZH/EN, 16 kHz)
  - **Optional text normalization** (WeTextProcessing) and
    **optional ZipEnhancer denoiser** for noisy refs
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
- `TripoSplat_Colab.ipynb` — **mesh quality completely overhauled**. The
  triangle mesh outputs (.glb / .fbx / .obj / _mesh.ply) were
  unusable due to four compounding problems:
    1. Poisson `scale=1.1` was clipping the surface to the unit-cube
       bbox. Now `scale=1.3` (30% padding) for proper implicit surface.
    2. `density_quantile=0.01` was cropping the mesh and creating holes.
       Now default 0.0 (no cropping; user post-processes if they want).
    3. `depth=10` (1024^3 octree) was over-fitting to noise. Now 8
       (256^3 octree, right resolution for 200k-point clouds).
    4. `max_faces=100_000` was over-decimating. Now 200_000.
  Plus two new pre-processing steps that fix the root cause of bad
  3DGS-derived meshes:
    - **Statistical outlier removal** (k=20, std=2.0) — drops the noisy
      edge Gaussians that 3DGS sometimes produces.
    - **Voxel pre-downsample** (auto-scaled to 0.5% of bbox) — evens out
      the 3DGS density gradient (denser at subject center, sparser at
      edges) which is what breaks Poisson.
  And **3 reconstruction methods** to choose from:
    - `alpha_shape` (default) — sharp, open surface, no "balloon"
      problem. Best for organic 3DGS subjects (people, animals).
    - `poisson` — smooth water-tight. Best for closed shapes.
    - `ball_pivoting` — robust to noise, good middle ground. New
      addition.
  All three methods exposed in Step 4 UI radio + Step 6 + Step 7 params.
- `TripoSplat_Colab.ipynb` — **T4 OOM crash after LOD chain**: the OBJ
  export was the #1 source of silent kernel kills even with
  `write_ascii=False` (open3d's OBJ writer can blow RAM during the
  encode step). OBJ is now **skipped by default**; production formats
  (GLB + FBX + mesh-PLY) are unaffected. Lowered `density_quantile`
  from 0.05 → 0.01 (was chopping off thin extremities, leaving holes
  around hands/feet/fur). Switched `linear_fit=False` → `True` for
  better Poisson vertex placement. Default `max_faces` 300_000 → 100_000
  so decimation triggers on typical 3DGS output (T4-safe). Bumped
  `max_points` 100_000 → 200_000 in QuickTest and Batch so Poisson
  gets a denser input cloud. Added `print_progress=False` to all
  open3d write calls. Beefed up the cleanup block to also trigger
  open3d TriangleMesh finalizers. Added missing `_lod_to_glb()`
  helper that the LOD chain was calling (function was defined nowhere
  — the call was raising `NameError` silently inside the try/except).
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
- README: tagline no longer mentions "image editing" (we don't have any)
  — now reads "3D generation, video creation, text-to-speech, and voice
  conversion"
- README: **Why This Exists** updated to mention all 4 modalities
  (3D + TTS + video + voice conversion) instead of just 3D + TTS
- README: TTS loader size estimate bumped from 35-40 GB to 40-45 GB
  (added VoxCPM2 + OpenVoice V2 weights)
- README: Notebook Overview table now includes the **VoxCPM2** row
  (was missing despite the per-notebook section existing)
- README: "30 multilingual (VoxCPM2)" → "30 languages (VoxCPM2)" in
  the TTS intro for clarity

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

### Changed
- **Gradio pinned version bumped from `5.33.0` to `5.49.1` in 15 notebooks**
  (16 occurrences). 5.49.1 is the last 5.x release (Oct 2025), includes bug
  fixes and performance improvements, and remains fully API-compatible with
  the 5.33 components we use (`gr.Blocks`, `gr.Image`, `gr.Audio`, `gr.Slider`,
  `gr.Model3D`, `gr.Tabs`, `gr.Progress`, `default_concurrency_limit`, etc.).
  No code changes needed beyond the version string. We deliberately stayed
  on 5.x rather than jumping to 6.x — Gradio 6.x adds MCP/accessibility/render
  perf features we don't use, and would conflict with Colab's default
  preinstalled 5.33. We also chose 5.49.1 over the latest 5.50.0 because
  5.50.0 is the first release of the backport branch (renamed
  `concurrency_limit` → `default_concurrency_limit`) and 5.49.1 is the most
  recent clean 5.x with our exact API. Notebooks affected:
  `Audio_PostProcessor_Colab`, `Dia`, `dots.tts-soar`, `Higgs-Audio`,
  `Hunyuan3D-2.1`, `Hunyuan3D-3`, `IndicF5`, `Kokoro-82M`, `MisoTTS`,
  `MOSS-TTS`, `Notebook_Generator`, `OpenVoice-V2`, `Pixal3D_Colab`,
  `Qwen3-TTS`, `Supertonic-3`, `TTS_Voice_Library`, `TripoSplat_Colab`,
  `VoxCPM2`, `Wan2.2`.
- **5 pre-existing `gr.X(...` syntax errors fixed** (the cells would have
  failed to run in Colab anyway, but `validate.py` was masking them with a
  too-permissive `_fix_empty` heuristic). One per notebook:
  `Audio_PostProcessor_Colab` (q_log missing `)`),
  `Kokoro-82M_Colab` (s_audio / s_log missing `,)`),
  `OpenVoice-V2_Colab` (c_src missing `,` to continue multi-line arg),
  `Qwen3-TTS_Colab` (clone_ref / clone_ref_text / clone_xv missing `,)`),
  `VoxCPM2_Colab` (t1_audio / t3_ref / t3_version / t4_ref / t4_prompt /
  t4_prompt_text / b_ref / t5_log each missing `,)`).
- **`tools/validate.py` `_fix_empty()` hardened**: previously treated any
  blank line as "missing body" and inserted a stray `pass` after control
  flow lines, breaking otherwise-valid multi-line `def foo(\n  arg,\n):`
  signatures. Now walks back through blank lines AND equal-indent signature
  continuation lines to find the true header indent before deciding whether
  a body is missing. Required for our `def synth(\n  ...\n):` style headers
  in MisoTTS, VoxCPM2, Qwen3-TTS, etc. to validate cleanly.

## [1.0.0] — 2026-05-29

### Added
- Initial release
- `Pixal3D_Colab.ipynb` — image-to-3D with PBR textures, 1536 px resolution
- `Pixal3D_Wheel_Builder.ipynb` — builds CUDA extension wheels for
  `o_voxel`, `cumesh`, `flex_gemm`, `nvdiffrec_render` and uploads to
  GitHub Releases
- GitHub Release assets: `wheels-a100-v1.0`, `wheels-l4-v1.0`,
  `wheels-t4-v1.0` (each ~40 MB, prebuilt for sm80/sm89/sm75)
