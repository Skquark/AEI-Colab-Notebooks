# Contributing

Thanks for your interest in the AEI Colab Notebooks project! This is a
collection of self-contained Google Colab notebooks for AI-powered 3D
generation and text-to-speech. Contributions are welcome.

## How to add a new notebook

The fastest way to add a new model is to use [`Notebook_Generator.ipynb`](./Notebook_Generator.ipynb):

1. Open it in Colab, run Steps 1-4, fill in the spec form (or pick a preset), and click **Generate**.
2. The generator emits a fully-scaffolded 9-cell notebook to `/content/<Name>_Colab.ipynb`.
3. Open the new notebook, fill in the model-specific `synth_*` / `gen_*` / `infer` function in Step 3, and the UI works.

Otherwise, every notebook in this repo follows the same 7-step pattern:

1. **STEP 1** — Install dependencies (form cell, `cellView="form"`)
2. **STEP 2** — Pre-cache model weights to Google Drive
3. **STEP 3** — Core functions (lazy model loading)
4. **STEP 4** — Gradio UI (`gr.Blocks`, with `info=` tooltips and `gr.Examples`)
5. **Step 5** — Keep alive (background `requests.get` thread)
6. **Step 6** — Quick test (one-off inference, no UI)
7. **Step 7** — Batch synthesis (wrap the per-line call in `try/except`)

A `TTS_Model_Loader.ipynb` entry is required so the new notebook can be
pre-cached once. If your notebook is voice-cloning-capable, also add a
row to `TTS_Voice_Library.ipynb` for reference clips.

## Coding conventions

- **Notebook metadata** must include `cellView="form"` on every `@title`
  form cell so Colab renders the form view by default.
- **Markdown cell at index 1** is the long-form blurb — explain what the
  model does, who made it, and the license. Use the optional **More info**
  section (HF card / GitHub / arXiv / BibTeX citation) so users can
  dig deeper without leaving the notebook.
- **Cell IDs** (in `cell.metadata.id`) must be stable and descriptive.
  Use the following scheme for the 7-step pattern:

  | Step | ID |
  |------|----|
  | STEP 1 — Install Dependencies | `step1-install` |
  | STEP 2 — Pre-cache Models | `step2-cache` |
  | STEP 3 — Core Functions | `step3-core` |
  | STEP 4 — Gradio UI | `step4-ui` |
  | Step 5 — Keep Alive | `step5-keepalive` |
  | Step 6 — Quick Test | `step6-quicktest` |
  | Step 7 — Batch Synthesis | `step7-batch` |

  Markdown cells should use `view-in-github` (the Colab badge cell) and
  `header` (the long-form blurb cell).
- **Launch pattern** must end with:
  ```python
  from IPython.display import clear_output as _clear
  _clear()
  demo.queue(max_size=8, concurrency_limit=2).launch(
      share=True, show_error=True, server_name="0.0.0.0", server_port=7860,
  )
  demo.load(lambda: "Your model ready message here.", outputs=[status_md])
  ```
- **Reproducibility**: wire `seed` through the Gradio UI, Step 6, and
  Step 7, and call `torch.manual_seed` inside the `synth()` function when
  the seed is non-negative.
- **Error handling**: every batch loop iteration must be wrapped in
  `try/except Exception as e: print(f"  -> EXCEPTION: {e}"); continue`.
- **Help text**: every `gr.Slider` should have an `info=` parameter
  explaining what the knob does.
- **Examples**: every Gradio UI should have a `gr.Examples` block with
  at least 3 click-to-load rows.

## Before submitting

1. Validate the new notebook with the project tools:
   ```bash
   python3 tools/validate.py     # JSON + AST-parse every code cell
   python3 tools/qa_check.py     # full polish audit (info tooltips, concurrency, etc)
   ```
   Both must exit 0 for the notebook to be mergeable.
2. Run the notebook in Colab on a fresh runtime, all cells green.
3. Add a row to the main `README.md` "Notebook Overview" table.
4. Add a row to the "Model Cards & Upstream Attribution" table in the README.
5. (If voice-cloning capable) register a row in `TTS_Voice_Library.ipynb`'s
   `VOICES` list.

## Reporting bugs

Open an issue on GitHub. Include:
- The notebook name
- The GPU you ran on (A100 / L4 / T4 / CPU)
- The error message and the cell number
- Steps to reproduce

## License

By contributing, you agree that your contributions will be licensed under
the project's MIT license (see `LICENSE`). Third-party model weights
remain under their original licenses.
