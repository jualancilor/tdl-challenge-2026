# Running the Polynormer benchmark on a free cloud GPU

Everything is packaged into **`polynormer_gpu_benchmark.ipynb`** — a single,
self-contained notebook. Upload it to Kaggle or Colab (GPU runtime) and run all
cells top to bottom. It clones TopoBench, injects our Polynormer files, builds a
Python 3.11 + CUDA env with `uv`, runs the full 72-run grid offline, and writes
`results.json`.

## Recommended: Kaggle (30 GPU-h/week, P100/T4)

1. Go to https://www.kaggle.com/code → **New Notebook** → **File → Import
   Notebook** → upload `polynormer_gpu_benchmark.ipynb`.
2. Right panel → **Settings → Accelerator → GPU P100** (or T4 x2).
   Also turn **Internet: On** (needed to clone the repo + install deps).
3. **Run All**. The env cell takes ~5–15 min; the grid takes a few hours.
4. When done, download `results.json` from `/kaggle/working/TopoBench/`
   (Output/Data panel).

## Alternative: Google Colab (T4)

1. https://colab.research.google.com → **Upload** `polynormer_gpu_benchmark.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**.
4. Last cell: `from google.colab import files; files.download('results.json')`.

## Notes

- **No Weights & Biases login needed** — the run uses `WANDB_MODE=offline`
  (same as the existing Polynormer PR #345).
- **GPU is required.** The challenge's default trainer is GPU
  (`configs/trainer/default.yaml`), and 72 runs × up to 500 epochs is impractical
  on CPU.
- **Session limits:** Kaggle sessions last up to 9 h, Colab up to 12 h. If the
  grid does not finish in one session, do a first pass with a single seed
  (edit `CHALLENGE_TRAIN_SEEDS = (42,)` in `2026_tdl_challenge/utils.py`, → 24
  runs) or run the two tasks separately. Keep all 3 seeds for the official
  submission.
- **Heads-up:** plain "Polynormer" for Track 1 is already an open submission
  (PR #345, team SweetLesson), plus a TopoPolynormer variant (PR #353). If this
  is for a challenge PR rather than coursework, consider differentiating.

## What's in this folder

- `polynormer_gpu_benchmark.ipynb` — the runnable notebook (use this).
- `topobench-repo/` — local TopoBench checkout with our implementation:
  - `topobench/nn/backbones/graph/polynormer.py` — backbone
  - `configs/model/graph/polynormer.yaml` — Hydra config
  - `test/nn/backbones/graph/test_polynormer.py` — 18 unit tests (100% file coverage)
- `polynormer-ref/` — official Polynormer reference code (for comparison).
