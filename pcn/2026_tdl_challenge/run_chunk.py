"""Run one chunk of the TDL challenge grid and persist its partial results.

The challenge notebook executes the whole 72-run grid inside a single
``run_challenge_grid`` call that accumulates results in memory only, so
any interruption discards every completed run. This runner calls the
exact same function with the same defaults, but restricted to one
(experiment mode, train seed) pair -- 12 runs -- and writes that chunk's
results to disk as soon as it finishes.

Per-run behaviour is unchanged: ``run_challenge_grid`` re-seeds via
``pl.seed_everything`` before every run and derives each run's Hydra
overrides solely from (setting, seed, model config), so a run yields the
same result regardless of which process executes it. ``merge_chunks.py``
concatenates the six chunks in canonical order and hands the combined
list to ``save_challenge_artifacts``.

Usage
-----
python run_chunk.py --mode community_detection --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CHALLENGE_DIR = Path(__file__).resolve().parent
_REPO = _CHALLENGE_DIR.parent
for _p in (str(_REPO), str(_CHALLENGE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import (  # noqa: E402
    DEFAULT_WANDB_PROJECT_CD,
    DEFAULT_WANDB_PROJECT_TRI,
    _json_sanitize,
    resolve_project_root,
    run_challenge_grid,
)

MODEL_CONFIG = "cell/pcn"
STUDY_PREFIX = "pcn_chunked"

#: Mirrors the default ``experiment_modes`` of ``run_challenge_grid``.
MODES: dict[str, tuple[str, str, str, list[str]]] = {
    "community_detection": (
        "community_detection",
        DEFAULT_WANDB_PROJECT_CD,
        "dataset=graph/graphuniverse_inductive",
        [],
    ),
    "triangle_counting": (
        "triangle_counting",
        DEFAULT_WANDB_PROJECT_TRI,
        "dataset=graph/graphuniverse_inductive_triangle",
        [],
    ),
}

CHUNK_DIR = _CHALLENGE_DIR / "outputs" / "pcn_chunks"


def chunk_path(mode: str, seed: int) -> Path:
    """Return the on-disk location of a chunk's partial results.

    Parameters
    ----------
    mode : str
        Experiment mode name.
    seed : int
        Training seed of the chunk.

    Returns
    -------
    pathlib.Path
        Path of the chunk's JSON file.
    """
    return CHUNK_DIR / f"chunk__{mode}__s{seed}.json"


def main() -> None:
    """Run the requested chunk unless its results already exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = chunk_path(args.mode, args.seed)
    if out_path.exists():
        print(f"chunk already complete, skipping: {out_path}", flush=True)
        return

    study_id = f"{STUDY_PREFIX}__{args.mode}__s{args.seed}"
    results, resolved_study_id = run_challenge_grid(
        project_root=resolve_project_root(_REPO),
        model_config=MODEL_CONFIG,
        experiment_modes=[MODES[args.mode]],
        train_seeds=(args.seed,),
        study_id=study_id,
        quiet=True,
    )

    payload = {
        "mode": args.mode,
        "seed": args.seed,
        "study_id": resolved_study_id,
        "model_config": MODEL_CONFIG,
        "n_runs": len(results),
        "results": results,
    }
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(_json_sanitize(payload), indent=2), encoding="utf-8"
    )
    tmp_path.replace(out_path)
    print(f"chunk complete: {out_path} ({len(results)} runs)", flush=True)


if __name__ == "__main__":
    main()
