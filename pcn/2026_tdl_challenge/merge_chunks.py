"""Merge the six challenge chunks into the final 72-run artifacts.

Concatenates the per-chunk result lists written by ``run_chunk.py`` in
the same order ``run_challenge_grid`` would have produced them (mode
outer, setting middle, seed inner) and hands the combined list to
``save_challenge_artifacts``, which is the only writer of
``results.json`` and the figure outputs.

Usage
-----
python merge_chunks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_CHALLENGE_DIR = Path(__file__).resolve().parent
_REPO = _CHALLENGE_DIR.parent
for _p in (str(_REPO), str(_CHALLENGE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_chunk import MODEL_CONFIG, MODES, chunk_path  # noqa: E402
from utils import (  # noqa: E402
    CHALLENGE_TRAIN_SEEDS,
    save_challenge_artifacts,
)

EXPECTED_RUNS = 72
OUT_DIR = _CHALLENGE_DIR / "outputs" / "pcn"


def load_chunks() -> list[dict[str, Any]]:
    """Load every chunk in canonical grid order.

    Returns
    -------
    list of dict
        The concatenated per-run result rows.

    Raises
    ------
    FileNotFoundError
        If any of the six chunks is missing.
    """
    ordered_modes = ["community_detection", "triangle_counting"]
    assert set(ordered_modes) == set(MODES)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for mode in ordered_modes:
        per_seed: dict[int, list[dict[str, Any]]] = {}
        for seed in CHALLENGE_TRAIN_SEEDS:
            path = chunk_path(mode, seed)
            if not path.exists():
                missing.append(path.name)
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            per_seed[seed] = payload["results"]
        if missing:
            continue
        # Interleave so that seeds vary fastest, matching the ordering of
        # the single-process grid (mode -> setting -> seed).
        n_settings = len(per_seed[CHALLENGE_TRAIN_SEEDS[0]])
        for setting_ix in range(n_settings):
            for seed in CHALLENGE_TRAIN_SEEDS:
                rows.append(per_seed[seed][setting_ix])

    if missing:
        raise FileNotFoundError(
            "missing chunk files: " + ", ".join(sorted(missing))
        )
    return rows


def main() -> None:
    """Merge the chunks and write the final challenge artifacts."""
    rows = load_chunks()
    if len(rows) != EXPECTED_RUNS:
        raise ValueError(
            f"expected {EXPECTED_RUNS} runs, merged {len(rows)}"
        )

    out_paths = save_challenge_artifacts(
        rows,
        out_dir=OUT_DIR,
        model_config=MODEL_CONFIG,
        study_id="pcn_chunked",
    )
    print(f"merged {len(rows)} runs")
    print(f"JSON: {out_paths['json']}")


if __name__ == "__main__":
    main()
