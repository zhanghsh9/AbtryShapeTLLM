#!/usr/bin/env python3
"""
Convert metasurface fine‑tuning data (JSON) to MATLAB .mat files.

Each JSON record must have:
  - "instruction": text containing a 4×4 grid like [[a,b,c,d],[...]]
  - "output":      text containing a 31‑value list like [t1,t2,…,t31]

Created: 4May2025
"""

import json
import re
import ast
from pathlib import Path
import numpy as np
from scipy.io import savemat


# --- just replace the two regex definitions near the top ---

# Capture the whole 4×4 grid ([[ … ]]) in a *non‑greedy* way, even over line‑breaks
GRID_RE = re.compile(r"\[\[.*?\]\]", re.DOTALL)

# Capture the first plain list of numbers (used on the *output* string only)
LIST_RE = re.compile(r"\[[0-9eE\.\,\s\-]+\]")


def extract_grid(instr: str) -> list[float]:
    """Return flattened 4×4 grid (length 16) from instruction string."""
    match = GRID_RE.search(instr)
    if not match:
        raise ValueError(f"Grid not found in instruction: {instr}")
    grid = ast.literal_eval(match.group(0))          # → list[list[float]]
    return [v for row in grid for v in row]          # flatten


def extract_spectrum(out: str) -> list[float]:
    """Return 31‑value transmission list from output string."""
    match = LIST_RE.search(out)
    if not match:
        raise ValueError(f"Spectrum not found in output: {out}")
    return list(ast.literal_eval(match.group(0)))    # → list[float]


def process(json_path: Path, mat_path: Path) -> None:
    """Convert one JSON file → .mat file with matrices `grids` and `TL`."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)              # expect a list of objects

    grids, spectra = [], []
    for record in data:
        grids.append(extract_grid(record["instruction"]))
        spectra.append(extract_spectrum(record["output"]))

    grids_arr   = np.asarray(grids,   dtype=float)   # shape: n × 16
    spectra_arr = np.asarray(spectra, dtype=float)   # shape: n × 31

    savemat(mat_path, {"grids": grids_arr, "TL": spectra_arr})
    print(f"Saved {mat_path}  (grids {grids_arr.shape}, TL {spectra_arr.shape})")


def main() -> None:
    process(Path("test.json"),  Path("test.mat"))
    process(Path("train.json"), Path("train.mat"))


if __name__ == "__main__":
    main()
