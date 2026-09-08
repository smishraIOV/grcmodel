#!/usr/bin/env python3
"""CLI entry point for the static Froot-Stein model (docs/quant-model.md 5.1).

Usage: uv run python scripts/run_static_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.static import run_convexity_sweep
from quant.device import get_device

if __name__ == "__main__":
    device = get_device()
    print(f"device: {device}")
    print(f"{'convexity':>10} | {'optimal g':>10} | {'firm value':>10}")
    for convexity, g_star, value_star in run_convexity_sweep():
        print(f"{convexity:>10.2f} | {g_star:>10.4f} | {value_star:>10.4f}")
