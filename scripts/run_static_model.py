#!/usr/bin/env python3
"""CLI entry point for the static Froot-Stein model (docs/quant-model.md 5.1).

Usage: uv run python scripts/run_static_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.static import main

if __name__ == "__main__":
    main()
