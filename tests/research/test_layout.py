"""Every research entry point imports and has a `main` (CLAUDE.md: a script
nothing imports can be broken for days)."""

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_every_research_entry_point_imports():
    scripts = sorted((ROOT / "scripts" / "research").glob("*.py"))
    assert scripts
    for path in scripts:
        module = importlib.import_module(f"scripts.research.{path.stem}")
        assert hasattr(module, "main"), path.name
