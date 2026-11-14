"""The tool's core does not depend on the research code (docs/plan-tool.md
section 1): nothing in `quant/` outside `quant/research/`, and no test outside
`tests/research/`, imports from `quant.research` or the research scripts.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ("quant.research", "scripts.research", "tests.research")


def imported(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_the_core_does_not_import_the_research_code():
    core = [p for p in (ROOT / "quant").glob("*.py")] + [p for p in (ROOT / "tests").glob("*.py")]
    core += [p for p in (ROOT / "scripts").glob("*.py")]
    assert len(core) > 10
    offenders = [
        f"{path.relative_to(ROOT)}: {name}"
        for path in core for name in imported(path) if name.startswith(RESEARCH)
    ]
    assert not offenders, offenders
