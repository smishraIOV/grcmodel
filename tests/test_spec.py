"""`docs/model-spec.md` is the specification; this is what stops it drifting.

Every document in this repo is written to be checked, and the specification is
the one that most needs it: a symbol table is exactly the kind of thing that
stays plausible for a year after the code moved underneath it. The failure is
silent by construction -- nothing errors, a reader simply believes a default
that has not been true for four commits.

So the partition is enforced in both directions. Every parameter named in the
spec must exist with the value quoted; every parameter that exists must either
be in the spec or be excused by name in its section 10. There is no third
option, which is the point: a new parameter cannot be added without a decision
about whether it belongs in the specification.

What this cannot check is whether the *equations* say what the code does. That
needs a reader. What it can check is that the vocabulary is real, which is the
part that rots without anyone noticing.
"""

import re
from dataclasses import fields
from pathlib import Path

import pytest

import quant.params as params

SPEC = Path(__file__).resolve().parent.parent / "docs" / "model-spec.md"

# The dataclasses the specification is about. `ModelParams` is a container and
# carries no scalars of its own.
PARAMETER_CLASSES = (
    "FirmParams",
    "FundingParams",
    "HazardParams",
    "CliffParams",
    "GrcAlphas",
    "AnnualRates",
    "SamplerParams",
    "ShockParams",
)


def spec_text() -> str:
    assert SPEC.exists(), f"the specification has moved or gone: {SPEC}"
    return SPEC.read_text()


def cited() -> dict[str, str]:
    """`{"FundingParams.deposit_capacity": "4.0"}` from the notation tables.

    A cited row is one whose fourth cell is a backticked `Class.field` and whose
    fifth is the default. Rows using an em dash for both -- the derived and
    state quantities -- carry no parameter and are skipped.
    """
    found = {}
    for line in spec_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        match = re.fullmatch(r"`([A-Z][A-Za-z]*\.[a-z_]+)`", cells[3])
        if match:
            found[match.group(1)] = cells[4]
    return found


def excused() -> set[str]:
    """Parameters section 10 explicitly declines to specify, with a reason."""
    body = spec_text().split("## 10.")[-1]
    out = set()
    for line in body.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or not cells[1]:
            continue
        match = re.fullmatch(r"`([A-Z][A-Za-z]*\.(?:[a-z_]+|\*))`", cells[0])
        if match:
            out.add(match.group(1))
    return out


def live() -> set[str]:
    """Every scalar parameter the code actually carries."""
    return {
        f"{name}.{field.name}"
        for name in PARAMETER_CLASSES
        for field in fields(getattr(params, name))
    }


def test_the_specification_cites_real_parameters():
    """Every `Class.field` the spec names has to exist."""
    missing = sorted(name for name in cited() if not _resolve(name))
    assert not missing, f"specification names parameters that do not exist: {missing}"


def test_the_quoted_defaults_are_the_actual_defaults():
    """...and the value beside it has to be the value.

    This is the row that rots. A default moves in a recalibration, the table
    keeps the old number, and the document reads as though nothing happened.
    """
    wrong = []
    for name, quoted in cited().items():
        actual = _resolve(name)
        if actual is None:
            continue
        if not _agrees(quoted, actual):
            wrong.append(f"{name}: spec says {quoted}, code says {actual}")
    assert not wrong, "specification is stale:\n  " + "\n  ".join(wrong)


def test_every_parameter_is_specified_or_excused():
    """The other direction, and the one that catches a *new* parameter.

    Adding a field to a parameter dataclass without deciding whether it belongs
    in the specification is how a specification stops being one. Either document
    it in section 1 or excuse it by name in section 10.
    """
    documented = set(cited())
    excuses = excused()
    wildcards = {e.split(".")[0] for e in excuses if e.endswith(".*")}

    orphans = sorted(
        name
        for name in live()
        if name not in documented
        and name not in excuses
        and name.split(".")[0] not in wildcards
    )
    assert not orphans, (
        "parameters neither specified nor excused:\n  "
        + "\n  ".join(orphans)
        + "\n\nAdd each to docs/model-spec.md section 1, or to its section 10 "
        "with a reason."
    )


def test_nothing_is_both_specified_and_excused():
    """A parameter cannot be in the specification and declared out of it."""
    both = sorted(set(cited()) & excused())
    assert not both, f"specified and excused at once: {both}"


def test_excuses_refer_to_parameters_that_exist():
    """An excuse for a parameter that has been deleted is dead text, and it is
    the kind that survives for years because nothing reads it."""
    classes = {name.split(".")[0] for name in live()}
    stale = sorted(
        e
        for e in excused()
        if (e.split(".")[0] not in classes)
        or (not e.endswith(".*") and e not in live())
    )
    assert not stale, f"section 10 excuses parameters that no longer exist: {stale}"


@pytest.mark.parametrize("family", ["credit", "operational", "compliance"])
def test_the_channel_counts_agree_with_the_measured_ones(family):
    """Section 6 quotes a channel count per family. tests/test_alphas.py measures
    it. They have to be the same number, or one of the two is lying."""
    from tests.test_alphas import channels_driven_by

    row = re.search(
        rf"^\|\s*{family}\s*\|\s*\*\*(\d+)\*\*", spec_text(), re.M
    )
    assert row, f"section 6 has no channel-count row for {family}"
    assert int(row.group(1)) == len(channels_driven_by(family))


def _resolve(dotted: str):
    """The default of `Class.field`, or None if either half is missing."""
    class_name, _, field_name = dotted.partition(".")
    cls = getattr(params, class_name, None)
    if cls is None:
        return None
    for field in fields(cls):
        if field.name == field_name:
            return field.default
    return None


def _agrees(quoted: str, actual) -> bool:
    try:
        return float(quoted) == pytest.approx(float(actual))
    except (TypeError, ValueError):
        return quoted == str(actual)
