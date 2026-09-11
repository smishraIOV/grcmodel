"""Shared entry-point plumbing: profile selection and the provenance header.

Every number this repo prints is conditional on a NumericsProfile, a torch
version and a commit. Printing that line costs nothing and removes a whole
class of "why doesn't this reproduce" — docs/quant-model.md requires quoted
numbers to come from a real run, and this is what identifies the run.
"""

import argparse

from quant.numerics import DEFAULT_PROFILE, PROFILES, NumericsProfile, get_profile


def parse_profile(description: str, argv: list[str] | None = None) -> NumericsProfile:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE.name,
        choices=sorted(PROFILES),
        help=(
            "numerics profile (default: %(default)s). Published numbers must come "
            "from 'reference'; the others trade precision or determinism for speed."
        ),
    )
    return get_profile(parser.parse_args(argv).profile)


def print_header(profile: NumericsProfile) -> None:
    print(f"run: {profile.fingerprint()}")
    if profile is not DEFAULT_PROFILE:
        print(
            f"warning: profile {profile.name!r} is not the reference profile; "
            "these numbers should not be quoted in docs/."
        )
    print()
