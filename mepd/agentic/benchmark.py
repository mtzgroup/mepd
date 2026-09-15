"""The fixed 3-reaction benchmark `mepd agentic tune` scores candidate
parameter sets against: a small oxy-Cope rearrangement, a Wittig reaction,
and a Dakin rearrangement -- chosen as a size/complexity spread (17-47
atoms) that's still cheap enough to re-run every tuning iteration.

The benchmark xyz files live in sibling directories of this repo (under
`neb-dynamics/`, not `neb-dynamics/mepd/`) -- a fragile assumption tied to
this exact checkout layout, overridable via `MEPD_AGENTIC_DATA_ROOT` if the
repo is laid out differently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _data_root() -> Path:
    override = os.environ.get("MEPD_AGENTIC_DATA_ROOT")
    if override:
        return Path(override)
    # This file lives at mepd/mepd/agentic/benchmark.py: parents[0]=agentic,
    # [1]=mepd (the importable package), [2]=mepd (the repo root, contains
    # pyproject.toml), [3]=neb-dynamics -- the sibling-directory root the
    # benchmark data actually lives under on this checkout.
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BenchmarkReaction:
    name: str
    start: Path
    end: Path


def _reactions(root: Path) -> list[BenchmarkReaction]:
    return [
        BenchmarkReaction(
            name="oxycope",
            start=root / "publication-data" / "oxycope" / "start_oxycope.xyz",
            end=root / "publication-data" / "oxycope" / "end_oxycope.xyz",
        ),
        BenchmarkReaction(
            name="wittig",
            start=root / "publication-data" / "wittig" / "start_wittig_ph.xyz",
            end=root / "publication-data" / "wittig" / "end_wittig_ph.xyz",
        ),
        BenchmarkReaction(
            name="dakin",
            start=root / "examples" / "charla_pr_data" / "start_dakin.xyz",
            end=root / "examples" / "charla_pr_data" / "end_dakin.xyz",
        ),
    ]


BENCHMARK_REACTIONS: list[BenchmarkReaction] = _reactions(_data_root())


def get_reaction(name: str) -> BenchmarkReaction:
    for reaction in BENCHMARK_REACTIONS:
        if reaction.name == name:
            return reaction
    available = ", ".join(r.name for r in BENCHMARK_REACTIONS)
    raise ValueError(f"Unknown benchmark reaction {name!r}. Available: {available}.")


def validate_benchmark_data() -> list[str]:
    """Human-readable problems (missing files); empty if every benchmark
    reaction's endpoints are present."""
    problems = []
    for reaction in BENCHMARK_REACTIONS:
        for label, fp in (("start", reaction.start), ("end", reaction.end)):
            if not fp.exists():
                problems.append(f"{reaction.name}: {label} endpoint not found at {fp}")
    return problems
