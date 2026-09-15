"""Subprocess entrypoint for one benchmark-reaction evaluation, spawned by
`mepd.agentic.evaluate.run_benchmark` -- one reaction per process, so a
pathological candidate parameter set (a hang, a crash) can't take down the
whole tuning loop or the other reactions running alongside it.

Deliberately `argparse`, not typer: this process is spawned once per
reaction per tuning iteration, so a lighter/faster-spawning entrypoint
matters more here than CLI ergonomics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from mepd.agentic import benchmark, evaluate
from mepd.errors import format_exception_message
from mepd.inputs import RunInputs


def _write_result(result_path: Path, result: evaluate.ReactionResult) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(dataclasses.asdict(result)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction", required=True, help="Benchmark reaction name.")
    parser.add_argument("--inputs", required=True, type=Path, help="Candidate RunInputs TOML.")
    parser.add_argument("--result-json", required=True, type=Path, help="Where to write the ReactionResult JSON.")
    parser.add_argument("--parallel-msmep", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        reaction = benchmark.get_reaction(args.reaction)
        run_inputs = RunInputs.open(args.inputs)
        result = evaluate.score_reaction(
            reaction,
            run_inputs,
            parallel_msmep=args.parallel_msmep,
            parallel_workers=args.parallel_workers,
        )
    except Exception as exc:
        # `score_reaction` itself never raises -- this is reserved for
        # genuinely-couldn't-even-start failures (bad CLI args, a
        # malformed inputs TOML, an import error). Still write a result
        # file before re-raising, so the parent's "missing result file"
        # handling stays the rare/exceptional case.
        _write_result(
            args.result_json,
            evaluate.ReactionResult(
                name=args.reaction,
                success=False,
                wall_clock_s=0.0,
                error_type=type(exc).__name__,
                error_msg=format_exception_message(exc),
            ),
        )
        raise

    _write_result(args.result_json, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
