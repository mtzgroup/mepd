"""A program-agnostic bundle of calculation arguments.

qcdata <=0.18 shipped `ProgramArgs`: the model/keywords half of a calculation,
with no `program`, no `calctype` and no structure attached. mepd leans on that
split -- `RunInputs.program_kwds` is filled in from TOML long before anyone
knows whether it will be used for an energy, a gradient or a hessian, and
`QCComputeEngine` tracks the program name separately in `Engine.program`.

qcdata 0.19 dropped that model. Its replacement, `ProgramSpec`, requires both
`program` and `calctype`, so it cannot stand in for a partially-specified
bundle. `ProgramArgs` therefore lives here now, and `to_spec()` turns it into
the 0.19 `ProgramSpec` at the point where the program and calctype are finally
known.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from qcdata.models.base_models import CalcType, Model
from qcdata.models.inputs import ProgramSpec


class ProgramArgs(BaseModel):
    """Calculation arguments that are independent of program and calctype."""

    # Mirrors the qcdata model config. `frozen` matters beyond immutability:
    # it makes instances hashable, which is what lets one serve as the default
    # for `QCComputeEngine.program_args` on a stdlib dataclass.
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, frozen=True
    )

    model: Model | None = None
    keywords: dict[str, Any] = {}
    files: dict[str, str | bytes] = {}
    cmdline_args: list[str] = []
    extras: dict[str, Any] = {}

    def to_spec(self, program: str, calctype: CalcType | str) -> ProgramSpec:
        """Bind these arguments to a program and calctype.

        Used to build the `subprograms` entries that qcdata 0.19 expects in
        place of the old `DualProgramInput.subprogram`/`subprogram_args` pair.
        """
        return ProgramSpec(
            program=program,
            calctype=calctype,
            model=self.model,
            keywords=dict(self.keywords),
        )
