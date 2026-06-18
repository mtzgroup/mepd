# MEPD

Minimum Energy Path Discovery tools for reaction-path calculations.

This public repository contains:

- NEB and related path minimizers
- MLPGI path support
- MSMEP recursive path splitting
- elementary-step analysis utilities
- engine, chain, node, optimizer, and visualization helpers needed by those workflows

## Installation

```bash
pip install "git+https://github.com/mtzgroup/mepd.git"
```

For local development:

```bash
uv sync
uv run pytest
```

## Quick Start

```python
from mepd import Chain, ChainInputs, NEB, NEBInputs, StructureNode
from mepd.engines.qcop import QCOPEngine
from mepd.optimizers.cg import ConjugateGradient
import mepd.chainhelpers as ch
from qcio import Structure

start = Structure.from_xyz("start.xyz")
end = Structure.from_xyz("end.xyz")

engine = QCOPEngine(compute_program="chemcloud")

start_node = StructureNode(structure=start)
end_node = StructureNode(structure=end)
start_opt = engine.compute_geometry_optimization(start_node)[-1]
end_opt = engine.compute_geometry_optimization(end_node)[-1]

chain = Chain.model_validate({
    "nodes": [start_opt, end_opt],
    "parameters": ChainInputs(k=0.1, delta_k=0.09),
})
initial_chain = ch.run_geodesic(chain, nimages=15)

neb = NEB(
    initial_chain=initial_chain,
    parameters=NEBInputs(v=True),
    optimizer=ConjugateGradient(timestep=0.5),
    engine=engine,
)
result = neb.optimize_chain()
```

## CLI

The public CLI exposes path minimization and NEB refinement commands:

```bash
mepd run --start start.xyz --end end.xyz --inputs inputs.toml
mepd run-refine examples/oxycope.xyz -i expensive.toml -ci cheap.toml --mode neb
mepd refine previous_result.xyz --inputs expensive.toml --mode neb
mepd make-default-inputs --name inputs.toml
mepd-elementarystep path.xyz
```

Use the Python API for lower-level NEB, MLPGI, MSMEP, and elementary-step workflows.

## Maintainers

For questions, contact:

- [Jan](mailto:jdep@stanford.edu)
- [Alessio](mailto:alevale@stanford.edu)
