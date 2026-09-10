# Performance / lightweighting ideas (deferred)

Ideas identified while auditing Phase 1 for weight/speed, deliberately **not**
implemented yet -- revisit once the port is fully stable (all phases landed,
real workloads exercised) since they carry more risk than the quick wins
already applied (lazy matplotlib/IPython imports, `model_copy` instead of
full `Structure` re-validation in `Node.update_coords`).

## 1. Replace/reduce the OpenBabel dependency via qcinf (biggest lever, needs research)

`openbabel.pybel` is the single heaviest import in the whole package --
measured at ~575ms of cumulative import time (`python -X importtime`), out of
~1.5-1.8s total for `import mepd; import mepd.engines`. It's there because
`mepd/qcdata_structure_helpers.py` and `mepd/OBH.py` do all structure<->
molecular-graph conversion (bond perception, `Molecule` construction) through
OpenBabel.

`qcinf` (already a dependency, already pulls in `rdkit`) *might* already
cover structure->connectivity-graph perception natively, which could let us
drop `openbabel-wheel` entirely -- saving the import cost and a whole heavy
binary dependency. This was flagged during the Phase 1 port
(`qcdata_structure_helpers.py`'s OpenBabel-based conversion was identified as
"a candidate for future qcinf-based simplification" but not swapped, since
a wrong swap changes chemistry outcomes, not just speed) and flagged again
during this perf pass.

**Before touching this**: audit exactly what `qcinf` provides for bond
perception/connectivity (`qcinf.algorithms.*`), confirm it produces
equivalent bond orders/connectivity to the current OpenBabel-based
`structure_to_molecule`/`molecule_to_structure` on a representative set of
real molecules (not just water), and only then consider swapping the
backend -- ideally behind a flag first, not a hard cutover.

## 2. Make `Node`'s molecular graph lazy instead of eager (real, measured win, deferred by choice)

`Node.update_coords()` used to (and still does, as of this writing) rebuild
the full OpenBabel-based connectivity graph via `structure_to_molecule` on
*every* coordinate update -- i.e. every optimizer step, for every image in
the chain. Measured at ~0.4-0.5ms/call on a 17-atom system. The graph is only
actually read at specific checkpoints (elementary-step/endpoint-identity
comparisons), not on every intermediate optimizer step that gets discarded.

Idea: make `Node.graph` a cached property that's invalidated (not
recomputed) on `update_coords`, and only actually rebuilt on next access.
This is a real win -- proportionally larger for fast engines (toy potentials,
ML potentials) where the graph rebuild can dominate over an actually-cheap
gradient call -- but touches a fairly central code path (`has_molecular_graph`
/ `disable_molecular_graphs` / serialization all interact with `.graph`), so
it deserves its own change with graph-correctness test coverage, once the
rest of the port (MSMEP, network-completion, TS optimization) has landed and
we're not still shaking out correctness bugs in the same code.

## Already done (for reference, not ideas)

- Lazy `matplotlib.pyplot`/`IPython.display` imports (moved from module top
  level into the specific plotting/animation functions that use them, across
  `neb.py`, `chain.py`, `chainhelpers.py`, `pathminimizers/pathminimizer.py`).
  Measured ~290ms removed from cumulative import self-time.
- `Node.update_coords` uses `structure.model_copy(update={"geometry": ...})`
  instead of `Structure(**dict)`, skipping full pydantic re-validation on
  every coordinate update. Verified bit-identical NEB output before/after on
  the real g-xTB oxy-Cope run.
