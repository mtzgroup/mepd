"""IRC (intrinsic reaction coordinate) path computation via geomeTRIC.

Extracted from the legacy `retropaths_workflow` module: this logic is
generic transition-state/IRC machinery with no retropaths-specific
coupling, it was simply defined alongside that module's network-growth
code upstream.
"""
from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM

from mepd.chain import Chain
from mepd.engines.engine import build_hessian_result_from_matrix
from mepd.nodes.node import StructureNode


def _compute_hessian_result_for_sampling(
    engine: Any,
    node: StructureNode,
    *,
    use_bigchem: bool | None = None,
) -> Any:
    if hasattr(engine, "_compute_hessian_result"):
        if use_bigchem is not None:
            with contextlib.suppress(Exception):
                import inspect

                signature = inspect.signature(engine._compute_hessian_result)
                params = signature.parameters.values()
                accepts_use_bigchem = "use_bigchem" in signature.parameters
                accepts_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in params
                )
                if accepts_use_bigchem or accepts_kwargs:
                    return engine._compute_hessian_result(
                        node,
                        use_bigchem=bool(use_bigchem),
                    )
        return engine._compute_hessian_result(node)
    hessian = np.asarray(engine.compute_hessian(node), dtype=float)
    return build_hessian_result_from_matrix(node=node, hessian=hessian)


def _extract_cartesian_hessian_matrix(
    hessres: Any,
    *,
    natoms: int | None = None,
) -> np.ndarray | None:
    candidates: list[Any] = []

    if hessres is None:
        return None

    candidates.append(getattr(hessres, "return_result", None))
    results = getattr(hessres, "results", None)
    if results is not None:
        candidates.append(getattr(results, "hessian", None))
        candidates.append(getattr(results, "return_result", None))
    if isinstance(hessres, dict):
        candidates.append(hessres.get("hessian"))
        candidates.append(hessres.get("return_result"))
        nested_results = hessres.get("results")
        if isinstance(nested_results, dict):
            candidates.append(nested_results.get("hessian"))
            candidates.append(nested_results.get("return_result"))

    ncart = (3 * int(natoms)) if natoms is not None else None
    for candidate in candidates:
        if candidate is None:
            continue
        with contextlib.suppress(Exception):
            arr = np.asarray(candidate, dtype=float)
            if arr.ndim == 1 and ncart is not None and arr.size == ncart * ncart:
                arr = arr.reshape((ncart, ncart))
            if arr.ndim != 2:
                continue
            if ncart is not None and arr.shape != (ncart, ncart):
                continue
            return arr
    return None


def compute_irc_chain_with_geometric(
    engine: Any,
    ts_node: StructureNode,
    *,
    keywords: dict[str, Any] | None = None,
    use_bigchem: bool = False,
) -> Chain:
    try:
        import geometric.engine  # type: ignore
        import geometric.molecule  # type: ignore
        import geometric.optimize  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Refinement IRC requires the `geometric` package."
        ) from exc

    class _GeometricRefineEngine(geometric.engine.Engine):  # type: ignore[name-defined]
        def __init__(self, molecule: Any, base_engine: Any, template_node: StructureNode):
            super().__init__(molecule)
            self.base_engine = base_engine
            self.template_node = template_node

        def copy_scratch(self, src: str, dest: str) -> None:
            del src, dest
            return None

        def calc_new(self, coords: Any, dirname: str) -> dict[str, Any]:
            del dirname
            updated = self.template_node.copy().update_coords(np.array(coords, dtype=float).reshape((-1, 3)))
            updated.has_molecular_graph = False
            updated.graph = None
            energy = float(self.base_engine.compute_energies([updated])[0])
            gradient = np.asarray(self.base_engine.compute_gradients([updated])[0], dtype=float)
            return {
                "energy": energy,
                "gradient": gradient.reshape(-1) * BOHR_TO_ANGSTROM,
            }

    irc_keywords: dict[str, Any] = {"irc": True, "coordsys": "dlc", "maxiter": 500}
    if keywords:
        irc_keywords.update(dict(keywords))
    irc_keywords["irc"] = True

    molecule = geometric.molecule.Molecule()  # type: ignore[name-defined]
    molecule.elem = list(ts_node.structure.symbols)
    # qcdata stores Structure geometry in Bohr; geomeTRIC Molecule expects Angstrom.
    molecule.xyzs = [np.array(ts_node.structure.geometry, dtype=float) * BOHR_TO_ANGSTROM]
    # GeomeTRIC IRC path expects fragment/topology metadata (`molecules`) to be present.
    with contextlib.suppress(Exception):
        molecule.build_topology(force_bonds=False)
    if not hasattr(molecule, "molecules"):
        with contextlib.suppress(Exception):
            molecule.build_topology()
    if not hasattr(molecule, "molecules"):
        raise RuntimeError(
            "Failed to initialize geomeTRIC molecule topology for IRC (missing `molecules`)."
        )

    ref_node = ts_node.copy()
    ref_node.has_molecular_graph = False
    ref_node.graph = None
    custom_engine = _GeometricRefineEngine(
        molecule=molecule,
        base_engine=engine,
        template_node=ref_node,
    )

    hessian_tmp_path: Path | None = None
    if bool(use_bigchem):
        irc_keywords.setdefault("bigchem", True)
        if "hessian" not in irc_keywords:
            try:
                hessres = _compute_hessian_result_for_sampling(
                    engine,
                    ref_node,
                    use_bigchem=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to compute a BigChem Hessian for IRC initialization."
                ) from exc
            hessian = _extract_cartesian_hessian_matrix(
                hessres,
                natoms=len(getattr(ref_node.structure, "symbols", []) or []),
            )
            if hessian is None:
                raise RuntimeError(
                    "BigChem Hessian result did not contain a usable Cartesian Hessian matrix."
                )
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as hess_file:
                np.savetxt(hess_file.name, hessian)
                hessian_tmp_path = Path(hess_file.name)
            irc_keywords["hessian"] = f"file:{hessian_tmp_path}"

    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmpf:
            output = geometric.optimize.run_optimizer(  # type: ignore[name-defined]
                customengine=custom_engine,
                input=tmpf.name,
                **irc_keywords,
            )
    finally:
        if hessian_tmp_path is not None:
            with contextlib.suppress(Exception):
                hessian_tmp_path.unlink()
    xyzs = list(getattr(output, "xyzs", []) or [])
    if len(xyzs) < 2:
        raise RuntimeError("Geometric IRC did not return a usable trajectory.")

    irc_nodes: list[StructureNode] = []
    for coords in xyzs:
        node = ref_node.copy().update_coords(np.array(coords, dtype=float) * ANGSTROM_TO_BOHR)
        node.has_molecular_graph = False
        node.graph = None
        irc_nodes.append(node)
    return Chain.model_validate({"nodes": irc_nodes})


def optimize_ts_and_irc(engine: Any, ts_guess: StructureNode):
    """Optimize `ts_guess` to a TS with `engine` and follow its IRC.

    Returns `(ts_node, irc_chain)`; `ts_node` is None if the engine can't
    optimize transition states or the optimization failed, and `irc_chain`
    is None if the IRC failed. Never raises."""
    compute_ts = getattr(engine, "compute_transition_state", None)
    if not callable(compute_ts):
        return None, None
    try:
        ts_node = compute_ts(node=ts_guess)
    except Exception:
        return None, None
    if not isinstance(ts_node, StructureNode):
        return None, None
    try:
        irc_fn = getattr(engine, "compute_irc_chain", None)
        irc_chain = irc_fn(ts_node) if callable(irc_fn) else compute_irc_chain_with_geometric(engine, ts_node)
    except Exception:
        return ts_node, None
    return ts_node, irc_chain


def irc_connects(irc_chain: Chain, start: StructureNode, end: StructureNode) -> bool:
    """Whether the IRC's two ends are `start` and `end` (same molecules,
    ignoring conformation), in either direction."""
    from mepd.nodes.nodehelpers import _connectivity_matches

    a, b, s, e = (
        StructureNode(structure=node.structure)
        for node in (irc_chain[0], irc_chain[-1], start, end)
    )
    return (_connectivity_matches(a, s) and _connectivity_matches(b, e)) or (
        _connectivity_matches(a, e) and _connectivity_matches(b, s)
    )
