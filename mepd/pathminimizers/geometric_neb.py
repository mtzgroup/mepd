from __future__ import annotations

import contextlib
import inspect
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qcio import Structure

from mepd.chain import Chain
from mepd.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from mepd.elementarystep import ElemStepResults, check_if_elem_step
from mepd.engines.engine import Engine
from mepd.errors import ElectronicStructureError
from mepd.nodes.node import StructureNode
from mepd.pathminimizers.pathminimizer import PathMinimizer
from mepd.scripts.progress import print_persistent, update_status

IS_ELEM_STEP = ElemStepResults(
    is_elem_step=True,
    is_concave=True,
    splitting_criterion=None,
    minimization_results=None,
    number_grad_calls=0,
)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _build_structure_chain(
    xyzs_angstrom: list[np.ndarray],
    *,
    symbols: list[str],
    charge: int,
    multiplicity: int,
    chain_parameters: Any,
) -> Chain:
    nodes = []
    for xyz_ang in xyzs_angstrom:
        struct = Structure(
            symbols=symbols,
            geometry=np.asarray(xyz_ang, dtype=float) * ANGSTROM_TO_BOHR,
            charge=charge,
            multiplicity=multiplicity,
        )
        nodes.append(StructureNode(structure=struct))
    return Chain.model_validate({"nodes": nodes, "parameters": chain_parameters})


def _history_chain_files(workdir: Path) -> list[Path]:
    files = sorted(workdir.glob("chain_*.xyz"))
    if files:
        return files
    return sorted(workdir.rglob("chain_*.xyz"))


def _read_chain_from_xyz_file(
    xyz_fp: Path,
    *,
    charge: int,
    multiplicity: int,
    chain_parameters: Any,
) -> Chain | None:
    text = xyz_fp.read_text(encoding="utf-8")
    structures = Structure.from_xyz_multi(
        text,
        charge=int(charge),
        multiplicity=int(multiplicity),
    )
    if len(structures) == 0:
        return None
    return Chain.model_validate(
        {
            "nodes": [StructureNode(structure=struct) for struct in structures],
            "parameters": chain_parameters,
        }
    )


@dataclass
class GeometricNEB(PathMinimizer):
    initial_chain: Chain
    engine: Engine
    parameters: object | None = None

    optimized: Chain | None = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    grad_calls_made: int = 0
    geom_grad_calls_made: int = 0

    def __post_init__(self):
        self._params = _as_dict(self.parameters)
        self._verbose = bool(self._params.get("v", False))
        self._sp_call_count = 0
        self._batch_cycle_count = 0

    def _status(self, message: str, *, persistent: bool = False) -> None:
        if self._verbose:
            print(message)
            return
        if persistent:
            print_persistent(message=message)
        else:
            update_status(message)

    def _build_geometric_kwargs(self, *, nimages: int) -> dict[str, Any]:
        climb_raw = self._params.get("climb", 0.5)
        if isinstance(climb_raw, bool):
            climb_value = 0.5 if climb_raw else 0.0
        else:
            climb_value = float(climb_raw)

        kwargs: dict[str, Any] = {
            "images": int(self._params.get("images", nimages)),
            "plain": int(self._params.get("plain", 0)),
            "maxg": float(self._params.get("max_rms_grad_thre", self._params.get("maxg", 0.05))),
            "avgg": float(self._params.get("rms_grad_thre", self._params.get("avgg", 0.025))),
            "guessk": float(self._params.get("guessk", 0.05)),
            "guessw": float(self._params.get("guessw", 0.1)),
            "nebk": float(self._params.get("nebk", 1.0)),
            "neb_maxcyc": int(self._params.get("max_steps", self._params.get("neb_maxcyc", 200))),
            "climb": climb_value,
            "ncimg": int(self._params.get("ncimg", 1)),
            "optep": _coerce_bool(self._params.get("optep"), default=False),
            "align": _coerce_bool(self._params.get("align"), default=True),
            "epsilon": float(self._params.get("epsilon", 1e-5)),
            "trust": float(self._params.get("trust", 0.1)),
            "tmax": float(self._params.get("tmax", 0.3)),
            "tmin": float(self._params.get("tmin", 1.2e-3)),
            "verbose": bool(self._params.get("v", False)),
        }
        if self._params.get("prefix") is not None:
            kwargs["prefix"] = str(self._params["prefix"])
        return kwargs

    def _load_history_chains(
        self,
        *,
        workdir: Path,
        charge: int,
        multiplicity: int,
        chain_parameters: Any,
    ) -> list[Chain]:
        chains: list[Chain] = []
        for fp in _history_chain_files(workdir):
            with contextlib.suppress(Exception):
                maybe_chain = _read_chain_from_xyz_file(
                    fp,
                    charge=charge,
                    multiplicity=multiplicity,
                    chain_parameters=chain_parameters,
                )
                if maybe_chain is not None:
                    chains.append(maybe_chain)
        return chains

    @staticmethod
    def _patch_geometric_recover_compat(geometric_neb_module: Any) -> None:
        """Handle geomeTRIC NEB versions where recover() is called with 3 args."""
        recover_fn = getattr(geometric_neb_module, "recover", None)
        if recover_fn is None:
            return
        try:
            params = list(inspect.signature(recover_fn).parameters.values())
        except Exception:
            return
        if len(params) != 2:
            return
        if getattr(recover_fn, "_mepd_recover_compat", False):
            return

        def _recover_compat(chain_hist, maybe_last_force=None, result=None):
            # Older code paths call recover(chain_hist, LastForce, result)
            # while recover is defined as recover(chain_hist, result=None).
            if result is None:
                result = maybe_last_force
            return recover_fn(chain_hist, result=result)

        setattr(_recover_compat, "_mepd_recover_compat", True)
        geometric_neb_module.recover = _recover_compat

    def optimize_chain(self) -> ElemStepResults:
        try:
            import geometric.engine
            import geometric.molecule
            import geometric.neb
            import geometric.params
        except ImportError as exc:
            raise ElectronicStructureError(
                msg="geomeTRIC NEB requires the `geometric` package to be installed.",
                obj=None,
            ) from exc
        self._patch_geometric_recover_compat(geometric.neb)

        work_chain = self.initial_chain.copy()
        chain_parameters = work_chain.parameters.copy()
        charge = int(work_chain[0].structure.charge)
        multiplicity = int(work_chain[0].structure.multiplicity)
        symbols = list(work_chain[0].structure.symbols)

        self.chain_trajectory = [work_chain.copy()]
        with contextlib.suppress(Exception):
            self.engine.compute_energies(work_chain)
            self.grad_calls_made += len(work_chain)

        molecule = geometric.molecule.Molecule()
        molecule.elem = symbols
        molecule.xyzs = [np.asarray(node.coords, dtype=float) * BOHR_TO_ANGSTROMS for node in work_chain]

        # Build simple topology metadata when possible; geomeTRIC can proceed in Cartesian
        # mode without this, but some molecule operations use this attribute when available.
        with contextlib.suppress(Exception):
            molecule.build_topology(force_bonds=False)
        if not hasattr(molecule, "molecules"):
            with contextlib.suppress(Exception):
                molecule.build_topology()

        parent = self

        class _GeometricEngine(geometric.engine.Engine):
            def __init__(self, geom_molecule, base_engine: Engine, template_node: StructureNode):
                super().__init__(geom_molecule)
                self.base_engine = base_engine
                self.template_node = template_node

            def __deepcopy__(self, memo):
                # geomeTRIC NEB deep-copies Chain objects during trust-step updates.
                # Base engines (for example QCOPEngine) can contain thread locks that
                # are not deepcopy/pickle safe, so keep a shared base engine reference.
                from copy import deepcopy

                cloned = self.__class__.__new__(self.__class__)
                memo[id(self)] = cloned
                cloned.M = deepcopy(self.M, memo)
                cloned.stored_calcs = deepcopy(self.stored_calcs, memo)
                cloned.base_engine = self.base_engine
                cloned.template_node = self.template_node
                cloned._mepd_batch_engine_calls = bool(
                    getattr(self, "_mepd_batch_engine_calls", False)
                )
                cloned._mepd_parent = getattr(self, "_mepd_parent", None)
                return cloned

            def copy_scratch(self, src, dest):
                del src, dest
                return None

            def calc_new(self, coords, dirname):
                del dirname
                curr_coords = np.asarray(coords, dtype=float).reshape((-1, 3))
                curr_node = self.template_node.copy().update_coords(curr_coords)
                curr_node.has_molecular_graph = False
                curr_node.graph = None

                parent._sp_call_count += 1
                if parent._sp_call_count == 1 or parent._sp_call_count % 10 == 0:
                    parent._status(
                        f"geomeTRIC NEB: evaluating image gradient #{parent._sp_call_count}",
                        persistent=True,
                    )

                gradient = np.asarray(self.base_engine.compute_gradients([curr_node])[0], dtype=float)
                parent.grad_calls_made += 1

                energy = getattr(curr_node, "_cached_energy", None)
                if energy is None:
                    energy = float(self.base_engine.compute_energies([curr_node])[0])
                    parent.grad_calls_made += 1
                else:
                    energy = float(energy)

                return {
                    "energy": energy,
                    # geomeTRIC expects gradient in Hartree/Angstrom.
                    "gradient": gradient.reshape(-1) * BOHR_TO_ANGSTROMS,
                }

        kwargs = self._build_geometric_kwargs(nimages=len(work_chain))
        params = geometric.params.NEBParams(**kwargs)
        use_batch = _coerce_bool(
            self._params.get("batch_engine_calls"),
            default=True,
        )

        self._status("geomeTRIC NEB: launching optimization")
        try:
            with tempfile.TemporaryDirectory(prefix="geometric-neb-") as td:
                run_dir = Path(td)
                custom_engine = _GeometricEngine(molecule, self.engine, work_chain[0].copy())
                custom_engine._mepd_batch_engine_calls = bool(use_batch)
                custom_engine._mepd_parent = self
                band = geometric.neb.ElasticBand(
                    molecule,
                    engine=custom_engine,
                    tmpdir=str(run_dir),
                    params=params,
                    plain=params.plain,
                )
                original_chain_compute = geometric.neb.Chain.ComputeEnergyGradient

                def _compute_energy_gradient_batched(chain_self, cyc=None, result=None):
                    engine_obj = getattr(chain_self, "engine", None)
                    if (
                        result is not None
                        or engine_obj is None
                        or not bool(getattr(engine_obj, "_mepd_batch_engine_calls", False))
                    ):
                        return original_chain_compute(chain_self, cyc=cyc, result=result)

                    owner = getattr(engine_obj, "_mepd_parent", None)
                    base_engine = getattr(engine_obj, "base_engine", None)
                    template_node = getattr(engine_obj, "template_node", None)
                    if owner is None or base_engine is None or template_node is None:
                        return original_chain_compute(chain_self, cyc=cyc, result=result)

                    owner._batch_cycle_count += 1
                    owner._status(
                        f"geomeTRIC NEB: batched gradient cycle {owner._batch_cycle_count}",
                        persistent=True,
                    )
                    eval_nodes: list[StructureNode] = []
                    for struct in chain_self.Structures:
                        coords_bohr = np.asarray(struct.cartesians, dtype=float).reshape((-1, 3))
                        node = template_node.copy().update_coords(coords_bohr)
                        node.has_molecular_graph = False
                        node.graph = None
                        eval_nodes.append(node)

                    gradients = np.asarray(base_engine.compute_gradients(eval_nodes), dtype=float)
                    owner.grad_calls_made += len(eval_nodes)

                    missing_inds = [
                        i for i, node in enumerate(eval_nodes)
                        if getattr(node, "_cached_energy", None) is None
                    ]
                    if missing_inds:
                        missing_nodes = [eval_nodes[i] for i in missing_inds]
                        missing_energies = base_engine.compute_energies(missing_nodes)
                        owner.grad_calls_made += len(missing_nodes)
                        for idx, ene in zip(missing_inds, missing_energies):
                            eval_nodes[idx]._cached_energy = float(ene)

                    for i, struct in enumerate(chain_self.Structures):
                        grad_cart = gradients[i].reshape(-1) * BOHR_TO_ANGSTROMS
                        ene = float(getattr(eval_nodes[i], "_cached_energy"))
                        struct.ComputeEnergyGradient(
                            result={"energy": ene, "gradient": grad_cart}
                        )
                    if cyc is not None:
                        for struct in chain_self.Structures:
                            struct.engine.number_output(struct.tmpdir, cyc)
                    chain_self.haveCalcs = True

                if use_batch:
                    geometric.neb.Chain.ComputeEnergyGradient = _compute_energy_gradient_batched
                try:
                    final_band, _cycles = geometric.neb.OptimizeChain(band, custom_engine, params)
                finally:
                    if use_batch:
                        geometric.neb.Chain.ComputeEnergyGradient = original_chain_compute

                history = self._load_history_chains(
                    workdir=run_dir,
                    charge=charge,
                    multiplicity=multiplicity,
                    chain_parameters=chain_parameters,
                )

                final_chain = _build_structure_chain(
                    [np.asarray(xyz, dtype=float) for xyz in list(final_band.M.xyzs)],
                    symbols=symbols,
                    charge=charge,
                    multiplicity=multiplicity,
                    chain_parameters=chain_parameters,
                )
                # Ensure the converged chain has energies before any downstream split logic
                # reads `chain.energies` from `chain_trajectory[-1]`.
                self.engine.compute_energies(final_chain)
                self.grad_calls_made += len(final_chain)

                if history:
                    self.chain_trajectory.extend(history)
                if (
                    len(self.chain_trajectory) == 0
                    or len(self.chain_trajectory[-1]) != len(final_chain)
                    or not np.allclose(self.chain_trajectory[-1].coordinates, final_chain.coordinates)
                ):
                    self.chain_trajectory.append(final_chain.copy())
                else:
                    # Replace the final parsed history entry with an energy-populated copy.
                    self.chain_trajectory[-1] = final_chain.copy()
                self.optimized = final_chain

        except ElectronicStructureError:
            raise
        except Exception as exc:
            raise ElectronicStructureError(
                msg=f"geomeTRIC NEB execution failed ({type(exc).__name__}): {exc}",
                obj=None,
            ) from exc

        if bool(self._params.get("do_elem_step_checks", True)):
            self._status("geomeTRIC NEB: running elementary-step checks")
            elem_step_results = check_if_elem_step(
                inp_chain=self.optimized,
                engine=self.engine,
                validate_minima_with_hessian=bool(
                    self._params.get("validate_minima_with_hessian", False)
                ),
                hessian_minimum_frequency_cutoff=float(
                    self._params.get("hessian_minimum_frequency_cutoff", 0.0)
                ),
                hessian_minima_rescue_displacement=float(
                    self._params.get("hessian_minima_rescue_displacement", 0.1)
                ),
            )
            self.geom_grad_calls_made += int(elem_step_results.number_grad_calls)
            return elem_step_results
        return IS_ELEM_STEP
