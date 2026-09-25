import traceback
import logging
import copy
import os
import time
import concurrent.futures
import contextlib
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from mepd.helper_functions import pairwise
from typing import Any, List, Sequence, Tuple

from mepd.nodes.node import Node, StructureNode
from mepd.nodes.nodehelpers import _is_connectivity_identical
from mepd.elementarystep import elem_step_check_kwargs, check_if_elem_step

from mepd.chain import Chain
import mepd.chainhelpers as ch
from mepd.elementarystep import ElemStepResults
from mepd.neb import NEB, NoneConvergedException
from mepd.nodes.nodehelpers import is_identical

from mepd.TreeNode import TreeNode
from mepd.errors import (
    ElectronicStructureError,
    extract_electronic_structure_error_details,
    format_exception_message,
)
from mepd.optimizers.cg import ConjugateGradient
from mepd.optimizers.vpo import VelocityProjectedOptimizer
from mepd.optimizers.lbfgs import LBFGS
from mepd.optimizers.adam import AdamOptimizer
from mepd.optimizers.amg import AdaptiveMomentumGradient
from mepd.optimizers.fire import FIREOptimizer
from mepd.optimizers.sgd import SGDOptimizer
from mepd.optimizers.gd import DeterministicGradientDescentOptimizer

from mepd.pathminimizers.fneb import FreezingNEB
from mepd.pathminimizers.geometric_neb import GeometricNEB
from mepd.pathminimizers.nebdlf import DLFindNEB
from mepd.inputs import RunInputs
from mepd.progress import (
    get_progress_printer,
    preserve_chain_snapshot,
    progress_monitor,
    start_status,
    update_status,
    stop_status,
)

def _get_verbose(inputs: RunInputs) -> bool:
    """Get verbosity from RunInputs."""
    return getattr(inputs.path_min_inputs, 'v', False)


def _disregard_stereochem(inputs: RunInputs) -> bool:
    return bool(getattr(inputs.path_min_inputs, "disregard_stereochem", False))


PATH_METHODS = ["NEB", "FNEB", "MLPGI", "NEB-DLF", "GEOMETRIC-NEB", "GSM"]
DEFAULT_CONSECUTIVE_SAME_PAIR_SPLIT_LIMIT = 5


def _normalize_path_method(path_min_method: str) -> str:
    method = str(path_min_method or "").strip().upper().replace("_", "-")
    aliases = {
        "NEBDLF": "NEB-DLF",
        "DLFNEB": "NEB-DLF",
        "DLFIND": "NEB-DLF",
        "DL-FIND": "NEB-DLF",
        "GEOMETRIC": "GEOMETRIC-NEB",
        "GEOMETRICNEB": "GEOMETRIC-NEB",
        "FSM": "FNEB",  # freezing string method
    }
    return aliases.get(method, method)


def _empty_leaf(index: int, status: str) -> TreeNode:
    node = TreeNode(data=None, children=[], index=index)
    setattr(node, "leaf_status", status)
    return node


def _renumber_depth_first(node: TreeNode, index: int) -> int:
    """Number `node`'s subtree in depth-first preorder from `index`; returns
    the next free index."""
    node.index = index
    index += 1
    for child in node.children:
        index = _renumber_depth_first(child, index)
    return index


def _failed_leaf(
    index: int,
    status: str,
    exc: BaseException,
    chain: Chain | None = None,
) -> TreeNode:
    node = _empty_leaf(index, status=status)
    setattr(node, "leaf_error_type", type(exc).__name__)
    setattr(node, "leaf_error", str(exc))
    setattr(node, "leaf_traceback", traceback.format_exc())
    if chain is not None:
        setattr(node, "failed_chain", chain.copy())
    return node


def _to_plain_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, SimpleNamespace):
        return copy.deepcopy(vars(value))
    if hasattr(value, "model_dump"):
        try:
            return copy.deepcopy(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return copy.deepcopy(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return copy.deepcopy(vars(value))
    return copy.deepcopy(value)


def _clone_run_inputs_for_worker(run_inputs: RunInputs) -> RunInputs:
    """An independent copy of `run_inputs` for one worker thread: the same
    engine, optimizer and settings as the serial run, with no mutable state
    (optimizer history, parameter namespaces) shared between branches. The
    engine is deep-copied when it can be and shared otherwise."""
    memo = {}
    engine = getattr(run_inputs, "engine", None)
    if engine is not None:
        try:
            memo[id(engine)] = copy.deepcopy(engine)
        except Exception:
            memo[id(engine)] = engine
    return copy.deepcopy(run_inputs, memo)


def _run_inputs_payload_for_worker(run_inputs: RunInputs) -> dict:
    """Build a serialization-safe RunInputs payload for workers."""
    return {
        "engine_name": getattr(run_inputs, "engine_name", "chemcloud"),
        "program": getattr(run_inputs, "program", "xtb"),
        "chemcloud_queue": getattr(run_inputs, "chemcloud_queue", None),
        "write_qcio": bool(getattr(run_inputs, "write_qcio", False)),
        "nanoreactor_inputs": _to_plain_dict(
            getattr(run_inputs, "nanoreactor_inputs", None)
        ),
        "path_min_method": getattr(run_inputs, "path_min_method", "NEB"),
        "path_min_inputs": _to_plain_dict(
            getattr(run_inputs, "path_min_inputs", None)
        ),
        "chain_inputs": _to_plain_dict(getattr(run_inputs, "chain_inputs", None)),
        "gi_inputs": _to_plain_dict(getattr(run_inputs, "gi_inputs", None)),
        "program_kwds": _to_plain_dict(getattr(run_inputs, "program_kwds", None)),
        "ase_engine_kwds": _to_plain_dict(getattr(run_inputs, "ase_engine_kwds", None)),
        "gxtb_engine_kwds": _to_plain_dict(getattr(run_inputs, "gxtb_engine_kwds", None)),
        "geometry_optimizer_kwds": _to_plain_dict(
            getattr(run_inputs, "geometry_optimizer_kwds", None)
        ),
        "optimizer_kwds": _to_plain_dict(getattr(run_inputs, "optimizer_kwds", None)),
        "atom_mapping_inputs": _to_plain_dict(getattr(run_inputs, "atom_mapping_inputs", None)),
        "print_stdout": bool(getattr(run_inputs, "print_stdout", False)),
    }


def _chain_payload_for_worker(input_chain: Chain) -> dict:
    payload = {
        "nodes": [
            node.to_serializable() if hasattr(node, "to_serializable") else copy.deepcopy(node)
            for node in input_chain.nodes
        ],
        "parameters": _to_plain_dict(input_chain.parameters),
    }
    lineage = getattr(input_chain, "_same_pair_split_lineage", None)
    if isinstance(lineage, dict):
        payload["same_pair_split_lineage"] = copy.deepcopy(lineage)
    return payload


def _chain_from_worker_payload(payload: dict) -> Chain:
    nodes = []
    for item in list(payload.get("nodes") or []):
        if isinstance(item, dict) and "structure" in item:
            nodes.append(StructureNode.from_serializable(copy.deepcopy(item)))
        elif hasattr(item, "copy"):
            nodes.append(item.copy())
        else:
            nodes.append(copy.deepcopy(item))
    chain = Chain.model_validate(
        {
            "nodes": nodes,
            "parameters": payload.get("parameters"),
        }
    )
    lineage = payload.get("same_pair_split_lineage")
    if isinstance(lineage, dict):
        chain._same_pair_split_lineage = copy.deepcopy(lineage)
    return chain


def _structure_node_from_attempt_payload(value: Any) -> StructureNode | None:
    if isinstance(value, StructureNode):
        return value.copy()
    if isinstance(value, dict) and "structure" in value:
        try:
            return StructureNode.from_serializable(copy.deepcopy(value))
        except Exception:
            return None
    if hasattr(value, "copy"):
        try:
            candidate = value.copy()
            if isinstance(candidate, StructureNode):
                return candidate
        except Exception:
            return None
    return None


def _leaf_chain_from_tree_node(node: TreeNode) -> Chain | None:
    if node is None or getattr(node, "data", None) is None:
        return None
    data = node.data
    chain_trajectory = getattr(data, "chain_trajectory", None) or []
    if chain_trajectory:
        return chain_trajectory[-1]
    optimized = getattr(data, "optimized", None)
    if optimized is not None:
        return optimized
    return None


def _concat_leaf_chains(chains: list[Chain], parameters) -> Chain:
    if len(chains) == 0:
        raise ValueError("Cannot concatenate empty chain list.")
    nodes = []
    for i, chain in enumerate(chains):
        chain_nodes = chain.nodes if i == 0 else chain.nodes[1:]
        nodes.extend([node.copy() for node in chain_nodes])
    return Chain.model_validate({"nodes": nodes, "parameters": parameters})


@dataclass
class MSMEP:
    """Class for running autosplitting MEP minimizations."""

    inputs: RunInputs
    path_minimizer = None

    def __post_init__(self):
        assert (
            self.inputs.path_min_method or self.path_minimizer is not None
        ), "Need to input a path_min_method or path minimizer"
        if self.path_minimizer is None:
            normalized_method = _normalize_path_method(self.inputs.path_min_method)
            assert (
                normalized_method in PATH_METHODS
            ), f"Invalid path method: {self.inputs.path_min_method}. Allowed are: {PATH_METHODS}"
        self._attempted_pairs_payload_ref = None
        self._attempted_pair_cache: list[
            tuple[StructureNode, StructureNode, bool, bool]
        ] = []

    def _resolve_recursive_split_max_depth(self, max_depth: int | None = None) -> int | None:
        if max_depth is not None:
            return max(0, int(max_depth))
        value = getattr(self.inputs.path_min_inputs, "recursive_split_max_depth", None)
        if value is None:
            return None
        return max(0, int(value))

    def _resolve_same_pair_split_limit(self) -> int:
        value = getattr(
            self.inputs.path_min_inputs,
            "recursive_same_pair_split_limit",
            DEFAULT_CONSECUTIVE_SAME_PAIR_SPLIT_LIMIT,
        )
        return max(1, int(value))

    def _build_neb_optimizer(self):
        kwds = dict(self.inputs.optimizer_kwds or {})
        optimizer_name = kwds.pop("name", "cg").lower()
        optimizer_map = {
            "cg": ConjugateGradient,
            "conjugate_gradient": ConjugateGradient,
            "vpo": VelocityProjectedOptimizer,
            "velocity_projected": VelocityProjectedOptimizer,
            "lbfgs": LBFGS,
            "adam": AdamOptimizer,
            "sgd": SGDOptimizer,
            "stochastic_gradient_descent": SGDOptimizer,
            "gd": DeterministicGradientDescentOptimizer,
            "gradient_descent": DeterministicGradientDescentOptimizer,
            "deterministic_gradient_descent": DeterministicGradientDescentOptimizer,
            "amg": AdaptiveMomentumGradient,
            "adaptive_momentum": AdaptiveMomentumGradient,
            "fire": FIREOptimizer,
        }
        if optimizer_name not in optimizer_map:
            available = ", ".join(sorted(set(optimizer_map.keys())))
            raise ValueError(
                f"Unsupported optimizer '{optimizer_name}'. Supported values: {available}"
            )
        return optimizer_map[optimizer_name](**kwds)

    def _should_disable_graphs(self) -> bool:
        return bool(getattr(self.inputs.engine, "disable_molecular_graphs", False))

    def _set_attempted_pairs_payload(self, payload: Any) -> None:
        setattr(self.inputs.path_min_inputs, "attempted_pairs_payload", payload)
        self._attempted_pairs_payload_ref = None
        self._attempted_pair_cache = []

    def _get_attempted_pairs_payload(self) -> list[dict[str, Any]]:
        path_min_inputs = getattr(self.inputs, "path_min_inputs", None)
        if path_min_inputs is None:
            return []
        payload = getattr(path_min_inputs, "attempted_pairs_payload", None)
        if not isinstance(payload, list):
            payload = []
            setattr(path_min_inputs, "attempted_pairs_payload", payload)
            self._attempted_pairs_payload_ref = None
            self._attempted_pair_cache = []
        return payload

    def _chain_attempt_payload(
        self,
        input_chain: Chain,
        directed: bool = False,
        *,
        converged: bool = False,
        is_elem_step: bool = False,
        status: str = "attempted",
    ) -> dict[str, Any] | None:
        if len(input_chain) < 2:
            return None
        start_node = input_chain[0]
        end_node = input_chain[-1]
        if not hasattr(start_node, "to_serializable") or not hasattr(
            end_node, "to_serializable"
        ):
            return None
        try:
            start_payload = start_node.to_serializable()
            end_payload = end_node.to_serializable()
        except Exception:
            return None
        return {
            "start": start_payload,
            "end": end_payload,
            "directed": bool(directed),
            "converged": bool(converged),
            "is_elem_step": bool(is_elem_step),
            "status": str(status),
        }

    def _record_attempted_pair(self, input_chain: Chain) -> dict[str, Any] | None:
        payload = self._get_attempted_pairs_payload()
        attempt_payload = self._chain_attempt_payload(input_chain)
        if attempt_payload is None:
            return None
        payload.append(attempt_payload)
        self._attempted_pairs_payload_ref = None
        return attempt_payload

    def _chain_result_converged(self, neb_obj: Any) -> bool:
        explicit = getattr(neb_obj, "converged", None)
        if explicit is not None:
            return bool(explicit)
        chain = getattr(neb_obj, "optimized", None)
        if chain is None:
            trajectory = getattr(neb_obj, "chain_trajectory", None) or []
            chain = trajectory[-1] if trajectory else None
        nodes = list(getattr(chain, "nodes", []) or [])
        if not nodes:
            return False
        return all(bool(getattr(node, "converged", False)) for node in nodes)

    def _mark_attempted_pair_result(
        self,
        attempt_payload: dict[str, Any] | None,
        neb_obj: Any,
        elem_step_results: Any,
    ) -> None:
        if attempt_payload is None:
            return
        converged = self._chain_result_converged(neb_obj)
        is_elem_step = bool(getattr(elem_step_results, "is_elem_step", False))
        attempt_payload.update(
            {
                "converged": bool(converged),
                "is_elem_step": bool(is_elem_step),
                "status": "completed",
                "skip_eligible": bool(converged and is_elem_step),
            }
        )
        self._attempted_pairs_payload_ref = None

    @staticmethod
    def _attempt_payload_skip_eligible(item: dict[str, Any]) -> bool:
        if "skip_eligible" in item:
            return bool(item.get("skip_eligible"))
        return bool(item.get("converged")) and bool(item.get("is_elem_step"))

    def _disable_molecular_graphs(self, chain: Chain) -> None:
        if not self._should_disable_graphs():
            return
        for node in chain:
            if hasattr(node, "has_molecular_graph"):
                node.has_molecular_graph = False
            if hasattr(node, "graph"):
                node.graph = None

    def _nodes_match_for_attempt_skip(self, a: StructureNode, b: StructureNode) -> bool:
        if a is None or b is None:
            return False
        try:
            if (
                list(a.structure.symbols) == list(b.structure.symbols)
                and np.allclose(a.coords, b.coords)
            ):
                return True
        except Exception:
            pass
        try:
            a_cmp = a.copy()
            b_cmp = b.copy()
            setattr(a_cmp, "disable_smiles", True)
            setattr(b_cmp, "disable_smiles", True)
            return bool(
                is_identical(
                    a_cmp,
                    b_cmp,
                    fragment_rmsd_cutoff=self.inputs.chain_inputs.node_rms_thre,
                    kcal_mol_cutoff=self.inputs.chain_inputs.node_ene_thre,
                    verbose=False,
                )
            )
        except Exception:
            return False

    def _pair_matches_lineage(
        self,
        input_chain: Chain,
        lineage: dict[str, Any] | None,
    ) -> bool:
        if len(input_chain) < 2 or not isinstance(lineage, dict):
            return False
        prior_start = _structure_node_from_attempt_payload(lineage.get("start"))
        prior_end = _structure_node_from_attempt_payload(lineage.get("end"))
        if prior_start is None or prior_end is None:
            return False
        start_node = input_chain[0]
        end_node = input_chain[-1]

        def _same_species(a: StructureNode, b: StructureNode) -> bool:
            if (
                getattr(a, "has_molecular_graph", False)
                and getattr(b, "has_molecular_graph", False)
                and getattr(a, "graph", None) is not None
                and getattr(b, "graph", None) is not None
            ):
                try:
                    return bool(
                        a.graph.remove_Hs().is_bond_isomorphic_to(
                            b.graph.remove_Hs()
                        )
                    )
                except Exception:
                    pass
            return self._nodes_match_for_attempt_skip(a, b)

        forward = _same_species(start_node, prior_start) and _same_species(
            end_node, prior_end
        )
        if forward:
            return True
        return _same_species(start_node, prior_end) and _same_species(
            end_node, prior_start
        )

    def _consecutive_same_pair_split_count(self, input_chain: Chain) -> int:
        lineage = getattr(input_chain, "_same_pair_split_lineage", None)
        if not self._pair_matches_lineage(input_chain, lineage):
            return 0
        try:
            return max(0, int(lineage.get("count", 0)))
        except Exception:
            return 0

    def _set_child_same_pair_split_lineage(
        self,
        child_chains: list[Chain],
        parent_chain: Chain,
        count: int,
    ) -> None:
        payload = self._chain_attempt_payload(parent_chain)
        if payload is None:
            return
        lineage = {
            "start": payload["start"],
            "end": payload["end"],
            "count": max(0, int(count)),
        }
        for child_chain in child_chains:
            child_chain._same_pair_split_lineage = copy.deepcopy(lineage)

    def _same_pair_split_limit_message(self, count: int) -> str:
        return (
            f"Stopping further splitting of this branch only after {int(count)} "
            "consecutive splits of the same endpoint pair (other branches, and "
            "the rest of autosplitting, are unaffected). For floppy systems "
            "this usually means `node_rms_thre` and/or `node_ene_thre` are too small."
        )

    def _refresh_attempted_pair_cache(self) -> None:
        payload = getattr(
            getattr(self.inputs, "path_min_inputs", None),
            "attempted_pairs_payload",
            None,
        )
        if payload is self._attempted_pairs_payload_ref:
            return
        self._attempted_pairs_payload_ref = payload
        self._attempted_pair_cache = []
        if not isinstance(payload, list):
            return
        for item in payload:
            if not isinstance(item, dict):
                continue
            start_node = _structure_node_from_attempt_payload(item.get("start"))
            end_node = _structure_node_from_attempt_payload(item.get("end"))
            if start_node is None or end_node is None:
                continue
            directed = bool(item.get("directed", False))
            skip_eligible = self._attempt_payload_skip_eligible(item)
            self._attempted_pair_cache.append(
                (start_node, end_node, directed, skip_eligible)
            )

    def _skip_chain_due_to_attempted_history(self, input_chain: Chain) -> bool:
        if len(input_chain) < 2:
            return False
        self._refresh_attempted_pair_cache()
        if not self._attempted_pair_cache:
            return False
        start_node = input_chain[0]
        end_node = input_chain[-1]
        for attempted_start, attempted_end, directed, skip_eligible in self._attempted_pair_cache:
            if not skip_eligible:
                continue
            forward_match = self._nodes_match_for_attempt_skip(
                start_node, attempted_start
            ) and self._nodes_match_for_attempt_skip(end_node, attempted_end)
            if forward_match:
                return True
            if directed:
                continue
            reverse_match = self._nodes_match_for_attempt_skip(
                start_node, attempted_end
            ) and self._nodes_match_for_attempt_skip(end_node, attempted_start)
            if reverse_match:
                return True
        return False

    def _endpoint_connectivity_status(self, chain: Chain) -> str:
        return "Checking endpoint connectivity"

    def _say(self, msg: str, *, snapshot: bool = False, warn: bool = False) -> None:
        """Print `msg` in verbose mode; otherwise show it as the live status
        line, or keep it in the scrollback (`snapshot`)."""
        if _get_verbose(self.inputs):
            print(f"Warning! {msg}" if warn else msg, flush=True)
        elif snapshot:
            preserve_chain_snapshot(note=msg)
        else:
            update_status(msg)

    def run_recursive_minimize(
        self,
        input_chain: Chain,
        tree_node_index=0,
        attempted_pairs_payload: list[dict[str, Any]] | None = None,
        tree_depth: int = 0,
        max_depth: int | None = None,
    ) -> TreeNode:
        """Will take a chain as an input and run NEB minimizations until it exits out.
        NEB can exit due to the chain being converged, the chain needing to be split,
        or the maximum number of alloted steps being met.

        Args:
            input_chain (Chain): _description_
            tree_node_index (int, optional): index of node minimization. Root node is 0. Defaults to 0.

        Returns:
            TreeNode: Tree containing the history of NEB optimizations. First node is the initial
            chain given and its corresponding neb minimization. Children are chains into which
            the root chain was split.
        """
        if isinstance(input_chain, list):
            input_chain = Chain.model_validate(
                {"nodes": input_chain, "parameters": self.inputs.chain_inputs})
        if attempted_pairs_payload is not None:
            self._set_attempted_pairs_payload(attempted_pairs_payload)
        max_depth = self._resolve_recursive_split_max_depth(max_depth)
        history, sequence_of_chains = self._guarded_step(
            input_chain, tree_node_index, tree_depth=tree_depth, max_depth=max_depth
        )

        new_tree_node_index = tree_node_index + 1
        for i, chain_frag in enumerate(sequence_of_chains, start=1):
            self._say(f"On chain {i} of {len(sequence_of_chains)}...")
            try:
                out_history = self.run_recursive_minimize(
                    chain_frag,
                    tree_node_index=new_tree_node_index,
                    tree_depth=tree_depth + 1,
                    max_depth=max_depth,
                )
            except Exception as child_exc:
                out_history = self._branch_failed(new_tree_node_index, child_exc, chain_frag)
            history.children.append(out_history)
            new_tree_node_index = out_history.max_index + 1
        return history

    def _guarded_step(
        self, input_chain: Chain, tree_node_index: int, tree_depth: int = 0,
        max_depth: int | None = None,
    ) -> tuple[TreeNode, list[Chain]]:
        """`_run_recursive_step`, with a failing branch turned into a failed
        leaf instead of an exception -- the same in serial and in parallel."""
        try:
            return self._run_recursive_step(
                input_chain, tree_node_index, tree_depth=tree_depth, max_depth=max_depth
            )
        except ElectronicStructureError as e:
            self._say(
                f"Electronic structure error in recursive branch {tree_node_index}: "
                f"{format_exception_message(e)}"
            )
            obj = getattr(e, "obj", None)
            if hasattr(obj, "save"):
                with contextlib.suppress(Exception):
                    obj.save("/tmp/failed_output.qcio")
            return _failed_leaf(
                tree_node_index, status="electronic_structure_error", exc=e, chain=input_chain
            ), []
        except Exception as e:
            return self._branch_failed(tree_node_index, e, input_chain), []

    def _branch_failed(self, index: int, exc: Exception, chain: Chain) -> TreeNode:
        if _get_verbose(self.inputs):
            print(traceback.format_exc())
            print(
                f"Warning! Recursive branch {index} failed "
                f"({type(exc).__name__}: {exc}). Continuing."
            )
        else:
            update_status(
                f"Branch {index} failed with {type(exc).__name__}; continuing recursive search."
            )
        return _failed_leaf(index, status="path_minimization_error", exc=exc, chain=chain)

    def _run_recursive_step(
        self,
        input_chain: Chain,
        tree_node_index: int,
        tree_depth: int = 0,
        max_depth: int | None = None,
    ) -> tuple[TreeNode, list[Chain]]:
        """Run a single recursive minimization step and return child fragments to continue."""
        if isinstance(input_chain, list):
            input_chain = Chain.model_validate(
                {"nodes": input_chain, "parameters": self.inputs.chain_inputs})
        resolved_max_depth = self._resolve_recursive_split_max_depth(max_depth)
        self._disable_molecular_graphs(input_chain)
        if self._skip_chain_due_to_attempted_history(input_chain):
            self._say("Endpoints already attempted elsewhere. Skipping chain.")
            return _empty_leaf(tree_node_index, status="attempted_elsewhere"), []

        identical_msg = "Endpoints are identical. Returning nothing"
        if getattr(self.inputs.path_min_inputs, "skip_identical_graphs", True) and input_chain[0].has_molecular_graph:
            if not _get_verbose(self.inputs):
                update_status(self._endpoint_connectivity_status(input_chain))
            if _is_connectivity_identical(
                input_chain[0],
                input_chain[-1],
                verbose=_get_verbose(self.inputs),
                disregard_stereochem=_disregard_stereochem(self.inputs),
            ):
                self._say(identical_msg)
                return _empty_leaf(tree_node_index, status="identical_endpoints"), []

        ch._reset_node_convergence(input_chain)
        self.inputs.engine.compute_gradients(input_chain)

        if is_identical(
            self=input_chain[0],
            other=input_chain[-1],
            fragment_rmsd_cutoff=self.inputs.chain_inputs.node_rms_thre,
            kcal_mol_cutoff=self.inputs.chain_inputs.node_ene_thre,
            verbose=False,
            disregard_stereochem=_disregard_stereochem(self.inputs),
        ):
            self._say(identical_msg)
            return _empty_leaf(tree_node_index, status="identical_endpoints"), []

        attempt_payload = self._record_attempted_pair(input_chain)
        root_neb_obj, elem_step_results = self.run_minimize_chain(
            input_chain=input_chain
        )
        self._mark_attempted_pair_result(
            attempt_payload, root_neb_obj, elem_step_results
        )
        history_node = TreeNode(data=root_neb_obj, children=[], index=tree_node_index)

        if elem_step_results.is_elem_step:
            return history_node, []

        def _record_split_count(n: int) -> None:
            history_node.consecutive_same_pair_splits = int(n)
            if attempt_payload is not None:
                attempt_payload["consecutive_same_pair_splits"] = int(n)

        same_pair_split_count = self._consecutive_same_pair_split_count(input_chain)
        _record_split_count(same_pair_split_count)
        if (
            same_pair_split_count >= self._resolve_same_pair_split_limit()
            and not elem_step_results.new_structures
        ):
            # Only enforce the stop when this attempt itself found nothing
            # chemically new (genuinely unproductive repetition) -- a split
            # that DID discover a new molecule/conformer this generation is
            # real progress and must not be discarded just because prior
            # generations repeated the same pair.
            self._say(
                self._same_pair_split_limit_message(same_pair_split_count),
                snapshot=True, warn=True,
            )
            history_node.leaf_status = "same_pair_split_limit_reached"
            return history_node, []

        if resolved_max_depth is not None and tree_depth >= resolved_max_depth:
            history_node.leaf_status = "max_depth_reached"
            return history_node, []

        chain_trajectory = getattr(root_neb_obj, "chain_trajectory", None) or []
        if not chain_trajectory:
            return history_node, []
        self._say(
            f"Splitting chains based on: {elem_step_results.splitting_criterion}",
            snapshot=True,
        )
        sequence_of_chains = self.make_sequence_of_chains(
            chain=chain_trajectory[-1],
            split_method=elem_step_results.splitting_criterion,
            minimization_results=elem_step_results.minimization_results,
        )
        self._set_child_same_pair_split_lineage(
            sequence_of_chains, input_chain, same_pair_split_count + 1
        )
        _record_split_count(same_pair_split_count + 1)
        return history_node, sequence_of_chains

    def run_parallel_recursive_minimize(
        self,
        input_chain: Chain,
        tree_node_index: int = 0,
        max_workers: int | None = None,
        attempted_pairs_payload: list[dict[str, Any]] | None = None,
    ) -> TreeNode:
        """Recursively autosplit NEBs, evaluating split branches in parallel."""
        if attempted_pairs_payload is not None:
            self._set_attempted_pairs_payload(attempted_pairs_payload)
        resolved_max_depth = self._resolve_recursive_split_max_depth()
        if max_workers is None:
            bounded_workers = min(4, max(1, int(os.cpu_count() or 1)))
        else:
            # Honor explicit user-requested parallelism. `os.cpu_count()` can
            # under-report available capacity in constrained launch contexts.
            bounded_workers = max(1, int(max_workers))

        progress_printer = get_progress_printer()
        progress_printer.clear_path_so_far()
        with progress_monitor(f"branch-{int(tree_node_index)}"):
            root_history, root_children = self._guarded_step(
                input_chain=input_chain,
                tree_node_index=tree_node_index,
                tree_depth=0,
                max_depth=resolved_max_depth,
            )
        if not root_children:
            root_history.parallel_failures = []
            return root_history

        next_tree_index = tree_node_index + 1
        completed_leaf_chains_by_index: dict[int, Chain] = {}
        pending: dict[concurrent.futures.Future, SimpleNamespace] = {}
        branch_failures: list[str] = []
        max_worker_attempts = 2
        engine_name = str(getattr(self.inputs, "engine_name", "") or "").strip().lower()
        compute_program = str(
            getattr(getattr(self.inputs, "engine", None), "compute_program", "") or ""
        ).strip().lower()
        use_process_workers = engine_name == "qccompute" and compute_program == "qccompute"
        run_inputs_payload = (
            _run_inputs_payload_for_worker(self.inputs) if use_process_workers else None
        )
        set_status = getattr(progress_printer, "set_monitor_status", None)
        # Branches share the machine: an engine left to size its own image
        # parallelism (n_parallel = 0) gets an equal slice per branch.
        branch_inputs = self.inputs
        if getattr(self.inputs.engine, "n_parallel", None) == 0:
            branch_inputs = _clone_run_inputs_for_worker(self.inputs)
            branch_inputs.engine.n_parallel = max(
                1, int(os.cpu_count() or 1) // bounded_workers
            )

        def _submit(executor: concurrent.futures.Executor, job: SimpleNamespace) -> None:
            job.submitted_at = time.time()
            if use_process_workers:
                future = executor.submit(
                    _parallel_recursive_step_worker_from_payload,
                    run_inputs_payload, _chain_payload_for_worker(job.chain),
                    job.index, job.depth, resolved_max_depth, job.worker_payload,
                )
            else:
                future = executor.submit(
                    _parallel_recursive_step_worker,
                    branch_inputs, job.chain,
                    job.index, job.depth, resolved_max_depth, job.worker_payload,
                )
            pending[future] = job

        def _submit_children(
            executor: concurrent.futures.Executor,
            parent_node: TreeNode,
            child_fragments: list[Chain],
            parent_depth: int,
        ) -> None:
            nonlocal next_tree_index
            parent_node.children = [None] * len(child_fragments)
            child_depth = parent_depth + 1
            for child_position, child_chain in enumerate(child_fragments):
                child_index = next_tree_index
                next_tree_index += 1
                if resolved_max_depth is not None and child_depth > resolved_max_depth:
                    parent_node.children[child_position] = _empty_leaf(
                        child_index, status="max_depth_reached"
                    )
                    continue
                worker_payload = copy.deepcopy(self._get_attempted_pairs_payload())
                if self._skip_chain_due_to_attempted_history(child_chain):
                    parent_node.children[child_position] = _empty_leaf(
                        child_index, status="attempted_elsewhere"
                    )
                    continue
                attempt_payload = self._record_attempted_pair(child_chain)
                progress_printer.mark_monitor_active(f"branch-{child_index}")
                if set_status:
                    set_status(
                        f"branch-{child_index}",
                        "Running in worker process" if use_process_workers else "Running",
                    )
                _submit(executor, SimpleNamespace(
                    parent=parent_node, position=child_position, index=child_index,
                    depth=child_depth, chain=child_chain, attempt=1,
                    worker_payload=worker_payload, attempt_payload=attempt_payload,
                ))

        executor_cls = (
            concurrent.futures.ProcessPoolExecutor
            if use_process_workers
            else concurrent.futures.ThreadPoolExecutor
        )
        with executor_cls(max_workers=bounded_workers) as executor:
            _submit_children(executor, root_history, root_children, parent_depth=0)
            while pending:
                done, _ = concurrent.futures.wait(
                    tuple(pending.keys()),
                    timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    if use_process_workers and set_status:
                        now = time.time()
                        for job in pending.values():
                            elapsed = max(0, int(now - job.submitted_at))
                            set_status(
                                f"branch-{job.index}",
                                f"Running in worker process (attempt {job.attempt}/{max_worker_attempts}, {elapsed}s)",
                            )
                    continue
                for future in done:
                    job = pending.pop(future)
                    try:
                        child_history, child_children = future.result()
                    except Exception as worker_exc:
                        worker_trace = traceback.format_exc().strip()
                        if job.attempt < max_worker_attempts:
                            job.attempt += 1
                            if set_status:
                                set_status(
                                    f"branch-{job.index}",
                                    f"Retrying (attempt {job.attempt}/{max_worker_attempts})",
                                )
                            _submit(executor, job)
                            continue
                        branch_failures.append(
                            f"branch-{job.index}: worker failed after {job.attempt} attempt(s) "
                            f"({type(worker_exc).__name__}: {worker_exc})\n"
                            f"worker traceback:\n{worker_trace}"
                        )
                        child_history = _failed_leaf(
                            job.index, status="worker_failure", exc=worker_exc, chain=job.chain,
                        )
                        child_children = []
                    job.parent.children[job.position] = child_history
                    leaf_status = str(getattr(child_history, "leaf_status", "") or "")
                    is_elem_step = (
                        not child_children
                        and bool(getattr(child_history, "data", None))
                        and leaf_status not in {"max_depth_reached", "same_pair_split_limit_reached"}
                    )
                    self._mark_attempted_pair_result(
                        job.attempt_payload,
                        getattr(child_history, "data", None),
                        SimpleNamespace(is_elem_step=is_elem_step),
                    )
                    progress_printer.mark_monitor_inactive(f"branch-{job.index}")
                    if child_children:
                        _submit_children(
                            executor, child_history, child_children, parent_depth=job.depth,
                        )
                        continue
                    leaf_chain = _leaf_chain_from_tree_node(child_history)
                    if leaf_chain is not None:
                        completed_leaf_chains_by_index[job.index] = leaf_chain
                        ordered_leaf_chains = [
                            completed_leaf_chains_by_index[idx]
                            for idx in sorted(completed_leaf_chains_by_index)
                        ]
                        progress_printer.update_path_so_far(
                            _concat_leaf_chains(ordered_leaf_chains, self.inputs.chain_inputs),
                            caption=f"{len(ordered_leaf_chains)} completed branch(es)",
                        )

        # Branches finish in any order; number the tree depth-first, exactly
        # as the serial run does, so both write the same node_<i> files.
        _renumber_depth_first(root_history, tree_node_index)
        root_history.parallel_failures = branch_failures
        return root_history

    def _create_interpolation(self, chain: Chain):
        logger = logging.getLogger(
            'mepd.geodesic_interpolation2.interpolation')
        logger.propagate = False


        from mepd.interpolation import initial_chain as _initial_chain, interpolation_method

        method = interpolation_method(self.inputs.chain_inputs)
        if method != "geodesic":
            # linear / LST / IDPP. A sub-path's ends are minima already found:
            # never re-align (move) them.
            interpolation = _initial_chain(chain, self.inputs.chain_inputs, self.inputs.gi_inputs, align=False)
            interpolation._zero_velocity()
        else:
            if chain.parameters.frozen_atom_indices:
                inds_frozen = chain.parameters.frozen_atom_indices
                if _get_verbose(self.inputs):
                    print("will be freezing inds:", inds_frozen,
                          ' during geodesic interpolation')
                    print(type(inds_frozen))
            else:
                inds_frozen = np.array([], dtype=int)

            interpolation, _ = ch.run_geodesic(
                chain=chain,
                chain_inputs=copy.deepcopy(self.inputs.chain_inputs),
                nimages=self.inputs.gi_inputs.nimages,
                friction=self.inputs.gi_inputs.friction,
                nudge=self.inputs.gi_inputs.nudge,
                random_seed=self.inputs.gi_inputs.random_seed,
                align=self.inputs.gi_inputs.align,
                ignore_atoms=inds_frozen,
                return_smoother=True,
                **self.inputs.gi_inputs.extra_kwds,

            )

            interpolation._zero_velocity()

        return interpolation

    def _construct_path_minimizer(self, initial_chain: Chain):
        path_method = _normalize_path_method(self.inputs.path_min_method)
        if path_method == "NEB":

            self._say("Using in-house NEB optimizer")
            optimizer = self._build_neb_optimizer()

            n = NEB(
                initial_chain=initial_chain,
                parameters=self.inputs.path_min_inputs,
                optimizer=optimizer,
                engine=self.inputs.engine,
            )
        # elif self.inputs.path_min_method.upper() == "PYGSM":

        elif path_method == "FNEB":
            self._say("Using Freezing NEB optimizer")
            optimizer = self._build_neb_optimizer()
            n = FreezingNEB(
                initial_chain=initial_chain,
                engine=self.inputs.engine,
                parameters=self.inputs.path_min_inputs,
                optimizer=optimizer,
                gi_inputs=self.inputs.gi_inputs
            )

        elif path_method == "MLPGI":
            from mepd.pathminimizers.mlpgi import MLPGI
            self._say("Using MLP Geodesic Optimizer")
            n = MLPGI(
                initial_chain=initial_chain,
                engine=self.inputs.engine,
                parameters=self.inputs.path_min_inputs,
            )
        elif path_method == "NEB-DLF":
            self._say("Using DL-Find NEB optimizer via TeraChem/QCCompute")
            n = DLFindNEB(
                initial_chain=initial_chain,
                engine=self.inputs.engine,
                parameters=self.inputs.path_min_inputs,
            )
        elif path_method == "GEOMETRIC-NEB":
            self._say("Using geomeTRIC NEB optimizer")
            n = GeometricNEB(
                initial_chain=initial_chain,
                engine=self.inputs.engine,
                parameters=self.inputs.path_min_inputs,
            )
        elif path_method == "GSM":
            from mepd.pathminimizers.gsm import GSM
            self._say("Using molecularGSM (Zimmerman lab growing string method)")
            n = GSM(
                initial_chain=initial_chain,
                engine=self.inputs.engine,
                parameters=self.inputs.path_min_inputs,
                gi_inputs=self.inputs.gi_inputs,
            )
        else:
            raise NotImplementedError(
                "Invalid path minimization method. Select from NEB, FNEB, MLPGI, NEB-DLF, GEOMETRIC-NEB, or GSM.")

        return n

    def run_minimize_chain(self, input_chain: Chain) -> Tuple[NEB, ElemStepResults]:
        if isinstance(input_chain, list):
            input_chain = Chain.model_validate(
                {"nodes": input_chain, "parameters": self.inputs.chain_inputs})
        self._disable_molecular_graphs(input_chain)

        # make sure the chain parameters are reset
        # if they come from a converged chain
        if len(input_chain) != self.inputs.gi_inputs.nimages:
            interpolation = self._create_interpolation(
                input_chain,
            )
            assert (
                len(interpolation) == self.inputs.gi_inputs.nimages
            ), f"Geodesic interpolation wrong length.\
                 Requested: {self.inputs.gi_inputs.nimages}. Given: {len(interpolation)}"

        else:
            interpolation = input_chain
        self._disable_molecular_graphs(interpolation)

        # Use spinner when v=0, print when v=1
        verbose = _get_verbose(self.inputs)
        if verbose:
            print("Running path minimization...")
        else:
            start_status("Minimizing path...")

        try:
            n = self._construct_path_minimizer(initial_chain=interpolation)
            elem_step_results = n.optimize_chain()
            setattr(n, "converged", True)
            out_chain = n.optimized

        except NoneConvergedException:
            setattr(n, "converged", False)
            print(traceback.format_exc())

            print(
                "\nWarning! A chain did not converge.\
                        Returning an unoptimized chain..."
            )
            out_chain = n.chain_trajectory[-1]
            if self.inputs.path_min_inputs.do_elem_step_checks:
                elem_step_results = check_if_elem_step(
                    out_chain,
                    engine=self.inputs.engine,
                    verbose=_get_verbose(self.inputs),
                    **elem_step_check_kwargs(self.inputs.path_min_inputs),
                )
            else:
                elem_step_results = ElemStepResults(
                    is_elem_step=True,
                    is_concave=None,
                    splitting_criterion=None,
                    minimization_results=None,
                    number_grad_calls=0,
                )

        except ElectronicStructureError as e:
            setattr(n, "converged", False)
            setattr(n, "failure_reason", "electronic_structure_error")
            setattr(n, "failure_exception", e)

            print(
                "\nWarning! A chain has electronic structure errors. \
                    Returning an unoptimized chain..."
            )
            print(f"ElectronicStructureError message: {format_exception_message(e)}")
            if e.__cause__ is not None:
                print(f"ElectronicStructureError cause: {type(e.__cause__).__name__}: {e.__cause__}")
            details = extract_electronic_structure_error_details(e.obj)
            if details:
                print("ElectronicStructureError details:")
                for i, detail in enumerate(details, start=1):
                    print(f"  [{i}] {detail}")
            if e.obj is not None:
                if isinstance(e.obj, (list, tuple)) and not details:
                    print(
                        f"ElectronicStructureError includes {len(e.obj)} result objects."
                    )
                elif not details:
                    print(
                        "ElectronicStructureError includes a result object "
                        f"of type {type(e.obj).__name__}."
                    )
            out_chain = n.chain_trajectory[-1]
            elem_step_results = ElemStepResults(
                is_elem_step=True,
                is_concave=None,
                splitting_criterion=None,
                minimization_results=None,
                number_grad_calls=0,
            )

        finally:
            if not verbose:
                stop_status()

        return n, elem_step_results

    def _make_chain_frag(self, chain: Chain, geom_pair, ind_pair):
        start_ind, end_ind = ind_pair
        opt_start, opt_end = geom_pair

        # JDEP 01132025: Going to not recycle fragment nodes. Want a fresh
        # interpolation
        chain_frag = chain.model_copy(update={
            "nodes": [opt_start, opt_end],
            "parameters": self.inputs.chain_inputs})

        return chain_frag

    def _do_minima_based_split(self, chain: Chain, minimization_results: List[Node]):

        ind_minima = list(ch._get_ind_minima(chain))
        if len(ind_minima) == 0:
            fallback = chain.copy()
            fallback.nodes = [chain[0], chain[-1]]
            return [fallback]

        raw_min_nodes = [chain[int(i)] for i in ind_minima]

        candidates = []
        for node in list(minimization_results or []):
            if node is None:
                continue
            if self._nodes_match_for_attempt_skip(node, chain[0]):
                continue
            if self._nodes_match_for_attempt_skip(node, chain[-1]):
                continue
            candidates.append(node)

        validate_minima_with_hessian = bool(
            getattr(
                self.inputs.path_min_inputs,
                "validate_minima_with_hessian",
                False,
            )
        )

        if validate_minima_with_hessian:
            # Hessian validation is authoritative: rejected optimized minima
            # are omitted, and neighboring segments collapse across them.
            split_nodes = candidates
        elif len(candidates) == len(raw_min_nodes):
            split_nodes = candidates
        elif len(candidates) == 0:
            split_nodes = raw_min_nodes
        else:
            # Mismatch between apparent minima count and optimizer-returned minima.
            # Keep splitting deterministic by falling back to path minima anchors.
            split_nodes = raw_min_nodes

        all_geometries = [chain[0], *split_nodes, chain[-1]]
        pairs_geoms = list(pairwise(all_geometries))

        chains = []
        seen_pairs = set()
        for geom_pair in pairs_geoms:
            start_node, end_node = geom_pair
            try:
                same_endpoints = np.allclose(
                    np.asarray(start_node.coords),
                    np.asarray(end_node.coords),
                    atol=1e-8,
                    rtol=0.0,
                )
            except Exception:
                same_endpoints = False
            if same_endpoints:
                continue

            pair_key = (
                tuple(np.round(np.asarray(start_node.coords).flatten(), 8).tolist()),
                tuple(np.round(np.asarray(end_node.coords).flatten(), 8).tolist()),
            )
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            chains.append(
                self._make_chain_frag(
                    chain=chain,
                    geom_pair=geom_pair,
                    ind_pair=(None, None),
                )
            )

        if not chains:
            fallback = chain.copy()
            fallback.nodes = [chain[0], chain[-1]]
            return [fallback]
        return chains

    def _do_maxima_based_split(
        self, chain: Chain, minimization_results: List[Node]
    ) -> List[Chain]:
        """
        Will take a chain that needs to be split based on 'maxima' criterion and outputs
        a list of Chains to be minimized.
        Args:
            chain (Chain): _description_
            minimization_results (List[Node]): a length-2 list containing the Reactant and Product
            geometries found through the `elemetnarystep.pseudo_irc()`

        Returns:
            List[Chain]: list of chains to be minimized
        """
        split_nodes = [
            node for node in list(minimization_results or [])
            if node is not None
        ]
        if not split_nodes:
            fallback = chain.copy()
            fallback.nodes = [chain[0], chain[-1]]
            return [fallback]

        segment_pairs = list(pairwise([chain[0], *split_nodes, chain[-1]]))

        def _pair_key(a: Node, b: Node):
            a_coords = np.asarray(a.coords).flatten()
            b_coords = np.asarray(b.coords).flatten()
            return (
                tuple(np.round(a_coords, 8).tolist()),
                tuple(np.round(b_coords, 8).tolist()),
            )

        chains_list = []
        seen_pairs = set()
        for start_node, end_node in segment_pairs:
            try:
                same_endpoints = np.allclose(
                    np.asarray(start_node.coords),
                    np.asarray(end_node.coords),
                    atol=1e-8,
                    rtol=0.0,
                )
            except Exception:
                same_endpoints = False
            if same_endpoints:
                continue

            key = _pair_key(start_node, end_node)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            chain_frag = chain.copy()
            chain_frag.nodes = [start_node, end_node]
            chains_list.append(chain_frag)

        if not chains_list:
            fallback = chain.copy()
            fallback.nodes = [chain[0], chain[-1]]
            return [fallback]

        return chains_list

    def make_sequence_of_chains(
        self, chain: Chain, split_method: str, minimization_results: List[Node]
    ) -> List[Chain]:
        """
        Takes an input chain, `chain`, that needs to be split accordint to `split_method` ("minima", "maxima")
        Returns a list of Chain objects to be minimzed.
        """
        if split_method == "minima":
            chains = self._do_minima_based_split(chain, minimization_results)

        elif split_method == "maxima":
            chains = self._do_maxima_based_split(chain, minimization_results)

        if getattr(getattr(self.inputs, "atom_mapping_inputs", None), "recheck_on_split", False):
            parent_ends = [chain[0], chain[-1]] if len(chain) >= 2 else []
            chains = [self._maybe_realign_chain_endpoints(c, forbidden=parent_ends) for c in chains]

        return chains

    def _maybe_realign_chain_endpoints(self, chain: Chain, forbidden: Sequence[Node] = ()) -> Chain:
        """`--atom-mapping-recheck-splits`: re-run the same best-of-N
        atom-mapping selection `--atom-mapping` runs once on the original
        --start/--end pair, but on THIS split's (reactant, product) pair.
        A no-op for splits whose pair has no SLAPMapper-detectable
        alternative mapping -- see `maybe_realign_pair`.

        `forbidden`: the parent chain's endpoints. A renumbering that makes
        the product side bond-for-bond (atom-indexed) identical to one of
        them is refused: when the intermediate C found on A -> B is the same
        molecule as B, renumbering it to B's numbering would recreate the
        parent pair A -> B, which splits at C again, forever."""
        from mepd.atom_mapping_selection import maybe_realign_pair
        from mepd.nodes.node import StructureNode

        try:
            realigned_structure, changed = maybe_realign_pair(
                chain[0].structure, chain[-1].structure, self.inputs
            )
        except Exception:
            return chain
        if changed and forbidden:
            from mepd.discovery.qct import bond_pattern

            def bonds(structure):
                return bond_pattern(list(structure.symbols), structure.geometry)

            try:
                new_bonds = bonds(realigned_structure)
                clash = any(
                    new_bonds == bonds(node.structure) != bonds(chain[-1].structure)
                    for node in forbidden
                    if getattr(node, "structure", None) is not None
                    and list(node.structure.symbols) == list(realigned_structure.symbols)
                )
            except Exception:
                clash = False
            if clash:
                if _get_verbose(self.inputs):
                    print("--atom-mapping-recheck-splits: kept a split's numbering "
                          "(the remap would recreate the parent's endpoint).")
                return chain
        if changed:
            if _get_verbose(self.inputs):
                print("--atom-mapping-recheck-splits: reindexed a split's product-side atoms.")
            chain.nodes = [chain.nodes[0], StructureNode(structure=realigned_structure)]
        return chain


def _parallel_recursive_step_worker(
    run_inputs: RunInputs,
    input_chain: Chain,
    tree_node_index: int,
    tree_depth: int = 0,
    max_depth: int | None = None,
    attempted_pairs_payload: list[dict[str, Any]] | None = None,
) -> tuple[TreeNode, list[Chain]]:
    local_inputs = _clone_run_inputs_for_worker(run_inputs)

    try:
        # Keep branch workers in non-verbose mode so transient rich panels from
        # elementary-step checks do not fight with the live ASCII monitors.
        local_inputs.path_min_inputs.v = False
    except Exception:
        pass

    # Isolate each branch worker from shared mutable endpoint/node state.
    # Split builders can reuse node instances across child chains (e.g. maxima
    # split middle points), which is safe in serial mode but can race in
    # parallel mode.
    try:
        local_chain = input_chain.copy()
    except Exception:
        try:
            local_chain = copy.deepcopy(input_chain)
        except Exception:
            local_chain = input_chain

    with progress_monitor(f"branch-{int(tree_node_index)}"):
        runner = MSMEP(inputs=local_inputs)
        if attempted_pairs_payload is not None:
            runner._set_attempted_pairs_payload(copy.deepcopy(attempted_pairs_payload))
        history, child_chains = runner._guarded_step(
            input_chain=local_chain,
            tree_node_index=tree_node_index,
            tree_depth=tree_depth,
            max_depth=max_depth,
        )
        try:
            if getattr(history, "data", None) is not None:
                history.data.engine = None
        except Exception:
            pass
        return history, child_chains


def _parallel_recursive_step_worker_from_payload(
    run_inputs_payload: dict,
    input_chain_payload: dict,
    tree_node_index: int,
    tree_depth: int = 0,
    max_depth: int | None = None,
    attempted_pairs_payload: list[dict[str, Any]] | None = None,
) -> tuple[TreeNode, list[Chain]]:
    local_inputs = RunInputs(**copy.deepcopy(run_inputs_payload))
    local_chain = _chain_from_worker_payload(input_chain_payload)
    # Subprocess workers should not write rich/live progress to the shared
    # terminal; keep rendering centralized in the parent scheduler process.
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return _parallel_recursive_step_worker(
                run_inputs=local_inputs,
                input_chain=local_chain,
                tree_node_index=tree_node_index,
                tree_depth=tree_depth,
                max_depth=max_depth,
                attempted_pairs_payload=attempted_pairs_payload,
            )
