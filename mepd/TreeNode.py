from dataclasses import dataclass
from functools import cached_property
from mepd.neb import NEB
from pathlib import Path
from typing import Any
import numpy as np
import networkx as nx
import shutil
from mepd.chain import Chain
from mepd.inputs import ChainInputs, NEBInputs, GIInputs
from mepd.optimizers.vpo import VelocityProjectedOptimizer
from mepd.nodes.nodehelpers import is_identical
from mepd.tsopt import TSOptResult


@dataclass
class GreedyTSOptCandidate:
    """One TS-guess drawn from `TreeNode.get_optimization_history()` and the
    outcome of optimizing it. `source_index` is the position of its source
    NEB within `get_optimization_history()` (stable for a given tree, useful
    for labeling output files)."""

    source_index: int
    ts_guess: Any
    result: TSOptResult


@dataclass
class TreeNode:
    data: NEB
    children: list
    index: int

    _force_tree_recalc: bool = False

    @property
    def max_index(self):
        max_found = 0
        for node in self.depth_first_ordered_nodes:
            if node.index > max_found:
                max_found = node.index
        return max_found

    @property
    def n_children(self):
        return len(self.children)

    @property
    def depth_first_ordered_nodes(self) -> list:
        if self.is_leaf and len(self.children) == 0:
            return [self]
        else:
            nodes = [self]
            for child in self.children:
                out_nodes = child.depth_first_ordered_nodes
                nodes.extend(out_nodes)
        return nodes

    @property
    def ordered_leaves(self):
        leaves = []
        for node in self.depth_first_ordered_nodes:
            if node.is_leaf and bool(node.data):
                leaves.append(node)
        return leaves

    @property
    def recoverable_ordered_leaves(self):
        child_nodes = []
        for child in self.children:
            child_nodes.extend(child.recoverable_ordered_leaves)
        if child_nodes:
            return child_nodes
        if bool(self.data):
            return [self]
        return []

    @classmethod
    def max_depth(cls, node, depth=0):
        if node.is_leaf:
            return depth
        else:
            max_depths = []
            for child in node.children:
                max_depths.append(cls.max_depth(child, depth + 1))

            return max(max_depths)

    @property
    def total_nodes(self):
        return len(self.depth_first_ordered_nodes)

    def get_num_opt_steps(self):
        return sum(
            [
                len(leaf.chain_trajectory)
                for leaf in self.get_optimization_history()
                if leaf
            ]
        )

    def get_num_grad_calls(self):
        return sum(
            [leaf.grad_calls_made for leaf in self.get_optimization_history()
             if leaf]
        )

    def get_nodes_at_depth(self, depth):
        curr_depth = 0
        nodes_to_iter_through = [self]
        while curr_depth < depth:
            new_nodes_to_iter_through = []
            for node in nodes_to_iter_through:
                new_nodes_to_iter_through.extend(node.children)
            curr_depth += 1
            nodes_to_iter_through = new_nodes_to_iter_through

        return nodes_to_iter_through

    def write_to_disk(self, folder_name: Path, write_qcio: bool = False):
        folder_name = Path(folder_name)

        if folder_name.exists():
            shutil.rmtree(folder_name)
        folder_name.mkdir()

        np.savetxt(fname=folder_name / "adj_matrix.txt", X=self.adj_matrix)

        for node in self.depth_first_ordered_nodes:
            i = node.index
            if node.data:
                node.data.write_to_disk(
                    fp=folder_name / f"node_{i}.xyz", write_history=True, write_qcio=write_qcio
                )
            elif getattr(node, "failed_chain", None) is not None:
                node.failed_chain.write_to_disk(
                    fp=folder_name / f"node_{i}_failed.xyz",
                    write_qcio=write_qcio,
                )

    def draw(self):
        foo = self.adj_matrix - np.identity(len(self.adj_matrix))
        g = nx.from_numpy_array(foo)
        nx.draw_networkx(g)

    def _update_adj_matrix(self, matrix, node):
        matrix_copy = matrix.copy()
        if node.data:
            matrix_copy[node.index, node.index] = 1
        if node.is_leaf:
            return matrix_copy
        else:
            for child in node.children:
                matrix_copy[node.index, child.index] = 1
                matrix_copy = self._update_adj_matrix(
                    matrix=matrix_copy, node=child)

        return matrix_copy

    @property
    def adj_matrix(self):
        # mat = np.zeros((self.total_nodes, self.total_nodes))
        mat = np.zeros((self.max_index + 1, self.max_index + 1))
        mat = self._update_adj_matrix(matrix=mat, node=self)

        return mat

    @classmethod
    def read_from_disk(
        cls,
        folder_name,
        neb_parameters=NEBInputs(),
        chain_parameters=ChainInputs(),
        gi_parameters=GIInputs(),
        optimizer=VelocityProjectedOptimizer(),
        engine=None,
        charge=0,
        multiplicity=1,
    ):
        if isinstance(folder_name, str):
            folder_name = Path(folder_name)
        adj_mat = np.loadtxt(folder_name / "adj_matrix.txt")
        if len(adj_mat.shape) > 0:

            nodes = list(folder_name.glob("node*.xyz"))
            true_node_indices = [int(p.stem.split("_")[1]) for p in nodes]
            node_list_indices = list(range(len(true_node_indices)))

            translator = {}
            for true_ind, local_ind in zip(true_node_indices, node_list_indices):
                translator[true_ind] = local_ind

            neb_nodes = [
                NEB.read_from_disk(
                    nodes[i],
                    chain_parameters=chain_parameters,
                    neb_parameters=neb_parameters,
                    gi_parameters=gi_parameters,
                    optimizer=optimizer,
                    engine=engine,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                for i in range(len(nodes))
            ]
            root = cls._get_node_helper(
                true_node_index=0,
                matrix=adj_mat,
                list_of_nodes=neb_nodes,
                indices_translator=translator,
            )
        else:
            neb_nodes = [
                NEB.read_from_disk(
                    folder_name / "node_0.xyz",
                    chain_parameters=chain_parameters,
                    neb_parameters=neb_parameters,
                    optimizer=optimizer,
                    engine=engine,
                    charge=charge,
                    multiplicity=multiplicity,
                )
            ]
            root = cls(data=neb_nodes[0], children=[], index=0)

        return root

    @classmethod
    def _get_node_helper(
        cls, true_node_index, matrix, list_of_nodes, indices_translator
    ):

        node = list_of_nodes[indices_translator[true_node_index]]
        row = matrix[true_node_index, true_node_index:]
        ind_nonzero_nodes = row.nonzero()[0] + true_node_index
        ind_children = ind_nonzero_nodes[1:]
        if len(ind_children):
            children = [
                cls._get_node_helper(
                    true_node_index=true_child_index,
                    matrix=matrix,
                    list_of_nodes=list_of_nodes,
                    indices_translator=indices_translator,
                )
                for true_child_index in ind_children
                if matrix[true_child_index]
                .nonzero()[0]
                .any()  # i.e. if it was not a 'None' Node
            ]
            return cls(data=node, children=children, index=true_node_index)
        else:
            return cls(data=node, children=[], index=true_node_index)

    # @property
    @cached_property
    def is_leaf(self):
        if self._force_tree_recalc:  # this should never be used but alas
            return self.data.chain_trajectory[-1].is_elem_step()[0]
        else:
            return len(self.children) == 0

    def get_optimization_history(self, node=None):
        if node:
            opt_history = [node.data]
            for child in node.children:
                if child.is_leaf and len(child.children) == 0:
                    opt_history.extend([child.data])
                else:
                    child_opt_history = self.get_optimization_history(child)
                    opt_history.extend(child_opt_history)
            return opt_history
        else:
            return self.get_optimization_history(node=self)

    def greedy_tsopt(
        self,
        engine: Any,
        *,
        run_irc: bool = False,
        dedup: bool = True,
        chain_inputs: ChainInputs = None,
    ) -> list["GreedyTSOptCandidate"]:
        """Greedily TS-opt (and optionally IRC) a guess from *every* NEB run
        in this tree's optimization history, not just the elem-step leaves.

        For each `neb` in `self.get_optimization_history()` with a non-empty
        `chain_trajectory`, takes `neb.chain_trajectory[-1].get_ts_node()` as
        a TS guess and attempts `tsopt.optimize_ts_and_irc` on it. This is a
        wider net than leaf-only TS-opt (e.g. `mepd run --use-tsopt`): it
        doesn't trust the tree's own elem-step/leaf classification and just
        lets TS-opt itself succeed or fail on every candidate.

        With `dedup=True` (default), a guess geometrically identical to one
        already attempted (per `ChainInputs.node_rms_thre`/`node_ene_thre`)
        is skipped -- multiple tree nodes commonly converge to the same
        underlying structure. Entries with falsy/missing `data` (failed
        splits) or an empty `chain_trajectory` are always skipped.

        Never raises: each candidate's outcome is captured in its
        `GreedyTSOptCandidate.result` (a `TSOptResult`), so one failure
        doesn't stop the rest.
        """
        from mepd.tsopt import optimize_ts_and_irc

        parameters = chain_inputs if chain_inputs is not None else ChainInputs()

        candidates: list[GreedyTSOptCandidate] = []
        seen_guesses: list = []
        for source_index, neb in enumerate(self.get_optimization_history()):
            if not neb or not getattr(neb, "chain_trajectory", None):
                continue
            guess = neb.chain_trajectory[-1].get_ts_node()

            if dedup and any(
                self._nodes_match(guess, seen, parameters) for seen in seen_guesses
            ):
                continue
            seen_guesses.append(guess)

            result = optimize_ts_and_irc(guess, engine, run_irc=run_irc)
            candidates.append(
                GreedyTSOptCandidate(
                    source_index=source_index, ts_guess=guess, result=result
                )
            )

        return candidates

    def get_adj_mat_leaves_indices(self):
        matrix = self.adj_matrix
        inds = []
        for i, row in enumerate(matrix):
            if len(row.nonzero()[0]) == 1:
                inds.append(i)
        return inds

    @staticmethod
    def _nodes_match(a, b, parameters: ChainInputs) -> bool:
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
            a_cmp = a.copy() if hasattr(a, "copy") else a
            b_cmp = b.copy() if hasattr(b, "copy") else b
            setattr(a_cmp, "disable_smiles", True)
            setattr(b_cmp, "disable_smiles", True)
            return bool(
                is_identical(
                    a_cmp,
                    b_cmp,
                    fragment_rmsd_cutoff=parameters.node_rms_thre,
                    kcal_mol_cutoff=parameters.node_ene_thre,
                    verbose=False,
                )
            )
        except Exception:
            return bool(
                list(a.structure.symbols) == list(b.structure.symbols)
                and np.allclose(a.coords, b.coords)
            )

    @classmethod
    def _prune_leaf_chain_loops(
        cls, chains: list[Chain], parameters: ChainInputs
    ) -> list[Chain]:
        if len(chains) <= 1:
            return chains

        nodes_seq = [chains[0][0].copy()]
        edge_chain_indices: list[int | None] = []
        for chain_index, chain in enumerate(chains):
            if len(chain.nodes) == 0:
                continue
            start_node = chain[0]
            end_node = chain[-1]
            if not cls._nodes_match(nodes_seq[-1], start_node, parameters):
                nodes_seq.append(start_node.copy())
                edge_chain_indices.append(None)
            nodes_seq.append(end_node.copy())
            edge_chain_indices.append(chain_index)

        pruned_nodes = [nodes_seq[0]]
        pruned_edges: list[int | None] = []
        for step_index in range(1, len(nodes_seq)):
            node = nodes_seq[step_index]
            edge_index = edge_chain_indices[step_index - 1]
            match_index = None
            for i, existing in enumerate(pruned_nodes):
                if cls._nodes_match(existing, node, parameters):
                    match_index = i
                    break
            if match_index is None:
                pruned_nodes.append(node)
                pruned_edges.append(edge_index)
                continue
            pruned_nodes = pruned_nodes[: match_index + 1]
            pruned_edges = pruned_edges[:match_index]

        used_chain_indices: list[int] = []
        for edge_idx in pruned_edges:
            if edge_idx is None:
                continue
            if used_chain_indices and used_chain_indices[-1] == edge_idx:
                continue
            used_chain_indices.append(edge_idx)

        if not used_chain_indices:
            return [chains[0]]
        return [chains[i] for i in used_chain_indices]

    @property
    def output_chain(self):
        chains = []
        for node in self.recoverable_ordered_leaves:
            data = node.data
            if getattr(data, "chain_trajectory", None):
                chains.append(data.chain_trajectory[-1])
            elif getattr(data, "optimized", None) is not None:
                chains.append(data.optimized)
        if len(chains) == 0:
            if self.data and getattr(self.data, "chain_trajectory", None):
                return self.data.chain_trajectory[-1]
            if self.data and getattr(self.data, "optimized", None) is not None:
                return self.data.optimized
            raise ValueError("TreeNode has no leaf chains and no root output chain.")
        chains = self._prune_leaf_chain_loops(chains, parameters=chains[0].parameters)
        out = Chain.from_list_of_chains(
            chains, parameters=chains[0].parameters)
        return out
