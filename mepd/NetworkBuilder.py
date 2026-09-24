import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import argrelmin

from mepd.chain import Chain
from mepd.inputs import ChainInputs, NetworkInputs
from mepd.pot import Pot
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import is_identical

from typing import Union


def _stereochemical_smiles_key(node: StructureNode) -> str:
    structure = getattr(node, "structure", None)
    if structure is None:
        return ""
    identifiers = getattr(structure, "identifiers", None)
    if identifiers is None:
        return ""
    for attr in (
        "canonical_isomeric_smiles",
        "canonical_isomeric_explicit_hydrogen_smiles",
        "canonical_isomeric_explicit_hydrogen_mapped_smiles",
        "smiles",
    ):
        value = getattr(identifiers, attr, None)
        if value:
            text = str(value).strip()
            if text:
                return text
    return ""


@dataclass
class NetworkBuilder:
    """
    Builds/dedupes a reaction network `Pot` from a set of already-completed
    MSMEP results (i.e. `mepd run --recursive`/`--parallel` output tree
    directories). Retropaths-agnostic: candidate reactant/product pairs and
    the MSMEP runs on them are entirely the caller's responsibility -- this
    class only consumes already-produced results.
    """

    data_dir: Path

    start: Union[StructureNode, str, None] = None
    end: Union[StructureNode, str, None] = None

    charge: int = 0
    spinmult: int = 1

    network_inputs: NetworkInputs = None
    chain_inputs: ChainInputs = None

    def __post_init__(self):
        if self.network_inputs is None:
            self.network_inputs = NetworkInputs()

        if self.chain_inputs is None:
            self.chain_inputs = ChainInputs()

    def _equality_function(self, a: StructureNode, b: StructureNode):
        return is_identical(a, b, fragment_rmsd_cutoff=self.chain_inputs.node_rms_thre, verbose=False,
                            kcal_mol_cutoff=self.chain_inputs.node_ene_thre)

    def _graph_equivalent(self, a: StructureNode, b: StructureNode) -> bool:
        if getattr(a, "graph", None) is None or getattr(b, "graph", None) is None:
            return self._equality_function(a, b)
        if not a.graph.is_isomorphic_to(b.graph):
            return False

        stereo_a = _stereochemical_smiles_key(a)
        stereo_b = _stereochemical_smiles_key(b)
        if stereo_a and stereo_b:
            return stereo_a == stereo_b
        return True

    def _get_ind_td(self, ref_list, td):
        inds = np.where(
            [self._graph_equivalent(a, b) for a, b in list(
                itertools.product([td], ref_list))]
        )[0]
        assert len(inds) >= 1, "No matches found. Network cannot be constructed."
        return inds[0]

    def _get_relevant_leaves(self, fp):
        adj_mat_fp = fp / "adj_matrix.txt"
        adj_mat = np.loadtxt(adj_mat_fp)
        if adj_mat.size == 1:
            return [
                Chain.from_xyz(fp / "node_0.xyz", self.chain_inputs)
            ]
        else:

            a = np.sum(adj_mat, axis=1)
            inds_leaves = np.where(a == 1)[0]
            chains = [
                Chain.from_xyz(
                    fp / f"node_{ind}.xyz",
                    self.chain_inputs,
                )
                for ind in inds_leaves
            ]
            return chains

    def _get_relevant_conformers(self, node_ind: int):
        all_conformers = []

        for key in self.leaf_objects.keys():

            pull_reactants = False
            pull_products = False
            vals = key.split("-")
            if node_ind == int(vals[0]):
                pull_reactants = True

            elif node_ind == int(vals[1]):
                pull_products = True

            if pull_reactants:
                for chain in self.leaf_objects[key]:
                    all_conformers.append(chain[0])

            elif pull_products:
                for chain in self.leaf_objects[key]:
                    all_conformers.append(chain[-1])

        return all_conformers

    def _get_relevant_edges(self, edge_dir, ind):
        relevant_keys = []
        for key, val in edge_dir.items():
            node_edges_str = key.split("-")
            node_edges = [int(v) for v in node_edges_str]
            if ind in node_edges:
                relevant_keys.append(key)
        return relevant_keys

    def _energies_fail(self, chain: Chain):
        try:
            chain.energies
            return False
        except Exception:
            return True

    def _load_network_data(self, msmep_paths: list[Path]):
        structures = []  # list of Node3D
        edges = {}
        leaf_objects = {}
        for fp in msmep_paths:
            if self.network_inputs.verbose:
                print(f"\tDoing: {fp}. Len: {len(structures)}")

            leaves = self._get_relevant_leaves(fp)
            for leaf in leaves:
                reactant = leaf[0]
                product = leaf[-1]
                out_leaf = leaf
                out_leaf_rev = out_leaf.copy()
                out_leaf_rev.nodes.reverse()
                if self._energies_fail(out_leaf):
                    if self.network_inputs.verbose:
                        print(
                            f"\t\t{fp} had a leaf with failed energies. Might result in disconnected nodes.")
                    continue

                if self.network_inputs.tolerate_kinks:
                    elementary_step = True
                else:
                    n_minima = len(argrelmin(out_leaf.energies)[0])

                    elementary_step = n_minima == 0
                if elementary_step:
                    reactant_comparison = all([not self._graph_equivalent(
                        reactant, reference) for reference in structures])
                    product_comparison = all([not self._graph_equivalent(
                        product, reference) for reference in structures])

                    if reactant_comparison or len(structures) == 0:
                        structures.append(reactant)
                    if product_comparison or len(structures) == 1:
                        structures.append(product)

                    ind_r = self._get_ind_td(ref_list=structures, td=reactant)
                    ind_p = self._get_ind_td(ref_list=structures, td=product)
                    eA = leaf.get_eA_chain()
                    edge_name = f"{ind_r}-{ind_p}"
                    rev_edge_name = f"{ind_p}-{ind_r}"
                    rev_eA = (leaf.energies.max() - leaf[-1].energy) * 627.5

                    if edge_name in edges.keys():
                        edges[edge_name].append(eA)
                        leaf_objects[edge_name].append(out_leaf)
                    else:
                        edges[edge_name] = [eA]
                        leaf_objects[edge_name] = [out_leaf]

                    if rev_edge_name in edges.keys():
                        edges[rev_edge_name].append(rev_eA)
                        leaf_objects[rev_edge_name].append(out_leaf_rev)

                    else:
                        edges[rev_edge_name] = [rev_eA]
                        leaf_objects[rev_edge_name] = [out_leaf_rev]

        self.leaf_objects = leaf_objects
        return structures, edges

    def _add_all_nodes(self, pot: Pot, structures: list):
        for i, mol_to_add in enumerate(structures):
            node_ind = i
            if self.network_inputs.verbose:
                print(f"Adding node {node_ind}")

            relevant_conformers = self._get_relevant_conformers(node_ind)
            if self.network_inputs.verbose:
                print(f"\tIt had {len(relevant_conformers)} conformers")

            pot.graph.add_node(
                node_ind,
                molecule=mol_to_add.graph,
                converged=False,
                td=mol_to_add,
                stereochemical_smiles=_stereochemical_smiles_key(mol_to_add),
                node_energy=mol_to_add.energy,
                node_energies=[
                    conformer.energy for conformer in relevant_conformers],
                conformers=relevant_conformers)

        return pot

    def _get_lowest_barrier_height(self, edges: dict, edgelabel: str):
        return float(np.min(edges[edgelabel]))

    def _add_all_edges(self, pot: Pot, structures: list, edges: dict):
        for i, _ in enumerate(structures):
            node_ind = i
            rel_edges = self._get_relevant_edges(edges, node_ind)
            for edgelabel in rel_edges:
                label = edgelabel
                vals = label.split("-")
                label_rev = f"{vals[1]}-{vals[0]}"
                lowest_barrier_height_fwd = self._get_lowest_barrier_height(
                    edges, edgelabel
                )
                lowest_barrier_height_rev = self._get_lowest_barrier_height(
                    edges, label_rev
                )

                if int(vals[1]) == node_ind:  # skipping to avoid self loops
                    continue

                if lowest_barrier_height_fwd <= self.network_inputs.maximum_barrier_height:

                    pot.graph.add_edge(
                        node_ind,
                        int(vals[1]),
                        reaction=f"eA ({edgelabel}): {np.min(edges[edgelabel])}",
                        list_of_nebs=[self.get_lowest_barrier_chain(label)],
                        barrier=lowest_barrier_height_fwd,
                        exp_neg_barrier=np.exp(-np.min(edges[edgelabel])),
                    )
                if lowest_barrier_height_rev <= self.network_inputs.maximum_barrier_height:
                    if int(vals[1]) == node_ind:
                        continue
                    pot.graph.add_edge(
                        int(vals[1]),
                        node_ind,
                        reaction=f"eA ({label_rev}):{np.min(edges[label_rev])}",
                        list_of_nebs=[self.get_lowest_barrier_chain(label_rev)],
                        barrier=lowest_barrier_height_rev,
                        exp_neg_barrier=np.exp(-np.min(edges[label_rev])),
                    )
        return pot

    def create_rxn_network_from_paths(self, msmep_paths: list[Path]):
        structures, edges = self._load_network_data(msmep_paths=msmep_paths)
        if not structures:
            raise ValueError(
                "No valid elementary-step leaves were found while building the reaction network."
            )
        pot = Pot.model_validate({'root': structures[0].graph})
        pot = self._add_all_nodes(pot, structures=structures)
        pot = self._add_all_edges(pot, structures=structures, edges=edges)
        return pot

    def get_lowest_barrier_chain(self, edge: str):
        edge_data = self.leaf_objects[edge]
        assert len(edge_data) >= 1, f"{edge} was not found in network."
        eAs = [c.get_eA_chain() for c in edge_data]
        best_ind = np.argmin(eAs)
        return edge_data[best_ind]

