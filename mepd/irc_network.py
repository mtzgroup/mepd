from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.molecule import Molecule
from mepd.pot import Pot


_EDGE_FILENAME_RE = re.compile(r"(?:^|_)edge_(?P<source>\d+)_(?P<target>\d+)(?:_|\.|$)")
HARTREE_TO_KCAL_MOL = 627.509474


@dataclass(frozen=True)
class IRCNetworkScan:
    pot: Pot
    xyz_files: tuple[Path, ...]
    skipped_xyz_files: tuple[Path, ...]


def _filename_edge_indices(path: Path) -> tuple[int, int] | None:
    match = _EDGE_FILENAME_RE.search(path.name)
    if match is None:
        return None
    return int(match.group("source")), int(match.group("target"))


def _load_irc_chain(
    xyz_path: Path,
    *,
    charge: int,
    multiplicity: int,
) -> Chain:
    energy_path = xyz_path.with_suffix("").with_suffix(".energies")
    if not energy_path.exists():
        # For foo.irc.xyz, Path.with_suffix("").with_suffix(".energies") gives
        # foo.energies. The MEPD convention is foo.irc.energies.
        energy_path = xyz_path.with_suffix(".energies")
    if not energy_path.exists():
        raise FileNotFoundError(f"No matching energy file for {xyz_path.name}")

    chain = Chain.from_xyz(
        xyz_path,
        parameters=ChainInputs(),
        charge=charge,
        spinmult=multiplicity,
    )
    energies = np.asarray(np.loadtxt(energy_path), dtype=float).reshape(-1)
    if len(chain.nodes) != len(energies):
        raise ValueError(
            f"{xyz_path.name} has {len(chain.nodes)} frames but "
            f"{energy_path.name} has {len(energies)} energies."
        )
    if not np.all(np.isfinite(energies)):
        raise ValueError(f"{energy_path.name} contains non-finite energies.")
    for node, energy in zip(chain.nodes, energies):
        node._cached_energy = float(energy)
    return chain


def _reverse_chain(chain: Chain) -> Chain:
    return Chain.model_validate(
        {
            "nodes": [node.copy() for node in reversed(chain.nodes)],
            "parameters": chain.parameters,
        }
    )


def _same_connectivity(node, reference) -> bool:
    node_graph = getattr(node, "graph", None)
    reference_graph = getattr(reference, "graph", None)
    if node_graph is None or reference_graph is None:
        return False
    node_heavy = node_graph.remove_Hs()
    reference_heavy = reference_graph.remove_Hs()
    if node_heavy.is_empty() and reference_heavy.is_empty():
        return node_graph.is_bond_isomorphic_to(reference_graph)
    return node_heavy.is_bond_isomorphic_to(reference_heavy)


def _register_species(node, species: list) -> int:
    for index, reference in enumerate(species):
        if _same_connectivity(node, reference):
            if (
                node._cached_energy is not None
                and (
                    reference._cached_energy is None
                    or float(node._cached_energy) < float(reference._cached_energy)
                )
            ):
                species[index] = node.copy()
            return index
    species.append(node.copy())
    return len(species) - 1


def _add_chain_edge(
    graph: nx.DiGraph,
    *,
    source: int,
    target: int,
    chain: Chain,
    barrier: float,
    reverse_barrier: float,
    reaction: str,
    xyz_path: Path,
    energy_path: Path,
    filename_edge: tuple[int, int] | None,
) -> None:
    if graph.has_edge(source, target):
        attrs = graph.edges[(source, target)]
        attrs.setdefault("list_of_nebs", []).append(chain)
        attrs.setdefault("source_files", []).append(str(xyz_path))
        attrs.setdefault("energy_files", []).append(str(energy_path))
        attrs.setdefault("filename_edges", []).append(filename_edge)
        if barrier < float(attrs.get("barrier", float("inf"))):
            attrs["barrier"] = barrier
            attrs["reverse_barrier"] = reverse_barrier
            attrs["pair_barrier_sum"] = barrier + reverse_barrier
            attrs["reaction"] = reaction
        return
    graph.add_edge(
        source,
        target,
        reaction=reaction,
        list_of_nebs=[chain],
        barrier=barrier,
        reverse_barrier=reverse_barrier,
        pair_barrier_sum=barrier + reverse_barrier,
        source_files=[str(xyz_path)],
        energy_files=[str(energy_path)],
        filename_edges=[filename_edge],
        generated_by="irc_network_scan",
    )


def build_irc_network(
    directory: str | Path,
    *,
    pattern: str = "*.xyz",
    recursive: bool = False,
    charge: int = 0,
    multiplicity: int = 1,
    strict: bool = False,
) -> IRCNetworkScan:
    """Build a standard MEPD ``Pot`` by scanning IRC XYZ/energy pairs.

    Network nodes are determined from endpoint molecular connectivity inferred
    from each geometry. Filename ``edge_<source>_<target>`` values are retained
    only as provenance metadata. XYZ files without a sibling ``.energies`` file
    are skipped, which naturally excludes TS-only refinement artifacts.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    candidates = sorted(
        directory.rglob(pattern) if recursive else directory.glob(pattern)
    )
    graph = nx.DiGraph()
    species: list = []
    loaded: list[Path] = []
    skipped: list[Path] = []
    first_source: int | None = None
    last_target: int | None = None

    for xyz_path in candidates:
        energy_path = xyz_path.with_suffix(".energies")
        if not energy_path.exists():
            skipped.append(xyz_path)
            continue
        try:
            chain = _load_irc_chain(
                xyz_path,
                charge=charge,
                multiplicity=multiplicity,
            )
        except Exception:
            if strict:
                raise
            skipped.append(xyz_path)
            continue

        source = _register_species(chain.nodes[0], species)
        target = _register_species(chain.nodes[-1], species)
        if source == target:
            if strict:
                raise ValueError(
                    f"{xyz_path.name} has connectivity-identical endpoints."
                )
            skipped.append(xyz_path)
            continue
        if first_source is None:
            first_source = source
        last_target = target
        peak = float(np.max(chain.energies))
        forward_barrier = max(
            0.0, (peak - float(chain.energies[0])) * HARTREE_TO_KCAL_MOL
        )
        reverse_barrier = max(
            0.0, (peak - float(chain.energies[-1])) * HARTREE_TO_KCAL_MOL
        )
        reaction = xyz_path.name.removesuffix(".xyz")
        filename_edge = _filename_edge_indices(xyz_path)
        _add_chain_edge(
            graph,
            source=source,
            target=target,
            chain=chain,
            barrier=forward_barrier,
            reverse_barrier=reverse_barrier,
            reaction=reaction,
            xyz_path=xyz_path,
            energy_path=energy_path,
            filename_edge=filename_edge,
        )
        _add_chain_edge(
            graph,
            source=target,
            target=source,
            chain=_reverse_chain(chain),
            barrier=reverse_barrier,
            reverse_barrier=forward_barrier,
            reaction=reaction,
            xyz_path=xyz_path,
            energy_path=energy_path,
            filename_edge=(
                (filename_edge[1], filename_edge[0])
                if filename_edge is not None
                else None
            ),
        )
        loaded.append(xyz_path)

    if not loaded:
        raise ValueError(
            f"No usable IRC XYZ/energy pairs found in {directory} with pattern {pattern!r}."
        )

    node_indices = list(range(len(species)))
    for node_index, node in enumerate(species):
        molecule = getattr(node, "graph", None) or Molecule()
        graph.add_node(
            node_index,
            molecule=molecule,
            td=node,
            converged=True,
            generated_by="irc_network_scan",
            root=node_index == first_source,
            requested_target=node_index == last_target,
            node_energy=float(node._cached_energy),
            connectivity_smiles=molecule.force_smiles(),
        )

    root_index = first_source if first_source is not None else node_indices[0]
    target_index = last_target if last_target is not None else node_indices[-1]
    if target_index == root_index and len(node_indices) > 1:
        target_index = next(
            index for index in reversed(node_indices) if index != root_index
        )
        graph.nodes[last_target]["requested_target"] = False
        graph.nodes[target_index]["requested_target"] = True
    pot = Pot(
        root=graph.nodes[root_index]["molecule"],
        target=graph.nodes[target_index]["molecule"],
        rxn_name=directory.name,
    )
    pot.graph = graph
    return IRCNetworkScan(
        pot=pot,
        xyz_files=tuple(loaded),
        skipped_xyz_files=tuple(skipped),
    )
