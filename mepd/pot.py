from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import networkx as nx
from pydantic import BaseModel, Field, field_serializer


from mepd.molecule import Molecule
from mepd.chain import Chain
from mepd.nodes.node import StructureNode

from typing import Optional


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


class Pot(BaseModel):
    """
    The pot object is the chemical reactor.
    """
    root: Molecule = Field(serialization_fn=lambda m: m.to_serializable())
    target: Molecule = Field(default_factory=lambda: Molecule(
    ), serialization=lambda m: m.to_serializable())
    # this needs to be here because it is needed when we calculates yield.
    multiplier: int = 1

    graph: nx.DiGraph = Field(default_factory=lambda: nx.DiGraph())
    run_time: Optional[float] = None
    rxn_name: Optional[str] = None

    def write_to_disk(self, fp: Path):
        if isinstance(fp, str):
            fp = Path(fp)

        dump = _json_safe(self.model_dump())
        tmp_fp = fp.with_suffix(fp.suffix + ".tmp")
        with open(tmp_fp, 'w+') as f:
            json.dump(dump, f)
        tmp_fp.replace(fp)

    @classmethod
    def read_from_disk(cls, fp: Path):
        if isinstance(fp, str):
            fp = Path(fp)
        loaded = json.load(open(fp))
        return cls.from_dict(loaded)

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _graph_link_entries(graph_payload: dict) -> list[dict]:
        if not isinstance(graph_payload, dict):
            return []
        links = graph_payload.get("links")
        if isinstance(links, list):
            return links
        edges = graph_payload.get("edges")
        if isinstance(edges, list):
            return edges
        return []

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        graph_payload = data.get("graph") or {}
        # Convert Molecule objects in graph nodes to serializable format
        data['root'] = data['root'].to_serializable()
        data['target'] = data['target'].to_serializable()
        for node_data in graph_payload.get("nodes", []):
            if 'molecule' in node_data and node_data['molecule'] is not None:
                node_data['molecule'] = node_data['molecule'].to_serializable()
            if 'environment' in node_data and node_data['environment'] is not None:
                node_data['environment'] = node_data['environment'].to_serializable()
            if "conformers" in node_data:
                for conformer in node_data["conformers"]:
                    if conformer.get('graph') is not None:
                        conformer['graph'] = conformer['graph'].to_serializable()
            if "td" in node_data:
                if node_data['td'].get('graph') is not None:
                    node_data['td']['graph'] = node_data['td']['graph'].to_serializable()

        link_data = self._graph_link_entries(graph_payload)
        for i, link in enumerate(link_data):
            list_of_nebs = link.get('list_of_nebs', [])
            link_data[i]['list_of_nebs'] = list_of_nebs
            source = link.get("source")
            target = link.get("target")
            if source is None or target is None:
                continue
            edge_key = (source, target)
            reverse_edge_key = (target, source)
            edge_attrs = {}
            if self.graph.has_edge(*edge_key):
                edge_attrs = self.graph.edges[edge_key]
            elif self.graph.has_edge(*reverse_edge_key):
                edge_attrs = self.graph.edges[reverse_edge_key]
            for j, _ in enumerate(list_of_nebs):
                nebs = edge_attrs.get('list_of_nebs', [])
                if j < len(nebs):
                    neb = nebs[j]
                    if hasattr(neb, "model_dump"):
                        link_data[i]['list_of_nebs'][j] = neb.model_dump()
                    else:
                        link_data[i]['list_of_nebs'][j] = _json_safe(neb)

        return data

    def model_post_init(self, __context):
        # Only synthesize the root node for newly-created empty graphs.
        # Deserialized graphs should retain their persisted node attributes.
        if self.graph.number_of_nodes() > 0:
            return
        multiplied_root = self.root * self.multiplier
        self.root = multiplied_root
        self.graph.add_node(
            0, molecule=multiplied_root, converged=False, root=True
        )  # root=True is for drawing

    @field_serializer("graph")
    def serialize_graph(self, graph: nx.DiGraph, _info):
        try:
            return nx.node_link_data(graph, edges="links")
        except TypeError:
            return nx.node_link_data(graph)

    @classmethod
    def from_dict(cls, data):
        graph_payload = data.get("graph") or {}
        edge_key = "links" if "links" in graph_payload else "edges"
        try:
            graph = nx.node_link_graph(graph_payload, edges=edge_key)
        except TypeError:
            graph = nx.node_link_graph(graph_payload)
        for node, node_data in graph.nodes(data=True):
            if 'molecule' in node_data:
                node_data['molecule'] = Molecule.from_serializable(
                    node_data['molecule'])
            if 'environment' in node_data:
                node_data['environment'] = Molecule.from_serializable(
                    node_data['environment'])

            if "conformers" in node_data:
                for i, conformer in enumerate(node_data["conformers"]):
                    node_data['conformers'][i] = StructureNode.from_serializable(
                        conformer)

            if "td" in node_data:
                node_data['td'] = StructureNode.from_serializable(
                    node_data['td'])

        link_data = cls._graph_link_entries(graph_payload)
        for i, _ in enumerate(link_data):
            list_of_nebs = link_data[i].get("list_of_nebs") or []
            link_data[i]["list_of_nebs"] = list_of_nebs
            for j, _ in enumerate(list_of_nebs):
                nodes = list_of_nebs[j].get("nodes", [])
                nodes = [StructureNode.from_serializable(n) for n in nodes]
                list_of_nebs[j]['nodes'] = nodes
                list_of_nebs[j] = Chain.model_validate(list_of_nebs[j])

        root = Molecule.from_serializable(data['root'])
        target = Molecule.from_serializable(data['target'])

        return cls(
            root=root,
            target=target,
            multiplier=data.get('multiplier', 1),
            graph=graph,
            run_time=data.get('run_time'),
            rxn_name=data.get('rxn_name')
        )

    @property
    def average_node_degree(self):
        """
        Average number of neighbors a node has
        """
        G = self.graph
        degrees = G.degree()
        sum_of_edges = sum([v for k, v in degrees])
        avg_node_degree = sum_of_edges / len(degrees)
        return avg_node_degree

    @property
    def number_of_nodes(self):
        """returns the number of nodes"""
        return len(self.graph.nodes)

    @property
    def clustering_coefficient(self):
        """
        Something about clusters in the graph
        """
        return nx.average_clustering(self.graph)

    def __str__(self):
        return f"POT {self.root.smiles}"

    def __repr__(self):
        return str(self)

    @property
    def leaves(self):
        """
        Returns a list of leaf nodes, defined as nodes with 'in degree' equal to zero
        """
        return [x[0] for x in self.graph.in_degree() if x[1] == 0]

    @property
    def reactions_in_the_pot(self):
        """it returs a list of the unique reactions that happened in the pot"""
        return sorted(
            list(set([self.graph.edges[x]["reaction"]
                 for x in self.graph.edges]))
        )

    def in_which_node_is_this_molecule(self, molecule: Molecule) -> list[int]:
        """
        we use this function to get number of node of where a mol is
        """
        where = []
        for node in self.graph.nodes:
            content = self.graph.nodes[node]["molecule"]
            boo = molecule.is_subgraph_isomorphic_to(content)
            if boo:
                where.append(node)
        return where

    @property
    def target_indexes(self) -> list[int]:
        if not self.target.is_empty():
            list_of_nodes = self.in_which_node_is_this_molecule(self.target)
            if len(list_of_nodes) == 0:
                raise ValueError(
                    "Target is not present in the reaction network.")
            return list_of_nodes

        else:
            raise ValueError("This pot has been created without a target.")

    @property
    def score(self):
        """
        this tries to calculate some scoring
        """
        denominator = len(self.leaves) if len(self.leaves) > 0 else 1
        how_many_times = sum(
            self.target.is_subgraph_isomorphic_to(
                self.graph.nodes[x]["molecule"])
            for x in self.leaves
        )
        score = how_many_times / denominator
        return score

