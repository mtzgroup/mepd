import threading
from contextlib import nullcontext

import networkx as nx
from timeout_timer import timeout


class TimeoutIsomorphism(Exception):
    pass


class SubGraphMatcher:
    """
    This class is the normal matcher, used by molecules that do not have R groups.
    This is also used by templates with R removed.
    Neighbors and charges are used towards isomorphism
    ISMAGS ON
    """

    GM = nx.isomorphism.ISMAGS

    def __init__(self, mol, verbosity=0, timeout_seconds=10):
        """
        Initialise this graph matcher with a particular mol
        """
        self.mol = mol
        self.verbosity = verbosity
        self.timeout_seconds = timeout_seconds

    def _node_matcher(self, n1, n2):
        nodes_equiv = (
            n1["element"] == n2["element"]
            and n1["neighbors"] == n2["neighbors"]
            and n1["charge"] == n2["charge"]
        )
        return nodes_equiv

    def _node_matcher_no_charge(self, n1, n2):
        nodes_equiv = (
            n1["element"] == n2["element"] and n1["neighbors"] == n2["neighbors"]
        )
        return nodes_equiv

    def _edge_matcher(self, e1, e2):
        edge_equiv = e1["bond_order"] == e2["bond_order"]
        return edge_equiv

    def _edge_match_by_existence(self, e1, e2):
        """
        will only check whether a bond exists in both cases.
        Adding it cause aromatics might be a double or a single
        but there is still a bond present
        """
        edge_equiv = bool(e1["bond_order"] and e2["bond_order"])
        return edge_equiv

    def _timeout_context(self):
        """
        SIGALRM-based timeout handling only works on the main thread.
        In worker threads (parallel autosplitting), run without this timeout
        guard to avoid `ValueError: signal only works in main thread`.
        """
        try:
            timeout_seconds = float(self.timeout_seconds)
        except Exception:
            timeout_seconds = 0.0
        if timeout_seconds <= 0:
            return nullcontext()
        if threading.current_thread() is not threading.main_thread():
            return nullcontext()
        return timeout(timeout_seconds, exception=TimeoutIsomorphism)

    @staticmethod
    def _safe_smiles_label(mol) -> str:
        """
        Best-effort label for timeout logging without any SMILES conversion.
        Timeout handlers must never trigger expensive/cascading chemistry work.
        """
        try:
            n_nodes = len(mol.nodes)
            n_edges = len(mol.edges)
            return f"<graph nodes={n_nodes} edges={n_edges}>"
        except Exception:
            return "<graph_unavailable>"

    def is_isomorphic(self, g):
        """
        returns a boolean to see if it's isomorphic.
        """
        try:
            with self._timeout_context():
                GM = self.GM(
                    self.mol,
                    g,
                    node_match=self._node_matcher,
                    edge_match=self._edge_matcher,
                )
                result = GM.is_isomorphic()
        except TimeoutIsomorphism:
            print(
                "A SubGraphMatcher timeout error occurred in is_isomorphic "
                f"{self._safe_smiles_label(self.mol)} -> {self._safe_smiles_label(g)}."
            )
            result = False
        return result

    def is_bond_isomorphic(self, g):
        """
        returns a boolean to see if it's isomorphic in connectivity
        """
        try:
            with self._timeout_context():
                GM = self.GM(
                    self.mol,
                    g,
                    node_match=self._node_matcher_no_charge,
                    edge_match=self._edge_match_by_existence,
                )
                result = GM.is_isomorphic()
        except TimeoutIsomorphism:
            print(
                "A SubGraphMatcher timeout error occurred in is_isomorphic "
                f"{self._safe_smiles_label(self.mol)} -> {self._safe_smiles_label(g)}."
            )
            result = False
        return result

    def is_subgraph_isomorphic(self, g):
        """
        Returns a boolean if self is subgraph isomorphic of g
        """
        try:
            with self._timeout_context():
                GM = self.GM(
                    self.mol,
                    g,
                    node_match=self._node_matcher,
                    edge_match=self._edge_matcher,
                )
                result = GM.subgraph_is_isomorphic()
        except TimeoutIsomorphism:
            print(
                "A SubGraphMatcher timeout error occurred in is_subgraph_isomorphic "
                f"{self._safe_smiles_label(self.mol)} -> {self._safe_smiles_label(g)}."
            )
            result = False
        return result

    def get_subgraph_isomorphisms(self, g):
        """
        Returns a list of dictionaries of node mappings.
        The keys of each dictionary are nodes in self.mol, while values are nodes in g.
        """
        try:
            with self._timeout_context():
                GM = self.GM(
                    self.mol,
                    g,
                    node_match=self._node_matcher,
                    edge_match=self._edge_matcher,
                )
                isos = [a for a in GM.subgraph_isomorphisms_iter()]
        except TimeoutIsomorphism:
            print(
                "A SubGraphMatcher timeout error occurred in get_subgraph_isomorphisms "
                f"{self._safe_smiles_label(self.mol)} -> {self._safe_smiles_label(g)}."
            )
            isos = []
        return isos

    def largest_common_subgraph(self, g):
        """
        This returns the ISMAGS largest common subgraph.
        """
        GM = self.GM(
            self.mol, g, node_match=self._node_matcher, edge_match=self._edge_matcher
        )
        return list(GM.largest_common_subgraph())


