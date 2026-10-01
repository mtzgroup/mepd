"""Multi-step route search over any single-step proposer.

An AND-OR tree in the manner of Retro* (Chen et al. 2020): a molecule (OR)
is solved by any one of its steps, a step (AND) needs all its precursors.
Costs: a step costs -log(score); a molecule in stock costs 0; a molecule not
yet expanded is estimated by `value` (from its SA score, or 0, which makes
the search Retro*-0); a molecule past `max_depth` with nothing in stock is a
dead end. Each iteration follows the cheapest partial route from the target
down to one of its unexpanded molecules and expands it; costs are then
updated back up to the target. Routes are read off the tree at the end, the
k cheapest solved ones first.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from mepd.retro import chem
from mepd.retro.proposers import Step
from mepd.retro.stock import Stock

INF = math.inf


@dataclass(eq=False)
class MolNode:
    smiles: str
    depth: int
    parent: Optional["RxnNode"] = None
    stock: Optional[str] = None           # "stock" | "small" | None
    children: list = field(default_factory=list)
    expanded: bool = False
    value: float = 0.0                    # cost to solve it, as far as is known

    @property
    def solved(self) -> bool:
        return self.stock is not None or any(r.solved for r in self.children)

    def ancestors(self):
        node = self
        while node.parent is not None:
            node = node.parent.parent
            yield node.smiles


@dataclass(eq=False)
class RxnNode:
    step: Step
    parent: MolNode
    children: list = field(default_factory=list)
    value: float = 0.0

    @property
    def solved(self) -> bool:
        return all(c.solved for c in self.children)


@dataclass
class SearchResult:
    target: str
    routes: list                          # route dicts, cheapest first
    solved: bool
    iterations: int
    expansions: int
    seconds: float
    molecules: int
    best_partial: Optional[dict] = None   # the cheapest route so far when none is solved


class RetroSearch:
    def __init__(self, proposer: Callable[[str, int], list[Step]], stock: Stock, *, max_depth: int = 6,
                 expansion_width: int = 10, value: str = "sa", value_weight: float = 0.5):
        self.proposer, self.stock = proposer, stock
        self.max_depth, self.width = int(max_depth), int(expansion_width)
        self.value_kind, self.value_weight = value, float(value_weight)
        self._cache: dict[str, list[Step]] = {}
        self.expansions = 0

    # ---------------------------------------------------------------- costs
    def _estimate(self, smiles: str) -> float:
        if self.value_kind == "zero":
            return 0.0
        return self.value_weight * max(0.0, chem.sa_score(smiles) - 1.0)

    def _leaf(self, smiles: str, depth: int, parent: Optional[RxnNode]) -> MolNode:
        node = MolNode(smiles, depth, parent, stock=self.stock.why(smiles))
        if node.stock is not None:
            node.value = 0.0
        elif depth >= self.max_depth:
            node.value, node.expanded = INF, True
        else:
            node.value = self._estimate(smiles)
        return node

    @staticmethod
    def _rxn_value(r: RxnNode) -> float:
        return r.step.cost + sum(c.value for c in r.children)

    def _update(self, node: MolNode) -> None:
        """Recompute costs from `node` up to the target."""
        while node is not None:
            if node.stock is None and node.expanded:
                node.value = min((r.value for r in node.children), default=INF)
            r = node.parent
            if r is None:
                return
            r.value = self._rxn_value(r)
            node = r.parent

    # --------------------------------------------------------------- search
    def _select(self, node: MolNode) -> Optional[MolNode]:
        """The unexpanded molecule on the cheapest partial route that still
        has one (once the cheapest route is solved, the next cheapest: more
        routes, not only the first)."""
        if node.value == INF or node.stock is not None:
            return None
        if not node.expanded:
            return node
        for r in sorted(node.children, key=lambda r: r.value):
            if r.value == INF:
                break
            # The hardest open precursor first: it decides whether the step works.
            for c in sorted(r.children, key=lambda c: -c.value):
                if c.solved and not c.children:
                    continue
                leaf = self._select(c)
                if leaf is not None:
                    return leaf
        return None

    def _expand(self, node: MolNode) -> None:
        if node.smiles not in self._cache:
            self._cache[node.smiles] = self.proposer(node.smiles, self.width)
            self.expansions += 1
        node.expanded = True
        banned = {node.smiles, *node.ancestors()}
        for step in self._cache[node.smiles]:
            if banned.intersection(step.reactants):
                continue   # a cycle back to a molecule on the way here
            r = RxnNode(step, node)
            r.children = [self._leaf(s, node.depth + 1, r) for s in step.reactants]
            r.value = self._rxn_value(r)
            node.children.append(r)
        self._update(node)

    def run(self, target: str, *, max_iterations: int = 200, time_limit: float = 300.0, routes: int = 5,
            stop_when_solved: bool = False, on_iteration: Optional[Callable] = None) -> SearchResult:
        target = chem.canonical(target)
        if target is None:
            raise ValueError("the target is not a valid SMILES")
        root = MolNode(target, 0)
        root.value = self._estimate(target)
        t0, it = time.time(), 0
        while it < max_iterations and time.time() - t0 < time_limit:
            leaf = self._select(root)
            if leaf is None:
                break
            self._expand(leaf)
            it += 1
            if on_iteration is not None:
                on_iteration(it, root)
            if stop_when_solved and root.solved:
                break
        found = extract_routes(root, routes)
        return SearchResult(target, found, root.solved, it, self.expansions, time.time() - t0, _count(root),
                            best_partial=None if found else best_partial(root))


def _count(root: MolNode) -> int:
    n, stack = 0, [root]
    while stack:
        m = stack.pop()
        n += 1
        for r in m.children:
            stack.extend(r.children)
    return n


# -------------------------------------------------------------------- routes
def _mol_json(m: MolNode, children: list) -> dict:
    d = {"smiles": m.smiles, "in_stock": m.stock}
    if children:
        d["children"] = children
    return d


def extract_routes(root: MolNode, k: int) -> list[dict]:
    """The k cheapest solved routes: {"cost", "tree", "steps", "leaves"}."""
    memo: dict = {}

    def best(m: MolNode) -> list[tuple[float, dict]]:
        if id(m) in memo:
            return memo[id(m)]
        if m.stock is not None:
            memo[id(m)] = [(0.0, _mol_json(m, []))]
            return memo[id(m)]
        options = []
        for r in m.children:
            if not r.solved:
                continue
            subs = [best(c) for c in r.children]
            for combo in itertools.islice(itertools.product(*subs), 2000):
                cost = r.step.cost + sum(c for c, _ in combo)
                options.append((cost, _mol_json(m, [{**r.step.to_json(), "reactants": [t for _, t in combo]}])))
        options.sort(key=lambda o: o[0])
        memo[id(m)] = options[:k]
        return memo[id(m)]

    return [route_record(tree, cost) for cost, tree in best(root)]


def best_partial(root: MolNode) -> Optional[dict]:
    """The cheapest route as far as it goes (unsolved molecules as leaves)."""
    def walk(m: MolNode) -> dict:
        r = min((r for r in m.children if r.value < INF), key=lambda r: r.value, default=None)
        if m.stock is not None or r is None:
            return _mol_json(m, [])
        return _mol_json(m, [{**r.step.to_json(), "reactants": [walk(c) for c in r.children]}])

    if not root.children:
        return None
    return route_record(walk(root), root.value)


def route_record(tree: dict, cost: float) -> dict:
    steps, leaves = [], []

    def walk(m: dict, depth: int) -> None:
        kids = m.get("children") or []
        if not kids:
            leaves.append({"smiles": m["smiles"], "in_stock": m.get("in_stock")})
            return
        rx = kids[0]
        steps.append({"product": m["smiles"], "reactants": [c["smiles"] for c in rx["reactants"]],
                      "score": rx["score"], "method": rx["method"], "depth": depth,
                      **({"info": rx["info"]} if rx.get("info") else {})})
        for c in rx["reactants"]:
            walk(c, depth + 1)

    walk(tree, 0)
    return {"cost": round(cost, 4), "score": round(math.exp(-cost), 6), "n_steps": len(steps),
            "solved": all(l["in_stock"] for l in leaves), "tree": tree, "steps": steps, "leaves": leaves}
