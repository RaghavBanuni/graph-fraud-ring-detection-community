"""Graphs, and the projection step where fraud detection usually goes wrong.

The raw data is **heterogeneous and bipartite**: accounts on one side, shared attributes on the other --
devices, cards, addresses, phone numbers, IP addresses. Accounts are never linked directly; they are linked
by sharing something. Turning that into an account-to-account graph is the projection, and how it is done
decides everything downstream.

The naive projection connects two accounts whenever they share any attribute, with weight equal to the number
of attributes shared. It fails immediately, and not subtly:

    a coffee-shop IP address shared by 4,000 accounts contributes 4,000 * 3,999 / 2 = 7,998,000 edges

That single attribute welds the entire graph into one component. Connected components then return "one ring
of 40,000 accounts", which is true and useless. `python -m fraudgraph.cli components` measures exactly this.

The fix is to weight a shared attribute by how surprising it is. This module implements **inverse-frequency
weighting**, the same idea as IDF in text retrieval:

    w(attribute) = log(N / n_attribute)      N accounts, n_attribute sharing this one

A card used by two accounts is worth `log(N/2)`; an IP used by 4,000 is worth almost nothing. Together with a
hard degree cap -- attributes above a size threshold are dropped as infrastructure rather than evidence --
this turns a useless projection into a usable one, and the tests assert the difference.

Nothing here is a graph library. Adjacency is a dict of dicts, which for a hundred thousand edges of pure
Python is the right trade: no dependency, and the data structure is visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class Graph:
    """Undirected weighted graph on hashable node labels, with self-loops allowed.

    Self-loops matter because Louvain's aggregation step creates them: when a community collapses into a
    single node, its internal edges become that node's self-loop. Storing them explicitly is what lets the
    aggregated graph have exactly the same modularity as the partition it came from -- a property the tests
    check, since it is the load-bearing invariant of the whole algorithm.
    """

    def __init__(self) -> None:
        self._adjacency: dict[object, dict[object, float]] = {}

    def add_node(self, node: object) -> None:
        self._adjacency.setdefault(node, {})

    def add_edge(self, first: object, second: object, weight: float = 1.0) -> None:
        """Add ``weight`` to the edge, creating it if absent. Repeated calls accumulate."""
        if weight <= 0.0:
            raise ValueError("edge weights must be positive")
        self.add_node(first)
        self.add_node(second)
        if first == second:
            self._adjacency[first][first] = self._adjacency[first].get(first, 0.0) + weight
            return
        self._adjacency[first][second] = self._adjacency[first].get(second, 0.0) + weight
        self._adjacency[second][first] = self._adjacency[second].get(first, 0.0) + weight

    @property
    def nodes(self) -> "list[object]":
        return list(self._adjacency)

    def __len__(self) -> int:
        return len(self._adjacency)

    def neighbours(self, node: object) -> "dict[object, float]":
        return self._adjacency[node]

    def weight(self, first: object, second: object) -> float:
        return self._adjacency.get(first, {}).get(second, 0.0)

    def self_loop(self, node: object) -> float:
        return self._adjacency.get(node, {}).get(node, 0.0)

    def degree(self, node: object) -> float:
        """Weighted degree, with the self-loop counted **twice**.

        This is the convention modularity requires: the degree must equal the number of edge endpoints at the
        node, and a self-loop contributes two endpoints. Getting it wrong makes the modularity of an
        aggregated graph differ from the partition it represents, and Louvain then optimises a quantity that
        drifts at every level.
        """
        edges = self._adjacency[node]
        return sum(edges.values()) + edges.get(node, 0.0)

    def total_weight(self) -> float:
        """``m``: the total edge weight, so that ``sum of degrees = 2m``."""
        return sum(self.degree(node) for node in self._adjacency) / 2.0

    def edges(self) -> "list[tuple[object, object, float]]":
        seen: set[tuple[object, object]] = set()
        output = []
        for node, neighbours in self._adjacency.items():
            for other, weight in neighbours.items():
                key = (node, other) if str(node) <= str(other) else (other, node)
                if key in seen:
                    continue
                seen.add(key)
                output.append((node, other, weight))
        return output

    def subgraph(self, nodes: "set[object]") -> "Graph":
        result = Graph()
        for node in nodes:
            result.add_node(node)
        for node in nodes:
            for other, weight in self._adjacency[node].items():
                if other in nodes and (str(node) <= str(other)):
                    result.add_edge(node, other, weight)
        return result

    def density(self, nodes: "set[object]") -> float:
        """Internal edge count over the possible count: the first thing to ask about a candidate ring."""
        if len(nodes) < 2:
            return 0.0
        internal = sum(
            1
            for node in nodes
            for other in self._adjacency[node]
            if other in nodes and other != node and str(node) < str(other)
        )
        possible = len(nodes) * (len(nodes) - 1) / 2
        return internal / possible


# ---------------------------------------------------------------------------------------------
# connected components, by union-find
# ---------------------------------------------------------------------------------------------


class UnionFind:
    """Union by size with full path compression: near-linear components, and the baseline to beat."""

    def __init__(self) -> None:
        self.parent: dict[object, object] = {}
        self.size: dict[object, int] = {}

    def add(self, item: object) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: object) -> object:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression, iterative to avoid recursion limits
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, first: object, second: object) -> None:
        left, right = self.find(first), self.find(second)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]

    def groups(self) -> "dict[object, set[object]]":
        output: dict[object, set[object]] = {}
        for item in self.parent:
            output.setdefault(self.find(item), set()).add(item)
        return output


def connected_components(graph: Graph) -> "list[set[object]]":
    """Components, largest first. The baseline fraud rule, and the one that collapses.

    "Accounts transitively linked by any shared attribute" is how most first-generation link analysis works.
    It is not wrong about the links; it is wrong about what a link means, because transitivity through one
    popular attribute connects everything to everything.
    """
    finder = UnionFind()
    for node in graph.nodes:
        finder.add(node)
    for first, second, _ in graph.edges():
        finder.union(first, second)
    return sorted(finder.groups().values(), key=len, reverse=True)


# ---------------------------------------------------------------------------------------------
# the heterogeneous graph and its projection
# ---------------------------------------------------------------------------------------------


@dataclass
class HeteroGraph:
    """Accounts linked to typed shared attributes -- the shape the raw data actually has."""

    links: "dict[str, set[str]]" = field(default_factory=dict)  # account -> attribute ids
    attribute_type: "dict[str, str]" = field(default_factory=dict)  # attribute id -> "device", "card", ...
    signup_time: "dict[str, float]" = field(default_factory=dict)  # account -> arrival time

    def add_link(self, account: str, attribute: str, kind: str) -> None:
        self.links.setdefault(account, set()).add(attribute)
        existing = self.attribute_type.get(attribute)
        if existing is not None and existing != kind:
            raise ValueError(f"attribute {attribute!r} declared as both {existing!r} and {kind!r}")
        self.attribute_type[attribute] = kind

    @property
    def accounts(self) -> "list[str]":
        return sorted(self.links)

    def holders(self) -> "dict[str, set[str]]":
        """Attribute -> the accounts sharing it. The inverted index the projection needs."""
        output: dict[str, set[str]] = {}
        for account, attributes in self.links.items():
            for attribute in attributes:
                output.setdefault(attribute, set()).add(account)
        return output

    def attribute_sizes(self) -> "dict[str, int]":
        return {attribute: len(accounts) for attribute, accounts in self.holders().items()}

    def summary(self) -> str:
        sizes = self.attribute_sizes()
        biggest = sorted(sizes.items(), key=lambda item: item[1], reverse=True)[:5]
        lines = [
            f"{len(self.links)} accounts, {len(sizes)} shared attributes",
            "largest attributes (these are the ones that decide the projection):",
        ]
        for attribute, size in biggest:
            pairs = size * (size - 1) // 2
            lines.append(
                f"  {attribute:<28} {self.attribute_type[attribute]:<9} {size:>6} accounts"
                f"  -> {pairs:>10,} account pairs"
            )
        return "\n".join(lines)


def project(
    hetero: HeteroGraph,
    weighting: str = "idf",
    max_attribute_size: int | None = 50,
    min_weight: float = 0.0,
) -> Graph:
    """Project the bipartite graph onto accounts.

    ``weighting``:

    * ``"count"`` -- one unit per shared attribute. The naive projection, kept because being able to
      reproduce the failure is the point.
    * ``"idf"`` -- each shared attribute contributes ``log(N / n_attribute)``, so evidence is weighted by
      how unlikely the coincidence is. Two accounts sharing a card nobody else uses get a large weight; two
      accounts sharing an airport wifi IP get one close to zero.
    * ``"jaccard"`` -- the weight is the Jaccard similarity of the two accounts' attribute sets, which
      normalises by how many attributes each account has. Useful when accounts differ wildly in how much
      data they carry.

    ``max_attribute_size`` drops attributes above a size, before any weighting. This is not an optimisation;
    it is a modelling statement -- an IP address with 4,000 accounts behind it is infrastructure, not
    evidence, and the quadratic edge count it produces will dominate everything if it is kept. IDF weighting
    alone shrinks those edges but does not remove them, and enough near-zero edges still merge communities.

    The two mechanisms are complementary and both are needed; ``python -m fraudgraph.cli components`` shows
    what each one does on its own.
    """
    if weighting not in {"count", "idf", "jaccard"}:
        raise ValueError("weighting must be 'count', 'idf' or 'jaccard'")

    holders = hetero.holders()
    total_accounts = max(len(hetero.links), 1)
    graph = Graph()
    for account in hetero.accounts:
        graph.add_node(account)

    for attribute, accounts in holders.items():
        size = len(accounts)
        if size < 2:
            continue
        if max_attribute_size is not None and size > max_attribute_size:
            continue
        if weighting == "count":
            weight = 1.0
        elif weighting == "idf":
            weight = math.log(total_accounts / size)
            if weight <= 0.0:
                continue  # an attribute held by every account carries no information at all
        else:
            weight = 1.0  # Jaccard is computed pairwise below

        ordered = sorted(accounts)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                if weighting == "jaccard":
                    left, right = hetero.links[first], hetero.links[second]
                    value = len(left & right) / len(left | right)
                else:
                    value = weight
                if value > min_weight:
                    graph.add_edge(first, second, value)

    if weighting == "jaccard":
        # Pairwise Jaccard is computed once per shared attribute, so an edge accumulates a multiple of the
        # true similarity. Divide it back out rather than leaving a quantity that is not what it claims.
        for first, second, accumulated in graph.edges():
            left, right = hetero.links[first], hetero.links[second]
            shared = len(left & right)
            if shared > 1:
                graph._adjacency[first][second] = accumulated / shared
                graph._adjacency[second][first] = accumulated / shared
    return graph


def projection_report(hetero: HeteroGraph) -> str:
    """Compare projections side by side: the edge count, the giant component, and what that costs."""
    lines = [f"{'projection':<34}{'edges':>12}{'largest component':>20}{'share of accounts':>20}"]
    lines.append("-" * len(lines[0]))
    total = len(hetero.accounts)
    for label, kwargs in (
        ("count, no cap (naive)", {"weighting": "count", "max_attribute_size": None}),
        ("idf, no cap", {"weighting": "idf", "max_attribute_size": None}),
        ("count, cap 50", {"weighting": "count", "max_attribute_size": 50}),
        ("idf, cap 50", {"weighting": "idf", "max_attribute_size": 50}),
    ):
        graph = project(hetero, **kwargs)
        components = connected_components(graph)
        largest = len(components[0]) if components else 0
        lines.append(
            f"{label:<34}{len(graph.edges()):>12,}{largest:>20,}{largest / total:>19.1%}"
        )
    return "\n".join(lines)
