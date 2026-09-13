"""Modularity, Louvain, label propagation -- and the resolution limit that bounds what any of them can find.

**Modularity** compares the edge weight inside communities against what a random graph with the same degrees
would produce:

    Q = sum over communities of [ in_c / 2m  -  gamma * (tot_c / 2m)^2 ]

``in_c`` is the internal weight counted twice (each edge from both ends, self-loops included), ``tot_c`` the
total degree of the community, ``m`` the total edge weight. The null model is the crucial half: a community
holding 30% of all edge endpoints is *expected* to contain about 9% of edges by chance, and only the excess
counts. Without that term the whole graph in one community would always win.

**Louvain** maximises it greedily in two alternating phases -- move each node to the neighbouring community
that improves ``Q`` most, then collapse each community into a single node and repeat. The move gain simplifies
to a form that costs one pass over a node's neighbours:

    dQ(i -> C) = w(i,C)/m  -  gamma * tot_C * k_i / (2 m^2)

which is derived in ``modularity_gain`` and checked in the tests against a full recomputation of ``Q``,
because a wrong gain formula produces a plausible partition that optimises nothing in particular.

**The resolution limit** (Fortunato & Barthelemy, 2007) is not a bug in Louvain, it is a property of
modularity itself: communities whose internal edge count is below roughly ``sqrt(2m)`` cannot be resolved,
because merging two of them *increases* Q. In fraud terms -- a five-account ring in a graph with a hundred
thousand edges will be merged into something larger no matter how tight it is, and no amount of restarts will
recover it. The fix is the resolution parameter ``gamma``: larger values shrink the communities modularity
prefers. `python -m fraudgraph.cli resolution` demonstrates the merge and its cure on a ring of cliques where
the right answer is known by construction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .graph import Graph

Partition = "dict[object, object]"


def modularity(graph: Graph, partition: "dict[object, object]", resolution: float = 1.0) -> float:
    """``Q`` for a partition, computed directly from the definition rather than incrementally.

    This is the reference implementation: slower than tracking ``Q`` through moves, and used in the tests to
    verify that the incremental gain formula agrees with it. When an optimiser and its objective are written
    twice, independently, agreement is evidence; a single implementation of both can only be self-consistent.
    """
    total = graph.total_weight()
    if total <= 0.0:
        return 0.0
    internal: dict[object, float] = {}
    degrees: dict[object, float] = {}
    for node in graph.nodes:
        community = partition[node]
        degrees[community] = degrees.get(community, 0.0) + graph.degree(node)
    for first, second, weight in graph.edges():
        if partition[first] == partition[second]:
            community = partition[first]
            internal[community] = internal.get(community, 0.0) + 2.0 * weight
    score = 0.0
    for community, degree in degrees.items():
        share = degree / (2.0 * total)
        score += internal.get(community, 0.0) / (2.0 * total) - resolution * share * share
    return score


def modularity_gain(
    weight_to_community: float,
    community_degree: float,
    node_degree: float,
    total_weight: float,
    resolution: float = 1.0,
) -> float:
    """Change in ``Q`` from moving an isolated node into a community.

    Starting from ``Q`` for the community with and without the node, everything except two terms cancels:

        dQ = (in_C + 2 w(i,C)) / 2m - gamma ((tot_C + k_i)/2m)^2
             - [ in_C/2m - gamma (tot_C/2m)^2 - gamma (k_i/2m)^2 ]
           = w(i,C)/m - gamma * tot_C * k_i / (2 m^2)

    Note ``tot_C`` must exclude the node being moved -- which is why the caller removes it from its own
    community before evaluating any candidate, including the one it came from. Skipping that removal is the
    single most common Louvain bug: every node then appears to be best off exactly where it already is, the
    first pass makes no moves, and the algorithm returns singletons while looking like it converged.
    """
    if total_weight <= 0.0:
        return 0.0
    return weight_to_community / total_weight - resolution * community_degree * node_degree / (
        2.0 * total_weight * total_weight
    )


@dataclass
class LouvainResult:
    """The final partition plus the level-by-level history, which is where the resolution limit shows."""

    partition: "dict[object, object]"
    modularity: float
    levels: "list[dict[object, object]]"
    resolution: float
    passes: int

    def communities(self) -> "list[set[object]]":
        grouped: dict[object, set[object]] = {}
        for node, community in self.partition.items():
            grouped.setdefault(community, set()).add(node)
        return sorted(grouped.values(), key=len, reverse=True)

    def sizes(self) -> "list[int]":
        return [len(community) for community in self.communities()]

    def summary(self) -> str:
        sizes = self.sizes()
        return (
            f"{len(sizes)} communities, Q = {self.modularity:.4f} at gamma = {self.resolution:g}, "
            f"{self.passes} levels; largest {sizes[0] if sizes else 0}, "
            f"median {sizes[len(sizes) // 2] if sizes else 0}"
        )


def _one_level(
    graph: Graph, resolution: float, rng: random.Random
) -> "tuple[dict[object, object], bool]":
    """Local moving until no single move improves ``Q``. Returns the partition and whether anything moved."""
    total = graph.total_weight()
    community = {node: node for node in graph.nodes}
    community_degree = {node: graph.degree(node) for node in graph.nodes}
    improved_at_all = False

    order = graph.nodes
    rng.shuffle(order)
    while True:
        improved = False
        for node in order:
            node_degree = graph.degree(node)
            own = community[node]

            links: dict[object, float] = {}
            for neighbour, weight in graph.neighbours(node).items():
                if neighbour != node:
                    target = community[neighbour]
                    links[target] = links.get(target, 0.0) + weight

            # Remove the node from its community first, so the candidate it came from is judged on the
            # same footing as every other candidate.
            community_degree[own] -= node_degree

            best_community = own
            best_gain = modularity_gain(
                links.get(own, 0.0), community_degree[own], node_degree, total, resolution
            )
            for candidate, weight in links.items():
                if candidate == own:
                    continue
                gain = modularity_gain(
                    weight, community_degree[candidate], node_degree, total, resolution
                )
                if gain > best_gain + 1e-12:
                    best_community, best_gain = candidate, gain

            community_degree[best_community] += node_degree
            community[node] = best_community
            if best_community != own:
                improved = True
                improved_at_all = True
        if not improved:
            break
    return community, improved_at_all


def aggregate(graph: Graph, partition: "dict[object, object]") -> Graph:
    """Collapse each community into one node; internal edges become that node's self-loop.

    The invariant that makes multi-level Louvain valid: the aggregated graph has the same total weight as the
    original, and the modularity of any partition of it equals the modularity of the corresponding partition
    of the original. ``test_aggregation_preserves_modularity`` asserts exactly that, and it is the only reason
    optimising the small graph is the same as optimising the big one.
    """
    result = Graph()
    for community in set(partition.values()):
        result.add_node(community)
    for first, second, weight in graph.edges():
        left, right = partition[first], partition[second]
        if left == right:
            # A self-loop's weight is already "internal twice" in the degree convention, so an internal edge
            # of weight w contributes w to the community self-loop, and the degree bookkeeping works out.
            result.add_edge(left, left, weight)
        else:
            result.add_edge(left, right, weight)
    return result


def louvain(graph: Graph, resolution: float = 1.0, seed: int = 0, max_passes: int = 20) -> LouvainResult:
    """Multi-level modularity optimisation.

    Louvain is greedy and order-dependent: a different node order gives a different partition, usually with
    similar ``Q``. That is worth stating plainly rather than hiding behind a fixed seed -- for a fraud queue
    it means a community boundary is not a fact about the graph, and an account near an edge may be flagged on
    one run and not the next. ``consensus`` below is the practical response.
    """
    if len(graph) == 0:
        return LouvainResult({}, 0.0, [], resolution, 0)
    rng = random.Random(seed)

    node_to_community = {node: node for node in graph.nodes}
    current = graph
    levels: list[dict[object, object]] = []
    best_modularity = modularity(graph, node_to_community, resolution)

    for pass_index in range(1, max_passes + 1):
        partition, improved = _one_level(current, resolution, rng)
        # translate the level's partition back to the original nodes
        mapped = {node: partition[node_to_community[node]] for node in graph.nodes}
        score = modularity(graph, mapped, resolution)
        if not improved or score <= best_modularity + 1e-12:
            break
        node_to_community = mapped
        best_modularity = score
        levels.append(dict(mapped))
        current = aggregate(current, partition)
        if len(current) == len(set(partition.values())) == len(partition):
            break  # nothing collapsed: no further level can change anything
        if len(current) <= 1:
            break

    return LouvainResult(
        partition=node_to_community,
        modularity=best_modularity,
        levels=levels,
        resolution=resolution,
        passes=len(levels),
    )


def label_propagation(graph: Graph, seed: int = 0, max_rounds: int = 50) -> "dict[object, object]":
    """Each node adopts its neighbours' heaviest label, asynchronously, until stable.

    Near-linear and parameter-free, which is why it is attractive at scale, and unstable for the same reason:
    it optimises nothing explicitly, so it has no objective to report and can oscillate between equally good
    labellings. Included as a second opinion -- where Louvain and label propagation agree on a community, the
    community is probably real; where they disagree, the boundary is an artefact of the algorithm.
    """
    rng = random.Random(seed)
    labels = {node: node for node in graph.nodes}
    order = graph.nodes
    for _ in range(max_rounds):
        rng.shuffle(order)
        changed = False
        for node in order:
            counts: dict[object, float] = {}
            for neighbour, weight in graph.neighbours(node).items():
                if neighbour == node:
                    continue
                counts[labels[neighbour]] = counts.get(labels[neighbour], 0.0) + weight
            if not counts:
                continue
            best = max(counts.values())
            candidates = sorted(
                [label for label, value in counts.items() if value >= best - 1e-12], key=str
            )
            chosen = rng.choice(candidates)  # random tie-break: the source of the instability
            if chosen != labels[node]:
                labels[node] = chosen
                changed = True
        if not changed:
            break
    return labels


def consensus(
    graph: Graph, runs: int = 5, resolution: float = 1.0, threshold: float = 0.8
) -> "dict[object, object]":
    """Keep only the co-membership that survives repeated runs from different seeds.

    Run Louvain ``runs`` times, count how often each edge's endpoints land together, keep the edges that
    co-occur at least ``threshold`` of the time, and take connected components of what remains. The result is
    a partition of the *stable* structure: pairs that every run agrees about. Accounts left as singletons are
    genuinely ambiguous, and telling an analyst that is more useful than handing them a boundary that a
    different random seed would have drawn elsewhere.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    from .graph import connected_components

    together: dict[tuple[object, object], int] = {}
    for seed in range(runs):
        partition = louvain(graph, resolution=resolution, seed=seed).partition
        for first, second, _ in graph.edges():
            if first != second and partition[first] == partition[second]:
                key = (first, second)
                together[key] = together.get(key, 0) + 1

    stable = Graph()
    for node in graph.nodes:
        stable.add_node(node)
    for (first, second), count in together.items():
        if count / runs >= threshold:
            stable.add_edge(first, second, float(count))

    labels: dict[object, object] = {}
    for index, component in enumerate(connected_components(stable)):
        for node in component:
            labels[node] = index
    return labels


def resolution_sweep(
    graph: Graph, values: "list[float]", seed: int = 0
) -> "list[tuple[float, int, float, int]]":
    """``(gamma, community count, Q at that gamma, largest community)`` across resolutions.

    There is no correct gamma. The sweep is the honest output: read off where the community count is stable
    across a range, because a structure that survives a change in resolution is more likely to be real than
    one that appears at a single setting.
    """
    output = []
    for gamma in values:
        result = louvain(graph, resolution=gamma, seed=seed)
        sizes = result.sizes()
        output.append((gamma, len(sizes), result.modularity, sizes[0] if sizes else 0))
    return output
