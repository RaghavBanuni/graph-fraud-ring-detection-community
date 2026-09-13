"""Demonstrations. Run them; the figures depend on the seed and are printed rather than promised.

    python -m fraudgraph.cli components   the naive projection, and why it returns one ring of everybody
    python -m fraudgraph.cli louvain      detection end to end, with the analyst queue and the truth
    python -m fraudgraph.cli resolution   the resolution limit on a graph whose answer is known
    python -m fraudgraph.cli stability     the same graph, different seeds: which boundaries are real
    python -m fraudgraph.cli compare      components vs label propagation vs Louvain, same data
    python -m fraudgraph.cli all
"""

from __future__ import annotations

import sys

from . import data, louvain as louvain_module, scoring
from .graph import connected_components, project, projection_report


def _rule(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def demo_components() -> None:
    _rule("The projection decides everything")
    scenario = data.planted_rings(seed=1)
    print(scenario.summary())
    print()
    print(scenario.hetero.summary())
    print()
    print(projection_report(scenario.hetero))
    print(
        "\nThe first row is the rule most link-analysis systems start with: connect accounts that share any\n"
        "attribute, then take connected components. One shared public IP is enough to merge most of the\n"
        "population into a single component, and 'a ring containing 40% of your customers' is not a case an\n"
        "analyst can open."
    )

    naive = connected_components(project(scenario.hetero, weighting="count", max_attribute_size=None))
    weighted = connected_components(project(scenario.hetero, weighting="idf", max_attribute_size=50))
    truth = list(scenario.rings.values())
    print("\nBoth projections scored as detectors, using connected components as the clustering:")
    print(f"  naive     {scoring.pairwise_scores(naive, truth).summary()}")
    print(f"  weighted  {scoring.pairwise_scores(weighted, truth).summary()}")
    print(
        "\nNote which number collapses. Recall stays high in the naive projection -- every ring pair really\n"
        "is inside the giant component -- while precision goes to almost nothing. Reporting recall alone\n"
        "would make the useless detector look like the better one."
    )


def demo_louvain() -> None:
    _rule("Detection end to end")
    scenario = data.planted_rings(seed=2)
    graph = project(scenario.hetero, weighting="idf", max_attribute_size=50)
    print(f"projected graph: {len(graph)} accounts, {len(graph.edges()):,} edges")

    result = louvain_module.louvain(graph, resolution=1.0, seed=0)
    print(result.summary())
    communities = result.communities()
    ranked = scoring.rank_communities(communities, graph, scenario.hetero)
    print()
    print(scoring.evaluation_report(communities, scenario, ranked, k=15))
    print("\nthe analyst queue, ranked by risk score:")
    print(scoring.queue_table(ranked, scenario))
    print(
        "\nRead the verdict column against the size column. Where a ring is recovered whole the score is\n"
        "high; where it is buried in a larger community the same ring is present and unusable."
    )


def demo_resolution() -> None:
    _rule("The resolution limit")
    graph = data.ring_of_cliques(clique_count=24, clique_size=5)
    print(
        "24 cliques of 5, joined in a cycle by single edges. The correct partition is not in doubt:\n"
        "one community per clique, 24 of them."
    )
    for gamma in (0.5, 1.0, 1.5, 2.0, 3.0):
        result = louvain_module.louvain(graph, resolution=gamma, seed=0)
        sizes = result.sizes()
        print(
            f"  gamma {gamma:>4}: {len(sizes):>3} communities, largest {sizes[0]:>3}, "
            f"Q = {result.modularity:.4f}"
        )
    print(
        "\nAt gamma = 1 adjacent cliques are merged, and this is not an optimisation failure: the merged\n"
        "partition genuinely has the higher modularity. The null model expects so few edges between two\n"
        "cliques that a single bridge reads as an excess. Only changing the objective recovers the truth,\n"
        "which is why 'run Louvain' is a modelling decision and not a default."
    )

    scenario = data.planted_rings(seed=3)
    graph = project(scenario.hetero, weighting="idf", max_attribute_size=50)
    print("\nand on the fraud graph, where the small rings are the ones at risk:")
    header = f"{'gamma':>7}{'communities':>14}{'largest':>10}{'Q':>10}   recall by ring size"
    print(header)
    print("-" * len(header))
    for gamma, count, quality, largest in louvain_module.resolution_sweep(
        graph, [0.5, 1.0, 2.0, 4.0, 8.0]
    ):
        communities = louvain_module.louvain(graph, resolution=gamma, seed=0).communities()
        by_size = scoring.recall_by_size(communities, scenario.rings)
        detail = " ".join(
            f"{size}:{found}/{total}" for size, (found, total) in sorted(by_size.items(), reverse=True)
        )
        print(f"{gamma:>7}{count:>14}{largest:>10}{quality:>10.4f}   {detail}")
    print(
        "\nThere is no gamma that is simply correct. Read where the answer is stable across a range, and\n"
        "treat a structure that appears at exactly one setting as an artefact of the setting."
    )


def demo_stability() -> None:
    _rule("Is a community boundary a fact about the graph?")
    scenario = data.planted_rings(seed=4)
    graph = project(scenario.hetero, weighting="idf", max_attribute_size=50)

    print("five Louvain runs, identical data, different node orders:")
    for seed in range(5):
        result = louvain_module.louvain(graph, resolution=1.0, seed=seed)
        sizes = result.sizes()
        print(
            f"  seed {seed}: {len(sizes):>4} communities, largest {sizes[0]:>4}, Q = {result.modularity:.4f}"
        )

    labels = louvain_module.consensus(graph, runs=5, resolution=1.0, threshold=0.8)
    grouped: dict[object, set[str]] = {}
    for account, label in labels.items():
        grouped.setdefault(label, set()).add(account)
    stable = [members for members in grouped.values() if len(members) >= 3]
    truth = list(scenario.rings.values())
    print(
        f"\nconsensus over those runs keeps {len(stable)} communities of three or more accounts"
    )
    print(f"  single Louvain run   {scoring.pairwise_scores(louvain_module.louvain(graph, seed=0).communities(), truth).summary()}")
    print(f"  consensus            {scoring.pairwise_scores(list(grouped.values()), truth).summary()}")
    print(
        "\nQ barely moves between seeds while the partitions differ, which is the point: modularity has\n"
        "many near-optimal solutions and the algorithm picks one. Consensus keeps only the co-membership\n"
        "every run agrees on, so an account that lands in a different community each time is returned as a\n"
        "singleton -- which is the honest answer for it."
    )


def demo_compare() -> None:
    _rule("Three clusterings, one graph")
    scenario = data.planted_rings(seed=5)
    graph = project(scenario.hetero, weighting="idf", max_attribute_size=50)
    truth = list(scenario.rings.values())

    components = connected_components(graph)
    propagation: dict[object, set[str]] = {}
    for account, label in louvain_module.label_propagation(graph, seed=0).items():
        propagation.setdefault(label, set()).add(account)
    modular = louvain_module.louvain(graph, resolution=2.0, seed=0).communities()

    for label, communities in (
        ("connected components", components),
        ("label propagation", list(propagation.values())),
        ("Louvain, gamma = 2", modular),
    ):
        sizes = sorted((len(community) for community in communities), reverse=True)
        score = scoring.pairwise_scores(communities, truth)
        ranked = scoring.rank_communities(communities, graph, scenario.hetero)
        precision, reviewed = scoring.precision_at_k(ranked, scenario, k=15)
        print(
            f"\n{label}\n  {len(sizes)} communities, largest {sizes[0]}\n  {score.summary()}\n"
            f"  precision@{reviewed} in the queue: {precision:.1%}"
        )
    print(
        "\nLabel propagation is near-linear and needs no parameter, and it optimises nothing explicitly, so\n"
        "it has no objective to report and no resolution to tune. It is a good second opinion and a poor\n"
        "primary: where it agrees with Louvain the community is probably real."
    )


DEMOS = {
    "components": demo_components,
    "louvain": demo_louvain,
    "resolution": demo_resolution,
    "stability": demo_stability,
    "compare": demo_compare,
}


def main(argv: "list[str] | None" = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    name = arguments[0]
    if name == "all":
        for demo in DEMOS.values():
            demo()
        return 0
    if name not in DEMOS:
        print(f"unknown demonstration {name!r}; choose from {', '.join(DEMOS)} or 'all'")
        return 2
    DEMOS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
