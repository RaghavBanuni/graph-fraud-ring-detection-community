"""Tests. Hand-computed modularity first, then the two identities the algorithm rests on.

The most important test in this file is `test_modularity_gain_equals_the_change_in_modularity`. The optimiser
uses an incremental gain formula; `modularity()` computes the objective from its definition. If those two
disagree, Louvain is optimising something that is not modularity, and the result still looks like a plausible
partition -- there is no symptom to notice. The second is `test_aggregation_preserves_modularity`, which is
what makes multi-level optimisation valid at all.

The resolution limit is tested **algebraically**, not through Louvain: on a ring of cliques, the partition that
merges clique pairs has strictly higher modularity than the correct one. That is a fact about the objective and
holds regardless of which optimiser is used or how many restarts it gets.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraudgraph import data, louvain as lv, scoring  # noqa: E402
from fraudgraph.graph import (  # noqa: E402
    Graph,
    HeteroGraph,
    UnionFind,
    connected_components,
    project,
    shared_attributes,
)


def two_edges() -> Graph:
    """Four nodes, two disjoint edges: small enough to do the modularity by hand."""
    graph = Graph()
    graph.add_edge("a", "b", 1.0)
    graph.add_edge("c", "d", 1.0)
    return graph


class TestGraph:
    def test_degree_counts_a_self_loop_twice(self):
        """The convention modularity requires: degree counts edge endpoints, and a loop has two."""
        graph = Graph()
        graph.add_edge("a", "b", 2.0)
        graph.add_edge("a", "a", 3.0)
        assert graph.degree("a") == pytest.approx(2.0 + 6.0)
        assert graph.degree("b") == pytest.approx(2.0)

    def test_total_weight_is_half_the_degree_sum(self):
        graph = Graph()
        graph.add_edge("a", "b", 2.0)
        graph.add_edge("b", "c", 1.5)
        graph.add_edge("c", "c", 0.5)
        assert graph.total_weight() == pytest.approx(
            sum(graph.degree(node) for node in graph.nodes) / 2.0
        )

    def test_repeated_edges_accumulate(self):
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("a", "b", 2.5)
        assert graph.weight("a", "b") == pytest.approx(3.5)
        assert graph.weight("b", "a") == pytest.approx(3.5)

    def test_edges_yields_each_undirected_edge_once(self):
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "c", 1.0)
        graph.add_edge("a", "a", 1.0)
        assert len(graph.edges()) == 3

    def test_non_positive_weights_are_rejected(self):
        graph = Graph()
        with pytest.raises(ValueError, match="positive"):
            graph.add_edge("a", "b", 0.0)

    def test_density_of_a_triangle_is_one(self):
        graph = Graph()
        for first, second in (("a", "b"), ("b", "c"), ("a", "c")):
            graph.add_edge(first, second, 1.0)
        assert graph.density({"a", "b", "c"}) == pytest.approx(1.0)
        assert graph.density({"a"}) == 0.0

    def test_components_separate_disconnected_pieces(self):
        components = connected_components(two_edges())
        assert sorted(len(component) for component in components) == [2, 2]

    def test_union_find_compresses_without_recursion(self):
        finder = UnionFind()
        for index in range(2000):  # a long chain: recursion would blow the stack here
            finder.union(index, index + 1)
        assert len(finder.groups()) == 1
        assert finder.find(0) == finder.find(2000)


class TestProjection:
    def test_an_attribute_held_by_everyone_creates_no_idf_edges(self):
        """log(N/N) = 0: a coincidence shared by the whole population is not evidence of anything."""
        hetero = HeteroGraph()
        for account in ("a", "b", "c"):
            hetero.add_link(account, "ip_shared", "ip")
        assert len(project(hetero, weighting="idf", max_attribute_size=None).edges()) == 0
        # the naive projection makes a triangle out of it
        assert len(project(hetero, weighting="count", max_attribute_size=None).edges()) == 3

    def test_idf_weight_matches_the_formula(self):
        """Four accounts, an attribute shared by two: weight = log(4/2) = log 2."""
        hetero = HeteroGraph()
        hetero.add_link("a", "card_1", "card")
        hetero.add_link("b", "card_1", "card")
        hetero.add_link("c", "card_c", "card")
        hetero.add_link("d", "card_d", "card")
        graph = project(hetero, weighting="idf", max_attribute_size=None)
        assert graph.weight("a", "b") == pytest.approx(math.log(2.0))

    def test_weights_from_several_shared_attributes_add(self):
        hetero = HeteroGraph()
        for attribute, kind in (("card_1", "card"), ("dev_1", "device")):
            hetero.add_link("a", attribute, kind)
            hetero.add_link("b", attribute, kind)
        for index in range(6):
            hetero.add_link(f"other_{index}", f"card_{index}", "card")
        graph = project(hetero, weighting="idf", max_attribute_size=None)
        expected = 2.0 * math.log(8.0 / 2.0)
        assert graph.weight("a", "b") == pytest.approx(expected)

    def test_the_size_cap_removes_the_hub_and_the_giant_component(self):
        """The headline failure, on data where the two structures are unambiguous."""
        hetero = HeteroGraph()
        for index in range(100):
            hetero.add_link(f"acct_{index}", "ip_cafe", "ip")
        hetero.add_link("ring_a", "card_x", "card")
        hetero.add_link("ring_b", "card_x", "card")

        naive = connected_components(project(hetero, weighting="count", max_attribute_size=None))
        assert len(naive[0]) == 100  # one shared IP, one component

        capped = connected_components(project(hetero, weighting="idf", max_attribute_size=50))
        assert max(len(component) for component in capped) == 2  # only the shared card survives

    def test_jaccard_uses_only_admissible_attributes(self):
        """The cap must apply before the similarity, not after: the denominator has to match the numerator.

        Accounts a and b share a capped-out hub and one real card. a carries nothing else, b carries one extra
        device. Admissible sets are {card_x} and {card_x, dev_b}, so the similarity is 1/2 -- not 2/3, which is
        what including the dropped hub would give.
        """
        hetero = HeteroGraph()
        for index in range(60):
            hetero.add_link(f"filler_{index}", "ip_hub", "ip")
        for account in ("a", "b"):
            hetero.add_link(account, "ip_hub", "ip")
            hetero.add_link(account, "card_x", "card")
        hetero.add_link("b", "dev_b", "device")
        graph = project(hetero, weighting="jaccard", max_attribute_size=50)
        assert graph.weight("a", "b") == pytest.approx(0.5)

    def test_shared_attributes_lists_each_pair_once(self):
        hetero = HeteroGraph()
        for account in ("a", "b", "c"):
            hetero.add_link(account, "dev_1", "device")
        pairs = shared_attributes(hetero, max_attribute_size=None)
        assert set(pairs) == {("a", "b"), ("a", "c"), ("b", "c")}

    def test_conflicting_attribute_types_are_rejected(self):
        hetero = HeteroGraph()
        hetero.add_link("a", "thing", "device")
        with pytest.raises(ValueError, match="declared as both"):
            hetero.add_link("b", "thing", "card")


class TestModularity:
    def test_two_disjoint_edges_by_hand(self):
        """m = 2. Each community: internal = 2, tot = 2, so Q_c = 2/4 - (2/4)^2 = 0.25, and Q = 0.5."""
        graph = two_edges()
        split = {"a": 0, "b": 0, "c": 1, "d": 1}
        assert lv.modularity(graph, split) == pytest.approx(0.5, abs=1e-12)

    def test_everything_in_one_community_scores_zero(self):
        """internal = 4, tot = 4: Q = 4/4 - (4/4)^2 = 0. The null model is what makes this the baseline."""
        graph = two_edges()
        merged = dict.fromkeys(graph.nodes, 0)
        assert lv.modularity(graph, merged) == pytest.approx(0.0, abs=1e-12)

    def test_singletons_score_negative(self):
        graph = data.two_clusters(size=6, bridges=1)
        singletons = {node: node for node in graph.nodes}
        assert lv.modularity(graph, singletons) < 0.0

    def test_resolution_scales_the_null_term_only(self):
        graph = two_edges()
        split = {"a": 0, "b": 0, "c": 1, "d": 1}
        at_one = lv.modularity(graph, split, resolution=1.0)
        at_two = lv.modularity(graph, split, resolution=2.0)
        # internal term unchanged, null term doubled: 0.5 -> 2*(0.5 - 2*0.25)
        assert at_one == pytest.approx(0.5)
        assert at_two == pytest.approx(2.0 * (0.5 - 2.0 * 0.25), abs=1e-12)

    def test_modularity_gain_equals_the_change_in_modularity(self):
        """The identity the optimiser depends on, checked against a full recomputation.

        Moving ``L0`` from its community into the other one. The gain formula is evaluated with the node
        already removed from its own community -- which is precisely what ``_one_level`` does -- and the
        difference of the two candidate gains must equal the change in Q computed from the definition.
        """
        graph = data.two_clusters(size=6, bridges=2, seed=0)
        partition = {node: ("L" if str(node).startswith("L") else "R") for node in graph.nodes}
        node = "L0"
        total = graph.total_weight()
        node_degree = graph.degree(node)

        weight_to: dict[str, float] = {"L": 0.0, "R": 0.0}
        for neighbour, weight in graph.neighbours(node).items():
            if neighbour != node:
                weight_to[partition[neighbour]] += weight

        degree_left = sum(graph.degree(n) for n in graph.nodes if partition[n] == "L") - node_degree
        degree_right = sum(graph.degree(n) for n in graph.nodes if partition[n] == "R")

        gain_stay = lv.modularity_gain(weight_to["L"], degree_left, node_degree, total)
        gain_move = lv.modularity_gain(weight_to["R"], degree_right, node_degree, total)

        before = lv.modularity(graph, partition)
        moved = dict(partition)
        moved[node] = "R"
        after = lv.modularity(graph, moved)
        assert gain_move - gain_stay == pytest.approx(after - before, abs=1e-12)

    def test_aggregation_preserves_modularity_and_weight(self):
        """What makes multi-level Louvain valid: the collapsed graph is the same optimisation problem."""
        graph = data.two_clusters(size=8, bridges=3, seed=1)
        partition = {node: ("L" if str(node).startswith("L") else "R") for node in graph.nodes}
        collapsed = lv.aggregate(graph, partition)
        assert collapsed.total_weight() == pytest.approx(graph.total_weight(), abs=1e-12)
        identity = {node: node for node in collapsed.nodes}
        assert lv.modularity(collapsed, identity) == pytest.approx(
            lv.modularity(graph, partition), abs=1e-12
        )


class TestResolutionLimit:
    def test_modularity_prefers_merging_cliques_in_a_large_ring(self):
        """Algebra, not optimisation: the merged partition has the higher objective value.

        Forty cliques of five in a cycle. The correct partition puts each clique in its own community; the
        merged one pairs adjacent cliques. At gamma = 1 the merged partition scores higher, so no optimiser
        of any quality can return the truth -- the truth is not the maximum. This is the resolution limit,
        and it is a property of modularity rather than of Louvain.
        """
        graph = data.ring_of_cliques(clique_count=40, clique_size=5)
        truth = {node: str(node).split("_")[0] for node in graph.nodes}
        merged = {node: f"pair{int(str(node).split('_')[0][1:]) // 2}" for node in graph.nodes}
        assert lv.modularity(graph, merged) > lv.modularity(graph, truth)

    def test_a_higher_resolution_reverses_the_preference(self):
        """The same two partitions, judged at gamma = 4: now the truth wins. That is the cure."""
        graph = data.ring_of_cliques(clique_count=40, clique_size=5)
        truth = {node: str(node).split("_")[0] for node in graph.nodes}
        merged = {node: f"pair{int(str(node).split('_')[0][1:]) // 2}" for node in graph.nodes}
        assert lv.modularity(graph, truth, resolution=4.0) > lv.modularity(
            graph, merged, resolution=4.0
        )

    def test_louvain_finds_more_communities_at_a_higher_resolution(self):
        graph = data.ring_of_cliques(clique_count=40, clique_size=5)
        coarse = len(lv.louvain(graph, resolution=1.0, seed=0).sizes())
        fine = len(lv.louvain(graph, resolution=4.0, seed=0).sizes())
        assert coarse < 40  # cliques merged at the default resolution
        assert fine > coarse


class TestLouvain:
    def test_two_cliques_are_separated(self):
        graph = data.two_clusters(size=15, bridges=1, seed=0)
        result = lv.louvain(graph, resolution=1.0, seed=0)
        assert len(result.sizes()) == 2
        for community in result.communities():
            prefixes = {str(node)[0] for node in community}
            assert len(prefixes) == 1  # no community mixes the two cliques

    def test_modularity_beats_both_trivial_partitions(self):
        graph = data.two_clusters(size=12, bridges=2, seed=2)
        result = lv.louvain(graph, seed=0)
        singletons = lv.modularity(graph, {node: node for node in graph.nodes})
        everything = lv.modularity(graph, dict.fromkeys(graph.nodes, 0))
        assert result.modularity > singletons
        assert result.modularity > everything

    def test_reported_modularity_matches_the_partition(self):
        """The returned number must be the modularity of the returned partition, recomputed from scratch."""
        graph = data.two_clusters(size=10, bridges=2, seed=3)
        result = lv.louvain(graph, resolution=1.5, seed=1)
        assert result.modularity == pytest.approx(
            lv.modularity(graph, result.partition, resolution=1.5), abs=1e-12
        )

    def test_every_node_is_assigned_exactly_once(self):
        graph = data.two_clusters(size=9, bridges=1, seed=4)
        result = lv.louvain(graph, seed=0)
        assert set(result.partition) == set(graph.nodes)
        assert sum(result.sizes()) == len(graph)

    def test_empty_graph_is_handled(self):
        result = lv.louvain(Graph())
        assert result.partition == {}
        assert result.modularity == 0.0

    def test_label_propagation_separates_disconnected_cliques(self):
        graph = data.two_clusters(size=10, bridges=0, seed=0)
        labels = lv.label_propagation(graph, seed=0)
        assert len({labels[node] for node in graph.nodes if str(node).startswith("L")}) == 1
        assert len({labels[node] for node in graph.nodes if str(node).startswith("R")}) == 1
        assert labels["L0"] != labels["R0"]

    def test_consensus_keeps_the_structure_every_run_agrees_on(self):
        graph = data.two_clusters(size=10, bridges=0, seed=0)
        labels = lv.consensus(graph, runs=3, resolution=1.0, threshold=1.0)
        assert len({labels[node] for node in graph.nodes}) == 2

    def test_consensus_rejects_an_impossible_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            lv.consensus(data.two_clusters(size=5), threshold=0.0)


class TestScoring:
    def test_pairwise_scores_by_hand(self):
        """Predicted {a,b,c} against truth {a,b} and {c,d}: 3 predicted pairs, 1 correct, 2 true pairs."""
        score = scoring.pairwise_scores([{"a", "b", "c"}], [{"a", "b"}, {"c", "d"}])
        assert score.predicted_pairs == 3
        assert score.true_pairs == 2
        assert score.correct_pairs == 1
        assert score.precision == pytest.approx(1.0 / 3.0)
        assert score.recall == pytest.approx(0.5)
        assert score.f1 == pytest.approx(2.0 * (1 / 3) * 0.5 / ((1 / 3) + 0.5))

    def test_a_perfect_partition_scores_one(self):
        truth = [{"a", "b", "c"}, {"d", "e"}]
        score = scoring.pairwise_scores(truth, truth)
        assert score.precision == 1.0
        assert score.recall == 1.0

    def test_a_pair_community_scores_zero_risk(self):
        """Two accounts sharing an address are a household until something else says otherwise."""
        scenario = data.planted_rings(n_legit=60, ring_sizes=(4,), household_count=5, seed=0)
        graph = project(scenario.hetero, weighting="idf", max_attribute_size=50)
        features = scoring.describe_community({"acct_00000", "acct_00001"}, graph, scenario.hetero, 365.0)
        assert features.risk_score() == 0.0

    def test_ring_recovery_reports_exact_matches(self):
        rings = {"ring_0": {"a", "b", "c"}}
        assert scoring.ring_recovery([{"a", "b", "c"}], rings)[0][2] == pytest.approx(1.0)
        # buried inside a much larger community: present, and unusable
        buried = scoring.ring_recovery([{"a", "b", "c"} | {f"x{index}" for index in range(97)}], rings)
        assert buried[0][2] == pytest.approx(3.0 / 100.0)

    def test_recall_by_size_splits_the_aggregate(self):
        rings = {"big": set("abcdefgh"), "small": {"x", "y", "z"}}
        communities = [set("abcdefgh")]  # the big ring found, the small one missed
        breakdown = scoring.recall_by_size(communities, rings)
        assert breakdown[8] == (1, 1)
        assert breakdown[3] == (0, 1)

    def test_precision_at_k_never_exceeds_the_queue_length(self):
        scenario = data.planted_rings(n_legit=120, ring_sizes=(6, 4), seed=1)
        graph = project(scenario.hetero, weighting="idf", max_attribute_size=50)
        communities = lv.louvain(graph, resolution=2.0, seed=0).communities()
        ranked = scoring.rank_communities(communities, graph, scenario.hetero)
        precision, reviewed = scoring.precision_at_k(ranked, scenario, k=10_000)
        assert reviewed == len(ranked)
        assert 0.0 <= precision <= 1.0

    def test_rank_communities_drops_pairs_and_sorts_by_score(self):
        scenario = data.planted_rings(n_legit=150, ring_sizes=(8, 5), seed=2)
        graph = project(scenario.hetero, weighting="idf", max_attribute_size=50)
        communities = lv.louvain(graph, resolution=2.0, seed=0).communities()
        ranked = scoring.rank_communities(communities, graph, scenario.hetero, min_size=3)
        assert all(features.size >= 3 for features in ranked)
        scores = [features.risk_score() for features in ranked]
        assert scores == sorted(scores, reverse=True)


class TestEndToEnd:
    def test_the_weighted_projection_beats_the_naive_one_on_precision(self):
        """The claim the whole repository is built on, measured rather than asserted.

        One infrastructure attribute behind half of a 400-account population, which is above the size cap and
        below it for the naive projection. Recall is not the interesting column: every ring pair survives in
        both. Precision is.
        """
        scenario = data.planted_rings(
            n_legit=400, ring_sizes=(10, 6, 4), hub_attributes=1, hub_share=0.5, seed=7
        )
        truth = list(scenario.rings.values())
        naive = connected_components(
            project(scenario.hetero, weighting="count", max_attribute_size=None)
        )
        weighted = connected_components(
            project(scenario.hetero, weighting="idf", max_attribute_size=50)
        )
        naive_score = scoring.pairwise_scores(naive, truth)
        weighted_score = scoring.pairwise_scores(weighted, truth)
        assert weighted_score.precision > naive_score.precision
        assert weighted_score.f1 > naive_score.f1

    def test_planted_rings_are_recovered_at_a_suitable_resolution(self):
        scenario = data.planted_rings(n_legit=400, ring_sizes=(12, 9, 7), seed=8)
        graph = project(scenario.hetero, weighting="idf", max_attribute_size=50)
        communities = lv.louvain(graph, resolution=2.0, seed=0).communities()
        recovered = [best for _, _, best, _ in scoring.ring_recovery(communities, scenario.rings)]
        assert max(recovered) > 0.5  # at least the largest ring comes out substantially intact

    def test_the_generator_reports_a_consistent_truth(self):
        scenario = data.planted_rings(n_legit=100, ring_sizes=(5, 3), seed=9)
        assert len(scenario.fraud_accounts) == 8
        for ring, members in scenario.rings.items():
            for account in members:
                assert scenario.ring_of(account) == ring
                assert account in scenario.hetero.links
        assert all(account in scenario.hetero.signup_time for account in scenario.hetero.accounts)
