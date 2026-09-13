"""Scoring candidate rings, and measuring detection the way a fraud team actually experiences it.

Community detection returns a partition, not an answer. Most communities are innocent -- households, shared
offices, coincidence -- so the output has to be **ranked**, because an analyst team can review a fixed number
of cases per day and nothing else about the model matters if the top of that queue is noise.

Two measurement points here deserve stating outright.

**Aggregate recall lies about small rings.** A detector that finds every ring of ten accounts and no ring of
three will post a respectable account-weighted recall, because the big rings contain most of the accounts.
``recall_by_size`` breaks it out, and the breakdown is where the resolution limit becomes visible.

**Precision at the review capacity is the metric with consequences.** Precision over all communities counts
work nobody will do. If the team can review twenty clusters this week, the question is how many of those
twenty contain fraud -- ``precision_at_k`` answers that, and it is the number to optimise.

The risk score itself is a **transparent heuristic**, not a fitted model, and that is a deliberate choice
rather than a shortcut: the features are the ones an analyst would name, and the weights are visible so a
disagreement can be argued about. A gradient-boosted scorer would do better on historical labels and worse on
the next fraud pattern, because it can only learn what was caught before. The features are the durable part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import Scenario
from .graph import Graph, HeteroGraph


@dataclass(frozen=True)
class RingFeatures:
    """What can be said about a candidate ring without knowing whether it is one."""

    members: "frozenset[str]"
    size: int
    density: float  # internal edges over possible internal edges
    mean_edge_weight: float  # in IDF units: how surprising the shared attributes are
    attribute_types: int  # devices AND cards AND addresses is stronger than three devices
    max_shared_fraction: float  # the largest share of members behind a single attribute
    signup_burstiness: float  # population signup spread over this community's spread
    external_ratio: float  # weight leaving the community over total weight touching it

    def risk_score(self) -> float:
        """A weighted sum of the features, with the reasoning for each weight in the source.

        Every term is bounded, so no single feature can dominate through scale alone:

        * ``mean_edge_weight`` (x 1.0) -- the core evidence. A high IDF mean means the members share
          attributes that almost nobody else shares.
        * ``density`` (x 0.8) -- rings share with each other, not through a chain of intermediaries.
        * ``attribute_types`` (x 0.5, capped) -- sharing a device *and* a card *and* an address is
          categorically stronger than sharing three devices, which can be one second-hand phone.
        * ``signup_burstiness`` (x 0.4, capped) -- accounts created within days of each other. Not graph
          structure at all, and one of the strongest signals available.
        * ``external_ratio`` (x -0.6) -- a community leaking most of its weight outwards is a slice of
          something bigger, not a ring.
        * ``size`` (log, x 0.2) -- larger is mildly more suspicious, kept small so a household of two with
          one shared card cannot rank above a genuine ring of eight.

        A two-account community scores low by construction. Two accounts sharing an address are a family
        until something else says otherwise, and a detector that flags them has found the electoral roll.
        """
        if self.size < 3:
            return 0.0
        return (
            1.0 * min(self.mean_edge_weight / 5.0, 1.0)
            + 0.8 * self.density
            + 0.5 * min(self.attribute_types / 3.0, 1.0)
            + 0.4 * min(self.signup_burstiness / 10.0, 1.0)
            - 0.6 * self.external_ratio
            + 0.2 * min(math.log(self.size) / math.log(20.0), 1.0)
        )


def describe_community(
    members: "set[str]", graph: Graph, hetero: HeteroGraph, population_spread: float
) -> RingFeatures:
    """Compute the features for one candidate community."""
    size = len(members)
    internal_weight = 0.0
    external_weight = 0.0
    internal_edges = 0
    seen: set[str] = set()
    for node in members:
        for other, weight in graph.neighbours(node).items():
            if other in members:
                if other != node and other not in seen:
                    internal_weight += weight
                    internal_edges += 1
            else:
                external_weight += weight
        seen.add(node)

    possible = size * (size - 1) / 2
    density = internal_edges / possible if possible else 0.0
    mean_weight = internal_weight / internal_edges if internal_edges else 0.0

    kinds = set()
    counts: dict[str, int] = {}
    for account in members:
        for attribute in hetero.links.get(account, set()):
            counts[attribute] = counts.get(attribute, 0) + 1
    for attribute, count in counts.items():
        if count >= 2:
            kinds.add(hetero.attribute_type[attribute])
    max_shared = max((count / size for count in counts.values()), default=0.0)

    times = [hetero.signup_time[account] for account in members if account in hetero.signup_time]
    spread = (max(times) - min(times)) if len(times) > 1 else population_spread
    burstiness = population_spread / max(spread, 1e-9)

    touching = internal_weight + external_weight
    return RingFeatures(
        members=frozenset(members),
        size=size,
        density=density,
        mean_edge_weight=mean_weight,
        attribute_types=len(kinds),
        max_shared_fraction=min(max_shared, 1.0),
        signup_burstiness=burstiness,
        external_ratio=external_weight / touching if touching else 0.0,
    )


def rank_communities(
    communities: "list[set[str]]", graph: Graph, hetero: HeteroGraph, min_size: int = 3
) -> "list[RingFeatures]":
    """Score every community and return them worst-first -- the analyst queue.

    Communities below ``min_size`` are dropped rather than scored low. Two accounts sharing an attribute is
    the single most common pattern in any real graph and the least informative; keeping them in the queue
    means the queue is mostly pairs.
    """
    times = list(hetero.signup_time.values())
    population_spread = (max(times) - min(times)) if len(times) > 1 else 1.0
    scored = [
        describe_community(members, graph, hetero, population_spread)
        for members in communities
        if len(members) >= min_size
    ]
    return sorted(scored, key=lambda features: features.risk_score(), reverse=True)


def queue_table(ranked: "list[RingFeatures]", scenario: Scenario, limit: int = 12) -> str:
    """The queue as an analyst would see it, with the planted truth in the last column."""
    header = (
        f"{'rank':>5}{'size':>6}{'density':>9}{'idf/edge':>10}{'types':>7}"
        f"{'burst':>8}{'ext':>7}{'score':>8}   verdict"
    )
    lines = [header, "-" * len(header)]
    for index, features in enumerate(ranked[:limit], start=1):
        rings = {scenario.ring_of(account) for account in features.members}
        rings.discard(None)
        fraud_members = len(features.members & scenario.fraud_accounts)
        if not rings:
            verdict = "innocent"
        elif fraud_members == len(features.members):
            verdict = f"pure {'+'.join(sorted(rings))}"
        else:
            verdict = f"{'+'.join(sorted(rings))} plus {len(features.members) - fraud_members} innocent"
        lines.append(
            f"{index:>5}{features.size:>6}{features.density:>9.2f}{features.mean_edge_weight:>10.2f}"
            f"{features.attribute_types:>7}{features.signup_burstiness:>8.1f}"
            f"{features.external_ratio:>7.2f}{features.risk_score():>8.3f}   {verdict}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseScore:
    """Precision and recall over *pairs* of accounts placed together.

    Pair counting is the right unit for comparing a partition against a truth that has no labels to match up:
    it needs no correspondence between predicted and true community identifiers, and it penalises both
    splitting a ring and merging two. Its known bias is towards large clusters, since a community of size k
    contributes k(k-1)/2 pairs -- one merged giant component can look respectable on recall alone, which is
    why precision is always reported with it.
    """

    precision: float
    recall: float
    true_pairs: int
    predicted_pairs: int
    correct_pairs: int

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0.0:
            return 0.0
        return 2.0 * self.precision * self.recall / (self.precision + self.recall)

    def summary(self) -> str:
        return (
            f"pairwise precision {self.precision:.3f}, recall {self.recall:.3f}, F1 {self.f1:.3f} "
            f"({self.correct_pairs:,} correct of {self.predicted_pairs:,} predicted, "
            f"{self.true_pairs:,} true)"
        )


def _pairs(groups: "list[set[str]]") -> "set[tuple[str, str]]":
    output: set[tuple[str, str]] = set()
    for group in groups:
        ordered = sorted(group)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                output.add((first, second))
    return output


def pairwise_scores(predicted: "list[set[str]]", truth: "list[set[str]]") -> PairwiseScore:
    predicted_pairs = _pairs(predicted)
    true_pairs = _pairs(truth)
    correct = predicted_pairs & true_pairs
    return PairwiseScore(
        precision=len(correct) / len(predicted_pairs) if predicted_pairs else 0.0,
        recall=len(correct) / len(true_pairs) if true_pairs else 0.0,
        true_pairs=len(true_pairs),
        predicted_pairs=len(predicted_pairs),
        correct_pairs=len(correct),
    )


def ring_recovery(
    communities: "list[set[str]]", rings: "dict[str, set[str]]"
) -> "list[tuple[str, int, float, int]]":
    """Per ring: ``(id, size, best Jaccard against any community, size of that community)``.

    Jaccard rather than "was it found", because the interesting failures are partial. A ring split across
    three communities and a ring buried inside a community of four hundred accounts are both misses, and they
    call for different fixes -- the first wants a lower resolution, the second a higher one.
    """
    output = []
    for ring, members in sorted(rings.items()):
        best = 0.0
        best_size = 0
        for community in communities:
            overlap = len(members & community)
            if not overlap:
                continue
            score = overlap / len(members | community)
            if score > best:
                best, best_size = score, len(community)
        output.append((ring, len(members), best, best_size))
    return output


def recall_by_size(
    communities: "list[set[str]]", rings: "dict[str, set[str]]", threshold: float = 0.5
) -> "dict[int, tuple[int, int]]":
    """``ring size -> (rings recovered above threshold, rings of that size)``.

    The breakdown that aggregate recall hides. Small rings failing while large ones succeed is the signature
    of the resolution limit, and it calls for a different resolution rather than a better optimiser.
    """
    output: dict[int, tuple[int, int]] = {}
    for _, size, best, _ in ring_recovery(communities, rings):
        found, total = output.get(size, (0, 0))
        output[size] = (found + (1 if best >= threshold else 0), total + 1)
    return output


def precision_at_k(
    ranked: "list[RingFeatures]", scenario: Scenario, k: int = 20, min_fraud: int = 2
) -> "tuple[float, int]":
    """Of the top ``k`` queued communities, the fraction containing at least ``min_fraud`` ring accounts.

    This is the metric with consequences attached. It is also the one that punishes a detector for returning
    one enormous community: a giant cluster contains fraud, so it counts as a hit, yet handing an analyst
    forty thousand accounts is not a case. ``min_fraud`` and the size cap in ``rank_communities`` together
    keep the measure honest, and the size of each hit is reported next to it in ``queue_table``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked[:k]
    hits = sum(
        1 for features in top if len(features.members & scenario.fraud_accounts) >= min_fraud
    )
    return (hits / len(top) if top else 0.0), len(top)


def evaluation_report(
    communities: "list[set[str]]", scenario: Scenario, ranked: "list[RingFeatures]", k: int = 20
) -> str:
    pairwise = pairwise_scores(communities, list(scenario.rings.values()))
    precision, reviewed = precision_at_k(ranked, scenario, k=k)
    lines = [
        pairwise.summary(),
        f"precision@{reviewed} in the analyst queue: {precision:.1%}",
        "",
        f"{'ring':<10}{'size':>6}{'best Jaccard':>15}{'in community of':>18}",
    ]
    lines.append("-" * len(lines[-1]))
    for ring, size, best, community_size in ring_recovery(communities, scenario.rings):
        lines.append(f"{ring:<10}{size:>6}{best:>15.2f}{community_size:>18}")
    lines.append("")
    lines.append("recall by ring size (recovered at Jaccard >= 0.5):")
    for size in sorted(recall_by_size(communities, scenario.rings), reverse=True):
        found, total = recall_by_size(communities, scenario.rings)[size]
        lines.append(f"  size {size:>3}: {found}/{total}")
    return "\n".join(lines)
