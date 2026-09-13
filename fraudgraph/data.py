"""Synthetic account graphs with planted rings -- and planted reasons to be wrong about them.

A generator that only plants rings tests nothing. Real fraud graphs are hard because innocent accounts share
things too, and the sharing looks identical in the data:

* **Infrastructure attributes.** A coffee-shop IP, a corporate NAT gateway, a shared office device. Thousands
  of unrelated accounts behind one attribute, which is what destroys the naive projection.
* **Households.** Two or three accounts sharing an address and a card, because they are a family. Small,
  dense, and structurally indistinguishable from a tiny ring -- the honest response is that a two-account
  cluster is not evidence, whatever the graph says.
* **Recycled attributes.** A phone number reassigned by the carrier, a device sold second-hand. A real link in
  the data corresponding to nothing in the world.

Each generator returns the ground truth alongside the graph, so precision and recall are computable rather
than asserted. ``planted_rings`` is the main one; ``ring_of_cliques`` exists for the resolution-limit
demonstration, where the correct answer is known by construction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .graph import Graph, HeteroGraph


@dataclass
class Scenario:
    """A graph plus the truth: which accounts belong to which planted ring."""

    hetero: HeteroGraph
    rings: "dict[str, set[str]]" = field(default_factory=dict)  # ring id -> accounts
    households: "list[set[str]]" = field(default_factory=list)  # innocent dense clusters
    notes: str = ""

    @property
    def fraud_accounts(self) -> "set[str]":
        return {account for members in self.rings.values() for account in members}

    def ring_of(self, account: str) -> str | None:
        for ring, members in self.rings.items():
            if account in members:
                return ring
        return None

    def summary(self) -> str:
        sizes = sorted((len(members) for members in self.rings.values()), reverse=True)
        return (
            f"{len(self.hetero.accounts)} accounts, {len(self.rings)} planted rings "
            f"covering {len(self.fraud_accounts)} accounts (sizes {sizes}), "
            f"{len(self.households)} innocent households\n{self.notes}"
        )


def planted_rings(
    n_legit: int = 1200,
    ring_sizes: "tuple[int, ...]" = (12, 9, 7, 6, 5, 4, 3),
    hub_attributes: int = 3,
    hub_share: float = 0.35,
    household_count: int = 60,
    recycled_attributes: int = 25,
    seed: int = 0,
) -> Scenario:
    """A population of ordinary accounts with fraud rings hidden inside it.

    Ring members share devices and cards heavily -- that is what makes them a ring -- and they also touch the
    same infrastructure everyone else does, so they cannot be found by looking for shared attributes alone.

    Ring sizes deliberately span 3 to 12. The small ones are there to be missed: at ``gamma = 1`` modularity
    cannot resolve a community whose internal edge count is far below ``sqrt(2m)``, so a three-account ring in
    a graph this size is beyond reach of the default resolution however tight it is. Reporting recall without
    breaking it down by ring size hides that, so ``scoring.recall_by_size`` does the breakdown.
    """
    if any(size < 2 for size in ring_sizes):
        raise ValueError("a ring needs at least two accounts")
    rng = random.Random(seed)
    hetero = HeteroGraph()

    legit = [f"acct_{index:05d}" for index in range(n_legit)]
    rings: dict[str, set[str]] = {}
    fraud: list[str] = []
    for ring_index, size in enumerate(ring_sizes):
        members = {f"fraud_{ring_index}_{position:02d}" for position in range(size)}
        rings[f"ring_{ring_index}"] = members
        fraud.extend(sorted(members))

    everyone = legit + fraud
    for account in everyone:
        hetero.signup_time[account] = rng.uniform(0.0, 365.0)
        hetero.links.setdefault(account, set())

    # --- infrastructure: a few attributes shared by a large share of the population
    hubs = [f"ip_public_{index}" for index in range(hub_attributes)]
    for account in everyone:
        if rng.random() < hub_share:
            hetero.add_link(account, rng.choice(hubs), "ip")

    # --- ordinary accounts: their own device and card, occasionally a second device
    for account in legit:
        hetero.add_link(account, f"dev_{account}", "device")
        hetero.add_link(account, f"card_{account}", "card")
        if rng.random() < 0.15:
            hetero.add_link(account, f"dev_{account}_b", "device")

    # --- households: two or three accounts sharing an address and sometimes a card
    households: list[set[str]] = []
    for index in range(household_count):
        size = rng.choice([2, 2, 3])
        members = set(rng.sample(legit, size))
        address = f"addr_home_{index}"
        for member in members:
            hetero.add_link(member, address, "address")
        if rng.random() < 0.5:
            card = f"card_home_{index}"
            for member in members:
                hetero.add_link(member, card, "card")
        households.append(members)

    # --- recycled attributes: a real link in the data that means nothing in the world
    for index in range(recycled_attributes):
        first, second = rng.sample(legit, 2)
        phone = f"phone_recycled_{index}"
        hetero.add_link(first, phone, "phone")
        hetero.add_link(second, phone, "phone")

    # --- the rings: dense internal sharing, plus the same infrastructure as everyone else
    for ring, members in rings.items():
        ordered = sorted(members)
        shared_devices = [f"dev_{ring}_{index}" for index in range(max(len(ordered) // 3, 1))]
        shared_cards = [f"card_{ring}_{index}" for index in range(max(len(ordered) // 4, 1))]
        burst_start = rng.uniform(0.0, 350.0)
        for position, account in enumerate(ordered):
            # signups cluster in time, which is a feature the graph does not carry
            hetero.signup_time[account] = burst_start + rng.uniform(0.0, 6.0)
            hetero.add_link(account, shared_devices[position % len(shared_devices)], "device")
            hetero.add_link(account, shared_cards[position % len(shared_cards)], "card")
            hetero.add_link(account, f"addr_{ring}", "address")
            if rng.random() < 0.4:  # some members also carry their own device: rings are not perfect cliques
                hetero.add_link(account, f"dev_{account}", "device")
            if rng.random() < hub_share:
                hetero.add_link(account, rng.choice(hubs), "ip")

    notes = (
        f"{hub_attributes} infrastructure attributes each behind roughly "
        f"{hub_share / hub_attributes:.0%} of accounts; households and recycled phone numbers are "
        "innocent dense structure that any detector will have to survive."
    )
    return Scenario(hetero=hetero, rings=rings, households=households, notes=notes)


def ring_of_cliques(clique_count: int = 24, clique_size: int = 5, seed: int = 0) -> Graph:
    """``clique_count`` cliques in a cycle, joined by single edges: the resolution-limit test case.

    Fortunato and Barthelemy's construction. The correct partition is obvious -- one community per clique --
    and modularity at ``gamma = 1`` does not return it: merging two adjacent cliques *increases* Q once the
    graph is large enough, because the null model expects so few edges between them that even one bridge looks
    like an excess. No amount of restarting or better optimisation helps, since the merged partition genuinely
    has the higher objective value. Only changing the objective, via ``gamma``, recovers the cliques.
    """
    if clique_size < 3 or clique_count < 3:
        raise ValueError("need at least three cliques of at least three nodes")
    graph = Graph()
    for clique in range(clique_count):
        members = [f"c{clique}_n{index}" for index in range(clique_size)]
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                graph.add_edge(first, second, 1.0)
    for clique in range(clique_count):
        following = (clique + 1) % clique_count
        graph.add_edge(f"c{clique}_n0", f"c{following}_n1", 1.0)
    return graph


def two_clusters(size: int = 20, bridges: int = 1, seed: int = 0) -> Graph:
    """Two cliques joined by ``bridges`` edges: a case where the right answer is unambiguous."""
    rng = random.Random(seed)
    graph = Graph()
    left = [f"L{index}" for index in range(size)]
    right = [f"R{index}" for index in range(size)]
    for group in (left, right):
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                graph.add_edge(first, second, 1.0)
    for _ in range(bridges):
        graph.add_edge(rng.choice(left), rng.choice(right), 1.0)
    return graph


SCENARIOS = {
    "planted": planted_rings,
    "cliques": ring_of_cliques,
    "two": two_clusters,
}
