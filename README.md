# Fraud Ring Detection on Heterogeneous Graphs

IDF-weighted bipartite projection, **Louvain modularity optimisation written from scratch** (local moving,
graph aggregation, resolution parameter), label propagation, consensus clustering across seeds, transparent
ring scoring, and evaluation against a review-capacity queue. **Pure Python, standard library only** -- no
networkx, no python-louvain, no NumPy.

## The whole problem in one number

Accounts are never linked directly. They are linked by sharing a device, a card, an address, an IP. So the
first step is projecting a bipartite graph onto accounts, and that step is where detection is usually decided
before any algorithm runs:

```
a coffee-shop IP address shared by 4,000 accounts
    => 4,000 x 3,999 / 2 = 7,998,000 account pairs from that one attribute
```

Connected components over those edges returns **one ring containing most of your customers**. It is not
wrong about the links; it is wrong about what a link means. `python -m fraudgraph.cli components` prints all
four projections side by side and then scores each as a detector:

```
projection                          edges   largest component   share of accounts
count, no cap (naive)             ~30,000              ~1,000                ~80%
idf, no cap                       ~30,000              ~1,000                ~80%
count, cap 50                      ~1,500                 ~14                 ~1%
idf, cap 50                        ~1,500                 ~14                 ~1%
```

Note which metric collapses and which does not. **Recall stays high in the naive projection** -- every ring
pair really is inside the giant component -- while precision goes to almost nothing. A report that quotes
recall alone will rank the useless detector first.

Two mechanisms fix it, and both are needed:

- **Inverse-frequency weighting.** `w(attribute) = log(N / n_attribute)`. A card shared by two accounts is
  strong evidence; an airport wifi IP is worth almost nothing. An attribute held by the entire population
  produces `log(1) = 0` and no edges at all, which is correct and is asserted in the tests.
- **A hard size cap.** IDF shrinks hub edges without removing them, and enough near-zero edges still merge
  communities. An IP with 4,000 accounts behind it is infrastructure, not evidence.

> Figures above illustrate the report format; they depend on the seed. Run it, and read the tests for the
> claims that are actually pinned down.

```bash
python -m fraudgraph.cli components   # the projection failure, measured
python -m fraudgraph.cli louvain      # detection end to end, with the analyst queue and the truth
python -m fraudgraph.cli resolution   # the resolution limit, on a graph whose answer is known
python -m fraudgraph.cli stability    # five seeds, five different partitions, near-identical Q
python -m fraudgraph.cli compare      # components vs label propagation vs Louvain
```

---

## Modularity, and why the optimiser is checked twice

```
Q = sum over communities of [ in_c / 2m  -  gamma * (tot_c / 2m)^2 ]
```

The null term is the substance: a community holding 30% of all edge endpoints is *expected* to contain about
9% of edges by chance, and only the excess counts. Without it, everything in one community always wins.

Louvain maximises `Q` greedily -- move each node to the neighbouring community that improves `Q` most, then
collapse each community into one node and repeat. The move gain reduces to

```
dQ(i -> C) = w(i,C)/m - gamma * tot_C * k_i / (2 m^2)
```

derived in `modularity_gain`. Two things about this implementation are deliberate:

**The objective is implemented twice.** Incrementally inside the optimiser, and directly from the definition
in `modularity()`. The tests assert they agree, because a wrong gain formula produces a perfectly plausible
partition that optimises nothing in particular -- there is no symptom to notice.

**`tot_C` excludes the node being moved**, which is why the code removes a node from its own community before
evaluating any candidate including the one it came from. Skipping that is the classic Louvain bug: every node
then appears best off where it already is, the first pass makes no moves, and the algorithm returns singletons
while looking converged.

Aggregation is checked as an identity too: the collapsed graph has the same total weight, and any partition of
it has the same modularity as the corresponding partition of the original. That is the only reason optimising
the small graph is the same problem as optimising the big one.

## The resolution limit is not a bug you can optimise away

Fortunato and Barthelemy, 2007: communities whose internal edge count falls below roughly `sqrt(2m)` cannot be
resolved by modularity, because **merging two of them increases Q**. Take forty 5-cliques in a cycle joined by
single edges, where the right answer is not in doubt:

```
gamma  1.0:  ~20 communities   (adjacent cliques merged)
gamma  4.0:  ~40 communities   (the cliques recovered)
```

The tests establish this **algebraically rather than through Louvain**: at `gamma = 1` the merged partition has
strictly higher modularity than the correct one, and at `gamma = 4` the preference reverses. No optimiser can
return a partition that is not the maximum, so no number of restarts helps. Only changing the objective does.

In fraud terms: **a three-account ring in a large graph is out of reach at the default resolution**, however
tight it is. That is why `recall_by_size` exists -- aggregate recall flatters a detector that finds only the
large rings, since those contain most of the accounts.

## A community boundary is not a fact about the graph

Louvain is order-dependent. Five runs on identical data give five different partitions with nearly identical
`Q`, because modularity has many near-optimal solutions. `consensus()` runs it repeatedly, keeps only the
co-membership that survives a threshold fraction of runs, and returns components of what remains. Accounts left
as singletons are genuinely ambiguous, and telling an analyst that is more useful than handing them a boundary
a different seed would have drawn elsewhere.

## Scoring, and the metric with consequences

Community detection returns a partition, not an answer: most communities are households, shared offices and
coincidence. The output is ranked by a **transparent heuristic** -- mean IDF weight per edge, internal density,
number of distinct attribute *types* shared (a device and a card and an address is categorically stronger than
three devices, which can be one second-hand phone), signup burstiness, and a penalty for weight leaking
outwards. Weights are visible in the source so a disagreement can be argued about.

A learned scorer would beat it on historical labels and lose on the next pattern, because it can only learn
what was already caught. The features are the durable part.

**A two-account community scores exactly zero.** Two accounts sharing an address are a family until something
else says otherwise, and a detector that flags them has rediscovered the electoral roll.

Evaluation:

| measure | what it is for |
|---|---|
| pairwise precision / recall / F1 | comparing a partition against truth with no identifier matching; penalises splitting a ring *and* merging two |
| `ring_recovery` (best Jaccard per ring) | distinguishes a ring split across three communities from a ring buried in a community of four hundred -- opposite fixes |
| `recall_by_size` | where the resolution limit becomes visible |
| `precision_at_k` | of the top *k* clusters an analyst will actually review, how many contain fraud. The number with consequences |

`precision_at_k` requires at least `min_fraud` ring accounts per hit and `rank_communities` drops communities
below three accounts, so returning one giant component cannot be scored as a success.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Hand-computed modularity (two disjoint edges: `Q = 0.5` split, `Q = 0` merged), the gain identity against full
recomputation, the aggregation identity, IDF weights against `log(N/n)`, the hub-and-cap failure on a graph
built to have exactly two structures in it, pairwise scores worked out on paper, and the resolution limit as
algebra.

## Limits

- **Static graph.** No time dimension in the structure: a device shared in 2019 and one shared yesterday are
  the same edge. Real systems window the graph, and burstiness here is only a scoring feature.
- **No supervised model, and no labels to train one.** The score is a heuristic and its weights are not fitted.
- **Modularity only.** No Leiden (which fixes Louvain's badly-connected-community defect), no stochastic block
  models, no Infomap, no spectral methods. Leiden is the first thing to add.
- **Undirected and unipartite after projection.** Money-flow direction and attribute types are used for
  scoring, not for the clustering itself; a heterogeneous GNN or metapath-based method uses them properly.
- **Pure Python and quadratic in attribute size.** An attribute with 4,000 accounts is dropped by the cap
  partly because it is meaningless and partly because eight million pairs is not tractable here.
- **Synthetic data only.** No real account, device or card data is used or implied.

## References

- Blondel, Guillaume, Lambiotte & Lefebvre (2008), *Fast unfolding of communities in large networks* -- Louvain.
- Newman & Girvan (2004), *Finding and evaluating community structure in networks* -- modularity.
- Fortunato & Barthelemy (2007), *Resolution limit in community detection*.
- Reichardt & Bornholdt (2006), *Statistical mechanics of community detection* -- the resolution parameter.
- Traag, Waltman & van Eck (2019), *From Louvain to Leiden: guaranteeing well-connected communities*.
- Raghavan, Albert & Kumara (2007), *Near linear time algorithm to detect community structures*.
- Lancichinetti & Fortunato (2012), *Consensus clustering in complex networks*.
- Akoglu, Tong & Koutra (2015), *Graph based anomaly detection and description: a survey*.

MIT licensed.
