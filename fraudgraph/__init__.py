"""Fraud ring detection on heterogeneous graphs, in pure Python.

    from fraudgraph import data, graph, louvain, scoring

    scenario   = data.planted_rings(seed=0)
    projected  = graph.project(scenario.hetero, weighting="idf", max_attribute_size=50)
    result     = louvain.louvain(projected, resolution=2.0)
    queue      = scoring.rank_communities(result.communities(), projected, scenario.hetero)

Read ``graph.project`` first. Everything downstream is decided there: a shared attribute behind thousands of
accounts implies millions of account pairs, and an unweighted projection turns the whole population into one
component before any community algorithm gets a say.
"""

from __future__ import annotations

from . import data, graph, louvain, scoring
from .graph import Graph, HeteroGraph, connected_components, project
from .louvain import consensus, label_propagation, modularity
from .louvain import louvain as detect_communities
from .scoring import pairwise_scores, rank_communities

__all__ = [
    "Graph",
    "HeteroGraph",
    "connected_components",
    "consensus",
    "data",
    "detect_communities",
    "graph",
    "label_propagation",
    "louvain",
    "modularity",
    "pairwise_scores",
    "project",
    "rank_communities",
    "scoring",
]
__version__ = "1.0.0"
