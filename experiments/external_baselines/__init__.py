"""Adapters for externally maintained single-cell benchmark baselines."""

from .online_graph_attention import (  # noqa: F401
    ONLINE_GRAPH_ATTENTION_METHODS,
    ONLINE_GRAPH_ATTENTION_SPECS,
    ExternalBaselineResult,
    train_online_graph_attention,
)
