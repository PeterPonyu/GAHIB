"""Adapters for externally maintained single-cell benchmark baselines."""

from .online_graph_attention import (  # noqa: F401
    ALL_GRAPH_ATTENTION_METHODS,
    ONLINE_GRAPH_ATTENTION_METHODS,
    ONLINE_GRAPH_ATTENTION_SPECS,
    PYTORCH_GRAPH_ATTENTION_METHODS,
    ExternalBaselineResult,
    train_online_graph_attention,
    train_pytorch_graph_attention_style,
)
