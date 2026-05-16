"""Adapters for externally maintained single-cell benchmark baselines."""

from .online_graph_attention import (  # noqa: F401
    ALL_GRAPH_ATTENTION_METHODS,
    ONLINE_GRAPH_ATTENTION_METHODS,
    ONLINE_GRAPH_ATTENTION_SPECS,
    PYTORCH_GRAPH_ATTENTION_METHODS,
    SOURCE_GRAPH_ATTENTION_METHODS,
    ExternalBaselineResult,
    resolve_online_graph_attention_method,
    train_external_online_graph_attention,
    train_online_graph_attention,
    train_pytorch_graph_attention_style,
)
