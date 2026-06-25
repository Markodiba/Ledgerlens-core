"""Graph Neural Network for structural wash-ring scoring (issue #129).

Uses GraphSAGE (Hamilton et al., 2017) to produce per-wallet wash-ring
probabilities that complement the SCC-based features in graph_engine.py.

Requires: torch, torch_geometric (PyTorch Geometric).
If these packages are absent the module degrades gracefully — callers
that guard with `try/except ImportError` receive `gnn_wash_ring_prob = 0.0`.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class WashRingGNN(torch.nn.Module):
        """3-layer GraphSAGE for node-level wash-ring binary classification."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 64,
            num_layers: int = 3,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.dropout = dropout
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, 1))

        def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
            for conv in self.convs[:-1]:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
            return torch.sigmoid(x).squeeze(-1)

        def architecture_summary(self) -> dict:
            n_params = sum(p.numel() for p in self.parameters())
            return {
                "model": "WashRingGNN (GraphSAGE)",
                "num_layers": len(self.convs),
                "hidden_channels": self.convs[0].out_channels if len(self.convs) > 1 else None,
                "dropout": self.dropout,
                "num_parameters": n_params,
            }

else:
    class WashRingGNN:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch and torch_geometric are required for WashRingGNN")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _graph_to_pyg(graph: Any, node_feature_dim: int = 4) -> Any:
    """Convert a NetworkX DiGraph (from graph_engine) to a PyG Data object.

    Node features: [log_out_degree, log_in_degree, log_total_volume, log_trade_count]
    """
    if not _TORCH_AVAILABLE or not _NX_AVAILABLE:
        raise ImportError("torch_geometric and networkx are required")

    nodes = list(graph.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    x_list = []
    for node in nodes:
        out_deg = graph.out_degree(node)
        in_deg = graph.in_degree(node)
        out_edges = graph.out_edges(node, data=True)
        total_vol = sum(d.get("total_volume", 0.0) for _, _, d in out_edges)
        trade_count = sum(d.get("trade_count", 0) for _, _, d in graph.out_edges(node, data=True))
        import math
        x_list.append([
            math.log1p(out_deg),
            math.log1p(in_deg),
            math.log1p(total_vol),
            math.log1p(trade_count),
        ])

    x = torch.tensor(x_list, dtype=torch.float)

    edge_src, edge_dst = [], []
    for u, v in graph.edges():
        if u in node_index and v in node_index:
            edge_src.append(node_index[u])
            edge_dst.append(node_index[v])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index), nodes


class TradeGraphDataset:
    """Converts a NetworkX trade graph into a PyG Data object."""

    def __init__(self, graph: Any) -> None:
        if not _TORCH_AVAILABLE or not _NX_AVAILABLE:
            raise ImportError("torch_geometric and networkx are required")
        self._graph = graph

    def to_pyg(self) -> tuple[Any, list[str]]:
        """Return (Data, node_list) for the stored graph."""
        return _graph_to_pyg(self._graph)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GNNTrainer:
    """Trains WashRingGNN on labelled synthetic trade graphs."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch and torch_geometric are required for GNNTrainer")
        self.model = WashRingGNN(in_channels, hidden_channels, num_layers, dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def train(self, data: Any, labels: "torch.Tensor") -> list[float]:
        """Train for `self.epochs` epochs on a single PyG Data object."""
        self.model.train()
        losses = []
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.binary_cross_entropy(out, labels.float())
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss))
        return losses

    def save(self, path: str | Path, version_hash: str = "v1") -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"gnn_{version_hash}.pt"
        torch.save(self.model.state_dict(), str(out))
        return out

    @classmethod
    def load(cls, checkpoint: str | Path, in_channels: int = 4, **kwargs: Any) -> "GNNTrainer":
        trainer = cls(in_channels=in_channels, **kwargs)
        state = torch.load(str(checkpoint), map_location="cpu")
        trainer.model.load_state_dict(state)
        return trainer


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class GNNInferenceEngine:
    """Scores all nodes in the current trade graph and returns wash-ring probs."""

    _instance: GNNInferenceEngine | None = None
    _lock = threading.Lock()

    def __init__(self, model: Any, in_channels: int = 4) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch and torch_geometric are required for GNNInferenceEngine")
        self._model = model
        self._model.eval()
        self._in_channels = in_channels
        self._last_inference_time: float | None = None

    @classmethod
    def get_instance(cls) -> "GNNInferenceEngine":
        with cls._lock:
            if cls._instance is None:
                raise RuntimeError("GNNInferenceEngine not initialised — call set_instance() first")
            return cls._instance

    @classmethod
    def set_instance(cls, engine: "GNNInferenceEngine") -> None:
        with cls._lock:
            cls._instance = engine

    def score_graph(self, graph: Any) -> dict[str, float]:
        """Return {wallet: wash_ring_probability} for all nodes in `graph`."""
        if not _NX_AVAILABLE:
            raise ImportError("networkx is required")
        dataset = TradeGraphDataset(graph)
        data, nodes = dataset.to_pyg()
        with torch.no_grad():
            probs = self._model(data.x, data.edge_index)
        self._last_inference_time = time.time()
        return {node: float(prob) for node, prob in zip(nodes, probs.tolist())}

    def stats(self) -> dict:
        import datetime as _dt
        last = (
            _dt.datetime.fromtimestamp(self._last_inference_time, tz=_dt.timezone.utc).isoformat()
            if self._last_inference_time
            else None
        )
        arch = self._model.architecture_summary() if hasattr(self._model, "architecture_summary") else {}
        return {
            "status": "available",
            "last_inference_time": last,
            **arch,
        }


# ---------------------------------------------------------------------------
# Convenience: load from disk
# ---------------------------------------------------------------------------

def load_gnn_engine(model_dir: str | Path, in_channels: int = 4) -> GNNInferenceEngine | None:
    """Load the latest gnn_*.pt checkpoint from `model_dir` and return an engine.

    Returns None (and logs a warning) if torch_geometric is unavailable or no
    checkpoint exists.
    """
    if not _TORCH_AVAILABLE:
        logger.warning("torch_geometric not installed; GNN scoring disabled")
        return None
    model_dir = Path(model_dir)
    checkpoints = sorted(model_dir.glob("gnn_*.pt"))
    if not checkpoints:
        logger.info("No GNN checkpoint found in %s; skipping GNN scoring", model_dir)
        return None
    ckpt = checkpoints[-1]
    model = WashRingGNN(in_channels=in_channels)
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    engine = GNNInferenceEngine(model, in_channels=in_channels)
    logger.info("Loaded GNN checkpoint %s", ckpt.name)
    return engine
