"""Single-relation input adapter around the CARE-GNN authors' model components.

The unmodified author model assumes exactly three relation-specific adjacency
lists. CAMERA supplies one observed relation, so this adapter retains the
authors' IntraAgg, filtering, RLModule, inter-relation aggregation, and
OneLayerCARE classifier while presenting exactly one relation. It is an
author-code adaptation, not an unmodified official reproduction.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
from torch import nn

CARE_REPO = Path(__file__).resolve().parent / "CARE-GNN"
sys.path.insert(0, str(CARE_REPO))
from layers import IntraAgg, RLModule, threshold_inter_agg  # noqa: E402
from model import OneLayerCARE  # noqa: E402


def adjacency_from_edge_index(edge_index: torch.Tensor, num_nodes: int) -> list[set[int]]:
    """Match the author preprocessing: undirected adjacency with self-loops."""
    adjacency = [{node} for node in range(num_nodes)]
    for source, target in edge_index.t().tolist():
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


class SingleRelationInterAgg(nn.Module):
    def __init__(
        self,
        raw_features: torch.Tensor,
        adjacency: list[set[int]],
        hidden_dim: int = 64,
        step_size: float = 0.02,
    ) -> None:
        super().__init__()
        if len(adjacency) != raw_features.size(0):
            raise ValueError("Adjacency and features have different node counts")
        self.features = nn.Embedding.from_pretrained(raw_features.detach().clone(), freeze=True)
        self.adjacency = adjacency
        self.intra = IntraAgg(self.features, raw_features.size(1), cuda=False)
        self.label_clf = nn.Linear(raw_features.size(1), 2)
        self.weight = nn.Parameter(torch.empty(raw_features.size(1), hidden_dim))
        nn.init.xavier_uniform_(self.weight)
        self.embed_dim = hidden_dim
        self.thresholds = [0.5]
        self.step_size = step_size
        self.batch_num = 0
        self.RL = True
        self.relation_score_log: list[list[float]] = []
        self.thresholds_log: list[list[float]] = [self.thresholds.copy()]

    def forward(
        self, nodes: list[int], labels: torch.Tensor, train_flag: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(nodes) < 2:
            raise ValueError("CARE-GNN author code needs batches of at least two nodes")
        neighbor_lists = [sorted(self.adjacency[node]) for node in nodes]
        unique_nodes = sorted(set(nodes).union(*neighbor_lists))
        mapping = {node: index for index, node in enumerate(unique_nodes)}
        device = self.features.weight.device
        indexed = torch.tensor(unique_nodes, dtype=torch.long, device=device)
        all_scores = self.label_clf(self.features(indexed))
        center_scores = all_scores[[mapping[node] for node in nodes]]
        neighbor_scores = [
            all_scores[[mapping[node] for node in neighbors]].reshape(-1, 2)
            for neighbors in neighbor_lists
        ]
        sample_count = [
            math.ceil(len(neighbors) * self.thresholds[0])
            for neighbors in neighbor_lists
        ]
        self.intra.cuda = device.type == "cuda"
        neighbor_features, selected_scores = self.intra.forward(
            nodes, neighbor_lists, center_scores, neighbor_scores, sample_count
        )
        center_features = self.features(
            torch.tensor(nodes, dtype=torch.long, device=device)
        )
        combined = threshold_inter_agg(
            1, center_features, neighbor_features, self.embed_dim,
            self.weight, self.thresholds, len(nodes), device.type == "cuda"
        )
        if self.RL and train_flag:
            if self.batch_num < 1 or not torch.any(labels == 1):
                raise ValueError("CARE-GNN RL requires a positive training node per batch")
            relation_scores, _, thresholds, stop_flag = RLModule(
                [selected_scores], self.relation_score_log, labels,
                self.thresholds, self.batch_num, self.step_size
            )
            self.thresholds = thresholds
            self.RL = stop_flag
            self.relation_score_log.append(relation_scores)
            self.thresholds_log.append(self.thresholds.copy())
        return combined, center_scores


def build_care_model(
    raw_features: torch.Tensor,
    edge_index: torch.Tensor,
    hidden_dim: int = 64,
    auxiliary_weight: float = 2.0,
) -> OneLayerCARE:
    adjacency = adjacency_from_edge_index(edge_index.cpu(), raw_features.size(0))
    inter = SingleRelationInterAgg(raw_features, adjacency, hidden_dim)
    model = OneLayerCARE(2, inter, auxiliary_weight)
    return model
