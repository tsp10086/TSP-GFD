"""Prototype matching, edge gating and branch fusion."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

DIM = 384
VARIANTS = {
    "structure_only": (False, False, "fixed"),
    "class_semantic_only": (True, False, "fixed"),
    "relation_condition_only": (False, True, "fixed"),
    "lsp_full": (True, True, "fixed"),
    "schema_only": (True, True, "fixed"),
    "learnable_fusion": (True, True, "learnable"),
}
class InterfaceGNN(nn.Module):
    def __init__(self, hidden: int, variant: str, proto: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.use_class, self.use_relation, self.fusion = VARIANTS[variant]
        self.dropout = dropout
        self.register_buffer("proto", F.normalize(proto, p=2, dim=1))
        self.inp = nn.Linear(DIM, hidden)
        self.gcn = GCNConv(hidden, hidden)
        self.cls = nn.Linear(hidden, 2)
        self.class_adapter = nn.Sequential(nn.Linear(DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.rel_adapter = nn.Linear(DIM, hidden)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.fusion_logits = nn.Parameter(torch.zeros(3)) if self.fusion == "learnable" else None

    def forward(self, raw: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h0 = F.relu(self.inp(raw))
        hx = F.relu(self.gcn(F.dropout(h0, p=self.dropout, training=self.training), edge_index))
        terms = [hx]
        if self.use_class:
            scores = raw @ self.proto[:2].T
            hc = self.class_adapter(scores @ self.proto[:2])
            terms.append(hc)
        if self.use_relation:
            src, dst = edge_index
            relation = self.rel_adapter(self.proto[2]).unsqueeze(0)
            gate = torch.sigmoid(((self.q(h0[src]) + relation) * self.k(h0[dst])).sum(dim=1) / (h0.shape[1] ** 0.5))
            agg = torch.zeros_like(h0)
            agg.index_add_(0, dst, h0[src] * gate.unsqueeze(1))
            degree = torch.zeros((h0.shape[0],), dtype=h0.dtype, device=h0.device)
            degree.index_add_(0, dst, gate)
            hr = agg / degree.clamp_min(1.0).unsqueeze(1)
            terms.append(hr)
        if self.fusion_logits is not None:
            active = self.fusion_logits[:len(terms)].softmax(dim=0)
            fused = sum(w * t for w, t in zip(active, terms))
        else:
            fused = sum(terms) / len(terms)
        return self.cls(F.dropout(F.relu(fused), p=self.dropout, training=self.training))

