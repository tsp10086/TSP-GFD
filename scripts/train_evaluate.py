"""Train and evaluate a TSP-GFD prototype interface on prepared graph tensors.

The input file is a dictionary containing x, edge_index and y. Node ordering
must match the fixed partition indices. Run with --help for arguments.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
from bwgnn import BWGNN
from prototype_interface import InterfaceGNN

class BWGNNHidden(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.model = BWGNN(hidden, hidden, hidden, d=2)

    def forward(self, features, edges):
        _, output = self.model(features, edges)
        return output

def metrics(labels, scores):
    pred = (scores >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "f1_macro": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)),
        "aucprc": float(average_precision_score(labels, scores)),
    }

def evaluate(model, x, edge, labels, indices):
    model.eval()
    with torch.no_grad():
        scores = model(x, edge)[indices].softmax(dim=1)[:, 1].cpu().numpy()
    return metrics(labels[indices].cpu().numpy(), scores)

def read_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["reddit", "instagram", "amazonvideo"], required=True)
    p.add_argument("--graph", type=Path, required=True, help="Prepared dictionary with x, edge_index, y")
    p.add_argument("--prototypes", type=Path, required=True, help="Locally prepared prototype bundle")
    p.add_argument("--test", action="store_true", help="Evaluate test nodes after validation selection")
    p.add_argument("--epochs", type=int, help="Override epoch cap for a smoke test")
    args = p.parse_args()

    cfg = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    graph = read_tensor(args.graph)
    x = F.normalize(graph["x"].float(), p=2, dim=1)
    edge = graph["edge_index"].long()
    labels = graph["y"].long()
    if x.ndim != 2 or x.shape[1] != cfg["embedding_dimension"] or labels.numel() != x.shape[0]:
        raise ValueError("Invalid x/y shapes or node alignment")
    part = ROOT / "splits" / args.dataset
    train, val = (read_tensor(part / n).long() for n in ("train_idx.pt", "val_idx.pt"))
    prototype_data = read_tensor(args.prototypes)
    prototype = prototype_data["llm"].float().clone()
    prototype = F.normalize(prototype, p=2, dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InterfaceGNN(cfg["hidden_dimension"], "lsp_full", prototype, cfg["dropout"])
    model.gcn = BWGNNHidden(cfg["hidden_dimension"])
    model = model.to(device)
    x, edge, labels, train, val = [v.to(device) for v in (x, edge, labels, train, val)]
    counts = torch.bincount(labels[train], minlength=2).float().clamp_min(1)
    weights = counts.sum() / (2 * counts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_auc, best, stale, best_epoch = float("-inf"), None, 0, 0
    for epoch in range(1, (args.epochs or cfg["epochs"][args.dataset]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, edge)[train], labels[train], weight=weights)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        result = evaluate(model, x, edge, labels, val)
        if result["auc"] > best_auc + 1e-8:
            best_auc, best_epoch, stale = result["auc"], epoch, 0
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg["early_stopping_patience"]:
            break
    if best is None:
        raise RuntimeError("No valid validation checkpoint")
    model.load_state_dict(best)
    output = {"dataset": args.dataset, "variant": "lsp_full", "best_epoch": best_epoch,
              "validation": evaluate(model, x, edge, labels, val)}
    if args.test:
        test = read_tensor(part / "test_idx.pt").long().to(device)
        output["test"] = evaluate(model, x, edge, labels, test)
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()