"""Same-protocol validation checkpoints for six Table 3 adapted baselines.

This suite never reads test indices, test labels, or prior test predictions.
All runs use the v2 graph/raw features/splits/seeds/optimizer/checkpoint rule.
CARE-GNN-style and FLAG-style retain their explicitly declared auxiliary losses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
import run_lsp_backbone_validation as training  # noqa: E402

METHODS = ("text_mlp", "gcn", "gat", "bwgnn", "care_gnn_style", "flag_style")
RUNS = HERE / "table3_baseline_validation"
CACHE = Path(r"LOCAL_PATH_TO_CONFIGURE")
FLAG_AUX_WEIGHT = 0.10
DIM = 384


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TextMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(training.DROPOUT), nn.Linear(DIM, training.HIDDEN), nn.ReLU(),
                                 nn.Dropout(training.DROPOUT), nn.Linear(training.HIDDEN, 2))

    def forward(self, raw: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        return self.net(raw)


class GCNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skip = nn.Linear(DIM, 2)
        self.conv1 = GCNConv(DIM, training.HIDDEN)
        self.conv2 = GCNConv(training.HIDDEN, 2)

    def forward(self, raw: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.conv1(F.dropout(raw, p=training.DROPOUT, training=self.training), edge))
        return self.conv2(F.dropout(hidden, p=training.DROPOUT, training=self.training), edge) + self.skip(raw)


class GATModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skip = nn.Linear(DIM, 2)
        self.gat = GATConv(DIM, training.HIDDEN, heads=2, concat=False, dropout=training.DROPOUT)
        self.out = nn.Linear(training.HIDDEN, 2)

    def forward(self, raw: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.gat(F.dropout(raw, p=training.DROPOUT, training=self.training), edge))
        return self.out(F.dropout(hidden, p=training.DROPOUT, training=self.training)) + self.skip(raw)


class BWGNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = training.BWGNN(DIM, training.HIDDEN, 2, d=training.FILTER_ORDER)

    def forward(self, raw: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        return self.model(raw, edge)[1]


class CAREGNNStyle(nn.Module):
    """Single-relation adaptation: neighbor gate and label-aware auxiliary head."""

    def __init__(self) -> None:
        super().__init__()
        h = training.HIDDEN
        self.input = nn.Linear(DIM, h)
        self.gate = nn.Sequential(nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, 1))
        self.update = nn.Sequential(nn.Linear(h * 2, h), nn.ReLU(), nn.Dropout(training.DROPOUT))
        self.classifier = nn.Linear(h, 2)
        self.label_head = nn.Linear(h, 2)
        self.skip = nn.Linear(DIM, 2)

    def forward(self, raw: torch.Tensor, edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.dropout(F.relu(self.input(raw)), p=training.DROPOUT, training=self.training)
        src, dst = edge
        gate = torch.sigmoid(self.gate(torch.cat((h[src], h[dst]), dim=1)))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, gate * h[src])
        degree = torch.zeros((h.size(0), 1), dtype=h.dtype, device=h.device)
        degree.index_add_(0, dst, gate)
        final = self.update(torch.cat((h, agg / degree.clamp_min(1.0)), dim=1))
        return self.classifier(final) + self.skip(raw), self.label_head(final)


class FLAGStyle(nn.Module):
    """FLAG-inspired dual-view GCN adaptation with fixed auxiliary weights."""

    def __init__(self) -> None:
        super().__init__()
        h = training.HIDDEN
        self.causal = GCNConv(DIM, h)
        self.background = GCNConv(DIM, h)
        self.causal_out = nn.Linear(h, 2)
        self.background_out = nn.Linear(h, 2)
        self.skip = nn.Linear(DIM, 2)

    def forward(self, matched: torch.Tensor, neutral: torch.Tensor, edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        causal = F.relu(self.causal(F.dropout(matched, p=training.DROPOUT, training=self.training), edge))
        background = F.relu(self.background(F.dropout(neutral, p=training.DROPOUT, training=self.training), edge))
        logits = self.causal_out(F.dropout(causal, p=training.DROPOUT, training=self.training)) + self.skip(matched)
        return logits, self.background_out(background), causal, background


def build_model(method: str) -> nn.Module:
    return {"text_mlp": TextMLP, "gcn": GCNModel, "gat": GATModel, "bwgnn": BWGNNModel,
            "care_gnn_style": CAREGNNStyle, "flag_style": FLAGStyle}[method]()


def load_flag_views(dataset: str, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("CACHE_FROZEN") is not True:
        raise RuntimeError("FLAG cache is not frozen")
    cache = CACHE / dataset
    cached_raw = F.normalize(torch.load(cache / "raw_minilm.pt", map_location="cpu", weights_only=True).float(), p=2, dim=1)
    if not torch.equal(cached_raw, raw):
        raise RuntimeError(f"FLAG raw cache differs from v2 raw input: {dataset}")
    matched = F.normalize(torch.load(cache / "matched_minilm.pt", map_location="cpu", weights_only=True).float(), p=2, dim=1)
    neutral = F.normalize(torch.load(cache / "neutral_minilm.pt", map_location="cpu", weights_only=True).float(), p=2, dim=1)
    if matched.shape != raw.shape or neutral.shape != raw.shape:
        raise RuntimeError(f"FLAG semantic view shape mismatch: {dataset}")
    hashes = {"flag_cache_manifest_sha256": sha_file(CACHE / "manifest.json"),
              "flag_matched_sha256": sha_file(cache / "matched_minilm.pt"),
              "flag_neutral_sha256": sha_file(cache / "neutral_minilm.pt")}
    return matched, neutral, hashes


def forward(model: nn.Module, method: str, raw: torch.Tensor, edge: torch.Tensor,
            views: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if method == "care_gnn_style":
        logits, aux = model(raw, edge)
        return logits, aux, None, None
    if method == "flag_style":
        assert views is not None
        return model(views[0], views[1], edge)
    return model(raw, edge), None, None, None


def run_one(dataset: str, method: str, seed: int) -> dict:
    if dataset not in training.DATASETS or method not in METHODS or seed not in training.SEEDS:
        raise ValueError("Outside prespecified Table 3 validation grid")
    target = RUNS / dataset / method / f"seed_{seed}"
    record_file, checkpoint = target / "result.json", target / "best_validation_checkpoint.pt"
    protocol, raw, _, hashes = training.load_inputs(dataset)
    views, view_hashes = None, {}
    if method == "flag_style":
        matched, neutral, view_hashes = load_flag_views(dataset, raw)
        views = (matched, neutral)
    settings = {**training.common_configuration(dataset, seed, hashes), "suite": "Table3_same_protocol_v2_extension",
                "method": method, "baseline_runner_sha256": sha_file(Path(__file__)), "view_hashes": view_hashes,
                "flag_aux_weight": FLAG_AUX_WEIGHT if method == "flag_style" else None,
                "care_aux_weight": FLAG_AUX_WEIGHT if method == "care_gnn_style" else None}
    config_hash = training.json_hash(settings)
    if record_file.exists():
        old = json.loads(record_file.read_text(encoding="utf-8"))
        if old.get("config_hash") != config_hash or old.get("TEST_READ") != "NO" or not checkpoint.is_file():
            raise RuntimeError(f"Refusing incompatible completed run: {target}")
        print(f"SKIP_VALID_EXISTING {dataset}/{method}/seed_{seed}", flush=True)
        return old
    if target.exists():
        raise RuntimeError(f"Refusing incomplete run: {target}")
    training.original.seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(method).to(device)
    raw, edge = raw.to(device), protocol["graph"]["edge_index"].long().to(device)
    train, val = protocol["train"].to(device), protocol["val"].to(device)
    train_y, val_y = protocol["train_y"].to(device), protocol["val_y"].to(device)
    if views is not None:
        views = (views[0].to(device), views[1].to(device))
    weights = training.original.class_weights(protocol["train_y"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.LR, weight_decay=training.WEIGHT_DECAY)
    best_auc, best_epoch, best_metrics, best_state, stale = -math.inf, 0, None, None, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, training.MAX_EPOCHS[dataset] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, aux, causal, background = forward(model, method, raw, edge, views)
        loss = F.cross_entropy(logits[train], train_y, weight=weights)
        if method == "care_gnn_style":
            assert aux is not None
            loss = loss + FLAG_AUX_WEIGHT * F.cross_entropy(aux[train], train_y, weight=weights)
        if method == "flag_style":
            assert aux is not None and causal is not None and background is not None
            uniform = torch.full_like(aux[train], 0.5)
            loss = loss + FLAG_AUX_WEIGHT * F.kl_div(F.log_softmax(aux[train], dim=1), uniform, reduction="batchmean")
            loss = loss + FLAG_AUX_WEIGHT * F.cosine_similarity(causal[train], background[train], dim=1).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss: {dataset}/{method}/seed_{seed}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            score = forward(model, method, raw, edge, views)[0][val].softmax(dim=1)[:, 1].cpu().numpy()
        metrics = training.original.metrics(val_y.cpu().numpy(), score)
        if metrics["auc"] > best_auc + 1e-8:
            best_auc, best_epoch, best_metrics, stale = metrics["auc"], epoch, metrics, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= training.PATIENCE:
            break
    assert best_state is not None and best_metrics is not None
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    target.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": best_state, "dataset": dataset, "method": method, "seed": seed,
                "best_epoch": best_epoch, "best_val_auc": best_auc, "config_hash": config_hash,
                "TEST_READ": "NO"}, checkpoint)
    record = {"dataset": dataset, "method": method, "seed": seed, **best_metrics,
              "best_epoch": best_epoch, "best_val_auc": best_auc, "epochs_executed": epoch,
              "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "runtime_seconds": elapsed, "peak_memory_bytes": peak, "device": str(device),
              "checkpoint_path": str(checkpoint), "config_hash": config_hash, "TEST_READ": "NO",
              "adapted_implementation": method in ("care_gnn_style", "flag_style"),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), **hashes, **view_hashes}
    record_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE {dataset}/{method}/seed_{seed} AUC={best_auc:.4f} AUCPRC={best_metrics['aucprc']:.4f} TEST_READ=NO", flush=True)
    del model, raw, edge, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def verify() -> None:
    rows = []
    for dataset in training.DATASETS:
        reference = json.loads((V2 / "results" / "lsp_backbone_validation" / dataset / "bwgnn" / "lsp_full" / "seed_REDACTED" / "result.json").read_text(encoding="utf-8"))
        for method in METHODS:
            items = []
            for seed in training.SEEDS:
                file = RUNS / dataset / method / f"seed_{seed}" / "result.json"
                if not file.is_file():
                    raise RuntimeError(f"Missing baseline: {file}")
                row = json.loads(file.read_text(encoding="utf-8"))
                if (row["dataset"], row["method"], row["seed"], row["TEST_READ"]) != (dataset, method, seed, "NO"):
                    raise RuntimeError(f"Baseline identity mismatch: {file}")
                for field in ("graph_hash", "raw_feature_hash", "train_split_hash", "validation_split_hash", "train_label_hash", "validation_label_hash"):
                    if row[field] != reference[field]:
                        raise RuntimeError(f"Input mismatch {field}: {file}")
                if row["best_val_auc"] != row["auc"] or not Path(row["checkpoint_path"]).is_file():
                    raise RuntimeError(f"Bad checkpoint: {file}")
                items.append(row)
            summary = {"dataset": dataset, "method": method, "n": 3, "TEST_READ": "NO"}
            for metric in ("accuracy", "f1_macro", "auc", "aucprc"):
                values = [float(item[metric]) for item in items]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_sample_std"] = statistics.stdev(values)
            rows.append(summary)
    path = HERE / "table3_baseline_validation_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"TABLE3_BASELINE_VALIDATION_PASS jobs=54 TEST_READ=NO summary={path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "train", "queue", "verify"), required=True)
    parser.add_argument("--dataset", choices=training.DATASETS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=training.SEEDS)
    args = parser.parse_args()
    if args.mode == "audit":
        for dataset in training.DATASETS:
            _, raw, _, hashes = training.load_inputs(dataset)
            matched, neutral, cache_hashes = load_flag_views(dataset, raw)
            print(json.dumps({"dataset": dataset, "nodes": len(raw), "matched_shape": list(matched.shape),
                              "neutral_shape": list(neutral.shape), "hashes": hashes,
                              "view_hashes": cache_hashes, "TEST_READ": "NO"}), flush=True)
        return
    if args.mode == "verify":
        verify()
        return
    if args.mode == "train":
        if args.dataset is None or args.method is None or args.seed is None:
            parser.error("train requires dataset, method, seed")
        run_one(args.dataset, args.method, args.seed)
        return
    status_file = HERE / "table3_baseline_queue_status.json"
    completed = 0
    try:
        for dataset in training.DATASETS:
            for seed in training.SEEDS:
                for method in METHODS:
                    status_file.write_text(json.dumps({"state": "RUNNING", "completed": completed, "total": 54,
                                                       "TEST_READ": "NO", "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
                    run_one(dataset, method, seed)
                    completed += 1
        verify()
        status_file.write_text(json.dumps({"state": "PASS", "completed": 54, "total": 54,
                                           "TEST_READ": "NO", "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    except BaseException as exc:
        status_file.write_text(json.dumps({"state": "FAIL", "completed": completed, "total": 54,
                                           "error": repr(exc), "TEST_READ": "NO", "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
