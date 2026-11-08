"""Validation-only, same-split FLAG author-code adaptation and MLED-style baselines.

This intentionally does not claim a faithful end-to-end FLAG reproduction:
the author training script uses Gemma-2-9B, dataset-specific samplers, and
alternating LoRA/GNN training. Here the *unaltered author model module* supplies
the downstream DualGNN/GCN stage, fed by the frozen local semantic cache.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
V2 = Path(r"LOCAL_PATH_TO_CONFIGURE")
EXTENSION = V2 / "extension_20260928"
FLAG_SOURCE = Path(r"LOCAL_PATH_TO_CONFIGURE")
METHODS = ("flag_author_code_adapted", "mled_style")
sys.path.insert(0, str(V2))
sys.path.insert(0, str(EXTENSION))
import run_lsp_backbone_validation as protocol_v2  # noqa: E402
import run_table3_baselines_validation as table3_v2  # noqa: E402


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_flag_class():
    if not FLAG_SOURCE.is_file():
        raise FileNotFoundError(FLAG_SOURCE)
    spec = importlib.util.spec_from_file_location("flag_author_models", FLAG_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FLAGAuthorCodeAdapted(nn.Module):
    """Author's unmodified DualGNN(GCN) downstream model, not its LLM stage."""

    def __init__(self) -> None:
        super().__init__()
        source = load_flag_class()
        self.model = source.DualGNN(2, source.GCN(384, protocol_v2.HIDDEN, 2, protocol_v2.DROPOUT))

    def forward(self, raw: torch.Tensor, matched: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        return self.model(raw, matched, edge)[1]


class MLEDStyle(nn.Module):
    """Paper Eqs. (1)-(8) with BWGNN as detector and one observed relation.

    The frozen v2 Qwen/MiniLM descriptions substitute for the paper's
    GPT-4/ada-002 source. This is an architectural adaptation, not MLED proper.
    """

    def __init__(self, prototypes: torch.Tensor) -> None:
        super().__init__()
        if prototypes.shape != (3, 384):
            raise ValueError("Expected two class descriptions and one relation description")
        hidden = protocol_v2.HIDDEN
        self.register_buffer("descriptions", prototypes.detach().clone())
        self.feature_map = nn.Linear(384, hidden)
        self.type_map = nn.Linear(384, hidden)
        self.relation_map = nn.Linear(384, hidden)
        self.type_gate = nn.Linear(hidden, 1)
        self.relation_gate = nn.Linear(hidden, 1, bias=False)
        self.type_weight = nn.Parameter(torch.tensor(1.0))
        self.relation_weight = nn.Parameter(torch.tensor(1.0))
        self.detector = protocol_v2.BWGNN(hidden, hidden, 2, d=protocol_v2.FILTER_ORDER)

    def forward(self, raw: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        node = F.leaky_relu(self.feature_map(raw))
        types = self.type_map(self.descriptions[:2])
        beta = torch.sigmoid(self.type_gate(node[:, None, :] + types[None, :, :]))
        type_context = (beta * types[None, :, :]).sum(dim=1) / types.size(0)
        relations = self.relation_map(self.descriptions[2:])
        alpha = F.leaky_relu(self.relation_gate(node[:, None, :] + relations[None, :, :]))
        delta = torch.softmax(alpha, dim=1)
        relation_context = (delta * relations[None, :, :]).sum(dim=1)
        fused = node + self.type_weight * type_context + self.relation_weight * relation_context
        return self.detector(fused, edge)[1]


def build_model(method: str, prototypes: torch.Tensor) -> nn.Module:
    if method == "flag_author_code_adapted":
        return FLAGAuthorCodeAdapted()
    if method == "mled_style":
        return MLEDStyle(prototypes)
    raise ValueError(method)


def run_one(dataset: str, method: str, seed: int, *, smoke: bool = False) -> dict:
    if dataset not in protocol_v2.DATASETS or method not in METHODS or seed not in protocol_v2.SEEDS:
        raise ValueError("Outside prespecified validation grid")
    target = HERE / ("smoke" if smoke else "validation_runs") / dataset / method / f"seed_{seed}"
    result_file = target / "result.json"
    checkpoint = target / "best_validation_checkpoint.pt"
    graph, raw, prototypes, input_hashes = protocol_v2.load_inputs(dataset)
    matched = None
    view_hashes = {}
    if method == "flag_author_code_adapted":
        matched, _, view_hashes = table3_v2.load_flag_views(dataset, raw)
    config = {
        **protocol_v2.common_configuration(dataset, seed, input_hashes),
        "method": method,
        "method_source_sha256": sha_file(Path(__file__)),
        "flag_author_model_sha256": sha_file(FLAG_SOURCE) if method == "flag_author_code_adapted" else None,
        "semantic_cache_hashes": view_hashes,
        "mled_description_source": "frozen_v2_Qwen2.5-1.5B_plus_MiniLM" if method == "mled_style" else None,
        "mled_relation_count": 1 if method == "mled_style" else None,
        "smoke": smoke,
        "effective_max_epochs": 2 if smoke else protocol_v2.MAX_EPOCHS[dataset],
        "TEST_READ": "NO",
    }
    config_hash = protocol_v2.json_hash(config)
    if result_file.exists():
        prior = json.loads(result_file.read_text(encoding="utf-8"))
        if prior.get("config_hash") != config_hash or prior.get("TEST_READ") != "NO" or not checkpoint.is_file():
            raise RuntimeError(f"Refusing incompatible completed run: {target}")
        print(f"SKIP_VALID_EXISTING {dataset}/{method}/seed_{seed}", flush=True)
        return prior
    if target.exists():
        raise RuntimeError(f"Refusing incomplete run: {target}")
    protocol_v2.original.seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(method, prototypes).to(device)
    raw = raw.to(device)
    edge = graph["graph"]["edge_index"].long().to(device)
    train, val = graph["train"].to(device), graph["val"].to(device)
    train_y, val_y = graph["train_y"].to(device), graph["val_y"].to(device)
    if matched is not None:
        matched = matched.to(device)
    weights = protocol_v2.original.class_weights(graph["train_y"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol_v2.LR, weight_decay=protocol_v2.WEIGHT_DECAY)

    def forward() -> torch.Tensor:
        if method == "flag_author_code_adapted":
            assert matched is not None
            return model(raw, matched, edge)
        return model(raw, edge)

    best_auc, best_epoch, best_metrics, best_state, stale = -math.inf, 0, None, None, 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cap = 2 if smoke else protocol_v2.MAX_EPOCHS[dataset]
    for epoch in range(1, cap + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = forward()
        loss = F.cross_entropy(logits[train], train_y, weight=weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss: {dataset}/{method}/seed_{seed}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            score = forward()[val].softmax(dim=1)[:, 1].cpu().numpy()
        metrics = protocol_v2.original.metrics(val_y.cpu().numpy(), score)
        if metrics["auc"] > best_auc + 1e-8:
            best_auc, best_epoch, best_metrics, stale = metrics["auc"], epoch, metrics, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= protocol_v2.PATIENCE:
            break
    assert best_state is not None and best_metrics is not None
    record = {
        "dataset": dataset, "method": method, "seed": seed, **best_metrics,
        "best_epoch": best_epoch, "best_val_auc": best_auc, "epochs_executed": epoch,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "checkpoint_path": str(checkpoint), "config_hash": config_hash,
        "implementation_status": "official_GNN_module_adapted" if method == "flag_author_code_adapted" else "paper_based_style",
        "TEST_READ": "NO", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **input_hashes, **view_hashes,
    }
    target.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": best_state, "config_hash": config_hash,
                "best_val_auc": best_auc, "best_epoch": best_epoch, "TEST_READ": "NO"}, checkpoint)
    result_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE {dataset}/{method}/seed_{seed} AUC={best_auc:.4f} AUCPRC={best_metrics['aucprc']:.4f} TEST_READ=NO", flush=True)
    del model, optimizer, raw, edge
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def verify() -> None:
    rows = []
    for dataset in protocol_v2.DATASETS:
        reference = json.loads((V2 / "results" / "lsp_backbone_validation" / dataset / "bwgnn" / "lsp_full" / "seed_REDACTED" / "result.json").read_text(encoding="utf-8"))
        for method in METHODS:
            run_rows = []
            for seed in protocol_v2.SEEDS:
                path = HERE / "validation_runs" / dataset / method / f"seed_{seed}" / "result.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                if (row["dataset"], row["method"], row["seed"], row["TEST_READ"]) != (dataset, method, seed, "NO"):
                    raise RuntimeError(f"Identity mismatch: {path}")
                for field in ("graph_hash", "raw_feature_hash", "train_split_hash", "validation_split_hash", "train_label_hash", "validation_label_hash"):
                    if row[field] != reference[field]:
                        raise RuntimeError(f"Protocol mismatch {field}: {path}")
                if row["auc"] != row["best_val_auc"] or not Path(row["checkpoint_path"]).is_file():
                    raise RuntimeError(f"Checkpoint mismatch: {path}")
                run_rows.append(row)
            item = {"dataset": dataset, "method": method, "n": 3, "TEST_READ": "NO"}
            for metric in ("accuracy", "f1_macro", "auc", "aucprc"):
                values = [float(row[metric]) for row in run_rows]
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_sample_std"] = statistics.stdev(values)
            rows.append(item)
    import csv
    output = HERE / "closest_baselines_validation_summary.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CLOSEST_BASELINES_VALIDATION_PASS jobs=18 TEST_READ=NO summary={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "smoke", "train", "queue", "verify"), required=True)
    parser.add_argument("--dataset", choices=protocol_v2.DATASETS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=protocol_v2.SEEDS)
    args = parser.parse_args()
    if args.mode == "audit":
        source = load_flag_class()
        for dataset in protocol_v2.DATASETS:
            graph, raw, prototypes, hashes = protocol_v2.load_inputs(dataset)
            matched, _, view_hashes = table3_v2.load_flag_views(dataset, raw)
            print(json.dumps({"dataset": dataset, "nodes": len(raw), "relations": 1,
                              "prototype_shape": list(prototypes.shape), "flag_dual_class": source.DualGNN.__name__,
                              "matched_shape": list(matched.shape), "input_hashes": hashes,
                              "view_hashes": view_hashes, "TEST_READ": "NO"}), flush=True)
        return
    if args.mode == "verify":
        verify()
        return
    if args.mode in ("smoke", "train"):
        if args.dataset is None or args.method is None or args.seed is None:
            parser.error("dataset, method, and seed are required")
        run_one(args.dataset, args.method, args.seed, smoke=args.mode == "smoke")
        return
    status = HERE / "queue_status.json"
    completed = 0
    try:
        for dataset in protocol_v2.DATASETS:
            for seed in protocol_v2.SEEDS:
                for method in METHODS:
                    status.write_text(json.dumps({"state": "RUNNING", "completed": completed, "total": 18,
                                                  "TEST_READ": "NO", "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
                    run_one(dataset, method, seed)
                    completed += 1
        verify()
        status.write_text(json.dumps({"state": "PASS", "completed": 18, "total": 18,
                                      "TEST_READ": "NO", "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    except BaseException as exc:
        status.write_text(json.dumps({"state": "FAIL", "completed": completed, "total": 18,
                                      "error": repr(exc), "TEST_READ": "NO", "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
