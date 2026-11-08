"""Validation-only author-code BWGNN and CARE-GNN baselines on frozen v2 data.

Official repositories are imported from pinned local clones. Their original
train.py/main.py scripts are deliberately not run: those scripts make their
own splits and evaluate test data during training. This runner reads only the
frozen train/validation protocol and selects checkpoints by validation AUC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BWGNN_REPO = HERE / "Rethinking-Anomaly-Detection"
CARE_REPO = HERE / "CARE-GNN"
PYTHON_DEPS = HERE / "python_deps"
LSP_SOURCE = Path(r"LOCAL_PATH_TO_CONFIGURE")
OUTPUT = HERE / "validation_runs"
DATASETS = ("reddit", "instagram", "amazonvideo")
METHODS = ("bwgnn_author", "care_gnn_author_r1")
SEEDS = (None, None, None)
HIDDEN = 64
LR = 0.005
WEIGHT_DECAY = 5e-4
PATIENCE = 60
MAX_EPOCHS = {"reddit": 300, "instagram": 300, "amazonvideo": 500}
CARE_BATCH_SIZE = {"reddit": 1024, "instagram": 1024, "amazonvideo": 256}
EXPECTED_COMMITS = {
    "bwgnn_author": "de0631f039bbd19c1890b483cc01f1007f596af7",
    "care_gnn_author_r1": "a64ff7523e187a24251f7ca88435d2c9d8f7dcd9",
}

sys.path.insert(0, str(LSP_SOURCE))
sys.path.insert(0, str(PYTHON_DEPS))
sys.path.insert(0, str(BWGNN_REPO))
import run_lsp_interface_ablation as protocol_source  # noqa: E402
import dgl  # noqa: E402
from BWGNN import BWGNN  # noqa: E402
from care_single_relation_adapter import build_care_model  # noqa: E402


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_tensor(value: torch.Tensor) -> str:
    return protocol_source.sha_tensor(value)


def repo_commit(path: Path) -> str:
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"Official source tree has uncommitted changes: {path}")
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def load_data(dataset: str) -> tuple[dict, torch.Tensor, dict[str, str]]:
    bundle = protocol_source.load_protocol(dataset)
    raw = protocol_source.raw_embeddings(dataset)
    graph_file = bundle["root"] / "graph_and_text_no_labels.pt"
    if raw.size(0) != int(bundle["graph"]["num_nodes"]):
        raise RuntimeError("Raw feature/node count mismatch")
    hashes = {
        "graph_hash": sha_file(graph_file),
        "raw_feature_hash": sha_tensor(raw),
        "train_split_hash": sha_tensor(bundle["train"]),
        "validation_split_hash": sha_tensor(bundle["val"]),
        "train_label_hash": sha_tensor(bundle["train_y"]),
        "validation_label_hash": sha_tensor(bundle["val_y"]),
    }
    return bundle, raw, hashes


def batches_at_least_two(nodes: list[int], size: int) -> list[list[int]]:
    chunks = [nodes[index:index + size] for index in range(0, len(nodes), size)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2].extend(chunks.pop())
    if not chunks or len(chunks[-1]) < 2:
        raise RuntimeError("Author CARE-GNN needs at least two nodes per batch")
    return chunks


def balanced_care_batches(
    train_nodes: list[int], train_labels: list[int], batch_size: int
) -> list[list[int]]:
    positives = [node for node, label in zip(train_nodes, train_labels) if label == 1]
    negatives = [node for node, label in zip(train_nodes, train_labels) if label == 0]
    if len(positives) < 2 or len(negatives) < len(positives):
        raise RuntimeError("CARE-GNN requires positive nodes and enough negatives")
    random.shuffle(positives)
    sampled_negatives = random.sample(negatives, len(positives))
    paired = [node for pair in zip(positives, sampled_negatives) for node in pair]
    chunks = batches_at_least_two(paired, batch_size)
    positive_set = set(positives)
    if any(len(chunk) < 2 or not any(node in positive_set for node in chunk) for chunk in chunks):
        raise RuntimeError("CARE-GNN RL batch lacks positive examples")
    return chunks


def validation_metrics(
    method: str, model: torch.nn.Module, raw: torch.Tensor,
    bundle: dict, device: torch.device, dataset: str
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        if method == "bwgnn_author":
            probabilities = model(raw).softmax(dim=1)
            scores = probabilities[bundle["val"], 1].cpu().numpy()
        else:
            nodes = bundle["val"].tolist()
            pieces = []
            for chunk in batches_at_least_two(nodes, CARE_BATCH_SIZE[dataset]):
                dummy = torch.zeros(len(chunk), dtype=torch.long, device=device)
                logits, _ = model.forward(chunk, dummy, train_flag=False)
                pieces.append(logits.softmax(dim=1)[:, 1].cpu())
            scores = torch.cat(pieces).numpy()
    labels = bundle["val_y"].cpu().numpy()
    if len(scores) != len(labels):
        raise RuntimeError("Validation score/label length mismatch")
    return protocol_source.metrics(labels, scores)


def run_one(
    dataset: str, method: str, seed: int, *, max_epochs_override: int | None = None,
    care_device: str = "cpu",
) -> dict:
    if dataset not in DATASETS or method not in METHODS or seed not in SEEDS:
        raise ValueError("Outside prespecified baseline grid")
    if max_epochs_override is not None and max_epochs_override < 1:
        raise ValueError("max_epochs_override must be positive")
    if care_device not in ("cpu", "cuda"):
        raise ValueError("care_device must be cpu or cuda")
    if care_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    bundle, raw_cpu, hashes = load_data(dataset)
    edge_cpu = bundle["graph"]["edge_index"].long().cpu()
    source = BWGNN_REPO if method == "bwgnn_author" else CARE_REPO
    source_commit = repo_commit(source)
    if source_commit != EXPECTED_COMMITS[method]:
        raise RuntimeError(f"Official source commit changed: {source_commit}")
    effective_max_epochs = max_epochs_override or MAX_EPOCHS[dataset]
    settings = {
        "family": "LSP_GFD_author_code_baselines_validation",
        "dataset": dataset, "method": method, "seed": seed,
        "official_repository": str(source), "official_commit": source_commit,
        "runner_hash": sha_file(Path(__file__)),
        "care_adapter_hash": sha_file(HERE / "care_single_relation_adapter.py"),
        "hidden_dim": HIDDEN, "optimizer": "AdamW", "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
        "max_epochs": effective_max_epochs,
        "checkpoint_selection": "validation_auc", "TEST_READ": "NO",
        "care_batch_size": CARE_BATCH_SIZE[dataset] if method == "care_gnn_author_r1" else None,
        "care_device": care_device if method == "care_gnn_author_r1" else None,
        "care_auxiliary_weight": 2.0 if method == "care_gnn_author_r1" else None,
        "care_rl_step_size": 0.02 if method == "care_gnn_author_r1" else None,
        "bwgnn_filter_order": 2 if method == "bwgnn_author" else None,
        "dgl_version": dgl.__version__ if method == "bwgnn_author" else None,
        **hashes,
    }
    config_hash = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_root = OUTPUT if max_epochs_override is None else HERE / "smoke_runs"
    target = run_root / dataset / method / f"seed_{seed}"
    record_file = target / "result.json"
    checkpoint = target / "best_validation_checkpoint.pt"
    if record_file.exists():
        previous = json.loads(record_file.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash or not checkpoint.is_file():
            raise RuntimeError(f"Refusing to reuse incompatible run: {target}")
        print(f"SKIP {dataset}/{method}/seed_{seed}", flush=True)
        return previous
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete run: {target}")

    protocol_source.seed_everything(seed)
    device = torch.device("cpu" if method == "bwgnn_author" else care_device)
    raw = raw_cpu.to(device)
    if method == "bwgnn_author":
        # The authors' homogeneous Dataset adapter adds one self-loop per node.
        source_graph = dgl.graph(
            (edge_cpu[0], edge_cpu[1]), num_nodes=raw_cpu.size(0)
        )
        source_graph = dgl.add_self_loop(dgl.remove_self_loop(source_graph))
        model = BWGNN(raw_cpu.size(1), HIDDEN, 2, source_graph, d=2)
        labels_by_node = None
        train_batches = None
    else:
        model = build_care_model(raw_cpu, edge_cpu, hidden_dim=HIDDEN)
        model = model.to(device)
        labels_by_node = torch.full((raw_cpu.size(0),), -1, dtype=torch.long)
        labels_by_node[bundle["train"]] = bundle["train_y"]
        train_batches = None
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    weights = protocol_source.class_weights(bundle["train_y"]).to(device)
    best_auc, best_epoch, best_metrics, best_state, best_thresholds = (
        -math.inf, 0, None, None, None
    )
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, effective_max_epochs + 1):
        model.train()
        if method == "bwgnn_author":
            optimizer.zero_grad(set_to_none=True)
            logits = model(raw)
            loss = F.cross_entropy(
                logits[bundle["train"]], bundle["train_y"], weight=weights
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite BWGNN loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        else:
            train_batches = balanced_care_batches(
                bundle["train"].tolist(), bundle["train_y"].tolist(),
                CARE_BATCH_SIZE[dataset],
            )
            model.inter1.batch_num = len(train_batches)
            for nodes in train_batches:
                labels = labels_by_node[nodes].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(nodes, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite CARE-GNN loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        metrics = validation_metrics(method, model, raw, bundle, device, dataset)
        if metrics["auc"] > best_auc + 1e-8:
            best_auc, best_epoch, best_metrics, stale = metrics["auc"], epoch, metrics, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            if method == "care_gnn_author_r1":
                best_thresholds = model.inter1.thresholds.copy()
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= PATIENCE:
            print(
                f"EPOCH {dataset}/{method}/seed_{seed} {epoch} "
                f"val_auc={metrics['auc']:.4f} best={best_auc:.4f}",
                flush=True,
            )
        if stale >= PATIENCE:
            break

    if best_metrics is None or best_state is None:
        raise RuntimeError("No valid checkpoint")
    target.mkdir(parents=True, exist_ok=False)
    torch.save({
        "state_dict": best_state,
        "care_thresholds": best_thresholds,
        "dataset": dataset, "method": method, "seed": seed,
        "best_epoch": best_epoch, "config_hash": config_hash, "TEST_READ": "NO",
    }, checkpoint)
    record = {
        **settings, **best_metrics,
        "config_hash": config_hash, "best_epoch": best_epoch,
        "epochs_executed": epoch, "runtime_seconds": time.perf_counter() - started,
        "device": str(device),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "checkpoint_path": str(checkpoint),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    record_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"DONE {dataset}/{method}/seed_{seed} "
        f"AUC={best_auc:.4f} AUCPRC={best_metrics['aucprc']:.4f} TEST_READ=NO",
        flush=True,
    )
    return record


def verify() -> None:
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                target = OUTPUT / dataset / method / f"seed_{seed}"
                record = json.loads((target / "result.json").read_text(encoding="utf-8"))
                checkpoint = torch.load(
                    target / "best_validation_checkpoint.pt",
                    map_location="cpu", weights_only=False,
                )
                if record["TEST_READ"] != "NO" or checkpoint["TEST_READ"] != "NO":
                    raise RuntimeError(f"Test access flag failed: {target}")
                if record["config_hash"] != checkpoint["config_hash"]:
                    raise RuntimeError(f"Checkpoint configuration mismatch: {target}")
                if record["official_commit"] != repo_commit(
                    BWGNN_REPO if method == "bwgnn_author" else CARE_REPO
                ):
                    raise RuntimeError(f"Official source commit changed: {target}")
    print("VERIFY PASS 18/18 validation cells; TEST_READ=NO", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-epochs-override", type=int)
    parser.add_argument("--care-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.verify:
        verify()
    elif args.dataset and args.method and args.seed:
        run_one(
            args.dataset, args.method, args.seed,
            max_epochs_override=args.max_epochs_override,
            care_device=args.care_device,
        )
    else:
        parser.error("Supply --verify or dataset, method, and seed")


if __name__ == "__main__":
    main()
