"""Validation-only, paired structure-backbone reproduction of the local LSP model.

This is a new experiment family. It is not a reconstruction of the historical
Table 3 checkpoint. Only train/validation protocol artifacts are opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = Path(r"LOCAL_PATH_TO_CONFIGURE")
LOCAL_LSP = Path(r"LOCAL_PATH_TO_CONFIGURE")
PROTOTYPE_FILE = LOCAL_LSP / "task_prototypes.pt"
SOURCE_FILE = LOCAL_LSP / "run_lsp_interface_ablation.py"
BWGNN_FILE = PROJECT / "bwgnn.py"
OUTPUT = HERE / "results" / "lsp_backbone_validation"
DATASETS = ("reddit", "instagram", "amazonvideo")
BACKBONES = ("gcn", "bwgnn")
VARIANTS = ("structure_only", "lsp_full")
SEEDS = (None, None, None)
HIDDEN = 64
DROPOUT = 0.30
LR = 0.005
WEIGHT_DECAY = 5e-4
PATIENCE = 60
MAX_EPOCHS = {"reddit": 300, "instagram": 300, "amazonvideo": 500}
FILTER_ORDER = 2

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(LOCAL_LSP))
from bwgnn import BWGNN  # noqa: E402
import run_lsp_interface_ablation as original  # noqa: E402


def file_hash(path: Path) -> str:
    return original.sha_file(path)


def tensor_hash(value: torch.Tensor) -> str:
    return original.sha_tensor(value)


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def node_order_hash(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        data = str(text).encode("utf-8")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


class BWGNNHidden(nn.Module):
    """Adapt the frozen repository BWGNN to the existing 64-d structure slot."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.model = BWGNN(hidden, hidden, hidden, d=FILTER_ORDER)

    def forward(self, value: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        _, output = self.model(value, edge_index)
        return output


def build_model(backbone: str, variant: str, prototype: torch.Tensor) -> nn.Module:
    if backbone not in BACKBONES or variant not in VARIANTS:
        raise ValueError("Unknown backbone or variant")
    # Construct the original model first. The shared projection, semantic
    # modules, fusion, and classifier therefore have identical seeded initial
    # values for both backbones; only the GCN module is replaced afterward.
    model = original.InterfaceGNN(HIDDEN, variant, prototype, DROPOUT)
    if backbone == "bwgnn":
        model.gcn = BWGNNHidden(HIDDEN)
    return model


def audit_shared_initialization(prototype: torch.Tensor, variant: str, seed: int) -> None:
    original.seed_everything(seed)
    gcn = build_model("gcn", variant, prototype)
    original.seed_everything(seed)
    bwgnn = build_model("bwgnn", variant, prototype)
    a, b = gcn.state_dict(), bwgnn.state_dict()
    shared = set(a).intersection(b) - {name for name in a if name.startswith("gcn.")}
    if not shared or any(not torch.equal(a[name], b[name]) for name in shared):
        raise RuntimeError("Shared LSP parameters differ between backbone initializations")


def load_inputs(dataset: str) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
    protocol = original.load_protocol(dataset)
    raw = original.raw_embeddings(dataset)
    if raw.shape != (int(protocol["graph"]["num_nodes"]), original.DIM):
        raise RuntimeError(f"{dataset}: node input shape mismatch")
    if not PROTOTYPE_FILE.is_file():
        raise FileNotFoundError(f"Task-level prototype file missing: {PROTOTYPE_FILE}")
    bundle = torch.load(PROTOTYPE_FILE, map_location="cpu", weights_only=False)
    prototype = bundle["llm"].detach().cpu().float()
    if prototype.shape != (3, original.DIM) or not torch.isfinite(prototype).all():
        raise RuntimeError("Expected three finite 384-d task-level prototypes")
    graph_file = protocol["root"] / "graph_and_text_no_labels.pt"
    hashes = {
        "graph_hash": file_hash(graph_file),
        "node_order_hash": node_order_hash(protocol["graph"]["raw_texts"]),
        "raw_feature_hash": tensor_hash(raw),
        "train_split_hash": tensor_hash(protocol["train"]),
        "validation_split_hash": tensor_hash(protocol["val"]),
        "train_label_hash": tensor_hash(protocol["train_y"]),
        "validation_label_hash": tensor_hash(protocol["val_y"]),
        "prototype_file_hash": file_hash(PROTOTYPE_FILE),
        "prototype_tensor_hash": tensor_hash(prototype),
    }
    return protocol, raw, prototype, hashes


def common_configuration(dataset: str, seed: int, hashes: dict) -> dict:
    return {
        "family": "LSP_GFD_Main_Reproduction_v2",
        "dataset": dataset,
        "seed": seed,
        "hidden_dim": HIDDEN,
        "dropout": DROPOUT,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss": "weighted_cross_entropy",
        "gradient_clip_norm": 5.0,
        "max_epochs": MAX_EPOCHS[dataset],
        "patience": PATIENCE,
        "checkpoint_selection": "validation_auc",
        "threshold": 0.5,
        "prototype_source": str(PROTOTYPE_FILE),
        "model_source_hash": file_hash(SOURCE_FILE),
        "bwgnn_source_hash": file_hash(BWGNN_FILE),
        **hashes,
    }


def run_path(dataset: str, backbone: str, variant: str, seed: int, smoke: bool) -> Path:
    root = HERE / "smoke" if smoke else OUTPUT
    return root / dataset / backbone / variant / f"seed_{seed}"


def run_one(dataset: str, backbone: str, variant: str, seed: int, *, smoke: bool = False) -> dict:
    if dataset not in DATASETS or backbone not in BACKBONES or variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("Run is outside the prespecified validation grid")
    target = run_path(dataset, backbone, variant, seed, smoke)
    record_file, checkpoint = target / "result.json", target / "best_validation_checkpoint.pt"
    protocol, raw, prototype, hashes = load_inputs(dataset)
    common = common_configuration(dataset, seed, hashes)
    common_hash = json_hash(common)
    run_config = {**common, "backbone": backbone, "variant": variant, "filter_order": FILTER_ORDER if backbone == "bwgnn" else None,
                  "smoke": smoke, "effective_max_epochs": 2 if smoke else MAX_EPOCHS[dataset]}
    config_hash = json_hash(run_config)
    if record_file.exists():
        previous = json.loads(record_file.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash or previous.get("TEST_READ") != "NO" or not checkpoint.is_file():
            raise RuntimeError(f"Refusing to overwrite incompatible completed run: {target}")
        print(f"SKIP_VALID_EXISTING {dataset}/{backbone}/{variant}/seed_{seed} TEST_READ=NO", flush=True)
        return previous
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete run directory: {target}")
    original.seed_everything(seed)
    model = build_model(backbone, variant, prototype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    raw = raw.to(device)
    edge = protocol["graph"]["edge_index"].long().to(device)
    train, val = protocol["train"].to(device), protocol["val"].to(device)
    train_y, val_y = protocol["train_y"].to(device), protocol["val_y"].to(device)
    weights = original.class_weights(protocol["train_y"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_auc, best_epoch, best_metrics, best_state, stale = -math.inf, 0, None, None, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cap = 2 if smoke else MAX_EPOCHS[dataset]
    for epoch in range(1, cap + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(raw, edge)
        loss = F.cross_entropy(logits[train], train_y, weight=weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss: {dataset}/{backbone}/{variant}/seed_{seed}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            score = model(raw, edge)[val].softmax(dim=1)[:, 1].cpu().numpy()
        values = original.metrics(val_y.cpu().numpy(), score)
        if values["auc"] > best_auc + 1e-8:
            best_auc, best_epoch, best_metrics, stale = values["auc"], epoch, values, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if not smoke and stale >= PATIENCE:
            break
    assert best_state is not None and best_metrics is not None
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    target.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": best_state, "best_epoch": best_epoch, "best_val_auc": best_auc,
                "dataset": dataset, "backbone": backbone, "variant": variant, "seed": seed,
                "config_hash": config_hash, "TEST_READ": "NO"}, checkpoint)
    record = {
        "dataset": dataset, "backbone": backbone, "variant": variant, "seed": seed,
        **best_metrics, "best_epoch": best_epoch, "best_val_auc": best_auc,
        "epochs_executed": epoch, "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "runtime_seconds": elapsed, "peak_memory_bytes": peak, "device": str(device),
        "config_hash": config_hash, "common_config_hash": common_hash,
        "checkpoint_path": str(checkpoint), "TEST_READ": "NO", "smoke": smoke,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), **hashes,
    }
    record_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE {dataset}/{backbone}/{variant}/seed_{seed} AUC={best_auc:.4f} AUCPRC={best_metrics['aucprc']:.4f} TEST_READ=NO", flush=True)
    del model, raw, edge, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def status(state: str, completed: int, detail: str = "") -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    value = {"state": state, "completed": completed, "total": 36,
             "detail": detail, "updated_at_utc": datetime.now(timezone.utc).isoformat(), "TEST_READ": "NO"}
    (HERE / "queue_status.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "smoke", "train", "queue"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.mode == "audit":
        for dataset in DATASETS:
            protocol, raw, prototype, hashes = load_inputs(dataset)
            audit_shared_initialization(prototype, "lsp_full", 11)
            print(json.dumps({"dataset": dataset, "nodes": len(raw), "edges": int(protocol["graph"]["edge_index"].shape[1]),
                              "prototype_shape": list(prototype.shape), "hashes": hashes, "TEST_READ": "NO"}), flush=True)
        return
    if args.mode in ("smoke", "train"):
        if any(value is None for value in (args.dataset, args.backbone, args.variant, args.seed)):
            parser.error("smoke/train require --dataset --backbone --variant --seed")
        run_one(args.dataset, args.backbone, args.variant, args.seed, smoke=args.mode == "smoke")
        return
    completed = 0
    status("RUNNING", completed)
    try:
        for dataset in DATASETS:
            for seed in SEEDS:
                for variant in VARIANTS:
                    for backbone in BACKBONES:
                        run_one(dataset, backbone, variant, seed)
                        completed += 1
                        status("RUNNING", completed, f"last={dataset}/{backbone}/{variant}/seed_{seed}")
        result = subprocess.run([sys.executable, str(HERE / "verify_lsp_backbone_validation.py"), "--aggregate"], check=True)
        status("PASS", completed, f"aggregation and verification exit={result.returncode}")
    except BaseException as error:
        status("FAIL", completed, repr(error))
        raise


if __name__ == "__main__":
    main()
