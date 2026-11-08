"""Freeze v2 validation checkpoints and evaluate 27 fixed-test cells once.

Only --fixed-test can load test indices/labels. All other modes are read-only
with respect to test artifacts. A started fixed-test attempt cannot be resumed.
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
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_lsp_backbone_validation as training  # noqa: E402
import verify_lsp_backbone_validation as validation_verify  # noqa: E402

MANIFEST = HERE / "final_manifest.json"
MANIFEST_SHA = HERE / "final_manifest.sha256"
TEST_ROOT = HERE / "fixed_test"
SELECTED = (("bwgnn", "structure_only"), ("bwgnn", "lsp_full"), ("gcn", "lsp_full"))
METRICS = ("accuracy", "f1_macro", "auc", "aucprc")
CODE_FILES = {
    "fixed_test_runner": HERE / "fixed_test_runner.py",
    "validation_runner": HERE / "run_lsp_backbone_validation.py",
    "validation_verifier": HERE / "verify_lsp_backbone_validation.py",
    "model_interface": training.SOURCE_FILE,
    "bwgnn": training.BWGNN_FILE,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def entry_key(dataset: str, backbone: str, variant: str, seed: int) -> str:
    return f"{dataset}/{backbone}/{variant}/seed_{seed}"


def selected_keys() -> list[str]:
    return [
        entry_key(dataset, backbone, variant, seed)
        for dataset in training.DATASETS
        for seed in training.SEEDS
        for backbone, variant in SELECTED
    ]


def build_manifest_snapshot() -> dict:
    """Reconstruct only from frozen validation artifacts; never load test."""
    queue = json.loads((HERE / "queue_status.json").read_text(encoding="utf-8"))
    verification = json.loads((HERE / "verification_status.json").read_text(encoding="utf-8"))
    require(queue.get("state") == "PASS" and queue.get("completed") == 36 and queue.get("TEST_READ") == "NO",
            "Validation queue is not fully verified")
    require(verification.get("status") == "PASS" and verification.get("runs") == 36
            and verification.get("TEST_READ") == "NO", "Validation verification is not PASS")
    records = validation_verify.read_records()
    selected = set(selected_keys())
    runs = []
    for dataset in training.DATASETS:
        for seed in training.SEEDS:
            for variant in training.VARIANTS:
                for backbone in training.BACKBONES:
                    key = entry_key(dataset, backbone, variant, seed)
                    result_path = training.run_path(dataset, backbone, variant, seed, False) / "result.json"
                    row = records[(dataset, backbone, variant, seed)]
                    checkpoint = Path(row["checkpoint_path"])
                    runs.append({
                        "key": key,
                        "selected_for_fixed_test": key in selected,
                        "result_path": str(result_path),
                        "result_sha256": file_hash(result_path),
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": file_hash(checkpoint),
                        "config_hash": row["config_hash"],
                        "common_config_hash": row["common_config_hash"],
                        "best_epoch": row["best_epoch"],
                        "best_val_auc": row["best_val_auc"],
                        "parameter_count": row["parameter_count"],
                        "TEST_READ": row["TEST_READ"],
                        "data_hashes": {name: row[name] for name in validation_verify.HASH_FIELDS},
                    })
    require(len(runs) == 36 and sum(item["selected_for_fixed_test"] for item in runs) == 27,
            "Expected 36 frozen runs and exactly 27 test-selected cells")
    datasets = {}
    for dataset in training.DATASETS:
        # Re-read only graph, raw features, prototypes, and train/validation
        # artifacts. This detects a changed input asset even when the old
        # validation result JSON and checkpoint bytes remain untouched.
        _, _, _, current_hashes = training.load_inputs(dataset)
        reference = records[(dataset, "gcn", "structure_only", 11)]
        require(all(current_hashes[name] == reference[name]
                    for name in validation_verify.HASH_FIELDS),
                f"{dataset}: current graph/node order/input/prototype differs from validation")
        protocol_root = training.original.PROTOCOL / dataset
        protocol_path = protocol_root / "manifest.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        require(protocol.get("test_status") == "LOCKED_NOT_READ_BY_VALIDATION_JOBS",
                f"{dataset}: unexpected test-lock protocol")
        expected = protocol.get("hashes", {})
        require(all(expected.get(name) for name in ("test_idx", "test_labels")),
                f"{dataset}: missing official test hashes in protocol manifest")
        datasets[dataset] = {
            "protocol_manifest_path": str(protocol_path),
            "protocol_manifest_sha256": file_hash(protocol_path),
            "test_lock_dir": str(protocol_root / "test_lock"),
            "expected_test_idx_sha256": expected["test_idx"],
            "expected_test_labels_sha256": expected["test_labels"],
            "expected_test_count": int(protocol["split_counts"]["test"]),
            "nodes": int(protocol["nodes"]),
        }
    return {
        "schema": "LSP_GFD_Main_Reproduction_v2.final_manifest.v1",
        "status": "FROZEN_VALIDATION",
        "experiment": "new_local_v2_not_historical_Table3",
        "checkpoint_selection": "validation_auc_only",
        "evaluation": "one_time_fixed_test_after_global_preflight",
        "no_hyperparameter_search_after_freeze": True,
        "validation_TEST_READ": "NO",
        "training_settings": {
            "hidden_dim": training.HIDDEN,
            "bwgnn_filter_order": training.FILTER_ORDER,
            "dropout": training.DROPOUT,
            "optimizer": "AdamW",
            "learning_rate": training.LR,
            "weight_decay": training.WEIGHT_DECAY,
            "patience": training.PATIENCE,
            "max_epochs_by_dataset": training.MAX_EPOCHS,
            "loss": "weighted_cross_entropy",
            "decision_threshold": 0.5,
        },
        "code_files": {name: {"path": str(path), "sha256": file_hash(path)}
                       for name, path in CODE_FILES.items()},
        "datasets": datasets,
        "selected_fixed_test_keys": selected_keys(),
        "validation_runs": runs,
    }


def load_frozen_manifest() -> tuple[dict, str]:
    require(MANIFEST.is_file() and MANIFEST_SHA.is_file(),
            "Final manifest or SHA sidecar missing; run --freeze first")
    current_sha = file_hash(MANIFEST)
    expected_sha = MANIFEST_SHA.read_text(encoding="ascii").strip().split()[0]
    require(current_sha == expected_sha, "Final manifest SHA256 mismatch")
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(value.get("schema") == "LSP_GFD_Main_Reproduction_v2.final_manifest.v1"
            and value.get("status") == "FROZEN_VALIDATION",
            "Unexpected final manifest schema/status")
    require(len(value.get("validation_runs", [])) == 36
            and len(value.get("selected_fixed_test_keys", [])) == 27,
            "Incomplete final manifest")
    return value, current_sha


def freeze() -> None:
    require(not TEST_ROOT.exists(), "Cannot freeze after a fixed-test attempt")
    if MANIFEST.exists() or MANIFEST_SHA.exists():
        value, digest = load_frozen_manifest()
        snapshot = build_manifest_snapshot()
        snapshot["created_at_utc"] = value.get("created_at_utc")
        require(value == snapshot, "Frozen manifest differs from current assets; refusing rewrite")
        print(f"ALREADY_FROZEN manifest={MANIFEST} sha256={digest} TEST_READ=NO", flush=True)
        return
    snapshot = build_manifest_snapshot()
    snapshot["created_at_utc"] = utc_now()
    dump_json_exclusive(MANIFEST, snapshot)
    digest = file_hash(MANIFEST)
    with MANIFEST_SHA.open("x", encoding="ascii") as stream:
        stream.write(f"{digest}  {MANIFEST.name}\n")
    print(f"FROZEN manifest={MANIFEST} sha256={digest} selected=27 all_validation=36 TEST_READ=NO", flush=True)


def preflight(*, require_unused_test: bool) -> tuple[dict, str]:
    manifest, digest = load_frozen_manifest()
    current = build_manifest_snapshot()
    current["created_at_utc"] = manifest.get("created_at_utc")
    require(current == manifest, "Source, data, validation result, or checkpoint changed after freeze")
    require(manifest["selected_fixed_test_keys"] == selected_keys(), "Test grid differs from frozen selection")
    for dataset, item in manifest["datasets"].items():
        lock_dir = Path(item["test_lock_dir"])
        require(lock_dir.is_dir(), f"{dataset}: test-lock directory missing")
        # Existence only: no test file is opened during freeze or preflight.
        require((lock_dir / "test_idx.pt").is_file() and (lock_dir / "test_labels.pt").is_file(),
                f"{dataset}: test-lock files missing")
    if require_unused_test:
        require(not TEST_ROOT.exists(), "One-time fixed-test directory already exists; refusing another attempt")
    print(f"PREFLIGHT_PASS jobs=27 checkpoints=36 manifest_sha256={digest} TEST_READ=NO", flush=True)
    return manifest, digest


def smoke_validation() -> None:
    manifest, _ = preflight(require_unused_test=True)
    key = entry_key("reddit", "bwgnn", "lsp_full", 11)
    frozen = next(item for item in manifest["validation_runs"] if item["key"] == key)
    protocol, raw, prototype, _ = training.load_inputs("reddit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = training.build_model("bwgnn", "lsp_full", prototype).to(device)
    checkpoint = torch.load(frozen["checkpoint_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        score = model(raw.to(device), protocol["graph"]["edge_index"].long().to(device))
        score = score[protocol["val"].to(device)].softmax(dim=1)[:, 1].cpu().numpy()
    measured = training.original.metrics(protocol["val_y"].numpy(), score)
    require(abs(measured["auc"] - frozen["best_val_auc"]) < 5e-5,
            "Validation-only checkpoint replay does not match frozen AUC")
    print(f"SMOKE_VALIDATION_PASS key={key} auc={measured['auc']:.6f} TEST_READ=NO", flush=True)


def load_test_once(dataset: str, frozen_dataset: dict, train_val: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """This function is reachable only after the one-time execution lock."""
    lock_dir = Path(frozen_dataset["test_lock_dir"])
    test_idx = torch.load(lock_dir / "test_idx.pt", map_location="cpu", weights_only=True).long()
    test_y = torch.load(lock_dir / "test_labels.pt", map_location="cpu", weights_only=True).long()
    require(training.tensor_hash(test_idx) == frozen_dataset["expected_test_idx_sha256"],
            f"{dataset}: test index hash mismatch")
    require(training.tensor_hash(test_y) == frozen_dataset["expected_test_labels_sha256"],
            f"{dataset}: test label hash mismatch")
    count = frozen_dataset["expected_test_count"]
    require(test_idx.numel() == test_y.numel() == count, f"{dataset}: test count mismatch")
    require(test_idx.min().item() >= 0 and test_idx.max().item() < frozen_dataset["nodes"],
            f"{dataset}: test index out of bounds")
    require(test_idx.unique().numel() == count, f"{dataset}: duplicate test indices")
    require(not (set(test_idx.tolist()) & set(train_val["train"].tolist())),
            f"{dataset}: train/test overlap")
    require(not (set(test_idx.tolist()) & set(train_val["val"].tolist())),
            f"{dataset}: validation/test overlap")
    require(set(test_y.tolist()) == {0, 1}, f"{dataset}: unexpected test labels")
    return test_idx, test_y, {"test_idx_sha256": training.tensor_hash(test_idx),
                              "test_labels_sha256": training.tensor_hash(test_y)}


def fixed_test() -> None:
    manifest, digest = preflight(require_unused_test=True)
    frozen_by_key = {item["key"]: item for item in manifest["validation_runs"]}
    # The exclusive directory and marker are created BEFORE opening test data.
    # Any interruption leaves a durable one-time lock; there is no auto-resume.
    TEST_ROOT.mkdir(exist_ok=False)
    dump_json_exclusive(TEST_ROOT / "execution_started.json",
                        {"started_at_utc": utc_now(), "manifest_sha256": digest,
                         "planned_jobs": 27, "TEST_READ": "NO_AT_START"})
    completed = 0
    read_counts = {}
    try:
        for dataset in training.DATASETS:
            protocol, raw, prototype, _ = training.load_inputs(dataset)
            test_idx, test_y, test_hashes = load_test_once(dataset, manifest["datasets"][dataset], protocol)
            read_counts[dataset] = 1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            raw_gpu = raw.to(device)
            edge_gpu = protocol["graph"]["edge_index"].long().to(device)
            test_gpu = test_idx.to(device)
            for seed in training.SEEDS:
                for backbone, variant in SELECTED:
                    key = entry_key(dataset, backbone, variant, seed)
                    frozen = frozen_by_key[key]
                    checkpoint = torch.load(frozen["checkpoint_path"], map_location="cpu", weights_only=False)
                    model = training.build_model(backbone, variant, prototype).to(device)
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    model.eval()
                    with torch.inference_mode():
                        logits = model(raw_gpu, edge_gpu)
                        score = logits[test_gpu].softmax(dim=1)[:, 1].cpu().numpy()
                    metrics = training.original.metrics(test_y.numpy(), score)
                    require(all(math.isfinite(value) for value in metrics.values()),
                            f"{key}: nonfinite test metric")
                    record = {
                        "key": key, "dataset": dataset, "backbone": backbone,
                        "variant": variant, "seed": seed, **metrics,
                        "selected_by": "validation_auc_only",
                        "best_epoch": frozen["best_epoch"],
                        "best_val_auc": frozen["best_val_auc"],
                        "config_hash": frozen["config_hash"],
                        "checkpoint_sha256": frozen["checkpoint_sha256"],
                        "final_manifest_sha256": digest, **test_hashes,
                        "test_label_read_count_for_dataset": 1,
                        "TEST_READ": "YES_ONE_TIME_FIXED",
                    }
                    path = TEST_ROOT / "results" / dataset / backbone / variant / f"seed_{seed}.json"
                    dump_json_exclusive(path, record)
                    completed += 1
                    print(f"FIXED_TEST_DONE {key} ({completed}/27)", flush=True)
                    del model, logits
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            del raw_gpu, edge_gpu, test_gpu, test_idx, test_y
            if device.type == "cuda":
                torch.cuda.empty_cache()
        require(completed == 27 and read_counts == {name: 1 for name in training.DATASETS},
                "Fixed-test coverage or read-count mismatch")
        dump_json_exclusive(TEST_ROOT / "execution_complete.json",
                            {"status": "COMPLETE", "jobs": 27, "test_lock_reads_by_dataset": read_counts,
                             "manifest_sha256": digest, "completed_at_utc": utc_now()})
        print("FIXED_TEST_COMPLETE jobs=27 test_lock_reads=1_per_dataset", flush=True)
    except BaseException as error:
        dump_json_exclusive(TEST_ROOT / "execution_failed.json",
                            {"status": "FAILED_LOCKED", "completed_jobs": completed,
                             "manifest_sha256": digest, "error": repr(error),
                             "failed_at_utc": utc_now()})
        raise


def test_records(manifest: dict, digest: str) -> list[dict]:
    complete = TEST_ROOT / "execution_complete.json"
    require(complete.is_file(), "No complete fixed-test attempt; aggregation is blocked")
    completion = json.loads(complete.read_text(encoding="utf-8"))
    require(completion.get("status") == "COMPLETE" and completion.get("jobs") == 27
            and completion.get("manifest_sha256") == digest,
            "Fixed-test completion marker mismatch")
    require(completion.get("test_lock_reads_by_dataset") == {name: 1 for name in training.DATASETS},
            "Test locks were not read exactly once per dataset")
    actual_paths = list((TEST_ROOT / "results").rglob("seed_*.json"))
    require(len(actual_paths) == 27, "Unexpected number of fixed-test result files")
    frozen_by_key = {item["key"]: item for item in manifest["validation_runs"]}
    rows = []
    for key in manifest["selected_fixed_test_keys"]:
        frozen = frozen_by_key[key]
        dataset, backbone, variant, seed_part = key.split("/")
        seed = int(seed_part.removeprefix("seed_"))
        path = TEST_ROOT / "results" / dataset / backbone / variant / f"seed_{seed}.json"
        require(path.is_file(), f"Missing fixed-test result: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        require((row.get("key"), row.get("dataset"), row.get("backbone"),
                 row.get("variant"), row.get("seed")) == (key, dataset, backbone, variant, seed),
                f"Fixed-test identity mismatch: {path}")
        require(row.get("TEST_READ") == "YES_ONE_TIME_FIXED"
                and row.get("selected_by") == "validation_auc_only"
                and row.get("test_label_read_count_for_dataset") == 1,
                f"Invalid test provenance: {path}")
        require(row.get("final_manifest_sha256") == digest
                and row.get("checkpoint_sha256") == frozen["checkpoint_sha256"]
                and row.get("config_hash") == frozen["config_hash"]
                and row.get("best_epoch") == frozen["best_epoch"]
                and row.get("best_val_auc") == frozen["best_val_auc"],
                f"Frozen checkpoint mismatch: {path}")
        require(file_hash(Path(frozen["checkpoint_path"])) == frozen["checkpoint_sha256"],
                f"Frozen checkpoint bytes changed: {path}")
        frozen_dataset = manifest["datasets"][dataset]
        require(row.get("test_idx_sha256") == frozen_dataset["expected_test_idx_sha256"]
                and row.get("test_labels_sha256") == frozen_dataset["expected_test_labels_sha256"],
                f"Test lock hash mismatch: {path}")
        for metric in METRICS:
            require(isinstance(row.get(metric), (int, float))
                    and math.isfinite(row[metric]) and 0 <= row[metric] <= 1,
                    f"Invalid {metric}: {path}")
        rows.append(row)
    require(len(rows) == 27, "Expected exactly 27 fixed-test result files")
    return rows


def build_tables(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    seed_rows = []
    for row in rows:
        seed_rows.append({name: row[name] for name in
                          ("dataset", "backbone", "variant", "seed", *METRICS,
                           "best_epoch", "best_val_auc", "config_hash",
                           "checkpoint_sha256", "final_manifest_sha256")})
    summary = []
    for dataset in training.DATASETS:
        for backbone, variant in SELECTED:
            group = [row for row in rows if row["dataset"] == dataset
                     and row["backbone"] == backbone and row["variant"] == variant]
            require(len(group) == None and {row["seed"] for row in group} == set(training.SEEDS),
                    f"Missing fixed-test seeds: {dataset}/{backbone}/{variant}")
            item = {"dataset": dataset, "backbone": backbone, "variant": variant, "n": 3}
            for metric in METRICS:
                values = [float(row[metric]) for row in group]
                mean, std = statistics.mean(values), statistics.stdev(values)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_sample_std"] = std
                item[f"{metric}_display"] = f"{mean:.4f} ± {std:.4f}"
            summary.append(item)
    by_key = {(row["dataset"], row["backbone"], row["variant"]): row for row in summary}
    comparison = []
    for dataset in training.DATASETS:
        gcn = by_key[(dataset, "gcn", "lsp_full")]
        full_bw = by_key[(dataset, "bwgnn", "lsp_full")]
        structure_bw = by_key[(dataset, "bwgnn", "structure_only")]
        item = {"dataset": dataset}
        for metric in ("auc", "aucprc"):
            item[f"full_gcn_{metric}"] = gcn[f"{metric}_mean"]
            item[f"full_bwgnn_{metric}"] = full_bw[f"{metric}_mean"]
            item[f"full_bwgnn_minus_gcn_{metric}"] = full_bw[f"{metric}_mean"] - gcn[f"{metric}_mean"]
            item[f"bwgnn_semantic_gain_{metric}"] = full_bw[f"{metric}_mean"] - structure_bw[f"{metric}_mean"]
        comparison.append(item)
    return seed_rows, summary, comparison


def csv_bytes(rows: list[dict]) -> bytes:
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def aggregate_or_verify(write: bool) -> None:
    manifest, digest = load_frozen_manifest()
    rows = test_records(manifest, digest)
    tables = build_tables(rows)
    paths = (TEST_ROOT / "fixed_test_seed_results.csv",
             TEST_ROOT / "fixed_test_summary.csv",
             TEST_ROOT / "fixed_test_comparison.csv")
    for path, table in zip(paths, tables):
        expected = csv_bytes(table)
        if write and not path.exists():
            with path.open("xb") as stream:
                stream.write(expected)
        else:
            require(path.is_file() and path.read_bytes() == expected,
                    f"Saved aggregate differs from frozen results: {path}")
    print(f"{'AGGREGATION' if write else 'VERIFY'}_PASS jobs=27 TEST_READ=NO_NEW_TEST_READ", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--smoke-validation", action="store_true")
    group.add_argument("--fixed-test", action="store_true")
    group.add_argument("--aggregate", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.preflight:
        preflight(require_unused_test=True)
    elif args.smoke_validation:
        smoke_validation()
    elif args.fixed_test:
        fixed_test()
    else:
        aggregate_or_verify(write=args.aggregate)


if __name__ == "__main__":
    main()
