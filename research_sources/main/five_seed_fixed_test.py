"""One-time, post-hoc five-seed fixed-test extension for the frozen Table 3 grid.

No training, tuning, or checkpoint selection occurs in this file. The earlier
36 compatible test records are reused; only 84 missing cells are evaluated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
FIVE = Path(r"LOCAL_PATH_TO_CONFIGURE")
V2 = Path(r"LOCAL_PATH_TO_CONFIGURE")
STANDARD = V2 / "extension_20260928"
CLOSEST = Path(r"LOCAL_PATH_TO_CONFIGURE")
AUTHOR = Path(r"LOCAL_PATH_TO_CONFIGURE")
for folder in (FIVE, V2, STANDARD, CLOSEST, AUTHOR):
    sys.path.insert(0, str(folder))
import audit_table3_five_seeds as audit  # noqa: E402
import run_lsp_backbone_validation as training  # noqa: E402
import run_table3_baselines_validation as standard  # noqa: E402
import run_closest_baselines_validation as closest  # noqa: E402
import run_author_code_baselines_validation as author  # noqa: E402
import fixed_test_runner as prior  # noqa: E402

MANIFEST = HERE / "final_manifest.json"
MANIFEST_SHA = HERE / "final_manifest.sha256"
TEST = HERE / "fixed_test"
METRICS = ("accuracy", "f1_macro", "auc", "aucprc")
HASHES = ("graph_hash", "raw_feature_hash", "train_split_hash",
          "validation_split_hash", "train_label_hash", "validation_label_hash")
STANDARD_METHODS = ("text_mlp", "gcn", "gat")
AUTHOR_METHODS = ("bwgnn_author", "care_gnn_author_r1")
CLOSEST_METHODS = ("flag_author_code_adapted", "mled_style")
FULL = "tsp_gfd_full_bwgnn"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def old_test_path(dataset: str, method: str, seed: int) -> Path | None:
    if seed not in (None, None, None):
        return None
    if method in STANDARD_METHODS:
        return STANDARD / "table3_fixed_test" / "results" / dataset / method / f"seed_{seed}.json"
    if method == FULL:
        return V2 / "fixed_test" / "results" / dataset / "bwgnn" / "lsp_full" / f"seed_{seed}.json"
    return None


def snapshot() -> dict:
    for name in ("queue_status.json", "table3_queue_status.json"):
        status = read_json(FIVE / name)
        require(status.get("state") == "PASS" and status.get("TEST_READ") == "NO",
                f"Five-seed validation queue not PASS/NO: {name}")
    old, old_sha = prior.load_frozen_manifest()
    prior_complete = read_json(V2 / "fixed_test" / "execution_complete.json")
    standard_complete = read_json(STANDARD / "table3_fixed_test" / "execution_complete.json")
    require(prior_complete.get("status") == "COMPLETE" and prior_complete.get("manifest_sha256") == old_sha,
            "Prior full fixed-test completion invalid")
    require(standard_complete.get("status") == "COMPLETE" and standard_complete.get("jobs") == 54,
            "Prior standard fixed-test completion invalid")
    require(author.repo_commit(author.BWGNN_REPO) == author.EXPECTED_COMMITS["bwgnn_author"],
            "BWGNN author commit changed")
    require(author.repo_commit(author.CARE_REPO) == author.EXPECTED_COMMITS["care_gnn_author_r1"],
            "CARE-GNN author commit changed")
    code = [Path(__file__), FIVE / "audit_table3_five_seeds.py", V2 / "run_lsp_backbone_validation.py",
            V2 / "fixed_test_runner.py", STANDARD / "run_table3_baselines_validation.py",
            CLOSEST / "run_closest_baselines_validation.py", AUTHOR / "run_author_code_baselines_validation.py",
            AUTHOR / "care_single_relation_adapter.py", CLOSEST / "../FLAG_official_reference_20260928/models.py"]
    code = [path.resolve() for path in code]
    code_hashes = {str(path): digest(path) for path in code}
    rows = []
    protocol_hashes = {}
    old_full_lookup = {item["key"]: item for item in old["validation_runs"]}
    for dataset in audit.DATASETS:
        protocol, _, _, current_hashes = training.load_inputs(dataset)
        protocol_hashes[dataset] = {name: current_hashes[name] for name in HASHES}
        require(Path(old["datasets"][dataset]["test_lock_dir"]).is_dir(), "Missing test lock directory")
        # Test files are only checked for existence here, never opened.
        lock = Path(old["datasets"][dataset]["test_lock_dir"])
        require((lock / "test_idx.pt").is_file() and (lock / "test_labels.pt").is_file(),
                f"Missing test lock files: {dataset}")
        for method in audit.METHODS:
            for seed in audit.SEEDS:
                folder = audit.record_dir(dataset, method, seed)
                result_path = folder / "result.json"
                checkpoint_path = folder / "best_validation_checkpoint.pt"
                record = read_json(result_path)
                require(record.get("dataset") == dataset and record.get("seed") == seed
                        and record.get("TEST_READ") == "NO", f"Invalid validation record: {result_path}")
                require(all(record.get(name) == protocol_hashes[dataset][name] for name in HASHES),
                        f"Validation protocol mismatch: {result_path}")
                if method == FULL:
                    require((record.get("backbone"), record.get("variant")) == ("bwgnn", "lsp_full"),
                            f"Full model identity mismatch: {result_path}")
                else:
                    require(record.get("method") == method, f"Method mismatch: {result_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                require(checkpoint["config_hash"] == record["config_hash"]
                        and checkpoint["best_epoch"] == record["best_epoch"]
                        and checkpoint.get("TEST_READ") == "NO", f"Checkpoint metadata mismatch: {checkpoint_path}")
                require(abs(float(record["auc"]) - float(record["best_val_auc"] if "best_val_auc" in record
                                    else record["auc"])) < 1e-9, f"Not validation-AUC selected: {result_path}")
                if "best_val_auc" in checkpoint:
                    require(abs(float(checkpoint["best_val_auc"]) - float(record["auc"])) < 1e-9,
                            f"Checkpoint validation AUC mismatch: {checkpoint_path}")
                if method in AUTHOR_METHODS:
                    require(record["official_commit"] == author.EXPECTED_COMMITS[method],
                            f"Author source mismatch: {result_path}")
                previous = old_test_path(dataset, method, seed)
                old_hash = None
                if previous is not None:
                    old_record = read_json(previous)
                    require(old_record["dataset"] == dataset and old_record["seed"] == seed
                            and old_record["checkpoint_sha256"] == digest(checkpoint_path)
                            and old_record["config_hash"] == record["config_hash"]
                            and old_record["selected_by"] == "validation_auc_only",
                            f"Earlier fixed-test is not checkpoint-compatible: {previous}")
                    require(old_record["test_idx_sha256"] == old["datasets"][dataset]["expected_test_idx_sha256"]
                            and old_record["test_labels_sha256"] == old["datasets"][dataset]["expected_test_labels_sha256"],
                            f"Earlier test split changed: {previous}")
                    old_hash = digest(previous)
                if method == FULL and seed in (None, None, None):
                    key = f"{dataset}/bwgnn/lsp_full/seed_{seed}"
                    require(old_full_lookup[key]["checkpoint_sha256"] == digest(checkpoint_path),
                            f"Old full checkpoint differs: {checkpoint_path}")
                rows.append({"dataset": dataset, "method": method, "seed": seed,
                             "result_path": str(result_path), "result_sha256": digest(result_path),
                             "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": digest(checkpoint_path),
                             "config_hash": record["config_hash"], "best_epoch": record["best_epoch"],
                             "best_val_auc": float(record["auc"]),
                             "old_test_path": str(previous) if previous else None,
                             "old_test_sha256": old_hash})
    require(len(rows) == 120 and sum(item["old_test_path"] is None for item in rows) == 84,
            "Expected 120 selected checkpoints and 84 missing test cells")
    return {"schema": "LSP_GFD_five_seed_fixed_test.v1", "status": "FROZEN_VALIDATION",
            "post_hoc_after_prior_test_view": True, "test_tuning": "FORBIDDEN",
            "checkpoint_selection": "validation_auc_only", "datasets": old["datasets"],
            "protocol_hashes": protocol_hashes, "prior_manifest_sha256": old_sha,
            "prior_complete_sha256": digest(V2 / "fixed_test" / "execution_complete.json"),
            "standard_complete_sha256": digest(STANDARD / "table3_fixed_test" / "execution_complete.json"),
            "five_seed_validation_summary_sha256": digest(FIVE / "table3_five_seed_validation_summary.csv"),
            "source_sha256": code_hashes, "runs": rows}


def freeze() -> None:
    require(not MANIFEST.exists() and not MANIFEST_SHA.exists() and not TEST.exists(),
            "Manifest/test already exists; refusing overwrite")
    value = snapshot()
    value["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_exclusive(MANIFEST, value)
    with MANIFEST_SHA.open("x", encoding="ascii") as stream:
        stream.write(digest(MANIFEST) + "  final_manifest.json\n")
    print(f"FROZEN runs=120 missing_test=84 manifest={MANIFEST} TEST_READ=NO", flush=True)


def preflight(unused: bool = False) -> tuple[dict, str]:
    require(MANIFEST.is_file() and MANIFEST_SHA.is_file(), "Run --freeze first")
    sha = digest(MANIFEST)
    require(sha == MANIFEST_SHA.read_text(encoding="ascii").split()[0], "Manifest SHA mismatch")
    frozen = read_json(MANIFEST)
    current = snapshot()
    current["created_at_utc"] = frozen["created_at_utc"]
    require(current == frozen, "Frozen validation/code/data/old-test asset changed")
    if unused:
        require(not TEST.exists(), "Fixed-test attempt already exists; refusing second read")
    print(f"PREFLIGHT_PASS runs=120 new_test=84 manifest_sha256={sha} TEST_READ=NO", flush=True)
    return frozen, sha


def make_model(dataset: str, method: str, prototype: torch.Tensor, raw: torch.Tensor,
               edge: torch.Tensor, checkpoint: dict, device: torch.device) -> torch.nn.Module:
    if method in STANDARD_METHODS:
        model = standard.build_model(method).to(device)
    elif method in CLOSEST_METHODS:
        model = closest.build_model(method, prototype).to(device)
    elif method == FULL:
        model = training.build_model("bwgnn", "lsp_full", prototype).to(device)
    elif method == "bwgnn_author":
        graph = author.dgl.graph((edge[0], edge[1]), num_nodes=raw.size(0))
        graph = author.dgl.add_self_loop(author.dgl.remove_self_loop(graph))
        model = author.BWGNN(raw.size(1), author.HIDDEN, 2, graph, d=2)
    elif method == "care_gnn_author_r1":
        model = author.build_care_model(raw, edge, hidden_dim=author.HIDDEN).to(device)
    else:
        raise ValueError(method)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if method == "care_gnn_author_r1":
        require(checkpoint["care_thresholds"] is not None, "CARE thresholds missing")
        model.inter1.thresholds = checkpoint["care_thresholds"].copy()
    model.eval()
    return model


def scores(model: torch.nn.Module, method: str, dataset: str, raw: torch.Tensor,
           edge: torch.Tensor, idx: torch.Tensor, views: tuple[torch.Tensor, torch.Tensor] | None,
           device: torch.device):
    with torch.inference_mode():
        if method == "care_gnn_author_r1":
            pieces = []
            for chunk in author.batches_at_least_two(idx.tolist(), author.CARE_BATCH_SIZE[dataset]):
                dummy = torch.zeros(len(chunk), dtype=torch.long, device=device)
                logits, _ = model.forward(chunk, dummy, train_flag=False)
                pieces.append(logits.softmax(dim=1)[:, 1].cpu())
            return torch.cat(pieces).numpy()
        if method == "flag_author_code_adapted":
            require(views is not None, "FLAG matched semantic view missing")
            logits = model(raw.to(device), views[0].to(device), edge.to(device))
        elif method in STANDARD_METHODS:
            logits = standard.forward(model, method, raw.to(device), edge.to(device), None)[0]
        else:
            logits = model(raw.to(device), edge.to(device)) if method in ("mled_style", FULL) else model(raw)
        return logits[idx.to(device)].softmax(dim=1)[:, 1].cpu().numpy()


def smoke() -> None:
    frozen, _ = preflight(unused=True)
    lookup = {(r["dataset"], r["method"], r["seed"]): r for r in frozen["runs"]}
    for dataset in audit.DATASETS:
        protocol, raw, prototype, _ = training.load_inputs(dataset)
        edge = protocol["graph"]["edge_index"].long().cpu()
        views = None
        author_protocol, author_raw, _ = author.load_data(dataset)
        require(torch.equal(raw, author_raw) and torch.equal(edge, author_protocol["graph"]["edge_index"].long()),
                f"Author-code input mismatch: {dataset}")
        for method in audit.METHODS:
            item = lookup[(dataset, method, 44)]
            ckpt = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
            device = torch.device("cpu" if method in AUTHOR_METHODS else
                                  ("cuda" if torch.cuda.is_available() else "cpu"))
            if method == "flag_author_code_adapted":
                matched, neutral, _ = standard.load_flag_views(dataset, raw)
                views = (matched, neutral)
            model = make_model(dataset, method, prototype, raw, edge, ckpt, device)
            measured = training.original.metrics(protocol["val_y"].numpy(),
                scores(model, method, dataset, raw, edge, protocol["val"], views, device))
            require(abs(measured["auc"] - item["best_val_auc"]) < 5e-5,
                    f"Validation replay failed: {dataset}/{method}/seed_REDACTED measured={measured['auc']} expected={item['best_val_auc']}")
            print(f"VALIDATION_REPLAY_PASS {dataset}/{method}/seed_REDACTED AUC={measured['auc']:.6f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print("SMOKE_PASS 24 validation checkpoints TEST_READ=NO", flush=True)


def fixed_test() -> None:
    frozen, sha = preflight(unused=True)
    lookup = {(r["dataset"], r["method"], r["seed"]): r for r in frozen["runs"]}
    TEST.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(TEST / "execution_started.json", {"manifest_sha256": sha, "planned_jobs": 84,
                         "post_hoc_after_prior_test_view": True, "started_at_utc": datetime.now(timezone.utc).isoformat()})
    completed = 0
    reads = {}
    try:
        for dataset in audit.DATASETS:
            protocol, raw, prototype, _ = training.load_inputs(dataset)
            edge = protocol["graph"]["edge_index"].long().cpu()
            test_idx, test_y, test_hashes = prior.load_test_once(dataset, frozen["datasets"][dataset], protocol)
            reads[dataset] = 1
            views = None
            for method in audit.METHODS:
                if method == "flag_author_code_adapted":
                    matched, neutral, _ = standard.load_flag_views(dataset, raw)
                    views = (matched, neutral)
                for seed in audit.SEEDS:
                    item = lookup[(dataset, method, seed)]
                    if item["old_test_path"] is not None:
                        continue
                    checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=False)
                    device = torch.device("cpu" if method in AUTHOR_METHODS else
                                          ("cuda" if torch.cuda.is_available() else "cpu"))
                    model = make_model(dataset, method, prototype, raw, edge, checkpoint, device)
                    metric = training.original.metrics(test_y.numpy(),
                        scores(model, method, dataset, raw, edge, test_idx, views, device))
                    require(all(math.isfinite(v) for v in metric.values()), "Nonfinite fixed-test metric")
                    path = TEST / "results" / dataset / method / f"seed_{seed}.json"
                    write_json_exclusive(path, {"dataset": dataset, "method": method, "seed": seed,
                                         **metric, **test_hashes, "selected_by": "validation_auc_only",
                                         "best_epoch": item["best_epoch"], "best_val_auc": item["best_val_auc"],
                                         "checkpoint_sha256": item["checkpoint_sha256"],
                                         "config_hash": item["config_hash"], "manifest_sha256": sha,
                                         "post_hoc_after_prior_test_view": True,
                                         "test_lock_reads_for_dataset_in_this_extension": 1,
                                         "TEST_READ": "YES_ONE_TIME_FIXED"})
                    completed += 1
                    print(f"FIXED_TEST_DONE {dataset}/{method}/seed_{seed} ({completed}/84)", flush=True)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            del test_idx, test_y
        require(completed == 84 and reads == {d: 1 for d in audit.DATASETS}, "Incomplete fixed-test grid")
        write_json_exclusive(TEST / "execution_complete.json", {"status": "COMPLETE", "jobs": 84,
            "manifest_sha256": sha, "test_lock_reads_by_dataset_in_this_extension": reads,
            "post_hoc_after_prior_test_view": True, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print("FIXED_TEST_COMPLETE 84 new + 36 reused = 120 post-hoc cells", flush=True)
    except BaseException as exc:
        write_json_exclusive(TEST / "execution_failed.json", {"status": "FAILED_LOCKED",
            "completed": completed, "error": repr(exc), "manifest_sha256": sha})
        raise


def aggregate() -> None:
    frozen, sha = preflight()
    complete = read_json(TEST / "execution_complete.json")
    require(complete["status"] == "COMPLETE" and complete["jobs"] == 84
            and complete["manifest_sha256"] == sha, "Fixed-test completion invalid")
    lookup = {(r["dataset"], r["method"], r["seed"]): r for r in frozen["runs"]}
    detailed, summary = [], []
    for dataset in audit.DATASETS:
        for method in audit.METHODS:
            records = []
            for seed in audit.SEEDS:
                item = lookup[(dataset, method, seed)]
                prior_record = item["old_test_path"] is not None
                path = Path(item["old_test_path"]) if prior_record else TEST / "results" / dataset / method / f"seed_{seed}.json"
                require(not prior_record or digest(path) == item["old_test_sha256"], f"Old record changed: {path}")
                row = read_json(path)
                require(row["dataset"] == dataset and row["seed"] == seed
                        and row["checkpoint_sha256"] == item["checkpoint_sha256"]
                        and row["config_hash"] == item["config_hash"]
                        and row["selected_by"] == "validation_auc_only",
                        f"Fixed-test record/checkpoint mismatch: {path}")
                require(row["test_idx_sha256"] == frozen["datasets"][dataset]["expected_test_idx_sha256"]
                        and row["test_labels_sha256"] == frozen["datasets"][dataset]["expected_test_labels_sha256"],
                        f"Test split mismatch: {path}")
                if not prior_record:
                    require(row["TEST_READ"] == "YES_ONE_TIME_FIXED" and row["manifest_sha256"] == sha,
                            f"New fixed-test marker mismatch: {path}")
                detailed.append({"dataset": dataset, "method": method, "seed": seed,
                                 "source": "prior_compatible_fixed_test" if prior_record else "new_one_time_fixed_test",
                                 "record_path": str(path), **{m: row[m] for m in METRICS}})
                records.append(row)
            agg = {"dataset": dataset, "method": method, "n_seeds": None,
                   "seeds": "REDACTED", "post_hoc_after_prior_test_view": True}
            for metric in METRICS:
                values = [float(record[metric]) for record in records]
                agg[metric + "_mean"] = statistics.mean(values)
                agg[metric + "_sample_sd"] = statistics.stdev(values)
            summary.append(agg)
    require(len(detailed) == 120 and len(summary) == 24,
            f"Expected 120 test records/24 summaries, got {len(detailed)}/{len(summary)}")
    for filename, rows in (("table3_five_seed_fixed_test_cells.csv", detailed),
                           ("table3_five_seed_final_summary.csv", summary)):
        path = HERE / filename
        require(not path.exists(), f"Refusing to overwrite {path}")
        with path.open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json_exclusive(HERE / "verification.json", {"status": "PASS", "records": 120,
        "old_compatible_records": 36, "new_fixed_test_records": 84, "summary_rows": 24,
        "manifest_sha256": sha, "post_hoc_after_prior_test_view": True,
        "test_tuning": "NO", "new_test_reads_per_dataset": complete["test_lock_reads_by_dataset_in_this_extension"]})
    print(f"AGGREGATE_PASS rows=24 cells=120 final={HERE / 'table3_five_seed_final_summary.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("freeze", "preflight", "smoke", "fixed-test", "aggregate"))
    mode = parser.parse_args().mode
    if mode == "freeze":
        freeze()
    elif mode == "preflight":
        preflight(unused=True)
    elif mode == "smoke":
        smoke()
    elif mode == "fixed-test":
        fixed_test()
    else:
        aggregate()


if __name__ == "__main__":
    main()
