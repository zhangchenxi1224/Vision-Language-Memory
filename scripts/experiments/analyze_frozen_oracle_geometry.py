"""Audit raw geometry arrays, report all denominators, and prepare A5/A8 cases."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision_memory.training.frozen_oracle_geometry import geometry_statistics, make_initial_array, wilson


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verified_array(record):
    if sha(record["path"]) != record["file_sha256"]:
        raise ValueError(f"array hash mismatch: {record['path']}")
    x = np.load(record["path"], allow_pickle=False)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite geometry array")
    return x


def prepare_evaluations(root, manifest, summaries):
    successful = sorted([s for s in summaries.values() if s["stage"] == "A2" and s.get("qa_pass")],
                        key=lambda s: s["run_id"])
    out = root / "evaluation_inputs"
    out.mkdir(exist_ok=True)
    cases = []
    reoptimization = []
    if not successful:
        write(out / "plan.json", {"status": "not_applicable", "reason": "no successful anchor endpoints",
                                  "cases": [], "reoptimization_runs": []})
        return None
    anchor = successful[0]
    anchor_x = np.load(anchor["optimized_xT_path"], allow_pickle=False)

    def add_case(case_id, space, value, metadata):
        path = out / (case_id + ".npy")
        with path.open("wb") as f:
            np.save(f, np.asarray(value, dtype=np.float32), allow_pickle=False)
        case = {"case_id": case_id, "space": space, "latent_path": str(path.resolve()),
                "latent_sha256": sha(path), "target_index": 1, **metadata}
        cases.append(case)
        return case

    for rho in manifest["local_basin"]["rhos"]:
        for direction in range(8):
            noise = make_initial_array(anchor_x.shape, "sphere", 10000 + direction)
            case = add_case(f"A5-rho{rho:g}-d{direction}", "xT", anchor_x + rho * noise,
                            {"kind": "perturbation", "rho": rho, "direction": direction,
                             "anchor_run_id": anchor["run_id"], "rho_unit": "coordinate RMS"})
            reoptimization.append({"run_id": case["case_id"] + "-reopt", "stage": "A5_reopt",
                                   "target_index": 1, "seed": 10000 + direction, "distribution": "sphere",
                                   "scale": rho, "repeat": None, "initial_xT_path": case["latent_path"],
                                   "initial_xT_file_sha256": case["latent_sha256"], "rho": rho,
                                   "anchor_run_id": anchor["run_id"]})
    for left, right in itertools.combinations(successful[:8], 2):
        for space, key in (("xT", "optimized_xT_path"), ("z", "endpoint_z_path")):
            a, b = np.load(left[key], allow_pickle=False), np.load(right[key], allow_pickle=False)
            for index, coefficient in enumerate(manifest["interpolation"]["lambdas"]):
                add_case(f"A8-{space}-s{left['seed']:02d}-s{right['seed']:02d}-l{index:02d}", space,
                         (1 - coefficient) * a + coefficient * b,
                         {"kind": "interpolation", "lambda": coefficient,
                          "left_run_id": left["run_id"], "right_run_id": right["run_id"]})
    write(out / "cases.json", {"cases": cases})
    eval_run = {"run_id": "A5-A8-functional-evaluation", "stage": "A5_A8", "target_index": 1,
                "seed": 0, "distribution": "gaussian", "scale": 1., "repeat": None,
                "evaluation_spec_path": str((out / "cases.json").resolve())}
    plan = {"status": "ready", "anchor_run_id": anchor["run_id"], "case_count": len(cases),
            "evaluation_run": eval_run, "reoptimization_runs": reoptimization}
    write(out / "plan.json", plan)
    return plan


def analyze(args):
    root, manifest = args.root.resolve(), read(args.manifest)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    summaries, ledger, arrays = {}, [], {}
    for spec in manifest["runs"]:
        summary_path = root / "runs" / spec["run_id"] / "summary.json"
        row = {**spec, "status": "missing", "qa_pass": False}
        if summary_path.exists():
            summary = read(summary_path)
            if summary.get("technical_pass") and summary.get("model_snapshot_end_verified") and summary.get("optimizer_steps") == 256:
                if any(summary.get(k) != v for k, v in spec.items() if k != "mode"):
                    raise ValueError("Run summary does not match prospective run spec")
                if sha(summary["manifest_path"]) != summary["manifest_sha256"]:
                    raise ValueError("Run manifest hash changed")
                index = read(summary["trajectory_index_path"])
                if [r["step"] for r in index] != list(range(257)):
                    raise ValueError(f"incomplete trajectory {spec['run_id']}")
                for record in index:
                    for space in ("xT", "z"):
                        verified_array(record[space])
                for key, record in (("optimized_xT_path", index[-1]["xT"]),
                                    ("endpoint_z_path", index[-1]["z"]),
                                    ("initial_xT_path", index[0]["xT"]),
                                    ("initial_endpoint_z_path", index[0]["z"])):
                    if Path(summary[key]).resolve() != Path(record["path"]).resolve():
                        raise ValueError("Summary paths do not match audited trajectory endpoints")
                arrays[spec["run_id"]] = {
                    "xT": verified_array(index[-1]["xT"]), "z": verified_array(index[-1]["z"]),
                    "delta_xT": verified_array(index[-1]["xT"]) - verified_array(index[0]["xT"]),
                    "delta_z": verified_array(index[-1]["z"]) - verified_array(index[0]["z"])}
                summaries[spec["run_id"]] = summary
                row.update(status="completed", qa_pass=bool(summary["qa_pass"]),
                           reader_margin=summary["reader_margin"], elapsed_seconds=summary["elapsed_seconds"])
            else:
                row["status"] = "technical_failure"
        ledger.append(row)
    write(out / "run_ledger.json", ledger)
    rates = []
    for study, members in manifest["study_memberships"].items():
        if study == "A1":
            continue
        for key, group in itertools.groupby(sorted([r for r in ledger if r["run_id"] in members],
                                                    key=lambda r: (r["distribution"], r["scale"])),
                                            key=lambda r: (r["distribution"], r["scale"])):
            rows = list(group)
            completed = [r for r in rows if r["status"] == "completed"]
            success = sum(r["qa_pass"] for r in completed)
            rates.append({"study": study, "distribution": key[0], "scale": key[1],
                          "planned": len(rows), "completed": len(completed), "successes": success,
                          "success_rate_completed": success / len(completed) if completed else None,
                          "wilson95_completed": wilson(success, len(completed)),
                          "unresolved": len(rows) - len(completed)})
    write(out / "success_rates.json", rates)
    if rates:
        with (out / "success_rates.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rates[0].keys())
            writer.writeheader()
            writer.writerows(rates)
    cross = [summaries[r] for r in manifest["study_memberships"]["A9"] if r in summaries]
    geometry = {}
    for population, selected in (("all_completed", cross), ("successful_only", [r for r in cross if r["qa_pass"]])):
        if len(selected) >= 2:
            geometry[population] = {space: geometry_statistics(
                np.stack([arrays[r["run_id"]][space] for r in selected]),
                [r["task_id"] for r in selected], [r["seed"] for r in selected])
                for space in ("xT", "z", "delta_xT", "delta_z")}
    write(out / "geometry_statistics.json", geometry)
    evaluation_path = root / "runs" / "A5-A8-functional-evaluation" / "summary.json"
    functionals = read(evaluation_path) if evaluation_path.exists() else None
    if functionals:
        expected_cases = read(root / "evaluation_inputs/cases.json")["cases"]
        observed_cases = [r["case"] for r in functionals.get("cases", [])]
        if not functionals.get("technical_pass") or not functionals.get("model_snapshot_end_verified"):
            raise ValueError("Functional evaluation did not pass technical/model integrity gates")
        if observed_cases != expected_cases:
            raise ValueError("Functional case set changed or is incomplete")
        for case in expected_cases:
            if sha(case["latent_path"]) != case["latent_sha256"]:
                raise ValueError("Functional input array changed")
        write(out / "functional_geometry.json", functionals)
    plan = prepare_evaluations(root, manifest, summaries) if args.prepare_evaluations else None
    reopt_complete = True
    plan_path = root / "evaluation_inputs/plan.json"
    if plan_path.exists() and read(plan_path).get("status") == "ready":
        reopt_ledger = []
        for spec in read(plan_path)["reoptimization_runs"]:
            result_path = root / "runs" / spec["run_id"] / "summary.json"
            result = read(result_path) if result_path.exists() else {}
            ok = bool(result.get("technical_pass") and result.get("model_snapshot_end_verified")
                      and result.get("optimizer_steps") == 256)
            if ok:
                trajectory = read(result["trajectory_index_path"])
                if [r["step"] for r in trajectory] != list(range(257)):
                    raise ValueError("Local reoptimization trajectory incomplete")
                for record in trajectory:
                    for space in ("xT", "z"):
                        verified_array(record[space])
            reopt_complete = reopt_complete and ok
            reopt_ledger.append({"run_id": spec["run_id"], "rho": spec["rho"], "completed": ok,
                                 "qa_pass": result.get("qa_pass") if ok else None})
        write(out / "local_reoptimization.json", reopt_ledger)
    complete = reopt_complete and len(summaries) == len(manifest["runs"]) and (functionals is not None or not any(
        s["stage"] == "A2" and s.get("qa_pass") for s in summaries.values()))
    decision = {"status": "complete" if complete else "incomplete",
                "bank_action": "pending_scientific_review" if complete else "blocked",
                "canonicalization": None, "controller_supervision": None,
                "reason": "A preregistered canonical rule and independent bank tasks are required; no outcome guessed",
                "completed_optimization_runs": len(summaries), "planned_optimization_runs": len(manifest["runs"]),
                "functional_geometry_available": functionals is not None,
                "full_ft_authorized_by_evidence": False}
    write(out / "geometry_decision.json", decision)
    report = ["# Frozen DreamLite Oracle Geometry", "",
              f"已完成 {len(summaries)}/{len(manifest['runs'])} 个预注册优化 run。", "",
              "主指标：第256步、4个未训练逆序选择题视图全部正确且margin>0。",
              "成功率仅描述本初始化分布、优化器和预算下的可达率。技术失败与缺失单列。", "",
              "Gaussian使用总体矩匹配；Sphere才固定样本RMS。PCA必须对照样本秩和随机基线。",
              "插值只检验有限网格上的直线路径。选择题成功不等于填空题或共享记忆成功。", "",
              "下一阶段状态：" + decision["bank_action"] + "。"]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"completed": len(summaries), "planned": len(manifest["runs"]),
                      "decision": decision, "evaluation_plan": plan["case_count"] if plan else None}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepare-evaluations", action="store_true")
    analyze(parser.parse_args())
