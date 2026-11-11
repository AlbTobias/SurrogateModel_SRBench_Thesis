#!/usr/bin/env python3
"""Resolve the newest protocol-compatible result for every trial combination."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


METRICS = (
    "r2", "rmse", "nrmse_range", "mae", "fit_seconds",
    "prediction_median_seconds", "prediction_microseconds_per_sample",
    "expression_extraction_seconds", "analysis_seconds", "expression_node_count",
    "expression_depth", "simplified_node_count", "simplified_depth",
    # Legacy recovery diagnostics are retained for sidecar and CSV schema
    # compatibility. The thesis's interpretability comparison uses the
    # structural expression metrics above, not these ground-truth fields.
    "simplified_to_ground_truth_size_ratio", "ground_truth_variable_recall",
    "training_scale_simplified_node_count", "training_scale_simplified_depth",
)
SCALINGS = ("raw", "domain_minmax")
SUITE_PATTERN = re.compile(r"benchmark_suite_v(\d+)$")


def suite_version(name: str) -> int:
    """Extract the numeric version from a supported benchmark-suite name."""

    match = SUITE_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Unsupported benchmark-suite name: {name}")
    return int(match.group(1))


def containing_suite(path: Path) -> str:
    """Return the benchmark-suite directory containing a result path."""

    for parent in path.parents:
        if SUITE_PATTERN.fullmatch(parent.name):
            return parent.name
    raise ValueError(f"Result is not contained in a benchmark suite: {path}")


def effective_algorithms(configuration: dict[str, object]) -> dict[str, object]:
    """Return the estimator parameter mappings used for compatibility checks."""

    algorithms = configuration["algorithms"]
    assert isinstance(algorithms, dict)
    return {
        name: details.get("parameters", {})
        for name, details in algorithms.items()
    }


def compatible_with_reference(
    candidate: dict[str, object], reference: dict[str, object], problem: str
) -> bool:
    """Check whether a suite can supply results for a reference problem.

    Compatibility requires the problem to be present and requires identical
    configuration schema, repetition protocol, execution controls, repetition
    labels, input-scaling conditions, effective algorithm parameters, and
    problem-specific dataset construction. Suite names and unrelated additional
    problems may differ because the benchmark collection was extended over time.
    """

    if problem not in candidate.get("problems", []):
        return False
    fixed_keys = (
        "configuration_schema_version", "repetition_protocol", "execution_controls",
        "seeds", "input_scalings",
    )
    if any(candidate.get(key) != reference.get(key) for key in fixed_keys):
        return False
    if effective_algorithms(candidate) != effective_algorithms(reference):
        return False
    candidate_generation = candidate.get("dataset_generation", {}).get(problem)
    reference_generation = reference.get("dataset_generation", {}).get(problem)
    return candidate_generation == reference_generation


def load_analysis(result_path: Path, result: dict[str, object]) -> None:
    """Attach a current analysis sidecar to a resolved successful result.

    A missing sidecar is reported through ``analysis_available``. A sidecar
    whose stored source hash differs from the selected result is rejected as
    stale, preventing symbolic statistics from a previous result from entering
    the resolved summary.
    """

    sidecar = result_path.with_name(f"{result_path.stem}.analysis.json")
    result["analysis_available"] = False
    if not sidecar.exists():
        return
    analysis = json.loads(sidecar.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if analysis.get("source_result_sha256") != expected_hash:
        raise ValueError(f"Stale analysis sidecar: {sidecar}")
    result.update(analysis)
    result["analysis_available"] = True


def resolve_trials(
    project_dir: Path, reference: dict[str, object]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Resolve every planned trial combination to its newest valid record.

    Successful and failed records are both candidates because a later rerun may
    legitimately change the outcome of an earlier attempt. Selection first
    favors the highest compatible suite version and then the most recently
    modified record within that suite. Combinations without either record are
    retained as missing in the manifest.
    """

    # Load every available suite configuration before inspecting result files.
    configurations: dict[str, dict[str, object]] = {}
    for path in (project_dir / "configs").glob("benchmark_suite_v*.json"):
        configuration = json.loads(path.read_text(encoding="utf-8"))
        configurations[str(configuration["name"])] = configuration

    # Define the complete set of trial combinations required by the reference.
    algorithms = tuple(effective_algorithms(reference))
    seeds = tuple(int(seed) for seed in reference["seeds"])
    resolved: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for problem in reference["problems"]:
        # Restrict candidates to suites whose protocol and dataset agree with
        # the reference configuration for this problem.
        problem_root = project_dir / "results" / str(problem)
        suites = sorted(
            (
                path for path in problem_root.glob("benchmark_suite_v*")
                if path.name in configurations
                and compatible_with_reference(configurations[path.name], reference, str(problem))
            ),
            key=lambda path: suite_version(path.name),
        )
        for scaling in SCALINGS:
            for algorithm in algorithms:
                for seed in seeds:
                    # Prefer the newest compatible suite, using modification
                    # time only to distinguish records within the same suite.
                    candidates: list[tuple[int, int, Path, str]] = []
                    for suite in suites:
                        success = suite / scaling / algorithm / f"seed-{seed}.json"
                        failure = suite / scaling / "failures" / algorithm / f"seed-{seed}.json"
                        if success.exists():
                            candidates.append((suite_version(suite.name), success.stat().st_mtime_ns,
                                               success, "success"))
                        if failure.exists():
                            candidates.append((suite_version(suite.name), failure.stat().st_mtime_ns,
                                               failure, "failed"))
                    combination = {
                        "problem": problem, "input_scaling": scaling,
                        "algorithm": algorithm, "seed": seed,
                    }
                    if not candidates:
                        manifest.append({**combination, "status": "missing", "suite": "", "path": ""})
                        continue
                    _, _, selected_path, status = max(candidates, key=lambda item: (item[0], item[1]))
                    record = json.loads(selected_path.read_text(encoding="utf-8"))
                    record.update(combination)
                    record["resolved_suite"] = containing_suite(selected_path)
                    record["resolved_path"] = str(selected_path.relative_to(project_dir))
                    if status == "success":
                        load_analysis(selected_path, record)
                    resolved.append(record)
                    manifest.append({
                        **combination, "status": status, "suite": record["resolved_suite"],
                        "path": record["resolved_path"],
                        "analysis_available": record.get("analysis_available", False),
                    })
    return resolved, manifest


def aggregate(
    records: list[dict[str, object]], reference: dict[str, object]
) -> list[dict[str, object]]:
    """Aggregate resolved records by problem, scaling condition, and algorithm."""

    # Establish the expected coverage and repetition type for every group.
    rows: list[dict[str, object]] = []
    expected = len(reference["seeds"])
    uncontrolled = set(reference.get("repetition_protocol", {}).get(
        "uncontrolled_repetition_algorithms", []
    ))
    for problem in reference["problems"]:
        for scaling in SCALINGS:
            for algorithm in effective_algorithms(reference):
                # Separate successful and failed records for this exact group.
                selected = [
                    record for record in records
                    if record["problem"] == problem
                    and record["input_scaling"] == scaling
                    and record["algorithm"] == algorithm
                ]
                trials = [record for record in selected if record.get("status") == "success"]
                failures = [record for record in selected if record.get("status") == "failed"]
                parsed = sum(record.get("expression_parse_success") is True for record in trials)
                complexity_valid = sum(record.get("complexity_valid") is True for record in trials)
                simplified = sum(record.get("complexity_source") == "simplified" for record in trials)
                fallback = sum(
                    record.get("complexity_source") == "unsimplified_fallback" for record in trials
                )
                # Preserve legacy exact-recovery counts for diagnostic
                # compatibility; they are not used in the thesis results.
                symbolic_trials = [
                    record for record in trials if record.get("symbolic_exact_match") is not None
                ]
                # Record coverage, parsing, and simplification counts.
                row: dict[str, object] = {
                    "problem": problem,
                    "algorithm": algorithm,
                    "input_scaling": scaling,
                    "repetition_type": "uncontrolled" if algorithm in uncontrolled else "seed-controlled",
                    "expected_trials": expected,
                    "successful_trials": len(trials),
                    "failed_trials": len(failures),
                    "missing_trials": expected - len(trials) - len(failures),
                    "analysis_trials": sum(record.get("analysis_available") is True for record in trials),
                    "parsed_expression_trials": parsed,
                    "complexity_valid_trials": complexity_valid,
                    "simplified_complexity_trials": simplified,
                    "unsimplified_fallback_trials": fallback,
                    "symbolic_exact_matches": sum(
                        record.get("symbolic_exact_match") is True for record in symbolic_trials
                    ),
                    "source_suites": ";".join(sorted(
                        {str(record["resolved_suite"]) for record in selected},
                        key=suite_version,
                    )),
                }
                # Summarize each numerical metric over successful trials only.
                for metric in METRICS:
                    values = [
                        float(record[metric]) for record in trials
                        if record.get(metric) is not None
                    ]
                    row[f"{metric}_mean"] = float(np.mean(values)) if values else None
                    row[f"{metric}_std"] = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None
                    )
                    row[f"{metric}_median"] = float(np.median(values)) if values else None
                    row[f"{metric}_min"] = float(np.min(values)) if values else None
                    row[f"{metric}_max"] = float(np.max(values)) if values else None
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to a deterministic CSV using their insertion-order fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_latest_summary(
    project_dir: Path, config_path: Path, output_dir: Path
) -> dict[str, int]:
    """Write the resolved manifest and aggregate summary, then return coverage."""

    # Resolve the latest compatible record for every planned trial combination.
    reference = json.loads(config_path.read_text(encoding="utf-8"))
    records, manifest = resolve_trials(project_dir, reference)

    # Aggregate the selected records and write both public CSV artefacts.
    rows = aggregate(records, reference)
    write_csv(output_dir / "summary.csv", rows)
    write_csv(output_dir / "manifest.csv", manifest)

    # Return coverage information for command-line reporting and plotting guards.
    statuses = {
        status: sum(row["status"] == status for row in manifest)
        for status in ("success", "failed", "missing")
    }
    statuses["summary_rows"] = len(rows)
    statuses["trial_combinations"] = len(manifest)
    statuses["analysis_missing"] = sum(
        row["status"] == "success" and not row.get("analysis_available", False)
        for row in manifest
    )
    return statuses


def main() -> None:
    """Provide the command-line entry point for final result resolution."""

    # Parse output locations, generate the resolved artefacts, and report coverage.
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=project_dir / "configs" / "benchmark_suite_v11.json",
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results" / "latest")
    args = parser.parse_args()
    statuses = write_latest_summary(project_dir, args.config, args.output_dir)
    print(
        f"Wrote {statuses['summary_rows']} summary rows and "
        f"{statuses['trial_combinations']} resolved trial combinations"
    )
    print("Coverage: " + ", ".join(f"{key}={value}" for key, value in statuses.items()))


if __name__ == "__main__":
    main()
