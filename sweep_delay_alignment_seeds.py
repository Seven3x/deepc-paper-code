import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime

import numpy as np

from compare_delay_alignment import SCENARIOS, make_args, parse_csv_list
from paths import RESULTS_DIR, ensure_output_dirs
from run_experiment import build_parser, run_single_experiment


def aggregate(rows):
    grouped = {}
    for row in rows:
        key = (row["trajectory"], row["scenario"], row["alignment_mode"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (trajectory, scenario, alignment_mode), items in grouped.items():
        rmse_position = [item["metrics"]["rmse_position"] for item in items]
        final_pos = [item["metrics"]["final_position_error_norm"] for item in items]
        max_pos = [item["metrics"]["max_abs_position_error"] for item in items]
        rmse_yaw = [item["metrics"]["rmse_yaw"] for item in items]
        success = [1.0 if item["metrics"]["all_finite"] else 0.0 for item in items]
        summary.append(
            {
                "trajectory": trajectory,
                "scenario": scenario,
                "alignment_mode": alignment_mode,
                "delay_steps": items[0]["measurement"]["delay_steps"],
                "burst_dropout_rate": items[0]["measurement"].get("burst_dropout_rate", 0.0),
                "num_seeds": len(items),
                "rmse_position_mean": float(np.mean(rmse_position)),
                "rmse_position_std": float(np.std(rmse_position)),
                "final_position_error_norm_mean": float(np.mean(final_pos)),
                "final_position_error_norm_std": float(np.std(final_pos)),
                "max_abs_position_error_mean": float(np.mean(max_pos)),
                "max_abs_position_error_std": float(np.std(max_pos)),
                "rmse_yaw_mean": float(np.mean(rmse_yaw)),
                "rmse_yaw_std": float(np.std(rmse_yaw)),
                "success_rate": float(np.mean(success)),
            }
        )
    summary.sort(key=lambda item: (item["trajectory"], item["scenario"], item["alignment_mode"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed DeePC delay-alignment sweep.")
    parser.add_argument("--reference-duration", type=float, default=4.0)
    parser.add_argument("--sampling-time", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--measurement-seeds", default="")
    parser.add_argument("--trajectories", default="step,figure8")
    parser.add_argument("--scenarios", default="nominal,delay_1,delay_2,burst_dropout_20pct")
    parser.add_argument("--alignment-modes", default="naive,time_aligned,async_masked")
    parser.add_argument("--tag", default="track1_async_delay_dropout")
    args = parser.parse_args()

    ensure_output_dirs()
    suite_name = f"delay_alignment_seed_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    suite_dir = RESULTS_DIR / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    base_parser = build_parser()
    base_args = base_parser.parse_args([
        "--controller", "deepc",
        "--trajectory", "step",
        "--reference-duration", str(args.reference_duration),
        "--sampling-time", str(args.sampling_time),
        "--dt", str(args.dt),
        "--quiet",
    ])

    seeds = [int(item) for item in parse_csv_list(args.seeds)]
    measurement_seeds = parse_csv_list(args.measurement_seeds)
    if measurement_seeds:
        measurement_seeds = [int(item) for item in measurement_seeds]
        if len(measurement_seeds) != len(seeds):
            raise ValueError("--measurement-seeds must match --seeds length when provided")
    else:
        measurement_seeds = seeds

    trajectories = parse_csv_list(args.trajectories)
    scenarios = parse_csv_list(args.scenarios)
    alignment_modes = parse_csv_list(args.alignment_modes)

    rows = []
    for seed, measurement_seed in zip(seeds, measurement_seeds):
        for scenario_name in scenarios:
            if scenario_name not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario_name}")
            for trajectory in trajectories:
                for alignment_mode in alignment_modes:
                    run_args = make_args(base_args, trajectory, scenario_name, alignment_mode, suite_name)
                    run_args.seed = seed
                    run_args.measurement_seed = measurement_seed
                    run_args.tag = f"{suite_name}_seed{seed}_{scenario_name}_{alignment_mode}_{trajectory}"
                    result = run_single_experiment(run_args)
                    result["scenario"] = scenario_name
                    result["alignment_mode"] = alignment_mode
                    rows.append(result)

    aggregate_rows = aggregate(rows)

    with open(suite_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": rows, "aggregate": aggregate_rows}, f, indent=2, ensure_ascii=False)

    with open(suite_dir / "aggregate.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trajectory",
                "scenario",
                "alignment_mode",
                "delay_steps",
                "burst_dropout_rate",
                "num_seeds",
                "rmse_position_mean",
                "rmse_position_std",
                "final_position_error_norm_mean",
                "final_position_error_norm_std",
                "max_abs_position_error_mean",
                "max_abs_position_error_std",
                "rmse_yaw_mean",
                "rmse_yaw_std",
                "success_rate",
            ]
        )
        for row in aggregate_rows:
            writer.writerow([row[key] for key in [
                "trajectory",
                "scenario",
                "alignment_mode",
                "delay_steps",
                "burst_dropout_rate",
                "num_seeds",
                "rmse_position_mean",
                "rmse_position_std",
                "final_position_error_norm_mean",
                "final_position_error_norm_std",
                "max_abs_position_error_mean",
                "max_abs_position_error_std",
                "rmse_yaw_mean",
                "rmse_yaw_std",
                "success_rate",
            ]])

    lines = [
        f"# Delay Alignment Seed Sweep: {suite_name}",
        "",
        "| Trajectory | Scenario | Alignment | Delay | Dropout | Seeds | RMSE Pos Mean | Final Pos Mean | Max Pos Mean | RMSE Yaw Mean | Success |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['trajectory']} | {row['scenario']} | {row['alignment_mode']} | "
            f"{row['delay_steps']} | {row['burst_dropout_rate']:.2f} | {row['num_seeds']} | {row['rmse_position_mean']:.4f} | "
            f"{row['final_position_error_norm_mean']:.4f} | {row['max_abs_position_error_mean']:.4f} | "
            f"{row['rmse_yaw_mean']:.4f} | {row['success_rate']:.2f} |"
        )

    with open(suite_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({
        "suite_name": suite_name,
        "suite_dir": str(suite_dir),
        "num_runs": len(rows),
        "num_aggregate_rows": len(aggregate_rows),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
