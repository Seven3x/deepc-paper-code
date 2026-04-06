import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime

from paths import RESULTS_DIR, ensure_output_dirs
from run_experiment import build_parser, run_single_experiment


SCENARIOS = {
    "nominal": {
        "measurement_delay_steps": 0,
        "measurement_burst_dropout_rate": 0.0,
        "measurement_burst_dropout_length": 0,
    },
    "delay_1": {
        "measurement_delay_steps": 1,
        "measurement_burst_dropout_rate": 0.0,
        "measurement_burst_dropout_length": 0,
    },
    "delay_2": {
        "measurement_delay_steps": 2,
        "measurement_burst_dropout_rate": 0.0,
        "measurement_burst_dropout_length": 0,
    },
    "burst_dropout_20pct": {
        "measurement_delay_steps": 0,
        "measurement_burst_dropout_rate": 0.2,
        "measurement_burst_dropout_length": 2,
    },
}


def parse_csv_list(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def make_args(base_args, trajectory, scenario_name, alignment_mode, suite_name):
    args = deepcopy(base_args)
    args.controller = "deepc"
    args.trajectory = trajectory
    args.deepc_history_alignment = alignment_mode
    for key, value in SCENARIOS[scenario_name].items():
        setattr(args, key, value)
    args.tag = f"{suite_name}_{scenario_name}_{alignment_mode}_{trajectory}"
    args.deepc_initial_controller = "random"
    args.deepc_random_excitation_amplitude = 0.35
    args.deepc_data_length_extra = 30

    if trajectory == "step":
        args.deepc_T_ini = 8
        args.deepc_N = 10
        args.deepc_lambda_y = 1000.0
        args.deepc_lambda_g = 10.0
    else:
        args.deepc_T_ini = 4
        args.deepc_N = 10
        args.deepc_lambda_y = 300.0
        args.deepc_lambda_g = 3.0

    return args


def main():
    parser = argparse.ArgumentParser(description="Run DeePC delay-alignment comparison suite.")
    parser.add_argument("--reference-duration", type=float, default=6.0)
    parser.add_argument("--sampling-time", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measurement-seed", type=int, default=0)
    parser.add_argument("--trajectories", default="step,figure8")
    parser.add_argument("--scenarios", default="nominal,delay_1,delay_2,burst_dropout_20pct")
    parser.add_argument("--alignment-modes", default="naive,time_aligned,async_masked")
    parser.add_argument("--tag", default="track1_async_delay_dropout_smoke")
    args = parser.parse_args()

    ensure_output_dirs()
    suite_name = f"delay_alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    suite_dir = RESULTS_DIR / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    base_parser = build_parser()
    base_args = base_parser.parse_args([
        "--controller", "deepc",
        "--trajectory", "step",
        "--reference-duration", str(args.reference_duration),
        "--sampling-time", str(args.sampling_time),
        "--dt", str(args.dt),
        "--seed", str(args.seed),
        "--measurement-seed", str(args.measurement_seed),
        "--quiet",
    ])

    trajectories = parse_csv_list(args.trajectories)
    scenarios = parse_csv_list(args.scenarios)
    alignment_modes = parse_csv_list(args.alignment_modes)
    rows = []

    for scenario_name in scenarios:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        for trajectory in trajectories:
            for alignment_mode in alignment_modes:
                run_args = make_args(base_args, trajectory, scenario_name, alignment_mode, suite_name)
                result = run_single_experiment(run_args)
                result["scenario"] = scenario_name
                result["alignment_mode"] = alignment_mode
                rows.append(result)

    with open(suite_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(suite_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "trajectory",
            "alignment_mode",
            "delay_steps",
            "burst_dropout_rate",
            "rmse_position",
            "rmse_yaw",
            "final_position_error_norm",
            "max_abs_position_error",
            "all_finite",
            "run_name",
        ])
        for row in rows:
            metrics = row["metrics"]
            writer.writerow([
                row["scenario"],
                row["trajectory"],
                row["alignment_mode"],
                row["measurement"]["delay_steps"],
                row["measurement"].get("burst_dropout_rate", 0.0),
                metrics["rmse_position"],
                metrics["rmse_yaw"],
                metrics["final_position_error_norm"],
                metrics["max_abs_position_error"],
                metrics["all_finite"],
                row["run_name"],
            ])

    lines = [
        f"# Delay Alignment Compare: {suite_name}",
        "",
        "| Scenario | Trajectory | Alignment | Delay | Dropout | RMSE Pos | RMSE Yaw | Final Pos Err | Max Pos Err | Finite |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['scenario']} | {row['trajectory']} | {row['alignment_mode']} | "
            f"{row['measurement']['delay_steps']} | {row['measurement'].get('burst_dropout_rate', 0.0):.2f} | {metrics['rmse_position']:.4f} | "
            f"{metrics['rmse_yaw']:.4f} | {metrics['final_position_error_norm']:.4f} | "
            f"{metrics['max_abs_position_error']:.4f} | {metrics['all_finite']} |"
        )

    with open(suite_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({
        "suite_name": suite_name,
        "suite_dir": str(suite_dir),
        "num_runs": len(rows),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
