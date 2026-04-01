import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime

from paths import RESULTS_DIR, ensure_output_dirs
from run_experiment import build_parser, run_single_experiment


SCENARIOS = {
    "nominal": {
        "measurement_noise_std": "0,0,0,0,0,0",
        "measurement_yaw_bias": 0.0,
        "measurement_yaw_drift_per_sec": 0.0,
    },
    "yaw_drift": {
        "measurement_noise_std": "0,0,0,0,0,0",
        "measurement_yaw_bias": 0.0,
        "measurement_yaw_drift_per_sec": 0.03,
    },
    "anisotropic_noise": {
        "measurement_noise_std": "0.005,0.005,0.12,0.01,0.01,0.01",
        "measurement_yaw_bias": 0.0,
        "measurement_yaw_drift_per_sec": 0.0,
    },
}


def parse_csv_list(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def make_args(base_args, trajectory, scenario_name, variant_name, suite_name, args):
    run_args = deepcopy(base_args)
    run_args.trajectory = trajectory
    run_args.tag = f"{suite_name}_{scenario_name}_{variant_name}_{trajectory}"
    run_args.output_set = "xyzpsi"
    run_args.deepc_data_length_extra = 0

    scenario = SCENARIOS[scenario_name]
    run_args.measurement_noise_std = scenario["measurement_noise_std"]
    run_args.measurement_yaw_bias = scenario["measurement_yaw_bias"]
    run_args.measurement_yaw_drift_per_sec = scenario["measurement_yaw_drift_per_sec"]

    if trajectory == "step":
        run_args.lqr_noise = 0.02
        run_args.deepc_T_ini = 8
        run_args.deepc_N = 10
        run_args.deepc_lambda_y = 1000.0
        run_args.deepc_lambda_g = 10.0
    else:
        run_args.lqr_noise = 0.02
        run_args.deepc_T_ini = 4
        run_args.deepc_N = 10
        run_args.deepc_lambda_y = 300.0
        run_args.deepc_lambda_g = 3.0

    run_args.deepc_solver = "CLARABEL"
    run_args.deepc_initial_controller = "lqr"
    run_args.deepc_random_excitation_amplitude = 0.15
    run_args.deepc_block_lambda_roll_pitch = None
    run_args.deepc_block_lambda_yaw = None
    run_args.deepc_block_lambda_position = None

    if variant_name == "uniform":
        run_args.deepc_regularization_mode = "uniform"
        run_args.deepc_attitude_slack_weight = 1.0
        run_args.deepc_position_slack_weight = 1.0
        run_args.deepc_output_slack_weights = "1,1,1,1,1,1"
    elif variant_name == "manual_grouped":
        run_args.deepc_regularization_mode = "manual_grouped"
        run_args.deepc_attitude_slack_weight = args.manual_attitude_weight
        run_args.deepc_position_slack_weight = args.manual_position_weight
        run_args.deepc_output_slack_weights = "1,1,1,1,1,1"
    elif variant_name == "manual_yaw_only":
        run_args.deepc_regularization_mode = "manual_output"
        run_args.deepc_attitude_slack_weight = 1.0
        run_args.deepc_position_slack_weight = 1.0
        run_args.deepc_output_slack_weights = args.manual_yaw_only_weights
    elif variant_name == "measurement_noise":
        run_args.deepc_regularization_mode = "measurement_noise"
        run_args.deepc_attitude_slack_weight = 1.0
        run_args.deepc_position_slack_weight = 1.0
        run_args.deepc_output_slack_weights = "1,1,1,1,1,1"
    elif variant_name == "residual_stats":
        run_args.deepc_regularization_mode = "residual_stats"
        run_args.deepc_attitude_slack_weight = 1.0
        run_args.deepc_position_slack_weight = 1.0
        run_args.deepc_output_slack_weights = "1,1,1,1,1,1"
    elif variant_name == "block_yaw_relaxed":
        run_args.deepc_regularization_mode = "block_l2"
        run_args.deepc_attitude_slack_weight = 1.0
        run_args.deepc_position_slack_weight = 1.0
        run_args.deepc_output_slack_weights = "1,1,1,1,1,1"
        run_args.deepc_block_lambda_roll_pitch = args.block_lambda_roll_pitch
        run_args.deepc_block_lambda_yaw = args.block_lambda_yaw
        run_args.deepc_block_lambda_position = args.block_lambda_position
    elif variant_name == "xyz_only":
        run_args.output_set = "xyz"
        run_args.deepc_initial_controller = "random"
        run_args.deepc_random_excitation_amplitude = args.xyz_random_excitation_amplitude
        run_args.deepc_regularization_mode = "uniform"
        run_args.deepc_attitude_slack_weight = 1.0
        run_args.deepc_position_slack_weight = 1.0
        run_args.deepc_output_slack_weights = "1,1,1"
        run_args.deepc_data_length_extra = 100
    else:
        raise ValueError(variant_name)

    return run_args


def classify(result, thresholds):
    metrics = result["metrics"]
    return (
        metrics["all_finite"]
        and metrics["max_abs_position_error"] <= thresholds.max_position_error
        and metrics["max_abs_yaw_error"] <= thresholds.max_yaw_error
        and metrics["final_position_error_norm"] <= thresholds.max_final_position_error
    )


def main():
    parser = argparse.ArgumentParser(description="Compare DeePC regularization variants under measurement mismatch.")
    parser.add_argument("--reference-duration", type=float, default=6.0)
    parser.add_argument("--sampling-time", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measurement-seed", type=int, default=0)
    parser.add_argument("--trajectories", default="step,figure8")
    parser.add_argument("--scenarios", default="nominal,yaw_drift,anisotropic_noise")
    parser.add_argument("--variants", default="uniform,manual_grouped")
    parser.add_argument("--manual-attitude-weight", type=float, default=0.2)
    parser.add_argument("--manual-position-weight", type=float, default=1.0)
    parser.add_argument("--manual-yaw-only-weights", default="1,1,0.2,1,1,1")
    parser.add_argument("--block-lambda-roll-pitch", type=float, default=1000.0)
    parser.add_argument("--block-lambda-yaw", type=float, default=250.0)
    parser.add_argument("--block-lambda-position", type=float, default=1000.0)
    parser.add_argument("--xyz-random-excitation-amplitude", type=float, default=0.2)
    parser.add_argument("--max-position-error", type=float, default=2.0)
    parser.add_argument("--max-yaw-error", type=float, default=1.0)
    parser.add_argument("--max-final-position-error", type=float, default=2.0)
    parser.add_argument("--tag", default="deepc_reg_compare")
    args = parser.parse_args()

    ensure_output_dirs()
    suite_name = f"deepc_reg_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
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
    variants = parse_csv_list(args.variants)
    rows = []

    for scenario_name in scenarios:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        for trajectory in trajectories:
            for variant_name in variants:
                run_args = make_args(base_args, trajectory, scenario_name, variant_name, suite_name, args)
                result = run_single_experiment(run_args)
                result["stable"] = classify(result, args)
                result["scenario"] = scenario_name
                result["variant"] = variant_name
                rows.append(result)

    with open(suite_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(suite_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "trajectory",
            "variant",
            "stable",
            "rmse_position",
            "rmse_yaw",
            "final_position_error_norm",
            "max_abs_position_error",
            "max_abs_yaw_error",
            "run_name",
            "deepc",
            "measurement",
        ])
        for row in rows:
            metrics = row["metrics"]
            writer.writerow([
                row["scenario"],
                row["trajectory"],
                row["variant"],
                row["stable"],
                metrics["rmse_position"],
                metrics["rmse_yaw"],
                metrics["final_position_error_norm"],
                metrics["max_abs_position_error"],
                metrics["max_abs_yaw_error"],
                row["run_name"],
                json.dumps(row["deepc"], ensure_ascii=False),
                json.dumps(row["measurement"], ensure_ascii=False),
            ])

    lines = [
        f"# DeePC Regularization Compare: {suite_name}",
        "",
        f"- reference_duration: {args.reference_duration}",
        f"- seed: {args.seed}",
        f"- measurement_seed: {args.measurement_seed}",
        f"- manual_attitude_weight: {args.manual_attitude_weight}",
        f"- manual_position_weight: {args.manual_position_weight}",
        "",
        "| Scenario | Trajectory | Variant | Stable | RMSE Pos | RMSE Yaw | Final Pos Err | Max Pos Err |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['scenario']} | {row['trajectory']} | {row['variant']} | {row['stable']} | "
            f"{metrics['rmse_position']:.4f} | {metrics['rmse_yaw']:.4f} | "
            f"{metrics['final_position_error_norm']:.4f} | {metrics['max_abs_position_error']:.4f} |"
        )

    with open(suite_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({
        "suite_name": suite_name,
        "suite_dir": str(suite_dir),
        "num_runs": len(rows),
        "stable_runs": sum(int(row["stable"]) for row in rows),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
