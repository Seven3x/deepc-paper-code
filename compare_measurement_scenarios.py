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
    "yaw_bias": {
        "measurement_noise_std": "0,0,0,0,0,0",
        "measurement_yaw_bias": 0.20,
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


def make_args(base_args, controller, trajectory, scenario_name, suite_name):
    args = deepcopy(base_args)
    args.controller = controller
    args.trajectory = trajectory
    args.tag = f"{suite_name}_{scenario_name}_{controller}_{trajectory}"

    scenario = SCENARIOS[scenario_name]
    args.measurement_noise_std = scenario["measurement_noise_std"]
    args.measurement_yaw_bias = scenario["measurement_yaw_bias"]
    args.measurement_yaw_drift_per_sec = scenario["measurement_yaw_drift_per_sec"]

    if controller == "lqr":
        args.lqr_noise = 0.0
    elif controller == "mpc":
        args.mpc_N = 12
        args.mpc_solver = "CLARABEL"
    elif controller == "deepc":
        if trajectory == "step":
            args.lqr_noise = 0.02
            args.deepc_T_ini = 8
            args.deepc_N = 10
            args.deepc_lambda_y = 1000.0
            args.deepc_lambda_g = 10.0
        else:
            args.lqr_noise = 0.02
            args.deepc_T_ini = 4
            args.deepc_N = 10
            args.deepc_lambda_y = 300.0
            args.deepc_lambda_g = 3.0
        args.deepc_solver = "CLARABEL"
    else:
        raise ValueError(controller)

    return args


def controller_config(args):
    if args.controller == "lqr":
        return {"lqr_noise": args.lqr_noise}
    if args.controller == "mpc":
        return {"N": args.mpc_N, "solver": args.mpc_solver}
    return {
        "T_ini": args.deepc_T_ini,
        "N": args.deepc_N,
        "lambda_y": args.deepc_lambda_y,
        "lambda_g": args.deepc_lambda_g,
        "solver": args.deepc_solver,
        "lqr_noise": args.lqr_noise,
    }


def classify(result, thresholds):
    metrics = result["metrics"]
    return (
        metrics["all_finite"]
        and metrics["max_abs_position_error"] <= thresholds.max_position_error
        and metrics["max_abs_yaw_error"] <= thresholds.max_yaw_error
        and metrics["final_position_error_norm"] <= thresholds.max_final_position_error
    )


def parse_csv_list(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run measurement heterogeneity comparison suite.")
    parser.add_argument("--reference-duration", type=float, default=6.0)
    parser.add_argument("--sampling-time", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measurement-seed", type=int, default=0)
    parser.add_argument("--trajectories", default="step,figure8")
    parser.add_argument("--controllers", default="deepc")
    parser.add_argument("--scenarios", default="nominal,yaw_bias,yaw_drift,anisotropic_noise")
    parser.add_argument("--max-position-error", type=float, default=2.0)
    parser.add_argument("--max-yaw-error", type=float, default=1.0)
    parser.add_argument("--max-final-position-error", type=float, default=2.0)
    parser.add_argument("--tag", default="measurement_smoke")
    args = parser.parse_args()

    ensure_output_dirs()
    suite_name = f"measurement_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
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
    controllers = parse_csv_list(args.controllers)
    scenarios = parse_csv_list(args.scenarios)
    rows = []

    for scenario_name in scenarios:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        for trajectory in trajectories:
            for controller in controllers:
                run_args = make_args(base_args, controller, trajectory, scenario_name, suite_name)
                result = run_single_experiment(run_args)
                result["stable"] = classify(result, args)
                result["scenario"] = scenario_name
                result["config"] = controller_config(run_args)
                rows.append(result)

    with open(suite_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(suite_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "trajectory",
            "controller",
            "stable",
            "rmse_position",
            "rmse_yaw",
            "final_position_error_norm",
            "max_abs_position_error",
            "max_abs_yaw_error",
            "run_name",
            "config",
            "measurement",
        ])
        for row in rows:
            metrics = row["metrics"]
            writer.writerow([
                row["scenario"],
                row["trajectory"],
                row["controller"],
                row["stable"],
                metrics["rmse_position"],
                metrics["rmse_yaw"],
                metrics["final_position_error_norm"],
                metrics["max_abs_position_error"],
                metrics["max_abs_yaw_error"],
                row["run_name"],
                json.dumps(row["config"], ensure_ascii=False),
                json.dumps(row["measurement"], ensure_ascii=False),
            ])

    lines = [
        f"# Measurement Compare: {suite_name}",
        "",
        f"- reference_duration: {args.reference_duration}",
        f"- seed: {args.seed}",
        f"- measurement_seed: {args.measurement_seed}",
        f"- controllers: {', '.join(controllers)}",
        "",
        "| Scenario | Trajectory | Controller | Stable | RMSE Pos | RMSE Yaw | Final Pos Err | Max Pos Err |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['scenario']} | {row['trajectory']} | {row['controller']} | {row['stable']} | "
            f"{metrics['rmse_position']:.4f} | {metrics['rmse_yaw']:.4f} | "
            f"{metrics['final_position_error_norm']:.4f} | {metrics['max_abs_position_error']:.4f} |"
        )

    lines.extend([
        "",
        "注：当前 `LQR` 与 `linear MPC` 实现仍按状态反馈运行，测量层扰动主要直接影响 `DeePC`。",
    ])

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
