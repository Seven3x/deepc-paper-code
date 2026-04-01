import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime

from paths import RESULTS_DIR, ensure_output_dirs
from run_experiment import build_parser, run_single_experiment


def make_args(base_args, controller, trajectory, suite_name):
    args = deepcopy(base_args)
    args.controller = controller
    args.trajectory = trajectory
    args.tag = f"{suite_name}_{controller}_{trajectory}"

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
        return {
            "lqr_noise": args.lqr_noise,
        }
    if args.controller == "mpc":
        return {
            "N": args.mpc_N,
            "solver": args.mpc_solver,
        }
    return {
        "T_ini": args.deepc_T_ini,
        "N": args.deepc_N,
        "lambda_y": args.deepc_lambda_y,
        "lambda_g": args.deepc_lambda_g,
        "solver": args.deepc_solver,
        "lqr_noise": args.lqr_noise,
    }


def classify(result, thresholds):
    m = result["metrics"]
    stable = (
        m["all_finite"]
        and m["max_abs_position_error"] <= thresholds.max_position_error
        and m["max_abs_yaw_error"] <= thresholds.max_yaw_error
        and m["final_position_error_norm"] <= thresholds.max_final_position_error
    )
    return stable


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison suite.")
    parser.add_argument("--reference-duration", type=float, default=6.0)
    parser.add_argument("--sampling-time", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trajectories", default="step,figure8")
    parser.add_argument("--max-position-error", type=float, default=2.0)
    parser.add_argument("--max-yaw-error", type=float, default=1.0)
    parser.add_argument("--max-final-position-error", type=float, default=2.0)
    parser.add_argument("--tag", default="baseline_compare")
    args = parser.parse_args()

    ensure_output_dirs()
    suite_name = f"baseline_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    suite_dir = RESULTS_DIR / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    base_parser = build_parser()
    base_args = base_parser.parse_args([
        "--controller", "lqr",
        "--trajectory", "step",
        "--reference-duration", str(args.reference_duration),
        "--sampling-time", str(args.sampling_time),
        "--dt", str(args.dt),
        "--seed", str(args.seed),
        "--quiet",
    ])

    trajectories = [item.strip() for item in args.trajectories.split(",") if item.strip()]
    controllers = ["lqr", "mpc", "deepc"]
    rows = []

    for trajectory in trajectories:
        for controller in controllers:
            run_args = make_args(base_args, controller, trajectory, suite_name)
            result = run_single_experiment(run_args)
            result["stable"] = classify(result, args)
            result["config"] = controller_config(run_args)
            rows.append(result)

    with open(suite_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(suite_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
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
        ])
        for row in rows:
            m = row["metrics"]
            writer.writerow([
                row["trajectory"],
                row["controller"],
                row["stable"],
                m["rmse_position"],
                m["rmse_yaw"],
                m["final_position_error_norm"],
                m["max_abs_position_error"],
                m["max_abs_yaw_error"],
                row["run_name"],
                json.dumps(row["config"], ensure_ascii=False),
            ])

    lines = [
        f"# Baseline Compare: {suite_name}",
        "",
        f"- reference_duration: {args.reference_duration}",
        f"- seed: {args.seed}",
        "",
        "| Trajectory | Controller | Stable | RMSE Pos | RMSE Yaw | Final Pos Err | Max Pos Err |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['trajectory']} | {row['controller']} | {row['stable']} | "
            f"{m['rmse_position']:.4f} | {m['rmse_yaw']:.4f} | "
            f"{m['final_position_error_norm']:.4f} | {m['max_abs_position_error']:.4f} |"
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
