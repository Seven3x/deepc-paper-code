import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime

from run_experiment import build_parser, run_single_experiment
from paths import RESULTS_DIR, ensure_output_dirs


def parse_list(raw, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def classify_run(result, thresholds):
    metrics = result["metrics"]
    stable = (
        metrics["all_finite"]
        and metrics["max_abs_position_error"] <= thresholds.max_position_error
        and metrics["max_abs_yaw_error"] <= thresholds.max_yaw_error
        and metrics["final_position_error_norm"] <= thresholds.max_final_position_error
    )
    return {
        "stable": stable,
        "stability_score": (
            metrics["rmse_position"]
            + 0.5 * metrics["rmse_yaw"]
            + 0.2 * metrics["final_position_error_norm"]
        ),
    }


def generate_ofat_runs(args):
    base = {
        "deepc_T_ini": args.base_T_ini,
        "deepc_N": args.base_N,
        "deepc_lambda_y": args.base_lambda_y,
        "deepc_lambda_g": args.base_lambda_g,
        "lqr_noise": args.base_lqr_noise,
    }
    runs = [("base", deepcopy(base))]

    for name, values in (
        ("deepc_T_ini", args.t_ini_values),
        ("deepc_N", args.N_values),
        ("deepc_lambda_y", args.lambda_y_values),
        ("deepc_lambda_g", args.lambda_g_values),
        ("lqr_noise", args.lqr_noise_values),
    ):
        for value in values:
            if value == base[name]:
                continue
            config = deepcopy(base)
            config[name] = value
            runs.append((f"{name}_{value}", config))

    return runs


def main():
    parser = argparse.ArgumentParser(description="Run OFAT DeePC smoke tests.")
    parser.add_argument("--trajectory", default="step", choices=["constant", "figure8", "step", "box"])
    parser.add_argument("--reference-duration", type=float, default=4.0)
    parser.add_argument("--sampling-time", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-T-ini", type=int, default=4)
    parser.add_argument("--base-N", type=int, default=10)
    parser.add_argument("--base-lambda-y", type=float, default=1000.0)
    parser.add_argument("--base-lambda-g", type=float, default=10.0)
    parser.add_argument("--base-lqr-noise", type=float, default=0.05)
    parser.add_argument("--t-ini-values", type=lambda x: parse_list(x, int), default="4,6,8")
    parser.add_argument("--N-values", type=lambda x: parse_list(x, int), default="8,10,12")
    parser.add_argument("--lambda-y-values", type=lambda x: parse_list(x, float), default="300,1000,3000")
    parser.add_argument("--lambda-g-values", type=lambda x: parse_list(x, float), default="3,10,30")
    parser.add_argument("--lqr-noise-values", type=lambda x: parse_list(x, float), default="0.02,0.05,0.1")
    parser.add_argument("--max-position-error", type=float, default=2.0)
    parser.add_argument("--max-yaw-error", type=float, default=1.0)
    parser.add_argument("--max-final-position-error", type=float, default=1.5)
    parser.add_argument("--tag", default="smoke")
    args = parser.parse_args()

    if isinstance(args.t_ini_values, str):
        args.t_ini_values = parse_list(args.t_ini_values, int)
    if isinstance(args.N_values, str):
        args.N_values = parse_list(args.N_values, int)
    if isinstance(args.lambda_y_values, str):
        args.lambda_y_values = parse_list(args.lambda_y_values, float)
    if isinstance(args.lambda_g_values, str):
        args.lambda_g_values = parse_list(args.lambda_g_values, float)
    if isinstance(args.lqr_noise_values, str):
        args.lqr_noise_values = parse_list(args.lqr_noise_values, float)

    ensure_output_dirs()

    base_parser = build_parser()
    base_args = base_parser.parse_args([
        "--controller", "deepc",
        "--trajectory", args.trajectory,
        "--reference-duration", str(args.reference_duration),
        "--sampling-time", str(args.sampling_time),
        "--dt", str(args.dt),
        "--seed", str(args.seed),
        "--quiet",
    ])

    suite_name = f"deepc_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    suite_dir = RESULTS_DIR / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for label, config in generate_ofat_runs(args):
        run_args = deepcopy(base_args)
        run_args.tag = f"{suite_name}_{label}"
        run_args.deepc_T_ini = config["deepc_T_ini"]
        run_args.deepc_N = config["deepc_N"]
        run_args.deepc_lambda_y = config["deepc_lambda_y"]
        run_args.deepc_lambda_g = config["deepc_lambda_g"]
        run_args.lqr_noise = config["lqr_noise"]

        result = run_single_experiment(run_args)
        result["sweep_label"] = label
        result["classification"] = classify_run(result, args)
        runs.append(result)

    runs.sort(key=lambda item: (not item["classification"]["stable"], item["classification"]["stability_score"]))

    with open(suite_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)

    with open(suite_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "label",
            "stable",
            "stability_score",
            "T_ini",
            "N",
            "lambda_y",
            "lambda_g",
            "lqr_noise",
            "rmse_position",
            "rmse_yaw",
            "final_position_error_norm",
            "max_abs_position_error",
            "max_abs_yaw_error",
            "run_name",
        ])
        for idx, result in enumerate(runs, start=1):
            deepc = result["deepc"]
            metrics = result["metrics"]
            classification = result["classification"]
            writer.writerow([
                idx,
                result["sweep_label"],
                classification["stable"],
                classification["stability_score"],
                deepc["T_ini"],
                deepc["N"],
                deepc["lambda_y"],
                deepc["lambda_g"],
                result["lqr_noise"],
                metrics["rmse_position"],
                metrics["rmse_yaw"],
                metrics["final_position_error_norm"],
                metrics["max_abs_position_error"],
                metrics["max_abs_yaw_error"],
                result["run_name"],
            ])

    print(json.dumps({
        "suite_name": suite_name,
        "suite_dir": str(suite_dir),
        "num_runs": len(runs),
        "best_run": runs[0]["run_name"] if runs else None,
        "stable_runs": sum(int(run["classification"]["stable"]) for run in runs),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
