import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from Controllers.deepc import DeePC
from Controllers.linear_mpc import LinearMPC
from Controllers.lqr_tracking import LQRTrackingController
from Controllers.lqr import LQR
from Simulator.simulation import Simulation, SimulationPlotter
from hdf5_reader import HDF5Reader
from paths import RESULTS_DIR, ensure_output_dirs
from quadcopter import Quadcopter
from trajectory_generator import TrajectoryGenerator


def parse_float_list(raw):
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_measurement_config(args):
    noise_std = parse_float_list(args.measurement_noise_std)
    if len(noise_std) == 1:
        noise_std = noise_std[0]
    elif len(noise_std) != 6:
        raise ValueError("--measurement-noise-std must provide either 1 value or 6 comma-separated values")

    return {
        "noise_std": noise_std,
        "yaw_bias": args.measurement_yaw_bias,
        "yaw_drift_per_sec": args.measurement_yaw_drift_per_sec,
        "seed": args.measurement_seed,
    }


def serialize_measurement_config(config):
    noise_std = config["noise_std"]
    if isinstance(noise_std, np.ndarray):
        noise_std = noise_std.tolist()
    return {
        "noise_std": noise_std,
        "yaw_bias": float(config["yaw_bias"]),
        "yaw_drift_per_sec": float(config["yaw_drift_per_sec"]),
        "seed": int(config["seed"]),
    }


def build_sigma_y_group_weights(args, system):
    if args.deepc_regularization_mode == "uniform":
        return None

    if args.deepc_regularization_mode == "manual_grouped":
        weights = np.ones(system.p)
        weights[:3] = args.deepc_attitude_slack_weight
        weights[3:] = args.deepc_position_slack_weight
        return weights

    if args.deepc_regularization_mode == "manual_output":
        weights = parse_float_list(args.deepc_output_slack_weights)
        if len(weights) != system.p:
            raise ValueError(f"--deepc-output-slack-weights must provide exactly {system.p} values")
        return np.asarray(weights, dtype=float)

    if args.deepc_regularization_mode == "measurement_noise":
        noise_std = np.asarray(system.measurement_config["noise_std"], dtype=float)
        if np.allclose(noise_std, 0.0):
            return np.ones(system.p)

        effective_std = np.maximum(noise_std, args.deepc_measurement_noise_floor)
        inv_std = 1.0 / effective_std
        median_inv_std = np.median(inv_std)
        weights = inv_std / median_inv_std
        return np.clip(weights, args.deepc_measurement_weight_min, args.deepc_measurement_weight_max)

    raise ValueError(f"Unsupported DeePC regularization mode: {args.deepc_regularization_mode}")


def build_controller(args, system, trajectory):
    if args.controller == "lqr":
        return LQRTrackingController(
            system=system,
            trajectory=trajectory,
            noise=args.lqr_noise,
            seed=args.seed,
        )

    if args.controller == "deepc":
        initial_controller = LQR(system, noise=args.lqr_noise, seed=args.seed)
        return DeePC(
            system,
            trajectory,
            initial_controller,
            is_regularized=not args.disable_regularization,
            prediction_horizon=args.deepc_N,
            t_ini=args.deepc_T_ini,
            lambda_y=args.deepc_lambda_y,
            lambda_g=args.deepc_lambda_g,
            solver=args.deepc_solver,
            regularization_mode=args.deepc_regularization_mode,
            sigma_y_group_weights=build_sigma_y_group_weights(args, system),
        )

    if args.controller == "mpc":
        return LinearMPC(
            system=system,
            trajectory=trajectory,
            horizon=args.mpc_N,
            solver=args.mpc_solver,
        )

    raise ValueError(f"Unsupported controller: {args.controller}")


def compute_metrics(result, trajectory):
    y = result["y"]
    total_steps = y.shape[1]
    ref = trajectory.extend_reference(trajectory.output_reference, total_steps)[:, :total_steps]
    err = y - ref

    pos_err = err[-3:, :]
    yaw_err = err[2, :]

    metrics = {
        "num_steps": int(total_steps),
        "rmse_all_outputs": float(np.sqrt(np.mean(np.square(err)))),
        "rmse_position": float(np.sqrt(np.mean(np.square(pos_err)))),
        "rmse_yaw": float(np.sqrt(np.mean(np.square(yaw_err)))),
        "final_position_error_norm": float(np.linalg.norm(pos_err[:, -1])),
        "max_abs_position_error": float(np.max(np.abs(pos_err))),
        "max_abs_yaw_error": float(np.max(np.abs(yaw_err))),
        "all_finite": bool(np.isfinite(y).all()),
    }
    return metrics


def run_single_experiment(args):
    ensure_output_dirs()
    measurement_config = build_measurement_config(args)

    system = Quadcopter(
        h=args.sampling_time,
        measurement_config=measurement_config,
    )
    has_initial_ref = args.controller == "deepc"
    trajectory = TrajectoryGenerator(
        sort=args.trajectory,
        system=system,
        duration=args.reference_duration,
        has_initial_ref=has_initial_ref,
    )
    controller = build_controller(args, system, trajectory)

    if args.controller == "deepc":
        t_final = controller.T * system.h + args.reference_duration
    else:
        t_final = args.reference_duration

    sim = Simulation(
        system,
        controller,
        dt=args.dt,
        t_final=t_final,
        verbose=not args.quiet,
    )
    result = sim.simulate()
    metrics = compute_metrics(result, trajectory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.controller}_{args.trajectory}_{timestamp}"
    if args.tag:
        run_name = f"{run_name}_{args.tag}"

    output = {
        "run_name": run_name,
        "controller": args.controller,
        "trajectory": args.trajectory,
        "sampling_time": args.sampling_time,
        "dt": args.dt,
        "reference_duration": args.reference_duration,
        "seed": args.seed,
        "lqr_noise": args.lqr_noise,
        "regularized": args.controller != "deepc" or not args.disable_regularization,
        "metrics": metrics,
        "measurement": serialize_measurement_config(measurement_config),
    }

    if args.controller == "deepc":
        output["deepc"] = {
            "T_ini": args.deepc_T_ini,
            "N": args.deepc_N,
            "lambda_y": args.deepc_lambda_y,
            "lambda_g": args.deepc_lambda_g,
            "solver": args.deepc_solver,
            "regularization_mode": args.deepc_regularization_mode,
            "attitude_slack_weight": args.deepc_attitude_slack_weight,
            "position_slack_weight": args.deepc_position_slack_weight,
            "output_slack_weights": parse_float_list(args.deepc_output_slack_weights),
            "data_length_T": int(controller.T),
        }
    if args.controller == "mpc":
        output["mpc"] = {
            "N": args.mpc_N,
            "solver": args.mpc_solver,
        }

    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if args.save_hdf5:
        reader = HDF5Reader(str(run_dir))
        reader.save_to_hdf5(result, run_name)

    if args.save_plots:
        plotter = SimulationPlotter(system)
        plotter.plot(result, trajectory)

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def build_parser():
    parser = argparse.ArgumentParser(description="Run a reproducible DeePC quadcopter experiment.")
    parser.add_argument("--controller", choices=["lqr", "mpc", "deepc"], required=True)
    parser.add_argument("--trajectory", choices=["constant", "figure8", "step", "box"], default="figure8")
    parser.add_argument("--reference-duration", type=float, default=12.0)
    parser.add_argument("--sampling-time", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lqr-noise", type=float, default=0.0)
    parser.add_argument("--disable-regularization", action="store_true")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--deepc-T-ini", dest="deepc_T_ini", type=int, default=6)
    parser.add_argument("--deepc-N", dest="deepc_N", type=int, default=25)
    parser.add_argument("--deepc-lambda-y", dest="deepc_lambda_y", type=float, default=1.0e4)
    parser.add_argument("--deepc-lambda-g", dest="deepc_lambda_g", type=float, default=30.0)
    parser.add_argument("--deepc-solver", choices=["CLARABEL", "ECOS", "SCS"], default="CLARABEL")
    parser.add_argument("--deepc-regularization-mode", choices=["uniform", "manual_grouped", "manual_output", "measurement_noise"], default="uniform")
    parser.add_argument("--deepc-attitude-slack-weight", type=float, default=1.0)
    parser.add_argument("--deepc-position-slack-weight", type=float, default=1.0)
    parser.add_argument("--deepc-output-slack-weights", default="1,1,1,1,1,1")
    parser.add_argument("--deepc-measurement-noise-floor", type=float, default=0.01)
    parser.add_argument("--deepc-measurement-weight-min", type=float, default=0.1)
    parser.add_argument("--deepc-measurement-weight-max", type=float, default=2.0)
    parser.add_argument("--mpc-N", dest="mpc_N", type=int, default=10)
    parser.add_argument("--mpc-solver", choices=["CLARABEL", "ECOS", "SCS"], default="CLARABEL")
    parser.add_argument("--tag", default="")
    parser.add_argument("--save-hdf5", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--measurement-noise-std", default="0,0,0,0,0,0")
    parser.add_argument("--measurement-yaw-bias", type=float, default=0.0)
    parser.add_argument("--measurement-yaw-drift-per-sec", type=float, default=0.0)
    parser.add_argument("--measurement-seed", type=int, default=0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_single_experiment(args)


if __name__ == "__main__":
    main()
