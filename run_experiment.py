import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from Controllers.deepc import DeePC
from Controllers.lqr_tracking import LQRTrackingController
from Controllers.lqr import LQR
from Simulator.simulation import Simulation, SimulationPlotter
from hdf5_reader import HDF5Reader
from paths import RESULTS_DIR, ensure_output_dirs
from quadcopter import Quadcopter
from trajectory_generator import TrajectoryGenerator


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
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible DeePC quadcopter experiment.")
    parser.add_argument("--controller", choices=["lqr", "deepc"], required=True)
    parser.add_argument("--trajectory", choices=["constant", "figure8", "step", "box"], default="figure8")
    parser.add_argument("--reference-duration", type=float, default=12.0)
    parser.add_argument("--sampling-time", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lqr-noise", type=float, default=0.0)
    parser.add_argument("--disable-regularization", action="store_true")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--tag", default="")
    parser.add_argument("--save-hdf5", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()

    system = Quadcopter(h=args.sampling_time)
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

    sim = Simulation(system, controller, dt=args.dt, t_final=t_final)
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


if __name__ == "__main__":
    main()
