import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from Controllers.deepc import DeePC
from Controllers.identified_linear_mpc import IdentifiedLinearMPC
from Controllers.linear_mpc import LinearMPC
from Controllers.additive_excitation import AdditiveExcitationController
from Controllers.lqr_tracking import LQRTrackingController
from Controllers.lqr import LQR
from Controllers.prbs_excitation import PRBSExcitationController
from Controllers.random_excitation import RandomExcitationController
from Controllers.reference_probe_excitation import ReferenceProbeExcitationController
from Simulator.simulation import Simulation, SimulationPlotter
from hdf5_reader import HDF5Reader
from paths import RESULTS_DIR, ensure_output_dirs
from quadcopter import Quadcopter
from trajectory_generator import TrajectoryGenerator


FROZEN_NAIVE_BASELINE = {
    "controller": "deepc",
    "output_set": "xyzpsi",
    "lqr_noise": 0.05,
    "deepc_T_ini": 8,
    "deepc_N": 10,
    "deepc_lambda_y": 1000.0,
    "deepc_lambda_g": 10.0,
    "deepc_data_length_extra": 30,
    "deepc_initial_controller": "lqr",
    "deepc_history_alignment": "naive",
    "deepc_health_mode": "nominal",
}

NAIVE_BASELINE_SANITY_CASES = (
    {"trajectory": "step", "reference_duration": 6.0, "rmse_position_threshold": 0.5},
    {"trajectory": "figure8", "reference_duration": 6.0, "rmse_position_threshold": 0.5},
    {"trajectory": "box", "reference_duration": 6.0, "rmse_position_threshold": 0.5},
)


def parse_float_list(raw):
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_measurement_config(args):
    noise_std = parse_float_list(args.measurement_noise_std)
    if len(noise_std) == 1:
        noise_std = noise_std[0]
    elif args.output_set == "xyz" and len(noise_std) == 6:
        noise_std = noise_std[3:]
    elif args.output_set == "xyz" and len(noise_std) != 3:
        raise ValueError("--measurement-noise-std must provide 1, 3, or 6 values when --output-set xyz")
    elif args.output_set == "xyzpsi" and len(noise_std) != 6:
        raise ValueError("--measurement-noise-std must provide either 1 value or 6 comma-separated values")

    return {
        "noise_std": noise_std,
        "yaw_bias": args.measurement_yaw_bias,
        "yaw_drift_per_sec": args.measurement_yaw_drift_per_sec,
        "seed": args.measurement_seed,
        "delay_steps": args.measurement_delay_steps,
        "async_period_steps": parse_int_list(args.measurement_async_period_steps),
        "burst_dropout_rate": args.measurement_burst_dropout_rate,
        "burst_dropout_length": args.measurement_burst_dropout_length,
    }


def apply_frozen_naive_baseline_args(args):
    for key, value in FROZEN_NAIVE_BASELINE.items():
        setattr(args, key, value)
    return args


def build_fault_config(args):
    fault_health_mode = "nominal" if args.deepc_health_mode == "nominal" else "degraded"
    return {
        "mode": args.fault_mode,
        "rotor_index": args.fault_rotor_index,
        "efficiency_scale": args.fault_efficiency_scale,
        "health_mode": fault_health_mode,
        "start_time": args.fault_start_time,
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
        "delay_steps": int(config.get("delay_steps", 0)),
        "async_period_steps": [int(item) for item in config.get("async_period_steps", [])],
        "burst_dropout_rate": float(config.get("burst_dropout_rate", 0.0)),
        "burst_dropout_length": int(config.get("burst_dropout_length", 0)),
    }


def serialize_fault_config(config):
    return {
        "mode": str(config["mode"]),
        "rotor_index": int(config["rotor_index"]),
        "efficiency_scale": float(config["efficiency_scale"]),
        "health_mode": str(config["health_mode"]),
        "start_time": float(config["start_time"]),
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

    if args.deepc_regularization_mode in {
        "residual_stats",
        "residual_variance",
        "residual_bias_variance",
        "robust_residual_stats",
    }:
        return np.ones(system.p)

    if args.deepc_regularization_mode == "yaw_selective_slack":
        return np.ones(system.p)

    if args.deepc_regularization_mode == "drop_yaw_past":
        return np.ones(system.p)

    if args.deepc_regularization_mode == "block_l2":
        return np.ones(system.p)

    raise ValueError(f"Unsupported DeePC regularization mode: {args.deepc_regularization_mode}")


def build_controller(args, system, trajectory):
    if args.controller == "lqr":
        return LQRTrackingController(
            system=system,
            trajectory=trajectory,
            noise=args.lqr_noise,
            seed=args.seed,
        )

    if args.controller == "identified_mpc":
        def build_id_excitation_controller():
            if args.identified_mpc_id_controller == "lqr":
                return LQR(system, noise=args.identified_mpc_id_noise, seed=args.seed)
            if args.identified_mpc_id_controller == "random":
                return RandomExcitationController(
                    system=system,
                    amplitude=args.identified_mpc_id_amplitude,
                    seed=args.seed,
                )
            if args.identified_mpc_id_controller == "prbs":
                return PRBSExcitationController(
                    system=system,
                    amplitude=args.identified_mpc_id_amplitude,
                    hold_steps=args.identified_mpc_prbs_hold_steps,
                    seed=args.seed,
                )
            if args.identified_mpc_id_controller == "lqr_random":
                return AdditiveExcitationController(
                    system=system,
                    base_controller=LQR(system, noise=0.0, seed=args.seed),
                    excitation_controller=RandomExcitationController(
                        system=system,
                        amplitude=args.identified_mpc_id_amplitude,
                        seed=args.seed,
                    ),
                )
            if args.identified_mpc_id_controller == "lqr_prbs":
                return AdditiveExcitationController(
                    system=system,
                    base_controller=LQR(system, noise=0.0, seed=args.seed),
                    excitation_controller=PRBSExcitationController(
                        system=system,
                        amplitude=args.identified_mpc_id_amplitude,
                        hold_steps=args.identified_mpc_prbs_hold_steps,
                        seed=args.seed,
                    ),
                )
            raise ValueError(f"Unsupported identified MPC ID controller: {args.identified_mpc_id_controller}")

        initial_controller = build_id_excitation_controller()
        id_data_length = max(
            (args.deepc_T_ini + args.deepc_N) * (1 + system.m + system.p) - 1,
            (system.m + 1) * (args.deepc_T_ini + args.deepc_N + system.n) - 1,
        ) + args.deepc_data_length_extra
        if args.identified_mpc_id_length is not None:
            id_data_length = int(args.identified_mpc_id_length)
        return IdentifiedLinearMPC(
            system=system,
            trajectory=trajectory,
            initial_controller=initial_controller,
            id_data_length=id_data_length,
            horizon=args.mpc_N,
            solver=args.mpc_solver,
            ridge=args.identified_mpc_ridge,
        )

    if args.controller == "deepc":
        if args.deepc_initial_controller == "lqr":
            initial_controller = LQR(system, noise=args.lqr_noise, seed=args.seed)
        elif args.deepc_initial_controller == "lqr_random":
            initial_controller = AdditiveExcitationController(
                system=system,
                base_controller=LQR(system, noise=0.0, seed=args.seed),
                excitation_controller=RandomExcitationController(
                    system=system,
                    amplitude=args.deepc_random_excitation_amplitude,
                    seed=args.seed,
                ),
            )
        elif args.deepc_initial_controller == "lqr_prbs":
            initial_controller = AdditiveExcitationController(
                system=system,
                base_controller=LQR(system, noise=0.0, seed=args.seed),
                excitation_controller=PRBSExcitationController(
                    system=system,
                    amplitude=args.deepc_random_excitation_amplitude,
                    hold_steps=args.deepc_prbs_hold_steps,
                    seed=args.seed,
                ),
            )
        elif args.deepc_initial_controller == "lqr_refprobe":
            initial_controller = ReferenceProbeExcitationController(
                system=system,
                base_controller=LQR(system, noise=0.0, seed=args.seed),
                sampling_time=system.h,
                position_amplitude=args.deepc_refprobe_position_amplitude,
                z_amplitude=args.deepc_refprobe_z_amplitude,
                yaw_amplitude=args.deepc_refprobe_yaw_amplitude,
                seed=args.seed,
            )
        elif args.deepc_initial_controller == "mpc":
            initial_controller = LinearMPC(
                system=system,
                trajectory=trajectory,
                horizon=args.mpc_N,
                solver=args.mpc_solver,
            )
        elif args.deepc_initial_controller == "random":
            initial_controller = RandomExcitationController(
                system=system,
                amplitude=args.deepc_random_excitation_amplitude,
                seed=args.seed,
            )
        else:
            raise ValueError(f"Unsupported DeePC initial controller: {args.deepc_initial_controller}")
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
            residual_weight_floor=args.deepc_residual_weight_floor,
            residual_weight_min=args.deepc_residual_weight_min,
            residual_weight_max=args.deepc_residual_weight_max,
            block_lambda_roll_pitch=args.deepc_block_lambda_roll_pitch,
            block_lambda_yaw=args.deepc_block_lambda_yaw,
            block_lambda_position=args.deepc_block_lambda_position,
            data_length_extra=args.deepc_data_length_extra,
            history_alignment=args.deepc_history_alignment,
            iv_projection_lag=args.deepc_iv_projection_lag,
            consistency_gate_lambda=args.deepc_consistency_gate_lambda,
            consistency_gate_clip=args.deepc_consistency_gate_clip,
            consistency_gate_eps=args.deepc_consistency_gate_eps,
            controller_health_mode=args.deepc_health_mode,
            bank_selection_mode=args.deepc_bank_selection,
            bank_transfer_mode=args.deepc_bank_transfer_mode,
            bank_transfer_interval_steps=args.deepc_bank_transfer_interval_steps,
            objective_variant=args.deepc_objective_variant,
            late_position_weight=args.deepc_late_position_weight,
            async_bootstrap_mode=args.deepc_async_bootstrap_mode,
            local_input_excitation_quantile=args.deepc_local_input_excitation_quantile,
        )

    if args.controller == "mpc":
        return LinearMPC(
            system=system,
            trajectory=trajectory,
            horizon=args.mpc_N,
            solver=args.mpc_solver,
        )

    raise ValueError(f"Unsupported controller: {args.controller}")


def compute_metrics(result, trajectory, system, eval_start_step=0, eval_num_steps=None, controller=None):
    y_eval_full = result.get("y_true", result["y"])
    total_steps = y_eval_full.shape[1]
    ref_full = trajectory.extend_reference(trajectory.output_reference, total_steps)[:, :total_steps]

    start = max(int(eval_start_step), 0)
    if eval_num_steps is None:
        stop = total_steps
    else:
        stop = min(start + max(int(eval_num_steps), 1), total_steps)
    if stop <= start:
        raise ValueError(f"Invalid evaluation window: start={start}, stop={stop}, total_steps={total_steps}")

    y_eval = y_eval_full[:, start:stop]
    ref = ref_full[:, start:stop]
    err = y_eval - ref

    position_rows = [i for i, idx in enumerate(system.output_indices) if idx in (9, 10, 11)]
    pos_err = err[position_rows, :]
    yaw_rows = [i for i, idx in enumerate(system.output_indices) if idx == 2]
    if yaw_rows:
        yaw_err = err[yaw_rows[0], :]
        has_yaw_output = True
    else:
        yaw_err = np.zeros(total_steps)
        has_yaw_output = False

    metrics = {
        "num_steps": int(y_eval.shape[1]),
        "evaluation_start_step": int(start),
        "evaluation_stop_step": int(stop),
        "rmse_all_outputs": float(np.sqrt(np.mean(np.square(err)))),
        "rmse_position": float(np.sqrt(np.mean(np.square(pos_err)))),
        "rmse_yaw": float(np.sqrt(np.mean(np.square(yaw_err)))),
        "p95_position_error_norm": float(np.percentile(np.linalg.norm(pos_err, axis=0), 95)),
        "max_position_error_norm": float(np.max(np.linalg.norm(pos_err, axis=0))),
        "final_position_error_norm": float(np.linalg.norm(pos_err[:, -1])),
        "max_abs_position_error": float(np.max(np.abs(pos_err))),
        "max_abs_yaw_error": float(np.max(np.abs(yaw_err))),
        "has_yaw_output": has_yaw_output,
        "all_finite": bool(np.isfinite(y_eval).all()),
    }
    if controller is not None and hasattr(controller, "qp_step_records"):
        eval_records = [
            record for record in controller.qp_step_records
            if start <= int(record["step"]) < stop
        ]
        if eval_records:
            qp_online = np.asarray([1.0 if record["qp_online"] else 0.0 for record in eval_records], dtype=float)
            pe_online = np.asarray([1.0 if record["pe_online"] else 0.0 for record in eval_records], dtype=float)
            first_qp_step = next(
                (int(record["step"]) for record in eval_records if record["qp_online"]),
                None,
            )
            metrics.update(
                {
                    "qp_online_ratio": float(np.mean(qp_online)),
                    "first_qp_step": first_qp_step,
                    "pe_online_ratio": float(np.mean(pe_online)),
                }
            )
        else:
            metrics.update(
                {
                    "qp_online_ratio": 0.0,
                    "first_qp_step": None,
                    "pe_online_ratio": 0.0,
                }
            )
    return metrics


def run_single_experiment(args):
    ensure_output_dirs()
    measurement_config = build_measurement_config(args)
    fault_config = build_fault_config(args)

    system = Quadcopter(
        h=args.sampling_time,
        measurement_config=measurement_config,
        output_set=args.output_set,
        fault_config=fault_config,
    )
    has_initial_ref = args.controller == "deepc"
    trajectory = TrajectoryGenerator(
        sort=args.trajectory,
        system=system,
        duration=args.reference_duration,
        has_initial_ref=has_initial_ref,
    )
    controller = build_controller(args, system, trajectory)

    task_start_steps = 0
    if hasattr(controller, "task_start_steps"):
        task_start_steps = int(getattr(controller, "task_start_steps"))
    elif hasattr(controller, "T") and args.controller == "deepc":
        task_start_steps = int(getattr(controller, "T"))
    task_start_time = task_start_steps * system.h
    effective_fault_config = dict(fault_config)
    effective_fault_config["start_time"] = float(args.fault_start_time) + float(task_start_time)
    system.fault_config = system._normalize_fault_config(effective_fault_config)

    if task_start_steps > 0:
        t_final = task_start_steps * system.h + args.reference_duration
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
    eval_num_steps = int(round(args.reference_duration / system.h))
    metrics = compute_metrics(
        result,
        trajectory,
        system,
        eval_start_step=task_start_steps,
        eval_num_steps=eval_num_steps,
        controller=controller,
    )

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
        "task_start_time": float(task_start_time),
        "task_start_steps": int(task_start_steps),
        "seed": args.seed,
        "output_set": args.output_set,
        "lqr_noise": args.lqr_noise,
        "regularized": args.controller != "deepc" or not args.disable_regularization,
        "metrics": metrics,
        "measurement": serialize_measurement_config(measurement_config),
        "fault": serialize_fault_config(system.fault_config),
    }

    if args.controller == "deepc":
        output["deepc"] = {
            "T_ini": args.deepc_T_ini,
            "N": args.deepc_N,
            "lambda_y": args.deepc_lambda_y,
            "lambda_g": args.deepc_lambda_g,
            "solver": args.deepc_solver,
            "initial_controller": args.deepc_initial_controller,
            "random_excitation_amplitude": args.deepc_random_excitation_amplitude,
            "prbs_hold_steps": int(args.deepc_prbs_hold_steps),
            "regularization_mode": args.deepc_regularization_mode,
            "attitude_slack_weight": args.deepc_attitude_slack_weight,
            "position_slack_weight": args.deepc_position_slack_weight,
            "output_slack_weights": parse_float_list(args.deepc_output_slack_weights),
            "residual_weight_floor": args.deepc_residual_weight_floor,
            "residual_weight_min": args.deepc_residual_weight_min,
            "residual_weight_max": args.deepc_residual_weight_max,
            "block_lambda_roll_pitch": args.deepc_block_lambda_roll_pitch,
            "block_lambda_yaw": args.deepc_block_lambda_yaw,
            "block_lambda_position": args.deepc_block_lambda_position,
            "data_length_extra": args.deepc_data_length_extra,
            "history_alignment": args.deepc_history_alignment,
            "iv_projection_lag": int(args.deepc_iv_projection_lag),
            "consistency_gate_lambda": args.deepc_consistency_gate_lambda,
            "consistency_gate_clip": args.deepc_consistency_gate_clip,
            "consistency_gate_eps": args.deepc_consistency_gate_eps,
            "effective_sigma_y_weights": np.asarray(controller.sigma_y_group_weights.value).reshape(-1).tolist(),
            "residual_stat_summary": controller.residual_stat_summary,
            "consistency_gate_summary": controller.consistency_gate_summary,
            "iv_projection_summary": controller.iv_projection_summary,
            "data_length_T": int(controller.T),
            "latest_measurement_metadata": controller.latest_measurement_metadata,
            "bank_mode": getattr(controller, "bank_mode", "single_bank"),
            "health_mode": getattr(controller, "health_mode", fault_config["health_mode"]),
            "controller_health_mode": getattr(controller, "controller_health_mode", fault_config["health_mode"]),
            "bank_selection_mode": getattr(controller, "bank_selection_mode", "fixed"),
            "bank_transfer_mode": getattr(controller, "bank_transfer_mode", "none"),
            "bank_transfer_interval_steps": int(getattr(controller, "bank_transfer_interval_steps", 1)),
            "objective_variant": args.deepc_objective_variant,
            "late_position_weight": float(args.deepc_late_position_weight),
            "async_bootstrap_mode": args.deepc_async_bootstrap_mode,
            "local_input_excitation_quantile": float(args.deepc_local_input_excitation_quantile),
            "plant_health_mode": getattr(controller, "plant_health_mode", "nominal"),
            "requested_bank_name": getattr(controller, "requested_bank_name", getattr(controller, "health_mode", "nominal")),
            "control_bank_name": getattr(controller, "control_bank_name", getattr(controller, "health_mode", "nominal")),
            "training_bank_name": getattr(controller, "training_bank_name", getattr(controller, "health_mode", "nominal")),
            "candidate_bank_scores": getattr(controller, "candidate_bank_scores", {}),
            "degraded_bank_bootstrapped": getattr(controller, "degraded_bank_bootstrapped", False),
            "degraded_bank_adaptation_steps": int(getattr(controller, "degraded_bank_adaptation_steps", 0)),
            "formal_hankel_column_count": int(
                len(controller.u_d) if isinstance(controller.u_d, list) else controller.u_d.shape[1]
            ),
            "bootstrap_hankel_column_count": int(
                len(controller.bootstrap_u_d) if isinstance(controller.bootstrap_u_d, list) else controller.bootstrap_u_d.shape[1]
            ),
            "bootstrap_partial_slot_count": int(getattr(controller, "bootstrap_partial_slot_count", 0)),
            "bootstrap_total_slot_count": int(getattr(controller, "bootstrap_total_slot_count", 0)),
            "formal_bank_accepts_partial": False,
            "bootstrap_contamination_ratio": (
                0.0
                if int(getattr(controller, "bootstrap_total_slot_count", 0)) == 0
                else float(getattr(controller, "bootstrap_partial_slot_count", 0))
                / float(getattr(controller, "bootstrap_total_slot_count", 0))
            ),
            "bootstrap_partial_excitation_scores_accepted": list(
                getattr(controller, "bootstrap_partial_excitation_scores_accepted", [])
            ),
            "bootstrap_partial_excitation_scores_rejected": list(
                getattr(controller, "bootstrap_partial_excitation_scores_rejected", [])
            ),
            "bootstrap_excitation_thresholds": list(
                getattr(controller, "bootstrap_excitation_thresholds", [])
            ),
        }
    if args.controller == "mpc":
        output["mpc"] = {
            "N": args.mpc_N,
            "solver": args.mpc_solver,
        }
    if args.controller == "identified_mpc":
        output["identified_mpc"] = {
            "N": args.mpc_N,
            "solver": args.mpc_solver,
            "id_controller": args.identified_mpc_id_controller,
            "id_amplitude": float(args.identified_mpc_id_amplitude),
            "id_noise": float(args.identified_mpc_id_noise),
            "prbs_hold_steps": int(args.identified_mpc_prbs_hold_steps),
            "ridge": float(args.identified_mpc_ridge),
            "id_length_override": (
                None if args.identified_mpc_id_length is None else int(args.identified_mpc_id_length)
            ),
            "online_state_source": "observer_from_measurements",
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
    parser.add_argument("--controller", choices=["lqr", "mpc", "identified_mpc", "deepc"], required=True)
    parser.add_argument("--trajectory", choices=["constant", "figure8", "step", "box"], default="figure8")
    parser.add_argument("--output-set", choices=["xyzpsi", "xyz"], default=FROZEN_NAIVE_BASELINE["output_set"])
    parser.add_argument("--reference-duration", type=float, default=12.0)
    parser.add_argument("--sampling-time", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lqr-noise", type=float, default=FROZEN_NAIVE_BASELINE["lqr_noise"])
    parser.add_argument("--disable-regularization", action="store_true")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--deepc-T-ini", dest="deepc_T_ini", type=int, default=FROZEN_NAIVE_BASELINE["deepc_T_ini"])
    parser.add_argument("--deepc-N", dest="deepc_N", type=int, default=FROZEN_NAIVE_BASELINE["deepc_N"])
    parser.add_argument("--deepc-lambda-y", dest="deepc_lambda_y", type=float, default=FROZEN_NAIVE_BASELINE["deepc_lambda_y"])
    parser.add_argument("--deepc-lambda-g", dest="deepc_lambda_g", type=float, default=FROZEN_NAIVE_BASELINE["deepc_lambda_g"])
    parser.add_argument("--deepc-solver", choices=["CLARABEL", "ECOS", "SCS"], default="CLARABEL")
    parser.add_argument("--deepc-initial-controller", choices=["lqr", "lqr_random", "lqr_prbs", "lqr_refprobe", "mpc", "random"], default=FROZEN_NAIVE_BASELINE["deepc_initial_controller"])
    parser.add_argument("--deepc-random-excitation-amplitude", type=float, default=0.15)
    parser.add_argument("--deepc-prbs-hold-steps", type=int, default=5)
    parser.add_argument("--deepc-refprobe-position-amplitude", type=float, default=0.12)
    parser.add_argument("--deepc-refprobe-z-amplitude", type=float, default=0.08)
    parser.add_argument("--deepc-refprobe-yaw-amplitude", type=float, default=0.20)
    parser.add_argument("--deepc-regularization-mode", choices=["uniform", "manual_grouped", "manual_output", "measurement_noise", "residual_stats", "residual_variance", "residual_bias_variance", "robust_residual_stats", "block_l2", "yaw_selective_slack", "drop_yaw_past"], default="uniform")
    parser.add_argument("--deepc-attitude-slack-weight", type=float, default=1.0)
    parser.add_argument("--deepc-position-slack-weight", type=float, default=1.0)
    parser.add_argument("--deepc-output-slack-weights", default="1,1,1,1,1,1")
    parser.add_argument("--deepc-measurement-noise-floor", type=float, default=0.01)
    parser.add_argument("--deepc-measurement-weight-min", type=float, default=0.1)
    parser.add_argument("--deepc-measurement-weight-max", type=float, default=2.0)
    parser.add_argument("--deepc-residual-weight-floor", type=float, default=0.01)
    parser.add_argument("--deepc-residual-weight-min", type=float, default=0.1)
    parser.add_argument("--deepc-residual-weight-max", type=float, default=2.0)
    parser.add_argument("--deepc-block-lambda-roll-pitch", type=float, default=None)
    parser.add_argument("--deepc-block-lambda-yaw", type=float, default=None)
    parser.add_argument("--deepc-block-lambda-position", type=float, default=None)
    parser.add_argument("--deepc-data-length-extra", type=int, default=FROZEN_NAIVE_BASELINE["deepc_data_length_extra"])
    parser.add_argument("--mpc-N", dest="mpc_N", type=int, default=10)
    parser.add_argument("--mpc-solver", choices=["CLARABEL", "ECOS", "SCS"], default="CLARABEL")
    parser.add_argument(
        "--identified-mpc-id-controller",
        choices=["lqr", "random", "prbs", "lqr_random", "lqr_prbs"],
        default="lqr_prbs",
    )
    parser.add_argument("--identified-mpc-id-amplitude", type=float, default=0.15)
    parser.add_argument("--identified-mpc-id-noise", type=float, default=0.0)
    parser.add_argument("--identified-mpc-prbs-hold-steps", type=int, default=5)
    parser.add_argument("--identified-mpc-ridge", type=float, default=1.0e-6)
    parser.add_argument("--identified-mpc-id-length", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--save-hdf5", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--measurement-noise-std", default="0,0,0,0,0,0")
    parser.add_argument("--measurement-yaw-bias", type=float, default=0.0)
    parser.add_argument("--measurement-yaw-drift-per-sec", type=float, default=0.0)
    parser.add_argument("--measurement-seed", type=int, default=0)
    parser.add_argument("--measurement-delay-steps", type=int, default=0)
    parser.add_argument("--measurement-async-period-steps", default="1,1,1,1,1,1")
    parser.add_argument("--measurement-burst-dropout-rate", type=float, default=0.0)
    parser.add_argument("--measurement-burst-dropout-length", type=int, default=0)
    parser.add_argument("--deepc-history-alignment", choices=["naive", "delay_ref_only", "time_aligned", "suffix_aligned", "consistency_gated_time_aligned", "async_masked", "iv_projected"], default=FROZEN_NAIVE_BASELINE["deepc_history_alignment"])
    parser.add_argument("--deepc-iv-projection-lag", type=int, default=1)
    parser.add_argument("--deepc-consistency-gate-lambda", type=float, default=3.0)
    parser.add_argument("--deepc-consistency-gate-clip", type=float, default=3.0)
    parser.add_argument("--deepc-consistency-gate-eps", type=float, default=1.0e-6)
    parser.add_argument("--deepc-objective-variant", choices=["baseline", "late_horizon_weight"], default="baseline")
    parser.add_argument("--deepc-late-position-weight", type=float, default=3.0)
    parser.add_argument(
        "--deepc-async-bootstrap-mode",
        choices=[
            "full_only",
            "obs_ge4",
            "pos2_obs4",
            "bootstrap_only_partial",
            "recent_consistent_partial_bootstrap",
            "bounded_stale_partial_bootstrap",
            "bounded_stale_minlen_bootstrap",
            "xy_full_minlen_bootstrap",
            "local_input_excitation_bootstrap",
        ],
        default="full_only",
    )
    parser.add_argument("--deepc-local-input-excitation-quantile", type=float, default=50.0)
    parser.add_argument("--fault-mode", choices=["nominal", "single_rotor_efficiency_drop"], default="nominal")
    parser.add_argument("--fault-rotor-index", type=int, default=0)
    parser.add_argument("--fault-efficiency-scale", type=float, default=1.0)
    parser.add_argument("--fault-start-time", type=float, default=0.0)
    parser.add_argument("--deepc-health-mode", choices=["nominal", "degraded", "health_gate"], default=FROZEN_NAIVE_BASELINE["deepc_health_mode"])
    parser.add_argument("--deepc-bank-selection", choices=["fixed", "oracle_minimal"], default="fixed")
    parser.add_argument(
        "--deepc-bank-transfer-mode",
        choices=["none", "warm_start_only", "adapt_only", "warm_start_adapt"],
        default="none",
    )
    parser.add_argument("--deepc-bank-transfer-interval-steps", type=int, default=10)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_single_experiment(args)


if __name__ == "__main__":
    main()
