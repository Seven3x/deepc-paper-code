import json
from copy import deepcopy

from run_experiment import (
    NAIVE_BASELINE_SANITY_CASES,
    apply_frozen_naive_baseline_args,
    build_parser,
    run_single_experiment,
)


def main():
    parser = build_parser()
    base_args = parser.parse_args(
        [
            "--controller",
            "deepc",
            "--trajectory",
            "step",
            "--reference-duration",
            "6.0",
            "--quiet",
        ]
    )
    apply_frozen_naive_baseline_args(base_args)

    rows = []
    all_pass = True
    for case in NAIVE_BASELINE_SANITY_CASES:
        args = deepcopy(base_args)
        args.trajectory = case["trajectory"]
        args.reference_duration = case["reference_duration"]
        args.tag = f"naive_nominal_regression_{case['trajectory']}"
        result = run_single_experiment(args)
        rmse_position = float(result["metrics"]["rmse_position"])
        threshold = float(case["rmse_position_threshold"])
        passed = rmse_position <= threshold
        all_pass = all_pass and passed
        rows.append(
            {
                "trajectory": case["trajectory"],
                "reference_duration": case["reference_duration"],
                "rmse_position": rmse_position,
                "threshold": threshold,
                "pass": bool(passed),
                "run_name": result["run_name"],
            }
        )

    print(json.dumps({"all_pass": bool(all_pass), "rows": rows}, indent=2, ensure_ascii=False))
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
