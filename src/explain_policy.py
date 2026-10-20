"""Placeholder CLI for policy explainability using SHAP.

This entry point validates the supplied checkpoint and state matrix.
It does not implement SHAP analysis or regenerate historical manuscript
results.

SHAP explanations of binary allocation states describe model outputs
with respect to those input features. They should not be interpreted
as direct explanations of hydraulic variables or contamination transport.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate inputs for a proposed SHAP analysis of a DQN "
            "sensor-placement policy. This placeholder does not execute "
            "SHAP or reproduce historical manuscript results."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the network-specific DQN checkpoint.",
    )

    parser.add_argument(
        "--states-csv",
        required=True,
        type=Path,
        help=(
            "Path to a CSV containing valid binary sensor-allocation "
            "states."
        ),
    )

    return parser.parse_args()


def validate_inputs(checkpoint_path, states_path):
    """Validate input-file existence and the state matrix format."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    if not states_path.is_file():
        raise FileNotFoundError(
            f"State matrix CSV not found: {states_path}"
        )

    try:
        states = pd.read_csv(states_path)
    except Exception as exc:
        raise ValueError(
            f"Could not read state matrix CSV: {exc}"
        ) from exc

    if states.empty:
        raise ValueError(
            "The state matrix CSV contains no data rows."
        )

    if states.shape[1] == 0:
        raise ValueError(
            "The state matrix CSV contains no feature columns."
        )

    if states.isnull().values.any():
        raise ValueError(
            "The state matrix contains missing values."
        )

    try:
        matrix = states.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Every state-matrix feature must be numeric."
        ) from exc

    if not np.isfinite(matrix).all():
        raise ValueError(
            "The state matrix contains non-finite values."
        )

    if not np.isin(matrix, [0.0, 1.0]).all():
        raise ValueError(
            "Expected binary allocation states containing only 0 and 1."
        )

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "states_csv": str(states_path.resolve()),
        "number_of_states": int(matrix.shape[0]),
        "number_of_features": int(matrix.shape[1]),
    }


def main():
    """Validate inputs without claiming to perform SHAP analysis."""

    args = parse_args()

    try:
        report = validate_inputs(
            args.checkpoint,
            args.states_csv,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Input validation failed: {exc}", file=sys.stderr)
        return 1

    print("Input validation successful.")
    print(f"Checkpoint: {report['checkpoint']}")
    print(f"State matrix: {report['states_csv']}")
    print(f"Number of states: {report['number_of_states']}")
    print(f"Number of features: {report['number_of_features']}")

    print(
        "\nSHAP analysis has NOT been executed. "
        "This entry point only validates the supplied inputs."
    )

    print(
        "Before implementing SHAP, verify checkpoint provenance, "
        "network-specific node ordering, feature dimensions, and the "
        "approved explanation protocol."
    )

    print(
        "Do not describe any subsequently generated analysis as a "
        "reproduction of historical manuscript results unless that "
        "reproduction has been independently established."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
