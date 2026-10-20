"""Run one new DQN training experiment.

Outputs from this script represent new experiments. They are not
automatically reproductions of historical manuscript results.
"""

import argparse
import csv
import json
import random
import sys
import time

from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

# Allow imports from the src directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from environment import SensorPlacementEnvironment
from dqn_agent import DQNAgent


FILES = {
    "Hanoi": ROOT / "networks" / "Hanoi.inp",
    "Net3": ROOT / "networks" / "Net3.inp",
    "C-Town": ROOT / "networks" / "C-Town.inp",
}

BUDGETS = {
    "Hanoi": 3,
    "Net3": 5,
    "C-Town": 10,
}


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a new DQN sensor-placement model. "
            "This script does not reconstruct historical experiments."
        )
    )

    parser.add_argument(
        "--network",
        choices=list(FILES),
        required=True,
        help="Benchmark network to use.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1).",
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help=(
            "Number of sensors to place. "
            "If omitted, the network-specific default is used."
        ),
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "results" / "new_runs",
        help="Directory for new experimental outputs.",
    )

    return parser.parse_args()


def set_random_seeds(seed):
    """Seed the random-number generators used by the training script."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise SystemExit(
            "Error: --episodes must be a positive integer."
        )

    if args.seed < 0:
        raise SystemExit(
            "Error: --seed must be a non-negative integer."
        )

    budget = (
        BUDGETS[args.network]
        if args.budget is None
        else args.budget
    )

    if budget <= 0:
        raise SystemExit(
            "Error: --budget must be a positive integer."
        )

    network_file = FILES[args.network]

    if not network_file.is_file():
        raise SystemExit(
            f"Error: benchmark network file not found: {network_file}"
        )

    set_random_seeds(args.seed)

    # Construct the environment and DQN agent.
    env = SensorPlacementEnvironment(
        network_file=network_file,
        sensor_budget=budget,
        coverage_weight=0.70,
        centrality_weight=0.30,
        coverage_radius=2,
    )

    agent = DQNAgent(
        state_dim=len(env.nodes),
        action_dim=len(env.nodes),
        lr=0.0005,
        gamma=0.99,
        replay_capacity=50000,
        batch_size=64,
        use_replay=True,
        use_target=True,
    )

    episode_rows = []
    best_episode_return = -float("inf")
    last_info = {}

    start_time = time.perf_counter()

    for episode in range(args.episodes):
        state = env.reset()
        done = False
        episode_return = 0.0
        episode_losses = []

        while not done:
            valid_actions = env.valid_actions()

            action = agent.act(
                state,
                valid_actions,
            )

            next_state, reward, done, info = env.step(action)

            # IMPORTANT:
            # remember() expects five separate arguments.
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            loss = agent.update()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_return += reward
            last_info = info

        best_episode_return = max(
            best_episode_return,
            episode_return,
        )

        agent.finish_episode(episode)

        episode_rows.append(
            {
                "episode": episode + 1,
                "episode_reward": episode_return,
                "epsilon": agent.epsilon,
                "mean_training_loss": (
                    float(np.mean(episode_losses))
                    if episode_losses
                    else None
                ),
            }
        )

    elapsed = time.perf_counter() - start_time

    # Create a new output directory without overwriting unrelated files.
    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{args.network.replace('-', '_')}"
        f"_budget{budget}"
        f"_seed{args.seed}"
    )

    history_path = args.outdir / f"{stem}_episode_history.csv"
    checkpoint_path = args.outdir / f"{stem}_qnetwork.pth"
    summary_path = args.outdir / f"{stem}_summary.json"

    # Save the episode history.
    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "episode_reward",
                "epsilon",
                "mean_training_loss",
            ],
        )

        writer.writeheader()
        writer.writerows(episode_rows)

    # Save the online Q-network parameters.
    torch.save(
        agent.online.state_dict(),
        checkpoint_path,
    )

    summary = {
        "network": args.network,
        "network_file": str(network_file.relative_to(ROOT)),
        "sensor_budget": budget,
        "seed": args.seed,
        "episodes": args.episodes,
        "learning_rate": 0.0005,
        "discount_factor": 0.99,
        "coverage_weight": 0.70,
        "centrality_weight": 0.30,
        "coverage_radius": 2,
        "replay_buffer_enabled": True,
        "target_network_enabled": True,
        "best_episode_return": best_episode_return,
        "final_episode_return": episode_rows[-1]["episode_reward"],
        "final_epsilon": agent.epsilon,
        "runtime_seconds": elapsed,
        "final_episode_nodes": last_info.get(
            "selected_nodes",
            [],
        ),
        "checkpoint_file": checkpoint_path.name,
        "history_file": history_path.name,
        "status": "new experiment; not a historical run reconstruction",
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
