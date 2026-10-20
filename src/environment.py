"""Graph-based placement environment for water distribution networks."""

from pathlib import Path

import networkx as nx
import numpy as np
import wntr


class SensorPlacementEnvironment:
    """Sequential sensor placement using graph-based rewards."""

    def __init__(
        self,
        network_file,
        sensor_budget,
        coverage_weight=0.70,
        centrality_weight=0.30,
        coverage_radius=2,
    ):
        network_path = Path(network_file)

        if not network_path.is_file():
            raise FileNotFoundError(
                f"Missing benchmark INP file: {network_path}"
            )

        if (
            isinstance(sensor_budget, bool)
            or not isinstance(sensor_budget, (int, np.integer))
        ):
            raise TypeError("sensor_budget must be an integer.")

        if sensor_budget < 1:
            raise ValueError("sensor_budget must be at least 1.")

        if (
            not isinstance(coverage_radius, (int, np.integer))
            or isinstance(coverage_radius, bool)
            or coverage_radius < 0
        ):
            raise ValueError(
                "coverage_radius must be a non-negative integer."
            )

        if coverage_weight < 0 or centrality_weight < 0:
            raise ValueError("Reward weights must be non-negative.")

        if coverage_weight + centrality_weight <= 0:
            raise ValueError(
                "At least one reward weight must be positive."
            )

        self.wn = wntr.network.WaterNetworkModel(str(network_path))

        self.graph = self.wn.get_graph()

        if not isinstance(self.graph, nx.Graph):
            raise TypeError(
                "WNTR did not return a NetworkX-compatible graph."
            )

        if self.graph.number_of_nodes() == 0:
            raise ValueError("The network graph contains no nodes.")

        if self.graph.number_of_edges() == 0:
            raise ValueError("The network graph contains no edges.")

        # Preserve a single node ordering for all state/action mappings.
        self.nodes = list(self.graph.nodes())
        self.node_to_index = {
            node: index for index, node in enumerate(self.nodes)
        }

        if sensor_budget > len(self.nodes):
            raise ValueError(
                f"Sensor budget ({sensor_budget}) exceeds "
                f"the number of available graph nodes ({len(self.nodes)})."
            )

        self.sensor_budget = int(sensor_budget)
        self.coverage_weight = float(coverage_weight)
        self.centrality_weight = float(centrality_weight)
        self.coverage_radius = int(coverage_radius)

        self.centrality = nx.degree_centrality(self.graph)

        self.reset()

    def reset(self):
        """Reset the environment for a new placement episode."""

        self.selected_nodes = []
        self.state = np.zeros(
            len(self.nodes),
            dtype=np.float32,
        )
        self.done = False

        return self.state.copy()

    def valid_actions(self):
        """Return indices corresponding to unselected sensor locations."""

        if self.done:
            return np.array([], dtype=int)

        return np.flatnonzero(self.state == 0)

    def coverage(self):
        """Calculate the fraction of nodes within the coverage radius."""

        covered = set()

        for node in self.selected_nodes:
            covered.update(
                nx.single_source_shortest_path_length(
                    self.graph,
                    node,
                    cutoff=self.coverage_radius,
                ).keys()
            )

        return len(covered) / len(self.nodes)

    def step(self, action):
        """Place one sensor and return state, reward, done and information."""

        if self.done:
            raise RuntimeError(
                "The episode is complete. Call reset() before another step."
            )

        if isinstance(action, bool) or not isinstance(
            action, (int, np.integer)
        ):
            raise TypeError("Action must be an integer node index.")

        action = int(action)

        if not 0 <= action < len(self.nodes):
            raise ValueError(
                f"Action {action} is outside the valid action range."
            )

        if self.state[action] != 0:
            raise ValueError(
                f"Node {self.nodes[action]} has already been selected."
            )

        selected_node = self.nodes[action]

        self.selected_nodes.append(selected_node)
        self.state[action] = 1.0

        coverage_value = self.coverage()

        mean_centrality = float(
            np.mean(
                [
                    self.centrality.get(node, 0.0)
                    for node in self.selected_nodes
                ]
            )
        )

        reward = (
            self.coverage_weight * coverage_value
            + self.centrality_weight * mean_centrality
        )

        self.done = (
            len(self.selected_nodes) >= self.sensor_budget
        )

        info = {
            "selected_nodes": [
                str(node) for node in self.selected_nodes
            ],
            "coverage": float(coverage_value),
            "mean_degree_centrality": mean_centrality,
            "sensor_count": len(self.selected_nodes),
            "sensor_budget": self.sensor_budget,
        }

        return (
            self.state.copy(),
            float(reward),
            self.done,
            info,
        )
