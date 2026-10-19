# Explainable Deep Reinforcement Learning for Sensor Placement in Water Distribution Networks

**Research manuscript:** Explainable Deep Reinforcement Learning for Sensor Placement in Water Distribution Networks: A Benchmark-Based Evaluation

## 1. Overview

This repository contains the source code and experimental configuration for a graph-based Deep Q-Network (DQN) framework for sensor placement in water distribution networks.

The study investigates how deep reinforcement learning can support the selection of sensor locations by combining graph-distance coverage and degree centrality in a multi-objective reward function. SHAP is used for post-hoc analysis of the learned policy.

The framework is evaluated using three benchmark water distribution networks:

- Hanoi
- Net3
- C-Town

The experimental design examines sensor placement, optimisation performance, computational requirements, and the interpretation of the learned policy.

## 2. Research Objectives

The experiment addresses four principal objectives:

1. Develop a DQN-based approach for optimising sensor placement in water distribution networks.
2. Evaluate the effect of sensor budget on DQN optimisation performance.
3. Examine the computational burden associated with increasing the sensor budget.
4. Investigate the interpretability of the learned placement policy using SHAP.

Contamination and leak detection are treated as separate engineering evaluation tasks rather than components of the DQN training reward.

## 3. Principal Experimental Configuration

The principal DQN experiments use the following network-specific configurations.

| Benchmark network | Sensor budget | Random seeds | Training episodes per run |
|---|---:|---:|---:|
| Hanoi | 3 | 1–10 | 1,000 |
| Net3 | 5 | 1–10 | 1,000 |
| C-Town | 10 | 1–10 | 1,000 |

Each network is evaluated using ten random seeds to examine variation in optimisation performance.

### 3.1 DQN configuration

The principal configuration uses a deep Q-network with three hidden layers containing 128, 64, and 32 neurons, respectively, with ReLU activations and a linear output layer.

The principal training parameters are:

| Parameter | Value |
|---|---:|
| Training episodes | 1,000 |
| Random seeds | 10 |
| Hidden-layer architecture | 128–64–32 |
| Learning rate | 0.0005 |
| Discount factor | 0.99 |
| Replay-buffer capacity | 50,000 |
| Batch size | 64 |
| Initial exploration rate | 1.0 |
| Minimum exploration rate | 0.01 |
| Exploration decay factor | 0.995 |
| Target-network update interval | 100 episodes |

### 3.2 Reward formulation

The DQN agent is trained using a weighted reward function:

`R = 0.70 × C + 0.30 × H`

where:

- `C` represents graph-distance coverage.
- `H` represents the mean degree centrality of the selected sensor locations.

Coverage is evaluated using a two-hop graph-distance criterion. A network node is considered covered when it is within two graph edges of at least one selected sensor.

The coverage measure is topological rather than hydraulic. Contamination and leak-detection outcomes are not included in the training reward.

## 4. Experimental Evaluation

The experiments examine the following aspects of the proposed framework.

### 4.1 Sensor-placement optimisation

DQN training is evaluated using the best reward achieved during training and the final reward. Results are summarised across the ten random seeds for each benchmark network.

### 4.2 Sensor-budget sensitivity

Additional experiments examine the effect of changing the number of sensors on optimisation performance.

The sensor-budget configurations are:

| Benchmark network | Sensor budgets |
|---|---|
| Hanoi | 2, 3, and 5 sensors |
| Net3 | 3, 5, and 8 sensors |
| C-Town | 5, 10, and 15 sensors |

Paired comparisons across matched random seeds are used to evaluate changes in optimisation reward between adjacent sensor budgets.

### 4.3 Computational performance

Training runtime is examined across the benchmark networks and sensor-budget configurations to assess how computational requirements change as the number of selected sensors increases.

### 4.4 Explainability analysis

SHAP is used to investigate the contribution of input features to the learned DQN policy. The analysis includes comparisons between SHAP-based feature rankings and policy-validation rankings.

The SHAP analysis is post hoc and concerns the model's learned allocation policy. It is not interpreted as evidence of hydraulic causality.

### 4.5 Engineering evaluation

Contamination and leak-detection tasks are considered separately from the graph-based training objective. This distinction allows optimisation performance to be discussed separately from engineering detection performance.

## 5. Repository Structure

```text
explainable-drl-sensor-placement/
├── networks/
├── src/
│   ├── dqn_agent.py
│   ├── environment.py
│   ├── explain_policy.py
│   └── train.py
├── README.md
└── requirements.txt
