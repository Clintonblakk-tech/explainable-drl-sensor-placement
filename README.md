# Explainable Deep Reinforcement Learning for Optimal Sensor Placement in Water Distribution Networks

This repository contains the official implementation of an Explainable Deep Reinforcement Learning (X-DRL) framework optimized for adaptive sensor placement within benchmark Water Distribution Networks (WDNs). 

By utilizing an action-masked Deep Q-Network (DQN) architecture, the agent optimizes multi-objective engineering constraints under a strict fiscal deployment budget.

## Core Architectural Features

*   **Decoupled State Space Formulation:** The state vector $s_t$ is restricted to a compact binary allocation grid $s_t \in \{0, 1\}^N$, eliminating streaming hydraulic solver overhead during policy selection.
*   **Algorithmic Action-Masking Layer:** Prevents duplicate sensor placement allocations by mapping a dynamic binary constraint mask vector $m_t \in \{0, 1\}^N$ directly to the Q-network policy layer, forcing invalid action probabilities to collapse to $-\infty$.
*   **Game-Theoretic Policy Transparency:** Post-hoc interpretability is achieved via a localized SHAP (SHapley Additive exPlanations) framework, mapping structural features to model decision transparency.

---

## Repository Structure

```text
├── README.md               # Execution guidelines and project documentation
├── requirements.txt         # Minimal dependency tracking file
└── src/
    ├── environment.py       # Custom multi-objective WDN environment wrapper
    ├── dqn_agent.py         # Action-masked Deep Q-Network implementation
    ├── train.py             # Execution pipeline for policy optimization
    └── explain_policy.py    # Post-hoc game-theoretic interpretability engine
Installation & Setup
Clone the repository directly to your workspace:
git clone [https://github.com/Clintonblakk-tech/explainable-drl-sensor-placement.git](https://github.com/Clintonblakk-tech/explainable-drl-sensor-placement.git)
cd explainable-drl-sensor-placement
Install the necessary mathematical modeling, deep learning, and interpretability dependencies:
pip install -r requirements.txt
Execution Guidelines
1. Training the Optimization Agent
To initiate the training pipeline on a specific network benchmark configuration (e.g., net3, hanoi, or ctown), execute the core orchestration script:
python src/train.py --network net3 --episodes 500
Generating Game-Theoretic Interpretability Profiles (SHAP)
To evaluate model decisions and extract localized hydraulic feature attributions, run the post-hoc explanation module:
python src/explain_policy.py --network net3
Code and Data Availability
All custom environment configurations, structural frameworks, and core execution scripts developed for this study are fully available in this repository. Benchmark network hydraulic properties are simulated utilizing the standard WNTR framework interfaces.
