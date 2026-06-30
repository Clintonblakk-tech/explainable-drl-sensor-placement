# Explainable Deep Reinforcement Learning for Optimal Sensor Placement and Decision Support in Water Distribution Networks

This repository contains the official Python implementation of the Explainable Deep Reinforcement Learning (X-DRL) framework for optimizing sensor allocations in municipal water infrastructure. By coupling an action-masked Deep Q-Network (DQN) with game-theoretic SHapley Additive exPlanations (SHAP), this framework balances complex hydraulic monitoring objectives while ensuring total decision transparency for water utility operators.

##  Framework Highlights
* **Multi-Objective Reward Shaping:** Dynamically optimizes for leak isolation metrics, chemical contamination containment, network topological centrality, and installation cost boundaries.
* **Algorithmic Safety Layer:** Incorporates an automated action-masking layer directly into the policy step to strictly eliminate redundant or duplicate node selections.
* **Policy Explainability:** Deploys `shap.DeepExplainer` to peel back the black-box layers of the deep neural network, mapping the structural and hydraulic drivers of each sensor placement decision.

##  Repository Structure
```text
├── networks/               # Public domain network configuration files (.inp)
│   ├── hanoi.inp
│   ├── net3.inp
│   └── ctown.inp
├── src/                    # Core Python source code
│   ├── dqn_agent.py        # DQN architecture and action-masking logic
│   ├── environment.py      # WNTR environment wrapper and reward formulation
│   ├── explain_policy.py   # SHAP post-hoc feature attribution pipeline
│   ├── train.py            # Model training execution script
│   └── evaluate.py         # Greedy policy extraction and validation script
├── requirements.txt        # Python package dependencies
└── README.md               # Repository documentation
1.Clone the Repository:
git clone [https://github.com/Clintonblakk-tech/explainable-drl-sensor-placement.git](https://github.com/Clintonblakk-tech/explainable-drl-sensor-placement.git)
cd explainable-drl-sensor-placement
2. Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Dependencies:
pip install -r requirements.txt
Running the Framework
To execute model training for a specific benchmark network:
python src/train.py --network net3 --episodes 500
python src/explain_policy.py --network net3
Data Attribution & Licensing
Hydraulic Engine: This project leverages the open-source Water Network Tool for Resilience (WNTR) framework to interface with EPANET simulation protocols.

Network Benchmarks: The benchmark network layout files (.inp) included in the networks/ folder are standard, public-domain engineering benchmarks originally developed by the United States Environmental Protection Agency (US EPA) and are redistributed here purely for scientific replication and validation purposes.

License: This source code is released under the open-source MIT License.
