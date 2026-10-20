# Explainable Deep Reinforcement Learning for Sensor Placement in Water Distribution Networks

## 1. Overview

This repository contains the source code and project documentation for a graph-based Deep Q-Network (DQN) framework for sensor placement in water distribution networks (WDNs).

The framework formulates sensor placement as a sequential decision-making problem. An agent selects sensor locations from available network nodes while optimising a reward based on graph-distance coverage and degree centrality.

The project includes three benchmark water distribution networks:

- Hanoi
- Net3
- C-Town

The repository provides an implementation for conducting new DQN training experiments. It should not be interpreted as a complete reproduction package for every historical experiment or result reported in the associated manuscript.

## 2. Methodology

### 2.1 Environment representation

Each water distribution network is loaded from an EPANET INP file using the Water Network Tool for Resilience (WNTR).

The network is represented as a graph. The environment maintains a binary state vector indicating which nodes have already been selected for sensor placement.

At each step, the agent selects an available node. The selected node is marked as occupied, and the environment calculates the resulting reward.

### 2.2 Reward function

The training reward combines graph-distance coverage and mean degree centrality:

\[
R_t = 0.70 C_t + 0.30 D_t
\]

where:

- \(C_t\) is the fraction of network nodes covered within the specified graph-distance radius.
- \(D_t\) is the mean degree centrality of the selected sensor locations.
- The reward weights are 0.70 for coverage and 0.30 for degree centrality.

The default coverage radius is two graph steps.

The reward is calculated after each sensor placement. The episode return is the sum of the step rewards across the episode.

**Important:** This is a graph-based optimisation objective. It does not directly measure contamination-detection accuracy, leak-detection accuracy, hydraulic performance, or water-quality performance. These outcomes require separate evaluation procedures.

### 2.3 DQN agent

The DQN agent uses a feed-forward neural network with three hidden layers:

- 128 neurons
- 64 neurons
- 32 neurons

ReLU activation functions are used between the hidden layers. The output layer produces a Q-value for each candidate sensor location.

The implementation includes:

- Epsilon-greedy action selection
- Experience replay
- A target network
- Adam optimisation
- Mean squared error for the temporal-difference learning objective
- Masking of previously selected sensor locations
- Periodic target-network synchronisation

## 3. Repository structure

The repository is organised as follows:

```text
explainable-drl-sensor-placement/
├── networks/
│   ├── C-Town.inp
│   ├── Hanoi.inp
│   └── Net3.inp
├── src/
│   ├── dqn_agent.py
│   ├── environment.py
│   ├── explain_policy.py
│   └── train.py
├── README.md
└── requirements.txt
```

### File descriptions

| File | Purpose |
|---|---|
| `networks/Hanoi.inp` | EPANET input file for the Hanoi benchmark |
| `networks/Net3.inp` | EPANET input file for the Net3 benchmark |
| `networks/C-Town.inp` | EPANET input file for the C-Town benchmark |
| `src/dqn_agent.py` | Defines the Q-network and DQN learning agent |
| `src/environment.py` | Implements the graph-based sensor-placement environment |
| `src/train.py` | Runs a new DQN training experiment and saves its outputs |
| `src/explain_policy.py` | Validates inputs for a proposed SHAP analysis; SHAP computation is not implemented in this entry point |
| `requirements.txt` | Lists the Python dependencies |

## 4. Software requirements

The implementation uses Python and the following principal packages:

- WNTR
- PyTorch
- NumPy
- NetworkX
- Pandas
- Matplotlib
- SHAP

The package specifications are provided in `requirements.txt`.

Compatibility between the specified package versions and the Python environment should be verified before conducting experiments.

## 5. Default training configuration

The current training script uses the following default settings:

| Parameter | Default value |
|---|---:|
| Training episodes | 1,000 |
| Random seed | 1 |
| Learning rate | 0.0005 |
| Discount factor | 0.99 |
| Replay-buffer capacity | 50,000 |
| Mini-batch size | 64 |
| Initial exploration rate | 1.0 |
| Minimum exploration rate | 0.01 |
| Exploration decay factor | 0.995 |
| Target-network update interval | 100 episodes |
| Coverage reward weight | 0.70 |
| Degree-centrality reward weight | 0.30 |
| Coverage radius | 2 graph steps |

The default sensor budgets are:

| Network | Default sensor budget |
|---|---:|
| Hanoi | 3 |
| Net3 | 5 |
| C-Town | 10 |

These are defaults for new runs. They should not be interpreted as proof that every historical experiment used the same configuration.

## 6. Running a new experiment

### 6.1 Install dependencies

Create and activate a Python environment appropriate for the specified package versions.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

### 6.2 Train the DQN agent

Run the training script from the repository root.

For Hanoi:

```bash
python src/train.py --network Hanoi
```

For Net3:

```bash
python src/train.py --network Net3
```

For C-Town:

```bash
python src/train.py --network C-Town
```

By default, each command runs one new experiment using 1,000 episodes and random seed 1.

### 6.3 Configure an experiment

The training script supports custom episode counts, random seeds, sensor budgets, and output directories.

For example:

```bash
python src/train.py --network Net3 --episodes 1000 --seed 2 --budget 5
```

To specify a different output directory:

```bash
python src/train.py --network Hanoi --episodes 1000 --seed 1 --outdir results/new_runs
```

Ensure that the required benchmark INP file exists at its expected location before starting training.

## 7. Training outputs

Each new training run generates three output files:

| Output | Description |
|---|---|
| Episode-history CSV | Records episode returns, exploration rates, and available mean training losses |
| Q-network checkpoint | Stores the final online Q-network parameters |
| JSON summary | Records the network, sensor budget, seed, configuration, runtime, and training-return summaries |

By default, the files are written to:

```text
results/new_runs/
```

Output filenames identify the network, sensor budget, and random seed.

### Interpreting the saved checkpoint

The current training script saves the online network at the end of training. It does not automatically restore the checkpoint from the episode with the highest return.

Consequently, the best episode return and the final checkpoint represent different quantities.

New training outputs are new experimental results. They must not be presented as historical manuscript results without a separate, documented verification process.

## 8. Explainability and SHAP

SHAP is relevant to the broader research framework for examining the relationship between model inputs and outputs.

However, the current public entry point, `src/explain_policy.py`, is an input-validation placeholder. It checks whether the supplied checkpoint and state-matrix CSV exist and whether the state matrix contains binary numerical values.

It does not:

- Load or evaluate the DQN checkpoint
- Calculate SHAP values
- Generate SHAP plots
- Reproduce the historical SHAP analysis

The presence of the SHAP dependency in `requirements.txt` does not mean that SHAP computation is implemented in this repository.

Any future SHAP implementation should verify checkpoint provenance, network-specific node ordering, feature dimensions, background-state selection, and the explanation protocol.

SHAP explanations of binary sensor-allocation states should not automatically be interpreted as causal explanations of hydraulic behaviour or contamination transport.

## 9. Experimental interpretation and reproducibility

The repository provides code for conducting new graph-based DQN sensor-placement experiments.

The following distinctions are important when interpreting experimental results:

1. Training reward measures the implemented graph-based optimisation objective.
2. Sensor-placement outputs identify selected graph nodes.
3. Contamination and leak detection require separate evaluation procedures.
4. Computational runtime depends on the hardware, software environment, network, sensor budget, and training configuration.
5. New experiments may differ from historical results because of changes in code, dependencies, random seeds, or experimental settings.

For reproducible experiments, record the Python and dependency versions, network input files, sensor budget, random seed, reward weights, training configuration, and generated outputs.

The current repository should not be described as independently reproducing all historical manuscript results until the corresponding implementations, configurations, and experimental records have been verified.

## 10. Scope and limitations

The current implementation has the following scope and limitations:

- Sensor placement is optimised using graph-based coverage and degree centrality.
- The training reward does not directly optimise contamination or leak detection.
- The current SHAP entry point does not perform explainability analysis.
- The repository does not automatically reconstruct historical multi-seed experiments.
- Separate hydraulic, water-quality, detection, or comparative evaluations require their corresponding implementations and experimental evidence.

These limitations should be considered when interpreting the code and planning further experiments.

## 11. Research context

This repository supports research into deep reinforcement learning for sensor placement in water distribution networks.

The associated research investigates the use of graph-based optimisation and explainability methods for analysing sensor-placement policies.

The code, documentation, and available experimental evidence should be considered together when assessing reproducibility and the scope of the reported findings.

## 12. Citation

If you use this repository in academic work, cite the associated research manuscript when its bibliographic information is available.

Please distinguish results obtained from this public implementation from results reported in earlier experimental records.

## 13. License and benchmark data

Before redistributing the benchmark input files or specifying a repository license, verify the applicable permissions and licensing conditions.

A repository license should only be added when the repository owner has confirmed the rights to license the included code and data.
