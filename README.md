Explainable Deep Reinforcement Learning for Sensor Placement in Water Distribution Networks

Research manuscript: Explainable Deep Reinforcement Learning for Sensor Placement in Water Distribution Networks: A Benchmark-Based Evaluation

1. Overview

This repository contains the source code, experimental configurations, and retained results associated with a graph-based Deep Q-Network (DQN) framework for sensor placement in water distribution networks (WDNs).

The study investigates how deep reinforcement learning can support sensor-placement optimisation by combining graph-distance coverage and degree centrality within a multi-objective reward function. SHapley Additive exPlanations (SHAP) is employed for post-hoc analysis of the learned placement policy.

The framework is evaluated using three benchmark water distribution networks:

Hanoi
Net3
C-Town

The experimental programme examines sensor-placement optimisation, the effect of sensor budget on optimisation performance, computational requirements, and the interpretability of the learned policy.

Contamination and leak detection are treated as separate engineering evaluation tasks. They are not components of the principal DQN training reward.

Important distinction: A higher DQN training reward indicates improved performance against the specified graph-based optimisation objective. It does not, by itself, establish improved contamination detection, leak detection, or hydraulic performance.

2. Research Objectives

The study addresses four principal objectives:

Develop a DQN-based approach for optimising sensor placement in water distribution networks.
Evaluate the effect of sensor budget on DQN optimisation performance.
Examine the computational burden associated with increasing the sensor budget.
Investigate the interpretability of the learned placement policy using SHAP.

A separate engineering evaluation examines contamination-detection performance under a specified Net3 simulation protocol.

The study distinguishes three types of evidence:

Optimisation evidence: Training rewards and final rewards obtained from DQN experiments.
Computational evidence: Recorded training runtimes across networks and sensor budgets.
Interpretability and engineering evidence: SHAP-based policy analysis and separately evaluated contamination or leak-detection outcomes.

These evidence categories are reported separately because they measure different aspects of the framework.

3. Principal Experimental Configuration

The principal DQN experiments use the following network-specific configurations.

Benchmark network	Sensor budget	Random seeds	Training episodes per run
Hanoi	3	1–10	1,000
Net3	5	1–10	1,000
C-Town	10	1–10	1,000

Ten random seeds are used for each network to examine variability in optimisation performance.

These are the principal benchmark configurations. Additional sensor-budget sensitivity experiments use different budgets and are described separately in Section 4.2.

### 3.1 DQN configuration

The DQN architecture consists of three hidden layers containing 128, 64, and 32 neurons, respectively, with ReLU activation functions and a linear output layer.

The principal training parameters implemented in the public repository are:

| Parameter | Value |
|---|---:|
| Training episodes per run | 1,000 (default) |
| Random seed | 1 (default in the training script) |
| Hidden-layer architecture | 128–64–32 |
| Learning rate | 0.0005 |
| Discount factor | 0.99 |
| Replay-buffer capacity | 50,000 |
| Batch size | 64 |
| Initial exploration rate | 1.0 |
| Minimum exploration rate | 0.01 |
| Exploration decay factor | 0.995 |
| Target-network update interval | 100 episodes |

The training script accepts a network, number of episodes, random seed, and optional sensor budget as command-line arguments. It runs one training experiment per invocation; the ten-seed benchmark requires separate runs with the corresponding seeds.

The implementation updates the target network by copying the online network parameters every 100 episodes. The replay buffer and target network can be enabled or disabled through the DQN agent configuration.

These settings describe the current public implementation. They should not be assumed to reproduce every historical experiment unless the relevant configuration and outputs have been independently verified.

3.2 Reward formulation

The principal DQN training objective combines graph-distance coverage and degree centrality:

$$ R(s,a)=0.70C(s,a)+0.30H(s,a) $$

where:

\(C(s,a)\) represents the graph-distance coverage score.
\(H(s,a)\) represents the mean degree centrality of the selected sensor locations.
The coefficients 0.70 and 0.30 determine the relative contributions of coverage and centrality.

Coverage is evaluated using a two-hop graph-distance criterion. A network node is considered covered when it is within two graph edges of at least one selected sensor.

The reward is therefore topological rather than hydraulic. It evaluates network structure and sensor allocation rather than directly modelling contaminant transport or leak propagation.

Contamination and leak-detection outcomes are not included in the principal training reward. They must be evaluated independently before drawing conclusions about engineering detection performance.

4. Experimental Evaluation
4.1 Sensor-placement optimisation

DQN optimisation performance is evaluated using the best reward achieved during training and the final reward.

Results are summarised across ten random seeds for each principal benchmark configuration.

The best training reward and final reward represent different quantities:

Best reward: The highest recorded training reward during a run.
Final reward: The reward associated with the final training episode.

Neither metric should be interpreted directly as a contamination-detection rate or a measure of hydraulic accuracy.

4.2 Sensor-budget sensitivity

Additional experiments investigate the effect of sensor budget on DQN optimisation performance.

The evaluated configurations are:

Benchmark network	Sensor budgets
Hanoi	2, 3, and 5 sensors
Net3	3, 5, and 8 sensors
C-Town	5, 10, and 15 sensors

The experiments examine changes in optimisation reward and computational requirements as the number of available sensors increases.

Paired comparisons across matched random seeds are used to evaluate changes in optimisation reward between adjacent sensor budgets.

The optimisation analysis is based on the graph-based training objective. An increase in training reward does not necessarily imply a corresponding increase in engineering detection performance.

4.3 Computational performance

Training runtime is examined across the benchmark networks and sensor-budget configurations.

The analysis investigates how recorded computational requirements change as the sensor budget increases.

Runtime comparisons should be interpreted within the context of the evaluated implementations, hardware, training configuration, and experimental conditions. They do not automatically establish general computational-complexity advantages over other optimisation methods.

### 4.4 Explainability analysis

The wider research project uses SHAP for post-hoc analysis of the learned DQN policy. The analysis investigates the contribution of binary sensor-allocation state features to model outputs. It does not establish hydraulic causality or directly explain physical variables such as pressure, demand, or contaminant concentration.

The current public repository includes `src/explain_policy.py` as a placeholder command-line entry point. This file does not currently execute SHAP calculations or regenerate the historical manuscript results.

Consequently, the historical SHAP findings should be interpreted using the analysis and evidence documented in the research manuscript. Reproducing those findings from the public repository requires implementation of the approved SHAP analysis procedure and access to the appropriate model checkpoints and valid sampled states.
4.5 Engineering evaluation

Contamination and leak detection are evaluated separately from the graph-based training objective.

The engineering evaluation is intended to distinguish topological optimisation performance from performance under a specified detection protocol.

Detection outcomes depend on the evaluation configuration, including the contamination-source scenarios, sensor locations, simulation settings, and detection threshold.

Consequently, engineering conclusions should be limited to the scenarios and retained evidence available for the corresponding experiment.

4.6 Corrected Net3 contamination-detection sensitivity experiment

A corrected Net3 experiment examined contamination-detection performance under sensor budgets of 3, 5, and 8 sensors.

The documented experimental configuration comprises:

Parameter	Configuration
Network	Net3
Sensor budgets	3, 5, and 8
Random seeds per budget	10
Training episodes per run	1,000
Coverage reward weight	0.70
Centrality reward weight	0.30
Replay buffer	Enabled
Target network	Enabled
Contamination-source scenarios	First 20 specified Net3 junctions
Configured source concentration	100 mg/L
Simulation duration	24 hours
Hydraulic, quality, and reporting timestep	1 hour
Water-quality mode	CHEMICAL
Configured detection threshold	Greater than 0.01 kg/m³

The configured source concentration and detection threshold describe the experimental settings. Because the generated EPANET inputs and raw water-quality outputs were not preserved, the exported concentration measurements and their units could not be independently verified by reproducing the underlying simulations.

4.6.1 Reported detection results

The retained corrected run-level records report the following results:

Sensor budget	Total event evaluations	Detected events	Detection rate
3	200	167	83.5%
5	200	172	86.0%
8	200	174	87.0%
Total	600	513	85.5%

Each budget comprises ten runs with 20 contamination-source scenarios per run.

The reported detection rate increases from 83.5% at three sensors to 87.0% at eight sensors, an observed increase of 3.5 percentage points.

However, paired comparisons across matched seeds did not remain statistically significant after Holm correction. The results therefore indicate a modest observed improvement under the tested protocol but do not establish a statistically robust general relationship between sensor budget and contamination-detection performance.

4.6.2 Reconciliation and reproducibility limitations

The retained corrected event-level and run-level records report consistent detected-event totals across the 30 budget–seed combinations, covering 600 event evaluations.

This record-level agreement supports consistency between the retained result summaries. It does not independently reproduce the underlying EPANET simulations.

The generated EPANET inputs and raw water-quality outputs required to independently verify the exported concentration measurements were not preserved in the available experimental archive.

Accordingly:

The reported detection rates describe the retained corrected experimental records.
The agreement of event-level and run-level detection totals should not be interpreted as independent validation of the underlying water-quality simulations.
The configured concentration threshold and source concentration should not be presented as independently verified measurements.
The results should not be generalised beyond the evaluated network, budgets, source scenarios, and simulation protocol without additional validation.

These limitations distinguish the available record-level evidence from full simulation reproducibility.

## 5. Repository Structure

The GitHub repository is organised as follows:

```text
.
├── networks/
├── src/
│   ├── dqn_agent.py
│   ├── environment.py
│   ├── explain_policy.py
│   └── train.py
├── README.md
└── requirements.txt

6. Reproducibility and Interpretation

Reproducing the reported experiments requires more than the DQN architecture alone. The relevant network files, training parameters, sensor budgets, random seeds, reward weights, and evaluation protocols must be consistent with the corresponding experiment.

For the principal benchmark experiments, the documented configuration uses ten random seeds and 1,000 training episodes per run.

The corrected Net3 contamination-detection sensitivity experiment additionally depends on its contamination-source scenarios, water-quality settings, detection threshold, and event-evaluation procedure.

The following distinctions are important when reproducing or interpreting the results:

Training reproducibility: Reproducing the DQN optimisation process with the corresponding configuration.
Result-record consistency: Confirming that retained event-level and run-level records agree.
Simulation reproducibility: Reconstructing the underlying hydraulic and water-quality simulations from their original inputs and settings.
Engineering validation: Establishing that detection outcomes are reliable under a specified and independently verifiable evaluation protocol.

These are related but distinct requirements. Agreement between retained result files does not, by itself, establish full simulation reproducibility.

Where required input files, raw outputs, or run-specific configuration records are unavailable, the resulting limitation should be stated explicitly rather than implying that the experiment has been independently reproduced.

7. Limitations

The following limitations should be considered when interpreting the study.

Topological reward. The principal DQN reward combines graph-distance coverage and degree centrality. It does not directly optimise hydraulic performance or contamination-detection accuracy.

Network and budget dependence. Results are obtained from three benchmark networks and the specified sensor-budget configurations. They should not automatically be generalised to all water distribution networks.

Explainability stability. SHAP-based feature importance describes the model and states examined. Differences between rankings may limit the strength of claims about stable or universally important sensor locations.

Engineering evaluation. Contamination and leak detection are separate from the principal training objective. Their performance must be assessed using the corresponding engineering evaluation protocol.

Net3 simulation reproducibility. The retained corrected Net3 event-level and run-level records report consistent detection totals, but the generated EPANET inputs and raw water-quality outputs needed to independently reproduce and verify the concentration measurements were not preserved.

Computational comparisons. Runtime results depend on the implementations and experimental settings used. They should not be interpreted as universal comparisons of algorithmic complexity without matched computational budgets and equivalent evaluation conditions.

8. Reporting Principles

The repository and associated manuscript distinguish the following claims:

Optimisation claims are supported by the retained DQN reward results.
Computational claims are supported by the recorded runtime measurements.
Explainability claims are based on the SHAP analysis and the comparisons performed.
Engineering detection claims are limited to the retained detection records and the documented evaluation protocol.
Reproducibility claims distinguish record-level consistency from independent reproduction of the underlying simulations.

These distinctions are intended to ensure that conclusions remain proportional to the available experimental evidence.

9. Research Manuscript

The repository accompanies the manuscript:

Explainable Deep Reinforcement Learning for Sensor Placement in Water Distribution Networks: A Benchmark-Based Evaluation.

The manuscript provides the detailed methodology, experimental results, statistical analyses, interpretation, and limitations associated with the study.

For interpretation of individual results, the manuscript's experiment-specific methods and results should be consulted alongside the corresponding retained code and experimental records.
