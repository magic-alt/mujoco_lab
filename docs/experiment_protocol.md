# Experiment protocol

## Minimum reproducibility record

Every reported experiment should preserve:

- git commit SHA;
- resolved YAML configuration;
- random seed;
- Python, MuJoCo, Gymnasium, training-framework and optional robosuite versions;
- robot model source and revision;
- training timesteps and wall-clock duration;
- hardware description;
- observation/action normalization state;
- checkpoint used for evaluation;
- evaluation seeds and episode count.

The bootstrap trainer already writes `resolved_config.yaml`, `metadata.json` and `vecnormalize.pkl` when normalization is enabled.

## Evaluation rule

Do not evaluate with the stochastic action mode used for exploration unless the experiment specifically studies stochastic policies. Keep evaluation outside the training rollout and use fixed evaluation seeds for comparisons.

For claims about learning performance, run multiple training seeds. RL returns have high variance; a single successful seed is an anecdote, not a benchmark.

## Humanoid metrics

A production G1/H1 locomotion report should include at least:

- command tracking error for forward/lateral/yaw velocity;
- fall rate / episode survival;
- distance or task progress;
- foot slip velocity during stance;
- contact timing statistics;
- joint limit violations;
- torque/action magnitude and action-rate statistics;
- robustness under pushes and randomized dynamics.

## Bimanual metrics

Use task success rate as the primary endpoint when the task has a discrete goal. Complement it with:

- time to success;
- collision/safety violations;
- grasp loss/drop rate;
- end-effector path length/action smoothness;
- return, with reward composition documented.

## Comparison discipline

When comparing algorithms or rewards, keep the environment, training budget and evaluation protocol fixed. Change one conceptual factor at a time where possible. If a change modifies simulator timestep, observation scaling, actuator limits or termination logic, treat it as a task change rather than a hyperparameter tweak.
