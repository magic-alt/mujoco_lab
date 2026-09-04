# 04 — AI-coding workflow for robotics research

AI coding is most useful here when it reduces mechanical work **without outsourcing experiment judgment**.

## 1. Start from an executable specification

Before asking an agent to code a new task, write an issue containing:

- robot/model source and revision;
- task statement;
- observation contract;
- action/control contract;
- reward terms;
- reset/termination behavior;
- metrics;
- tests and acceptance gates.

This prevents an agent from inventing convenient semantics halfway through implementation.

## 2. Ask for one vertical slice

Example sequence for G1:

1. model resolver + model-inspection test;
2. standing environment with zero/random action smoke tests;
3. observation/action builder;
4. decomposed rewards with unit tests;
5. PPO wiring and short smoke train;
6. multi-seed training report;
7. domain randomization only after nominal locomotion works.

Each slice should be reviewable independently.

## 3. Use tests as agent boundaries

Robotics tests should include more than Python type/shape checks. Add invariants such as:

- named joint/site exists;
- actuator count matches controlled DoFs;
- nominal pose is inside limits;
- zero action maps to nominal target;
- action ±1 maps to expected physical range;
- reward components are finite;
- fall termination triggers under a synthetic fallen state where possible;
- reset with a fixed seed is repeatable within upstream guarantees.

## 4. Make source verification explicit

When an AI agent integrates MuJoCo, robosuite, Gymnasium, Menagerie or MJX APIs, require it to verify the current official docs/source. API hallucination is especially costly in simulation code because an incorrect default can still run.

## 5. Separate code review from experiment review

A PR can be code-correct and research-wrong. Review twice:

**Software review** — interfaces, tests, dependencies, error handling, docs.

**Experiment review** — reward incentives, action scale, termination, contacts, training budget, seeds, metrics, failure modes.

## 6. Preserve failed hypotheses

If a reward term causes hopping, foot skating or a local optimum, record the experiment and why it failed. Do not let an agent erase the trail by replacing weights until one video looks good. Negative evidence is part of the project knowledge base.

## 7. Definition of done for an AI-generated PR

- diff is scoped to one issue;
- tests pass;
- tutorial/config docs match code;
- new external APIs have source links;
- experiment claims identify what was actually run;
- no unverified “works” language;
- `docs/development-log.md` is updated.
