# ADR 0002 — Keep curated robot assets external and versioned

- Status: Accepted
- Date: 2026-09-04

## Context

High-quality humanoid and manipulator models include meshes and model-specific licenses. Copying them into this tutorial repository creates provenance, update and repository-size problems.

## Decision

Prefer maintained upstream assets such as MuJoCo Menagerie and robosuite models. Do not commit third-party meshes or whole external repositories. A task that depends on an external robot must record upstream URL, revision/tag, license and expected model path.

Phase 2 will add an explicit asset resolver/cache for G1 rather than relying on a manually cloned directory.

## Consequences

- repository remains small;
- model updates are deliberate;
- licensing stays visible;
- offline use requires a documented asset-fetch/cache step.
