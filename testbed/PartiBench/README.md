# PartiBench — Package Overview

This directory contains the core Python package that runs inside the worker Docker image,
along with the placement tooling that runs on the host.

## Core Modules

| File | Role |
|---|---|
| `core.py` | Shared inference primitives: model block execution, tensor transport, pipeline bridges |
| `benchmark.py` | Per-node profiling — measures execution time and memory for every feasible partition of the target model |
| `place.py` | Placement algorithms: hardcoded split, greedy, and dynamic programming |
| `local_deploy.py` | Worker entry point — receives a deployment plan over a socket and launches the assigned pipeline components |

## Tools (host side)

| File | Role |
|---|---|
| `tools/create_graph.py` | Builds the compute-continuum graph from a cluster profile JSON file |
| `tools/merge_bench.py` | Merges per-node benchmark JSON outputs into a single file for placement |
| `tools/setup_hardcoded.py` | Generates the compute-continuum graph and hardcoded strategy file for two-node deployments |

## Generated Files

The tools write their outputs to `generated/`. This directory is created
automatically and is not tracked by version control.

## Container

| File | Role |
|---|---|
| `Dockerfile` | Worker image used by the Kubernetes automation |
| `requirements.txt` | Python dependencies for the PartiBench runtime and tools |
