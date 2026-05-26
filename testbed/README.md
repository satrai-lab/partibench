# PartiBench — Artifact Evaluation Guide

This repository contains the testbed artifact accompanying the PartiBench paper.
It provides a complete, self-contained environment for reproducing split-inference
experiments on a local Kubernetes cluster created with [Kind](https://kind.sigs.k8s.io/).
The testbed emulates a multi-tier compute continuum (IoT / Edge / Cloud) on a single
machine without requiring dedicated hardware.

---

## Repository Structure

```
UPATRAS-IMT/
├── PartiBench/                        # Core Python package (executes inside Docker)
│   ├── benchmark.py                   # Profiles a model partition-by-partition on one node
│   ├── core.py                        # Shared inference primitives (blocks, bridges, producers)
│   ├── local_deploy.py                # Worker entry point — receives placement and runs pipeline
│   ├── place.py                       # Placement algorithms (hardcoded, greedy, DP)
│   └── tools/
│       ├── create_graph.py            # Builds the compute-continuum graph from a cluster profile
│       ├── merge_bench.py             # Merges per-node benchmark JSON files into one
│       └── setup_hardcoded.py         # Generates graph and hardcoded strategy for two-node setups
│
├── manifests/
│   ├── namespaces/                    # Kubernetes namespace definitions
│   ├── services/                      # Kubernetes Services for inter-worker communication
│   ├── benchmark/
│   │   ├── jobs/                      # Benchmark Job manifests (per tier)
│   │   ├── pvc/                       # PersistentVolumeClaims for benchmark output
│   │   └── readers/                   # Reader pods for copying benchmark results off the PVC
│   └── workers/
│       ├── two-node/                  # Worker Deployments and placement Jobs for two-node setups
│       └── three-node/                # Worker Deployments and placement Job template for three-node setups
│
├── scripts/
│   ├── setup.sh                       # Installs prerequisites and creates the Kind cluster
│   ├── kind-cluster-template.json     # Kind configuration template used by setup.sh
│   ├── cluster-profiles/              # Node layout JSON files (CPU and memory per tier)
│   │   ├── partibench-two-node-local.json
│   │   ├── partibench-two-node.json
│   │   └── partibench-three-node-auto.json
│   └── experiments/
│       ├── run.sh                     # Unified experiment runner (build, benchmark, deploy, place, reset)
│       ├── configs/
│       │   ├── partibench-two-node-hardcoded.env  # Edge + Cloud, configurable model, fixed split
│       │   ├── partibench-two-node-auto.env        # Edge + Cloud, configurable model, DP / greedy
│       │   └── partibench-three-node-auto.env      # IoT + Edge + Cloud, auto DP
│       └── partibench/
│           ├── benchmark.sh               # Runs benchmark jobs sequentially and collects results
│           ├── deploy_hardcoded_split.sh  # Generates and deploys a fixed split strategy
│           └── deploy_auto.sh             # Runs DP / greedy placement for any node topology
```

---

## How the Testbed Works

Kind creates a local Kubernetes cluster in which each worker node represents one tier
of the compute continuum. Nodes are labelled with a `testbed-role` tag (`IoT`, `Edge`,
or `Cloud`); pod `nodeSelector` rules pin workloads to the correct tier. CPU quotas set
through the cluster profile cause Kubernetes's CFS scheduler to enforce a genuine
performance gap between tiers, replicating the heterogeneous compute capacity of a real
multi-tier deployment.

The experiment workflow proceeds in four phases:

1. **Benchmark** — a profiling job runs on each node, measuring per-block execution time
   and memory for every feasible partition of the target model.
2. **Placement** — a placement algorithm (hardcoded, greedy, or DP) reads the benchmark
   data and decides which model blocks run on which node.
3. **Deployment** — worker pods receive their assigned partitions and begin serving
   inference requests in a pipelined fashion across nodes.
4. **Inference** — the pipeline executes repeatedly, printing per-block compute time and
   end-to-end latency for analysis.

---

## Memory Requirements

Memory is the primary constraint when selecting a cluster profile. The table below shows
peak memory consumption per node for each supported model.

| Model | Benchmark peak (per node) | Inference peak (per node) |
|---|---|---|
| ResNet-152 | ~5 Gi | ~5 Gi |
| VGG-16 | ~10 Gi | ~6 Gi |

Benchmark jobs run **sequentially** (one node at a time), so the host only needs to
satisfy the per-node figure at any one moment — not the sum across all nodes.

Inference workers run **concurrently**. The host must be able to satisfy all deployed
workers simultaneously.

---

## Prerequisites

### Docker

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER   # log out and back in after this
docker run hello-world          # verify
```

### Kind and kubectl

`setup.sh` installs both automatically if they are not present.
To install them manually:

```bash
# Kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.27.0/kind-linux-amd64
chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/kubectl
```

### jq

`setup.sh` installs `jq` automatically if not present. It is used to generate the Kind
cluster configuration from a cluster profile JSON file.

### Python — host side (placement only)

All benchmark and inference workloads run inside Docker. Only the placement helper
scripts run on the host and require `networkx`:

```bash
python3 -m pip install networkx
```

---

## Cluster Profiles

A cluster profile is a JSON file that defines the nodes created by `setup.sh`.
The script reads it and generates the Kind configuration automatically, computing
`system-reserved` values so that Kubernetes reports the requested CPU and memory
as allocatable on each node.

### Profile format

```json
{
  "cluster_name": "eab-pandora-testbed",
  "nodes": [
    { "name": "Controler", "role": "control-plane", "cpu": "1", "memory": "2Gi" },
    { "name": "IoT",       "role": "worker",        "cpu": "1", "memory": "AGi" },
    { "name": "Edge",      "role": "worker",        "cpu": "2", "memory": "BGi" },
    { "name": "Cloud",     "role": "worker",        "cpu": "4", "memory": "CGi" }
  ]
}
```

The `name` field becomes the `testbed-role` label used by pod node selectors.
IoT and Edge nodes are optional — remove them to create a simpler topology.

**CPU:** Each tier should have fewer CPUs than the next tier up (`IoT < Edge < Cloud`).
This causes the CFS scheduler to enforce a real performance difference between tiers.

**Memory:** Allocate at least the benchmark peak for the intended model (see the table
above) plus 300–500 MiB headroom per node for system daemons.

### Provided profiles

| Profile | Topology | Suitable for |
|---|---|---|
| `partibench-two-node-local.json` | Edge 1 CPU / 5 Gi + Cloud 2 CPU / 5 Gi | ResNet-152, memory-constrained machines |
| `partibench-two-node.json` | Edge 4 CPU / 14 Gi + Cloud 8 CPU / 14 Gi | ResNet-152, VGG-16 |
| `partibench-three-node-auto.json` | IoT 2 CPU / 14 Gi + Edge 4 CPU / 14 Gi + Cloud 8 CPU / 14 Gi | Three-tier deployments |

---

## Experiment Scenarios

Three ready-to-run configurations are provided under `scripts/experiments/configs/`.
Both two-node configurations accept runtime overrides for the model and placement strategy.

| Configuration | Topology | Default model | Placement |
|---|---|---|---|
| `partibench-two-node-hardcoded.env` | Edge + Cloud | ResNet-152 | Fixed split after block 3 |
| `partibench-two-node-auto.env` | Edge + Cloud | ResNet-152 | Automatic — Dynamic Programming |
| `partibench-three-node-auto.env` | IoT + Edge + Cloud | VGG-16 | Automatic — Dynamic Programming |

**Runtime overrides** — model and strategy can be changed without editing the config file:

```bash
# VGG-16 with automatic DP placement
PARTIBENCH_MODEL_NAME=vgg16 \
  ./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-auto.env run-place

# ResNet-152 with a fixed split after block 3
PARTIBENCH_MODEL_NAME=resnet152 SPLIT_AFTER_BLOCK=3 \
  ./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env run-place

# Greedy placement instead of DP
AUTO_PLACE_METHOD=greedy \
  ./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-auto.env run-place
```

---

## Running an Experiment

All scenarios follow the same six-step workflow. Replace `<CONFIG>` with the path to
any `.env` file under `scripts/experiments/configs/`.

```bash
# 1. Create the Kind cluster (installs Kind and kubectl if missing)
./scripts/setup.sh <path-to-cluster-profile.json>

# 2. Build the worker Docker image
./scripts/experiments/run.sh <CONFIG> build-image

# 3. Load the image into every Kind node
./scripts/experiments/run.sh <CONFIG> load-image

# 4. Collect per-node benchmark data (runs sequentially, one node at a time)
./scripts/experiments/run.sh <CONFIG> benchmark

# 5. Deploy worker pods
./scripts/experiments/run.sh <CONFIG> deploy-workers

# 6. Run placement and start inference
./scripts/experiments/run.sh <CONFIG> run-place
```

> **Note:** Always use `run.sh` to submit benchmark jobs. The script substitutes the
> correct model name at runtime; applying benchmark manifests directly with
> `kubectl apply -f` will run the wrong model and may cause OOM errors.

To re-run placement without rebuilding or re-benchmarking (e.g., to compare strategies):

```bash
./scripts/experiments/run.sh <CONFIG> reset
./scripts/experiments/run.sh <CONFIG> deploy-workers
./scripts/experiments/run.sh <CONFIG> run-place
```

---

## Placement Strategies

### Hardcoded split

The split point is fixed by setting `SPLIT_AFTER_BLOCK=N`. Blocks 0 through N execute
on the first node (Edge or IoT); the remaining blocks execute on Cloud. This is
appropriate when the desired partition is known in advance or when reproducibility
independent of benchmark timing is required.

### Greedy (locally optimal)

Assigns each block to the node that minimises the cost of placing that block, given
the current output location. Runs in O(B × P) time but is not globally optimal — a
locally cheap choice can force an expensive transmission at a later block.

### Dynamic programming (globally optimal)

Tracks the minimum accumulated cost to reach every feasible (block, node) state and
recovers the globally optimal assignment by back-tracking. Negligibly slower than greedy
for the node counts used in practice (N = 2–5). This is the recommended default.

Set the strategy via `AUTO_PLACE_METHOD` in the config or as a runtime override:

```bash
AUTO_PLACE_METHOD="dp"      # globally optimal (default)
AUTO_PLACE_METHOD="greedy"  # fast, locally optimal
```

---

## Monitoring

```bash
# Live inference logs per node
kubectl logs -n edge  deployment/edge-worker  -f
kubectl logs -n cloud deployment/cloud-worker -f
kubectl logs -n iot   deployment/iot-worker   -f   # three-node only

# Placement decision log
kubectl logs -n edge job/partibench-place

# Pod status across all namespaces
./scripts/experiments/run.sh <CONFIG> status

# All logs at once
./scripts/experiments/run.sh <CONFIG> logs
```

---

## Understanding the Output

Each inference cycle prints timing at every stage of the pipeline:

```
PRODUCER: Inference started!
BLOCK TIME: 85.5 ms        ← compute time for this node's assigned partition
...
PRODUCER: Inference took: 440.2 ms    ← end-to-end wall time including network
PRODUCER: Results:
liner, ocean liner: 97.56%
```

**BLOCK TIME** is the pure compute time for the layers assigned to that node.
**Inference took** is the total end-to-end latency including network transfer of
intermediate tensors between nodes. The difference between the two quantities
represents the communication overhead introduced by the split.

The pipeline executes **16 inferences** automatically (one every 5 seconds).
The first inference is slower due to model initialisation; inferences 2–16 represent
steady-state performance and should be used for analysis.

---

## Troubleshooting

### Benchmark pod stuck in Pending

Check that the node has sufficient allocatable CPU and memory:

```bash
kubectl describe node <node-name>
```

If resources are insufficient, increase the values in the cluster profile and recreate
the cluster with `./scripts/setup.sh`.

### Worker OOMEvicted during inference

The placement algorithm assigned all blocks to a single node, requiring the full model
to be loaded in one process. Either increase the memory allocated to that node in the
cluster profile, or use the hardcoded split to distribute the model across nodes.

### Worker stuck at "Waiting for component deployment info..."

Workers listen on port 6999 for exactly one strategy push per pod lifetime. If the
placement job already ran against a previous pod instance, reset and redeploy:

```bash
./scripts/experiments/run.sh <CONFIG> reset
./scripts/experiments/run.sh <CONFIG> deploy-workers
./scripts/experiments/run.sh <CONFIG> run-place
```
