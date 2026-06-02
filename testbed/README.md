# PartiBench — Testbed (testbed)

← [Back to main README](../README.md)

Runs the same **2-node pipelined VGG16 example** described in the main README inside a local [Kind](https://kind.sigs.k8s.io/) Kubernetes cluster instead of two terminals on one machine. Each virtual node (`edge`, `cloud`) becomes a Kubernetes worker pod with its own CPU quota, namespace, and service DNS name.

---

## Repository Structure

```
testbed/
│
├── # ── Modified files (originals unchanged, see justification below) ──
├── core.py                  # Retry logic added; sim_delay made optional
├── local_deploy.py          # Auto-start inference; asset paths fixed; split receive functions
├── place.py                 # dest_ip bug fixed; mutable default fixed; K8s CLI defaults
├── merge_bench.py           # --bench-dir / --output CLI args added
│
├── # ── Unchanged files referenced via Dockerfile (no copies kept here) ──
│   benchmarking/benchmark.py       ← used directly from parent tree
│   benchmarking/memory_profiler.py ← used directly from parent tree
│   data/                           ← used directly as ./assets/ in the image
│
├── Dockerfile               # Build context = partibench/ root; COPYs from both trees
├── requirements.txt         # Same deps as main; pinned here for the image build
│
├── tools/
│   └── setup_hardcoded.py   # Generates cc_graph.gml, hs.txt, and net_rules/ TSVs
│
├── manifests/
│   ├── namespaces/          # edge, cloud Kubernetes Namespaces
│   ├── services/            # ClusterIP Services exposing ports 5000-5040 + 6999
│   ├── benchmark/
│   │   ├── pvc/             # PersistentVolumeClaims for benchmark output (50 Mi each)
│   │   ├── jobs/            # Benchmark Job manifests (one per tier)
│   │   └── readers/         # Reader pods for copying results off the PVCs
│   └── workers/
│       ├── edge-worker.yaml # Deployment — pinned to Edge node via nodeSelector
│       ├── cloud-worker.yaml
│       └── place-job.yaml   # Job that runs place.py and pushes assignments to workers
│
└── scripts/
    ├── setup.sh                          # Installs Kind/kubectl/jq; creates the cluster
    ├── kind-cluster-template.json        # Kind config skeleton
    ├── cluster-profiles/
    │   └── partibench-two-node.json      # Edge 4 CPU / 14 Gi  +  Cloud 8 CPU / 14 Gi
    └── experiments/
        ├── run.sh                        # Unified experiment runner
        ├── configs/
        │   └── partibench-two-node-hardcoded.env
        └── partibench/
            ├── benchmark.sh             # Runs benchmark Jobs and collects output
            └── deploy_hardcoded_split.sh # Generates strategy, creates ConfigMaps, runs place
```

---

## Why These Files Exist

The core Python logic is **identical** to the original. Only the minimum set of changes required to run inside Kubernetes was applied. Each modified file lives here so the originals in the parent tree are never touched.

### [core.py](core.py) — 2 changes from [../core.py](../core.py)

| Change | Reason |
|---|---|
| Added `CONNECT_RETRY_DELAY_SEC` and `CONNECT_MAX_RETRIES` constants + retry loop in `send_output` | In K8s all pods start in parallel. The receiving pod may not be listening yet when the sender first connects. Without retries the first `connect()` raises `ConnectionRefusedError` and the pipeline dies. |
| `sim_delay` moved to the end of `receive_input` and `send_output` and made optional (`sim_delay=False`) | The original positional `sim_delay` forced every caller to pass it explicitly. Making it a keyword argument with a safe default means existing callers in the original tree are unaffected, and testbed callers only pass it where they need it. |

Everything else — `delays_dict`, `time.sleep`, all model and partitioning math — is **byte-for-byte identical** to the original.

### [local_deploy.py](local_deploy.py) — 6 changes from [../orchestrator/local_deploy.py](../orchestrator/local_deploy.py)

| Change | Reason |
|---|---|
| `BASE_DIR` / `ASSETS_DIR` constants added | The original resolves asset paths relative to CWD. In K8s the benchmark step changes CWD to `/output`; using `__file__`-relative paths guarantees `ship.png` and `image_net_labels.json` are always found regardless of CWD. |
| `--ip` argument removed; always binds `0.0.0.0` | The original `--ip` flag was needed to distinguish two loopback addresses on one machine. In K8s each pod already has its own network namespace, so binding `0.0.0.0` is always correct. |
| `receive_text` rewrites recv as a loop until connection closes | The original single `recv(num_bytes)` can silently truncate large deployment JSON split across TCP packets. `place.py` closes the socket after sending, so looping until `b""` reliably collects the complete message. |
| `receive_result` added (single `recv`) | The final BRIDGE keeps its results socket **open** across all 16 inferences. A close-loop would block after the first result. A single `recv(1024)` reads exactly one result string per inference. PRODUCER now calls this instead of `receive_text`. |
| `input()` replaced with automatic queue of 16 signals | Kubernetes pods have no TTY. `input()` blocks forever. The parent process queues 16 `"s"` signals automatically with 5 s sleep between each. |
| `sim_delay=True` passed to BLOCK's `send_output` and `receive_input` | BLOCK is the only component that crosses pod boundaries. Passing `sim_delay=True` activates the delay lookup in `core.py` so configured delays are honoured. BRIDGE and PRODUCER communicate locally and stay at the default `False`. |

### [place.py](place.py) — 3 changes from [../orchestrator/place.py](../orchestrator/place.py)

| Change | Reason |
|---|---|
| `produce_possible_splits` mutable default argument fixed (`partial=None, results=None`) | The original `partial=[]` is a Python anti-pattern: the list is shared across all calls, making the function return wrong results after the first invocation. |
| `_node_ip(g, node, local_node)` helper added; all `dest_ip` assignments use it | The original hardcoded `"127.0.0.1"` for no-split BLOCK destinations. In a real multi-node K8s cluster this breaks inter-pod communication. `_node_ip` returns `"127.0.0.1"` when both components are on the same pod, and the K8s service DNS name otherwise. |
| CLI defaults changed (`--start-node edge`, `--bench/--graph/--hs` point to `/config/`) | The place.py Job receives its input files from a ConfigMap mounted at `/config/`. Pointing the defaults there avoids having to pass every flag explicitly in the Job manifest. |

### [merge_bench.py](merge_bench.py) — 3 changes from [../benchmarking/merge_bench.py](../benchmarking/merge_bench.py)

| Change | Reason |
|---|---|
| `--bench-dir` and `--output` CLI arguments added | The original hardcoded `"benchmarks/nodes"`. The automation scripts need to pass the path at runtime since output lands in a different directory depending on how the benchmark Job writes to the PVC. |
| Skips `benchmark.json` when scanning the directory | If a previous run's merged output is still in the folder it would be re-read and corrupt the result. |
| `assert False` replaced with `raise ValueError` | Gives a readable error message when an unexpected file appears in the benchmark directory. |

### `net_rules.tsv` — generated by the Dockerfile, not a tracked file

`core.py` reads `net_rules.tsv` at **import time** (module level). If the file does not exist the import raises `FileNotFoundError` and nothing starts. The Dockerfile creates a header-only fallback inline with `RUN printf ...` so no extra file needs to live in the repository. Worker pods that have not yet received their node-specific ConfigMap mount use this empty file (`delays_dict` stays empty, no delays applied). Once `deploy_hardcoded_split.sh` creates the ConfigMaps and restarts the pods, `core.py` reads the real per-node delay table from the mounted file.

---

## How the Delay Simulation Works

`setup_hardcoded.py` writes `generated/net_rules/edge_rules.tsv` and `generated/net_rules/cloud_rules.tsv`. Each file lists the **K8s service DNS names** as destination IPs:

```
DEST_NAME   IP                                      BANDWIDTH   DELAY
cloud       cloud-worker.cloud.svc.cluster.local    1000mbit    1ms
```

`deploy_hardcoded_split.sh` creates a ConfigMap from each file and mounts it into the matching worker pod at `/app/net_rules.tsv`. When BLOCK calls `send_output(dest_ip="cloud-worker.cloud.svc.cluster.local", sim_delay=True)`, core.py looks up that key in `delays_dict` and sleeps for the configured duration before sending.

---

## Dependencies

### Host machine (runs scripts and placement)

```bash
# Docker (running)
# Kind — installed automatically by setup.sh if missing
# kubectl — installed automatically by setup.sh if missing
# jq — installed automatically by setup.sh if missing

pip install networkx   # only needed for tools/setup_hardcoded.py
```

### Inside the container (all installed by the Dockerfile)

```
numpy<1.24   mxnet==1.9.1   gluoncv   networkx   psutil   matplotlib
```

VGG16 weights (~500 MB) are downloaded by GluonCV on first benchmark run inside the container.

---

## Cluster Profiles

A cluster profile defines the Kind nodes created by `setup.sh`. The provided profile allocates enough memory for VGG16 benchmarking and inference:

| Profile | Edge | Cloud |
|---|---|---|
| `partibench-two-node.json` | 4 CPU / 14 Gi | 8 CPU / 14 Gi |

CPU counts create a genuine performance gap between tiers via the Linux CFS scheduler — mirroring the heterogeneous hardware of a real Computing Continuum.

---

## Running the Experiment

All commands are run from the `testbed/` directory.

### Step 1 — Create the Kind cluster

```bash
./scripts/setup.sh scripts/cluster-profiles/partibench-two-node.json
```

Installs Kind and kubectl if missing, generates the Kind config from the profile, creates the cluster, and applies the namespace manifests.

### Step 2 — Build and load the Docker image

```bash
./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env build-image
./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env load-image
```

The build context is the `partibench/` root so the Dockerfile can reference both `testbed/` (modified files) and the original source tree (`benchmarking/`, `data/`).

### Step 3 — Run PartiBench (benchmark)

Profiles all VGG16 blocks on each node sequentially (~10 min per node on VGG16):

```bash
./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env benchmark
```

Runs the `benchmark-edge` and `benchmark-cloud` Jobs, collects JSON output from the PVCs, and merges them into `/tmp/partibench-benchmark.json`.

### Step 4 — Deploy worker pods

```bash
./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env deploy-workers
```

Applies the Services and worker Deployments. Pods start listening on port 6999 for their component assignments.

### Step 5 — Run placement and start inference

```bash
./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env run-place
```

This single command:
1. Calls `setup_hardcoded.py` → generates `cc_graph.gml`, `hs.txt`, and per-node `net_rules/` TSVs
2. Creates `edge-net-rules` and `cloud-net-rules` ConfigMaps and restarts the workers
3. Creates the placement ConfigMap and launches the `partibench-place` Job
4. `place.py` sends component assignments to both pods over TCP port 6999
5. The PRODUCER on edge auto-starts 16 inferences (one every 5 s)

### Monitoring

```bash
# Live logs
./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env logs

# Pod status
./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env status
```

### Re-running with a different split point

```bash
./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env reset
./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env deploy-workers

SPLIT_AFTER_BLOCK=3 \
  ./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env run-place
```

---

## Understanding the Output

Each inference cycle prints on the edge worker:

```
PRODUCER: Inference started!
~>~<~>~<~> BLOCK TIME  312.4 ms      ← pure compute on this pod's assigned blocks
PRODUCER: Inference took: 441.7 ms   ← end-to-end wall time including inter-pod transfer
PRODUCER: Results:
container ship, container vessel: 84.32%
wreck: 5.17%
lifeboat: 3.91%
```

**BLOCK TIME** — compute only. **Inference took** − **BLOCK TIME** ≈ communication overhead (network transfer + configured delay simulation). The pipeline runs 16 times automatically; the first is slower due to model initialisation.

---

## Notes

- **Benchmark peak memory:** VGG16 requires ~10 Gi per node during benchmarking. If a node has less, the pod will be OOMEvicted — increase the cluster profile values.
- **Model weights:** Downloaded on first benchmark run into the container's writable layer (`/app/vgg16_data/`). Not cached between runs.
- **Delay simulation:** Configured via the per-node `net_rules.tsv` mounted from a ConfigMap. Bandwidth values affect the delay formula in `place.py` cost estimation only; actual bandwidth is not shaped (no `tc` in Kind). To simulate bandwidth add Linux `tc` rules or use Chaos Mesh.
- **Ports:** Component ports are allocated sequentially from 5000 by `place.py`. The Services expose ports 5000–5040 and 6999. If a strategy produces more than 40 ports, add them to the Service manifests.
