#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
TOOLS_DIR="${PROJECT_ROOT}/tools"
GENERATED_DIR="${PROJECT_ROOT}/generated"

BENCH_JSON="${1:-/tmp/partibench-benchmark.json}"
SPLIT_AFTER_BLOCK="${2:-6}"
PARTIBENCH_MODEL_NAME="${3:-vgg16}"
NAMESPACE="edge"
CONFIGMAP_NAME="partibench-place-config"
JOB_NAME="partibench-place"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PLACE_JOB_MANIFEST="${PROJECT_ROOT}/manifests/workers/place-job.yaml"

if [[ ! -f "$BENCH_JSON" ]]; then
  echo "Benchmark JSON not found: $BENCH_JSON" >&2; exit 1
fi

mkdir -p "${GENERATED_DIR}"

# Verify Python can import networkx (needed by setup_hardcoded.py)
if ! "${PYTHON_BIN}" -c "import networkx" >/dev/null 2>&1; then
  echo "Missing Python dependency 'networkx'. Install with:" >&2
  echo "  ${PYTHON_BIN} -m pip install networkx" >&2
  exit 1
fi

# Generate cc_graph.gml, hs.txt, and net_rules/ TSVs for the 2-node K8s setup
"${PYTHON_BIN}" "${TOOLS_DIR}/setup_hardcoded.py" \
  --graph "${GENERATED_DIR}/cc_graph.gml" \
  --hs    "${GENERATED_DIR}/hs.txt" \
  --bench "${BENCH_JSON}" \
  --model "${PARTIBENCH_MODEL_NAME}" \
  --split-after-block "${SPLIT_AFTER_BLOCK}"

# ── net_rules ConfigMaps ───────────────────────────────────────────────────────
# Each worker pod mounts its own net_rules.tsv so core.py can build delays_dict
# with the correct K8s service DNS names as keys.
# Workers declare the volume as optional=true, so they start even if this
# ConfigMap doesn't exist yet (using the image's empty fallback).
# After we create the ConfigMaps here we restart the workers so they pick up
# the real delay tables before place.py connects to them.

echo "Creating net_rules ConfigMaps..."
kubectl delete configmap edge-net-rules   -n edge  --ignore-not-found
kubectl delete configmap cloud-net-rules  -n cloud --ignore-not-found

kubectl create configmap edge-net-rules  -n edge \
  --from-file=net_rules.tsv="${GENERATED_DIR}/net_rules/edge_rules.tsv"

kubectl create configmap cloud-net-rules -n cloud \
  --from-file=net_rules.tsv="${GENERATED_DIR}/net_rules/cloud_rules.tsv"

echo "Restarting workers to pick up net_rules ConfigMaps..."
kubectl rollout restart deployment/edge-worker  -n edge
kubectl rollout restart deployment/cloud-worker -n cloud
kubectl rollout status  deployment/edge-worker  -n edge  --timeout=120s
kubectl rollout status  deployment/cloud-worker -n cloud --timeout=120s

# ── Placement ConfigMap + Job ──────────────────────────────────────────────────
kubectl delete job       "${JOB_NAME}"       -n "${NAMESPACE}" --ignore-not-found
kubectl delete configmap "${CONFIGMAP_NAME}" -n "${NAMESPACE}" --ignore-not-found

kubectl create configmap "${CONFIGMAP_NAME}" -n "${NAMESPACE}" \
  --from-file=cc_graph.gml="${GENERATED_DIR}/cc_graph.gml" \
  --from-file=hs.txt="${GENERATED_DIR}/hs.txt" \
  --from-file=benchmark.json="${BENCH_JSON}"

# Substitute model name in the Job manifest at runtime
TEMP_JOB="$(mktemp)"
sed "s/vgg16/${PARTIBENCH_MODEL_NAME}/g" "${PLACE_JOB_MANIFEST}" > "${TEMP_JOB}"
kubectl apply -f "${TEMP_JOB}"
rm -f "${TEMP_JOB}"

echo "Waiting for ${JOB_NAME} to complete..."
kubectl wait --for=condition=complete "job/${JOB_NAME}" -n "${NAMESPACE}" --timeout=180s

echo "Placement job logs:"
kubectl logs "job/${JOB_NAME}" -n "${NAMESPACE}"
