#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
PARTIBENCH_DIR="${PROJECT_ROOT}/PartiBench"
TOOLS_DIR="${PARTIBENCH_DIR}/tools"
GENERATED_DIR="${PARTIBENCH_DIR}/generated/k8s_two_node"
BENCH_JSON="${1:-/tmp/partibench-benchmark.json}"
SPLIT_AFTER_BLOCK="${2:-6}"
PARTIBENCH_MODEL_NAME="${3:-vgg16}"
NAMESPACE="${4:-edge}"
CONFIGMAP_NAME="partibench-place-config"
JOB_NAME="partibench-place"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PLACE_JOB_MANIFEST="${PROJECT_ROOT}/manifests/workers/two-node/place-job.yaml"

if [[ ! -f "$BENCH_JSON" ]]; then
  echo "Benchmark JSON not found: $BENCH_JSON" >&2
  exit 1
fi

mkdir -p "${GENERATED_DIR}"

if ! "${PYTHON_BIN}" -c "import networkx" >/dev/null 2>&1; then
  echo "Python dependency check failed for ${PYTHON_BIN}: missing 'networkx'." >&2
  echo "Install host-side PartiBench dependencies with:" >&2
  echo "  ${PYTHON_BIN} -m pip install -r ${PARTIBENCH_DIR}/requirements.txt" >&2
  echo "Or run with a prepared interpreter, for example:" >&2
  echo "  PYTHON_BIN=/path/to/python ./scripts/experiments/run.sh <config> run-place" >&2
  exit 1
fi

"${PYTHON_BIN}" "${TOOLS_DIR}/setup_hardcoded.py" \
  --graph "${GENERATED_DIR}/cc_graph.gml" \
  --hs "${GENERATED_DIR}/hs.txt" \
  --bench "${BENCH_JSON}" \
  --model "${PARTIBENCH_MODEL_NAME}" \
  --split-after-block "${SPLIT_AFTER_BLOCK}"

kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}" --ignore-not-found
kubectl delete configmap "${CONFIGMAP_NAME}" -n "${NAMESPACE}" --ignore-not-found

kubectl create configmap "${CONFIGMAP_NAME}" -n "${NAMESPACE}" \
  --from-file=cc_graph.gml="${GENERATED_DIR}/cc_graph.gml" \
  --from-file=hs.txt="${GENERATED_DIR}/hs.txt" \
  --from-file=benchmark.json="${BENCH_JSON}"

TEMP_JOB_MANIFEST="$(mktemp)"
sed "s/- vgg16/- ${PARTIBENCH_MODEL_NAME}/" "${PLACE_JOB_MANIFEST}" > "${TEMP_JOB_MANIFEST}"
kubectl apply -f "${TEMP_JOB_MANIFEST}"
rm -f "${TEMP_JOB_MANIFEST}"

echo "Waiting for ${JOB_NAME} to complete..."
kubectl wait --for=condition=complete "job/${JOB_NAME}" -n "${NAMESPACE}" --timeout=180s

echo "Job logs:"
kubectl logs "job/${JOB_NAME}" -n "${NAMESPACE}"
