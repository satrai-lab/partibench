#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
PARTIBENCH_DIR="${PROJECT_ROOT}/PartiBench"
TOOLS_DIR="${PARTIBENCH_DIR}/tools"

CONFIG_FILE="${1:-}"
BENCH_JSON="${2:-/tmp/partibench-benchmark.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${CONFIG_FILE}" ]]; then
  echo "Usage: ./scripts/experiments/partibench/deploy_auto.sh <config-file> [benchmark-json]" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  if [[ -f "${PROJECT_ROOT}/${CONFIG_FILE}" ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
  else
    echo "Config file not found: ${CONFIG_FILE}" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "${CONFIG_FILE}"

PARTIBENCH_MODEL_NAME="${PARTIBENCH_MODEL_NAME:-vgg16}"
AUTO_PLACE_METHOD="${AUTO_PLACE_METHOD:-dp}"
AUTO_PLACE_START_NODE="${AUTO_PLACE_START_NODE:-iot}"
AUTO_PLACE_JOB_NAMESPACE="${AUTO_PLACE_JOB_NAMESPACE:-edge}"
AUTO_PLACE_JOB_NAME="${AUTO_PLACE_JOB_NAME:-partibench-place-auto}"
AUTO_PLACE_CONFIGMAP_NAME="${AUTO_PLACE_CONFIGMAP_NAME:-partibench-auto-place-config}"
AUTO_PLACE_JOB_MANIFEST="${AUTO_PLACE_JOB_MANIFEST:-manifests/workers/three-node/place-job.yaml}"
AUTO_PLACE_JOB_NODE_SELECTOR="${AUTO_PLACE_JOB_NODE_SELECTOR:-Edge}"
AUTO_PLACE_GRAPH_PATH="${AUTO_PLACE_GRAPH_PATH:-${PARTIBENCH_DIR}/generated/three_node_auto/cc_graph.gml}"
AUTO_PLACE_COMPONENTS_OUT="${AUTO_PLACE_COMPONENTS_OUT:-/tmp/partibench-auto-components.json}"
AUTO_PLACE_STRATEGY_OUT="${AUTO_PLACE_STRATEGY_OUT:-/tmp/partibench-auto-strategy.jsonl}"

if [[ ! -f "${BENCH_JSON}" ]]; then
  echo "Benchmark JSON not found: ${BENCH_JSON}" >&2
  exit 1
fi

mkdir -p "$(dirname "${AUTO_PLACE_GRAPH_PATH}")"

if ! "${PYTHON_BIN}" -c "import networkx" >/dev/null 2>&1; then
  echo "Python dependency check failed for ${PYTHON_BIN}: missing 'networkx'." >&2
  echo "Install host-side PartiBench dependencies with:" >&2
  echo "  ${PYTHON_BIN} -m pip install -r ${PARTIBENCH_DIR}/requirements.txt" >&2
  exit 1
fi

"${PYTHON_BIN}" "${TOOLS_DIR}/create_graph.py" \
  --nodes "${PROJECT_ROOT}/${NODES_FILE}" \
  --output "${AUTO_PLACE_GRAPH_PATH}"

kubectl delete job "${AUTO_PLACE_JOB_NAME}" -n "${AUTO_PLACE_JOB_NAMESPACE}" --ignore-not-found
kubectl delete configmap "${AUTO_PLACE_CONFIGMAP_NAME}" -n "${AUTO_PLACE_JOB_NAMESPACE}" --ignore-not-found

kubectl create configmap "${AUTO_PLACE_CONFIGMAP_NAME}" -n "${AUTO_PLACE_JOB_NAMESPACE}" \
  --from-file=cc_graph.gml="${AUTO_PLACE_GRAPH_PATH}" \
  --from-file=benchmark.json="${BENCH_JSON}"

TEMP_JOB_MANIFEST="$(mktemp)"
sed \
  -e "s/__JOB_NAME__/${AUTO_PLACE_JOB_NAME}/g" \
  -e "s/__JOB_NAMESPACE__/${AUTO_PLACE_JOB_NAMESPACE}/g" \
  -e "s/__NODE_SELECTOR__/${AUTO_PLACE_JOB_NODE_SELECTOR}/g" \
  -e "s/__START_NODE__/${AUTO_PLACE_START_NODE}/g" \
  -e "s/__MODEL_NAME__/${PARTIBENCH_MODEL_NAME}/g" \
  -e "s/__CONFIGMAP_NAME__/${AUTO_PLACE_CONFIGMAP_NAME}/g" \
  -e "s/__PLACEMENT_METHOD__/${AUTO_PLACE_METHOD}/g" \
  "${PROJECT_ROOT}/${AUTO_PLACE_JOB_MANIFEST}" > "${TEMP_JOB_MANIFEST}"

kubectl apply -f "${TEMP_JOB_MANIFEST}"
rm -f "${TEMP_JOB_MANIFEST}"

echo "Waiting for ${AUTO_PLACE_JOB_NAME} to complete..."
kubectl wait --for=condition=complete "job/${AUTO_PLACE_JOB_NAME}" -n "${AUTO_PLACE_JOB_NAMESPACE}" --timeout=180s

echo "Job logs:"
kubectl logs "job/${AUTO_PLACE_JOB_NAME}" -n "${AUTO_PLACE_JOB_NAMESPACE}"
