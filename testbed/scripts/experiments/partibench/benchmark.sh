#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/experiments/partibench/benchmark.sh <config-file> [output-file]

The config file must define:
  BENCHMARK_TARGETS
  BENCHMARK_JOB_MANIFESTS
  BENCHMARK_READER_MANIFESTS

Optional:
  PARTIBENCH_MODEL_NAME
  BENCHMARK_OUTPUT_DIR
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CONFIG_FILE="$1"
OUTPUT_FILE="${2:-/tmp/partibench-benchmark.json}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  if [[ -f "${PROJECT_ROOT}/${CONFIG_FILE}" ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
  else
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${PROJECT_ROOT}/manifests/benchmark/output}"
PARTIBENCH_MODEL_NAME="${PARTIBENCH_MODEL_NAME:-resnet152}"
mkdir -p "${OUTPUT_DIR}"

if [[ ${#BENCHMARK_TARGETS[@]} -eq 0 ]]; then
  echo "BENCHMARK_TARGETS is empty." >&2
  exit 1
fi

if [[ ${#BENCHMARK_TARGETS[@]} -ne ${#BENCHMARK_JOB_MANIFESTS[@]} ]] || [[ ${#BENCHMARK_TARGETS[@]} -ne ${#BENCHMARK_READER_MANIFESTS[@]} ]]; then
  echo "BENCHMARK_TARGETS, BENCHMARK_JOB_MANIFESTS, and BENCHMARK_READER_MANIFESTS must have the same length." >&2
  exit 1
fi

kubectl apply -f "${PROJECT_ROOT}/manifests/benchmark/pvc"

for index in "${!BENCHMARK_TARGETS[@]}"; do
  target="${BENCHMARK_TARGETS[$index]}"
  job_manifest="${PROJECT_ROOT}/${BENCHMARK_JOB_MANIFESTS[$index]}"
  reader_manifest="${PROJECT_ROOT}/${BENCHMARK_READER_MANIFESTS[$index]}"

  namespace="${target}"
  job_name="benchmark-${target}"
  reader_pod_name="benchmark-output-reader-${target}"
  temp_job_manifest="$(mktemp)"

  kubectl delete job "${job_name}" -n "${namespace}" --ignore-not-found
  sed "s/\"vgg16\"/\"${PARTIBENCH_MODEL_NAME}\"/" "${job_manifest}" > "${temp_job_manifest}"
  kubectl apply -f "${temp_job_manifest}"
  rm -f "${temp_job_manifest}"

  echo "Waiting for ${job_name} job to complete..."
  kubectl wait --for=condition=complete "job/${job_name}" -n "${namespace}" --timeout=3600s

  kubectl delete pod "${reader_pod_name}" -n "${namespace}" --ignore-not-found
  kubectl apply -f "${reader_manifest}"
  kubectl wait --for=condition=Ready "pod/${reader_pod_name}" -n "${namespace}" --timeout=120s

  kubectl cp \
    "${namespace}/${reader_pod_name}:/output/${target}.json" \
    "${OUTPUT_DIR}/${target}.json"
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/PartiBench/tools/merge_bench.py" --bench-dir "${OUTPUT_DIR}"
cp "${OUTPUT_DIR}/benchmark.json" "${OUTPUT_FILE}"

echo "Merged benchmark for model ${PARTIBENCH_MODEL_NAME} saved to ${OUTPUT_FILE}"
