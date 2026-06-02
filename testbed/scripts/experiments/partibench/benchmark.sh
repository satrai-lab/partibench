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
  BENCHMARK_TARGETS            (array of namespace/node-role names, e.g. "edge" "cloud")
  BENCHMARK_JOB_MANIFESTS      (array of Job manifest paths, one per target)
  BENCHMARK_READER_MANIFESTS   (array of reader-pod manifest paths, one per target)

Optional:
  PARTIBENCH_MODEL_NAME        (default: vgg16)
  BENCHMARK_OUTPUT_DIR         (default: manifests/benchmark/output)
EOF
}

if [[ $# -lt 1 ]]; then usage; exit 1; fi

CONFIG_FILE="$1"
OUTPUT_FILE="${2:-/tmp/partibench-benchmark.json}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  if [[ -f "${PROJECT_ROOT}/${CONFIG_FILE}" ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
  else
    echo "Config file not found: $CONFIG_FILE" >&2; exit 1
  fi
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${PROJECT_ROOT}/manifests/benchmark/output}"
PARTIBENCH_MODEL_NAME="${PARTIBENCH_MODEL_NAME:-vgg16}"
mkdir -p "${OUTPUT_DIR}"

[[ ${#BENCHMARK_TARGETS[@]} -eq 0 ]] && { echo "BENCHMARK_TARGETS is empty." >&2; exit 1; }

# Apply PVCs (idempotent)
kubectl apply -f "${PROJECT_ROOT}/manifests/benchmark/pvc"

for index in "${!BENCHMARK_TARGETS[@]}"; do
  target="${BENCHMARK_TARGETS[$index]}"
  job_manifest="${PROJECT_ROOT}/${BENCHMARK_JOB_MANIFESTS[$index]}"
  reader_manifest="${PROJECT_ROOT}/${BENCHMARK_READER_MANIFESTS[$index]}"

  namespace="${target}"
  job_name="benchmark-${target}"
  reader_pod="benchmark-output-reader-${target}"

  # Delete any previous run of this job
  kubectl delete job "${job_name}" -n "${namespace}" --ignore-not-found

  # Substitute model name at runtime so the same YAML works for any model
  temp_job="$(mktemp)"
  sed "s/vgg16/${PARTIBENCH_MODEL_NAME}/g" "${job_manifest}" > "${temp_job}"
  kubectl apply -f "${temp_job}"
  rm -f "${temp_job}"

  echo "Waiting for ${job_name} to complete (this can take ~10 min on VGG16)..."
  kubectl wait --for=condition=complete "job/${job_name}" -n "${namespace}" --timeout=3600s

  # Spin up reader pod to copy output off the PVC
  kubectl delete pod "${reader_pod}" -n "${namespace}" --ignore-not-found
  kubectl apply -f "${reader_manifest}"
  kubectl wait --for=condition=Ready "pod/${reader_pod}" -n "${namespace}" --timeout=120s

  kubectl cp "${namespace}/${reader_pod}:/output/${target}.json" "${OUTPUT_DIR}/${target}.json"
  echo "Collected ${target}.json"
done

# Merge all per-node JSONs into a single benchmark.json
"${PYTHON_BIN}" "${PROJECT_ROOT}/merge_bench.py" \
  --bench-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/benchmark.json"

cp "${OUTPUT_DIR}/benchmark.json" "${OUTPUT_FILE}"
echo "Merged benchmark written to ${OUTPUT_FILE}"
