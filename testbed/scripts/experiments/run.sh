#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/experiments/run.sh <config-file> <action>

Actions:
  recreate-cluster
  build-image
  load-image
  benchmark
  deploy-workers
  redeploy-workers
  run-place
  logs
  status
  reset
  all

The config file is a sourced bash file that defines the experiment-specific
variables used by this runner.
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

CONFIG_FILE="$1"
ACTION="$2"

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

required_vars=(
  EXPERIMENT_NAME
  CLUSTER_NAME
  IMAGE_NAME
  IMAGE_CONTEXT
  NODES_FILE
  BENCHMARK_HELPER
  PLACE_HELPER
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required config variable: ${var_name}" >&2
    exit 1
  fi
done

run_build_image() {
  docker build -t "${IMAGE_NAME}" "${PROJECT_ROOT}/${IMAGE_CONTEXT}"
}

run_load_image() {
  kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}"
}

run_recreate_cluster() {
  kind delete cluster --name "${CLUSTER_NAME}" || true
  "${PROJECT_ROOT}/scripts/setup.sh" "${NODES_FILE}"
}

run_deploy_workers() {
  for manifest in "${WORKER_MANIFESTS[@]}"; do
    kubectl apply -f "${PROJECT_ROOT}/${manifest}"
  done

  for target in "${ROLLOUT_TARGETS[@]}"; do
    namespace="${target%%:*}"
    resource="${target#*:}"
    kubectl rollout status "${resource}" -n "${namespace}"
  done
}

run_redeploy_workers() {
  run_reset
  run_deploy_workers
}

run_benchmark() {
  "${PROJECT_ROOT}/${BENCHMARK_HELPER}" "${BENCHMARK_HELPER_ARGS[@]}"
}

run_place() {
  "${PROJECT_ROOT}/${PLACE_HELPER}" "${PLACE_HELPER_ARGS[@]}"
}

run_logs() {
  for target in "${LOG_TARGETS[@]}"; do
    namespace="${target%%:*}"
    resource="${target#*:}"
    echo "=== ${namespace}:${resource} ==="
    kubectl logs -n "${namespace}" "${resource}" --tail=50 || true
  done
}

run_status() {
  for namespace in "${STATUS_NAMESPACES[@]}"; do
    echo "=== namespace/${namespace} ==="
    kubectl get pods -n "${namespace}" -o wide || true
  done
}

run_reset() {
  for target in "${RESET_JOBS[@]}"; do
    namespace="${target%%:*}"
    name="${target#*:}"
    kubectl delete job "${name}" -n "${namespace}" --ignore-not-found
  done

  for target in "${RESET_CONFIGMAPS[@]}"; do
    namespace="${target%%:*}"
    name="${target#*:}"
    kubectl delete configmap "${name}" -n "${namespace}" --ignore-not-found
  done

  for manifest in "${RESET_MANIFESTS[@]}"; do
    kubectl delete -f "${PROJECT_ROOT}/${manifest}" --ignore-not-found=true || true
  done
}

case "${ACTION}" in
  recreate-cluster)
    run_recreate_cluster
    ;;
  build-image)
    run_build_image
    ;;
  load-image)
    run_load_image
    ;;
  benchmark)
    run_benchmark
    ;;
  deploy-workers)
    run_deploy_workers
    ;;
  redeploy-workers)
    run_redeploy_workers
    ;;
  run-place)
    run_place
    ;;
  logs)
    run_logs
    ;;
  status)
    run_status
    ;;
  reset)
    run_reset
    ;;
  all)
    run_recreate_cluster
    run_build_image
    run_load_image
    run_benchmark
    run_deploy_workers
    run_place
    run_logs
    ;;
  *)
    usage
    exit 1
    ;;
esac
