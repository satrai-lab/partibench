#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/experiments/run.sh <config-file> <action>

Actions:
  build-image       Build the Docker worker image
  load-image        Load the image into every Kind node
  benchmark         Profile all nodes sequentially and collect results
  deploy-workers    Apply worker Deployments and Services
  redeploy-workers  Reset workers, then deploy fresh
  run-place         Generate strategy and push component assignments to workers
  logs              Tail logs from all worker pods
  status            Show pod status across all namespaces
  reset             Delete placement job, ConfigMap, and worker Deployments
  all               Run: build-image → load-image → benchmark → deploy-workers → run-place
EOF
}

if [[ $# -lt 2 ]]; then usage; exit 1; fi

CONFIG_FILE="$1"
ACTION="$2"

if [[ ! -f "$CONFIG_FILE" ]]; then
  if [[ -f "${PROJECT_ROOT}/${CONFIG_FILE}" ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
  else
    echo "Config file not found: $CONFIG_FILE" >&2; exit 1
  fi
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

required_vars=(EXPERIMENT_NAME CLUSTER_NAME IMAGE_NAME IMAGE_CONTEXT
               BENCHMARK_HELPER PLACE_HELPER)
for var in "${required_vars[@]}"; do
  [[ -z "${!var:-}" ]] && { echo "Missing required config variable: ${var}" >&2; exit 1; }
done

run_build_image() {
  local context="${PROJECT_ROOT}/${IMAGE_CONTEXT}"
  local dockerfile="${context}/${DOCKERFILE:-Dockerfile}"
  docker build -f "${dockerfile}" -t "${IMAGE_NAME}" "${context}"
}
run_load_image()     { kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}"; }

run_recreate_cluster() {
  kind delete cluster --name "${CLUSTER_NAME}" || true
  "${PROJECT_ROOT}/scripts/setup.sh" "${NODES_FILE}"
}

run_deploy_workers() {
  for manifest in "${WORKER_MANIFESTS[@]}"; do
    kubectl apply -f "${PROJECT_ROOT}/${manifest}"
  done
  for target in "${ROLLOUT_TARGETS[@]}"; do
    ns="${target%%:*}"; resource="${target#*:}"
    kubectl rollout status "${resource}" -n "${ns}"
  done
}

run_redeploy_workers() { run_reset; run_deploy_workers; }

run_benchmark() { "${PROJECT_ROOT}/${BENCHMARK_HELPER}" "${BENCHMARK_HELPER_ARGS[@]}"; }

run_place()     { "${PROJECT_ROOT}/${PLACE_HELPER}" "${PLACE_HELPER_ARGS[@]}"; }

run_logs() {
  for target in "${LOG_TARGETS[@]}"; do
    ns="${target%%:*}"; resource="${target#*:}"
    echo "=== ${ns}:${resource} ==="
    kubectl logs -n "${ns}" "${resource}" --tail=80 || true
  done
}

run_status() {
  for ns in "${STATUS_NAMESPACES[@]}"; do
    echo "=== namespace/${ns} ==="
    kubectl get pods -n "${ns}" -o wide || true
  done
}

run_reset() {
  for target in "${RESET_JOBS[@]:-}"; do
    ns="${target%%:*}"; name="${target#*:}"
    kubectl delete job "${name}" -n "${ns}" --ignore-not-found
  done
  for target in "${RESET_CONFIGMAPS[@]:-}"; do
    ns="${target%%:*}"; name="${target#*:}"
    kubectl delete configmap "${name}" -n "${ns}" --ignore-not-found
  done
  for manifest in "${RESET_MANIFESTS[@]:-}"; do
    kubectl delete -f "${PROJECT_ROOT}/${manifest}" --ignore-not-found=true || true
  done
}

case "${ACTION}" in
  build-image)       run_build_image ;;
  load-image)        run_load_image ;;
  recreate-cluster)  run_recreate_cluster ;;
  benchmark)         run_benchmark ;;
  deploy-workers)    run_deploy_workers ;;
  redeploy-workers)  run_redeploy_workers ;;
  run-place)         run_place ;;
  logs)              run_logs ;;
  status)            run_status ;;
  reset)             run_reset ;;
  all)
    run_build_image
    run_load_image
    run_benchmark
    run_deploy_workers
    run_place
    ;;
  *) usage; exit 1 ;;
esac
