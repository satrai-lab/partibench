#!/bin/bash
set -e

log_info()  { echo -e "\033[1;34m[INFO]\033[0m $1"; }
log_warn()  { echo -e "\033[1;33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; exit 1; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEMPLATE_FILE="${SCRIPT_DIR}/kind-cluster-template.json"
OUTPUT_FILE="${SCRIPT_DIR}/kind-cluster-config.json"

CONFIG_FILE="${1:-${SCRIPT_DIR}/cluster-profiles/partibench-two-node.json}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  if [[ -f "${PROJECT_ROOT}/$CONFIG_FILE" ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/$CONFIG_FILE"
  else
    log_error "Cluster profile not found: $CONFIG_FILE"
  fi
fi

log_info "Using cluster profile: $CONFIG_FILE"

# ── Verify Docker ──────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || log_error "Docker is not installed."
docker info >/dev/null 2>&1     || log_error "Docker is not running."

OS=$(uname -s)
ARCH=$(uname -m)
log_info "OS: $OS  ARCH: $ARCH"

# ── Install Kind if missing ────────────────────────────────────────────────────
if ! command -v kind >/dev/null 2>&1; then
  log_info "Installing Kind..."
  KIND_VERSION="v0.22.0"
  ARCH_DL="$ARCH"
  [[ "$ARCH" == "x86_64" ]]          && ARCH_DL="amd64"
  [[ "$ARCH" == "aarch64" ]]         && ARCH_DL="arm64"
  [[ "$OS"   == "Linux"  ]]          && curl -Lo kind "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-${ARCH_DL}"
  [[ "$OS"   == "Darwin" ]]          && curl -Lo kind "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-darwin-${ARCH_DL}"
  chmod +x kind
  sudo mv kind /usr/local/bin/kind || mv kind "$HOME/.local/bin/kind"
  log_info "Kind installed."
else
  log_info "Kind: $(kind --version)"
fi

# ── Install kubectl if missing ─────────────────────────────────────────────────
if ! command -v kubectl >/dev/null 2>&1; then
  log_info "Installing kubectl..."
  ARCH_DL="$ARCH"
  [[ "$ARCH" == "x86_64"  ]] && ARCH_DL="amd64"
  [[ "$ARCH" == "aarch64" ]] && ARCH_DL="arm64"
  STABLE=$(curl -sL https://dl.k8s.io/release/stable.txt)
  [[ "$OS" == "Darwin" ]] && curl -LO "https://dl.k8s.io/release/${STABLE}/bin/darwin/${ARCH_DL}/kubectl"
  [[ "$OS" == "Linux"  ]] && curl -LO "https://dl.k8s.io/release/${STABLE}/bin/linux/${ARCH_DL}/kubectl"
  chmod +x kubectl
  sudo mv kubectl /usr/local/bin/kubectl || mv kubectl "$HOME/.local/bin/kubectl"
  log_info "kubectl installed."
else
  log_info "kubectl already installed."
fi

# ── Install jq if missing ──────────────────────────────────────────────────────
if ! command -v jq >/dev/null 2>&1; then
  log_warn "jq not found. Attempting to install..."
  if   [[ "$OS" == "Linux"  ]]; then sudo apt-get update && sudo apt-get install -y jq
  elif [[ "$OS" == "Darwin" ]]; then brew install jq
  else log_error "Please install jq manually."
  fi
fi

# ── Generate Kind cluster config from profile ──────────────────────────────────
TOTAL_CPU=$(docker info --format '{{.NCPU}}')
MEM_BYTES=$(docker info --format '{{.MemTotal}}')
TOTAL_MEM_GIB=$(awk "BEGIN {printf \"%.2f\", $MEM_BYTES / (1024*1024*1024)}" | sed 's/,/./')

NODES=$(jq --argjson cpu_total "$TOTAL_CPU" --argjson mem_total_gib "$TOTAL_MEM_GIB" '
  [.nodes[] |
    .cpu_float = (if (.cpu|test("^[0-9.]+$")) then (.cpu|tonumber) else 0 end) |
    .mem_mib = (
      if (.memory|test("Gi$")) then (.memory|sub("Gi";"") |tonumber * 1024)
      elif (.memory|test("Mi$")) then (.memory|sub("Mi";"") |tonumber)
      else 0 end) |
    .cpu_reserved = ($cpu_total - .cpu_float) |
    .mem_reserved = (($mem_total_gib * 1024) - .mem_mib) |
    {
      role: .role,
      labels: { "testbed-role": .name },
      kubeadmConfigPatches: [
        "apiVersion: kubeadm.k8s.io/v1beta4\nkind: \"" +
        (if .role == "control-plane" then "Init" else "Join" end) +
        "Configuration\"\nnodeRegistration:\n  kubeletExtraArgs:\n    system-reserved: \"cpu=" +
        (.cpu_reserved|tostring) + ",memory=" + (.mem_reserved|tostring) + "Mi\"\n    eviction-hard: \"memory.available<100Mi,nodefs.available<5%,nodefs.inodesFree<3%\""
      ]
    }
  ]' "$CONFIG_FILE")

jq --argjson nodes "$NODES" '.nodes = $nodes' "$TEMPLATE_FILE" > "$OUTPUT_FILE"
log_info "Kind config written to $OUTPUT_FILE"

# ── Create the cluster ─────────────────────────────────────────────────────────
CLUSTER_NAME=$(jq -r '.cluster_name' "$CONFIG_FILE")

if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
  log_info "Cluster '${CLUSTER_NAME}' exists. Deleting first..."
  kind delete cluster --name "$CLUSTER_NAME"
fi

kind create cluster --name "$CLUSTER_NAME" --config "$OUTPUT_FILE" \
  || log_error "Cluster creation failed."

kubectl config use-context "kind-${CLUSTER_NAME}" \
  || log_error "Failed to set kubectl context."

kubectl wait --for=condition=Ready node --all --timeout=120s \
  || log_error "Cluster nodes did not become Ready."

# ── Apply namespaces ───────────────────────────────────────────────────────────
kubectl apply -f "${PROJECT_ROOT}/manifests/namespaces/"

log_info "Cluster '${CLUSTER_NAME}' is ready."
log_info "Next: ./scripts/experiments/run.sh scripts/experiments/configs/partibench-two-node-hardcoded.env build-image"
