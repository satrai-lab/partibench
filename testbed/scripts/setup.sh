#!/bin/bash
set -e

# ───────────────────────────────────────────────────────────────
#  PANDORA Testbed Setup Script
# ───────────────────────────────────────────────────────────────

echo ""
echo "██████╗  █████╗ ███╗   ██╗██████╗  ██████  ██████╗  █████╗ "
echo "██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══╗██║██╔══██╗██╔══██╗"
echo "██████╔╝███████║██╔██╗ ██║██║  ██║██║  ║██║██████╔╝███████║"
echo "██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██║  ║██║██╔═██║ ██╔══██║"
echo "██║     ██║  ██║██║ ╚████║██████╔╝╚██████╔╝██║ ██║ ██║  ██║"
echo "╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚═╝ ╚═╝ ╚═╝  ╚═╝"
echo ""

# ───────────────────────────────────────────────────────────────
#  Environment Check
# ───────────────────────────────────────────────────────────────

log_info()  { echo -e "\033[1;34m[INFO]\033[0m $1"; }
log_warn()  { echo -e "\033[1;33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; exit 1; }

OS=$(uname -s)
ARCH=$(uname -m)

log_info "Detected OS: $OS"
log_info "Detected Arch: $ARCH"

command -v docker >/dev/null 2>&1 || log_error "Docker is not installed. Install Docker Desktop."
docker info >/dev/null 2>&1 || log_error "Docker is not running. Please start Docker."

CPU_LIMIT=$(docker info --format '{{.NCPU}}')
MEM_BYTES=$(docker info --format '{{.MemTotal}}')
MEM_GB=$(awk "BEGIN {printf \"%.2f\", $MEM_BYTES / (1024 * 1024 * 1024)}")

log_info "Docker CPUs  : $CPU_LIMIT"
log_info "Docker Memory: $MEM_GB GiB"

# ───────────────────────────────────────────────────────────────
#  Kind Installation
# ───────────────────────────────────────────────────────────────

if ! command -v kind >/dev/null 2>&1; then
  log_info "Kind not found. Installing Kind..."

  KIND_VERSION="v0.22.0"
  ARCH_DL=$ARCH
  if [[ "$ARCH" == "x86_64" ]]; then
    ARCH_DL="amd64"
  elif [[ "$ARCH" == "arm64" ]]; then
    ARCH_DL="arm64"
  fi

  if [[ "$OS" == "Linux" ]]; then
    curl -Lo kind https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-${ARCH_DL}
  elif [[ "$OS" == "Darwin" ]]; then
    curl -Lo kind https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-darwin-${ARCH_DL}
  else
    log_error "Unsupported OS for automatic Kind install. Please install manually."
  fi


  chmod +x kind
  sudo mv kind /usr/local/bin/kind || mv kind "$HOME/.local/bin/kind"
  log_info "Kind installed successfully."
else
  log_info "Kind is already installed: $(kind --version)"
fi

# ───────────────────────────────────────────────────────────────
#  kubectl Installation
# ───────────────────────────────────────────────────────────────

if ! command -v kubectl >/dev/null 2>&1; then
  log_info "kubectl not found. Installing kubectl..."

  # Map uname -m to download architecture
  ARCH_DL=$ARCH
  if [[ "$ARCH" == "x86_64" ]]; then
    ARCH_DL="amd64"
  elif [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    ARCH_DL="arm64"
  else
    log_error "Unsupported architecture for kubectl: $ARCH"
  fi


  if [[ "$OS" == "Darwin" ]]; then
    curl -LO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/darwin/${ARCH_DL}/kubectl"
  elif [[ "$OS" == "Linux" ]]; then
    curl -LO "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/${ARCH_DL}/kubectl"
  else
    log_error "Unsupported OS for automatic kubectl install. Please install manually."
  fi

  # Verify it's a binary
  if file kubectl | grep -qi "text"; then
    log_error "Downloaded kubectl is not a binary. Download may have failed."
  fi

  chmod +x kubectl
  sudo mv kubectl /usr/local/bin/kubectl || mv kubectl "$HOME/.local/bin/kubectl"
  log_info "kubectl installed successfully."
else
  log_info "kubectl is already installed"
fi

log_info "Your environment is ready!"


# ───────────────────────────────────────────────────────────────
#  Generate Kind Cluster Config
# ───────────────────────────────────────────────────────────────

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONFIG_FILE="${1:-nodes.json}"
TEMPLATE_FILE="${SCRIPT_DIR}/kind-cluster-template.json"
OUTPUT_FILE="${SCRIPT_DIR}/kind-cluster-config.json"
CONFIG_PATH="$CONFIG_FILE"

if [[ ! -f "$CONFIG_PATH" ]]; then
  if [[ -f "${SCRIPT_DIR}/$CONFIG_FILE" ]]; then
    CONFIG_PATH="${SCRIPT_DIR}/$CONFIG_FILE"
  else
    log_error "Node config file not found: $CONFIG_FILE"
  fi
fi

if ! command -v jq >/dev/null 2>&1; then
  log_warn "jq not found. Attempting to install..."

  if [[ "$OS" == "Darwin" ]]; then
    if ! command -v brew >/dev/null 2>&1; then
      log_info "Homebrew not found. Installing Homebrew..."
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      eval "$(/opt/homebrew/bin/brew shellenv)" || eval "$(/usr/local/bin/brew shellenv)"
    fi
    brew install jq || log_error "Failed to install jq with Homebrew"
  elif [[ "$OS" == "Linux" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y jq || log_error "Failed to install jq via apt"
    else
      log_error "Unsupported Linux package manager. Please install jq manually."
    fi
  else
    log_error "Unsupported OS. Please install jq manually."
  fi
else
  log_info "jq is already installed: $(jq --version)"
fi

TOTAL_CPU=$(docker info --format '{{.NCPU}}')
MEM_BYTES=$(docker info --format '{{.MemTotal}}')
TOTAL_MEM_GIB=$(awk "BEGIN {printf \"%.2f\", $MEM_BYTES / (1024*1024*1024)}" | sed 's/,/./')



NODES=$(jq --argjson cpu_total "$TOTAL_CPU" --argjson mem_total_gib "$TOTAL_MEM_GIB" '
  [.nodes[] |
    .cpu_float = (if (.cpu|test("^[0-9.]+$")) then (.cpu|tonumber) else 0 end) |
    .mem_mib = (
      if (.memory|test("Gi$")) then
        (.memory|sub("Gi"; "")|tonumber * 1024)
      elif (.memory|test("Mi$")) then
        (.memory|sub("Mi"; "")|tonumber)
      else 0 end
    ) |
    .cpu_reserved = ($cpu_total - .cpu_float) |
    .mem_reserved = (($mem_total_gib * 1024) - .mem_mib) |
    {
      role: .role,
      labels: { "testbed-role": .name },
      kubeadmConfigPatches: [
        "apiVersion: kubeadm.k8s.io/v1beta3\nkind: \"" +
        (if .role == "control-plane" then "Init" else "Join" end) +
        "Configuration\"\nnodeRegistration:\n  kubeletExtraArgs:\n    system-reserved: \"cpu=" +
        (.cpu_reserved|tostring) + ",memory=" + (.mem_reserved|tostring) + "Mi\"\n    eviction-hard: \"memory.available<100Mi,nodefs.available<5%,nodefs.inodesFree<3%\""
      ]
    }
  ]' "$CONFIG_PATH")

jq --argjson nodes "$NODES" '.nodes = $nodes' "$TEMPLATE_FILE" > "$OUTPUT_FILE"
log_info "Kind config written to $OUTPUT_FILE"


# ───────────────────────────────────────────────────────────────
#  Publish NodePorts to the host
# ───────────────────────────────────────────────────────────────
EXTRA_PORTS='[
  {"containerPort":30000,"hostPort":30000,"protocol":"TCP"},
  {"containerPort":30001,"hostPort":30001,"protocol":"TCP"},
  {"containerPort":30002,"hostPort":30002,"protocol":"TCP"},
  {"containerPort":30003,"hostPort":30003,"protocol":"TCP"},
  {"containerPort":30004,"hostPort":30004,"protocol":"TCP"},
  {"containerPort":30005,"hostPort":30005,"protocol":"TCP"},
  {"containerPort":30006,"hostPort":30006,"protocol":"TCP"},
  {"containerPort":30007,"hostPort":30007,"protocol":"TCP"},
  {"containerPort":30008,"hostPort":30008,"protocol":"TCP"},
  {"containerPort":30009,"hostPort":30009,"protocol":"TCP"},
  {"containerPort":30010,"hostPort":30010,"protocol":"TCP"},
  {"containerPort":30011,"hostPort":30011,"protocol":"TCP"},
  {"containerPort":30012,"hostPort":30012,"protocol":"TCP"},
  {"containerPort":30093,"hostPort":30093,"protocol":"TCP"},
  {"containerPort":30180,"hostPort":30180,"protocol":"TCP"},
  {"containerPort":30090,"hostPort":30090,"protocol":"TCP"},
  {"containerPort":30091,"hostPort":30091,"protocol":"TCP"}
]'

tmpfile="${OUTPUT_FILE}.tmp"
jq --argjson ports "$EXTRA_PORTS" '
  .nodes = (
    .nodes
    | map(if .role == "control-plane"
          then . + {extraPortMappings:
                     ( ( .extraPortMappings // [] ) + $ports )
                   }
          else .
          end)
  )
' "$OUTPUT_FILE" > "$tmpfile" && mv "$tmpfile" "$OUTPUT_FILE"

log_info "Added extraPortMappings to control-plane for NodePort access"


# ───────────────────────────────────────────────────────────────
#  Create and Configure the Cluster
# ───────────────────────────────────────────────────────────────

KIND_CLUSTER_NAME="eab-pandora-testbed"
KIND_CONFIG_FILE="${SCRIPT_DIR}/kind-cluster-config.json"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || log_error "Failed to move to project root"

log_info "Creating Kind cluster '${KIND_CLUSTER_NAME}'..."
if kind get clusters | grep -q "${KIND_CLUSTER_NAME}"; then
  log_info "Cluster exists. Deleting first..."
  kind delete cluster --name "$KIND_CLUSTER_NAME"
fi

kind create cluster --name "$KIND_CLUSTER_NAME" --config "$KIND_CONFIG_FILE" || log_error "Cluster creation failed"
kubectl config use-context "kind-${KIND_CLUSTER_NAME}" || log_error "Failed to set kubectl context"
kubectl wait --for=condition=Ready node --all --timeout=90s || log_error "Cluster nodes not ready"

log_info "Applying Kubernetes manifests..."
kubectl apply -f "$PROJECT_ROOT/manifests/namespaces" || log_error "Failed to apply namespaces"


log_info "Cluster setup complete!"
log_info "You can now deploy workloads manually or run more setup scripts as needed."
