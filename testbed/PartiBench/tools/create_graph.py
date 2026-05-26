import argparse
import json
from pathlib import Path

import networkx as nx
import yaml


ROLE_MAP = {
    "IoT": {
        "node_name": "iot",
        "namespace": "iot",
        "service": "iot-worker",
    },
    "Iot": {
        "node_name": "iot",
        "namespace": "iot",
        "service": "iot-worker",
    },
    "Edge": {
        "node_name": "edge",
        "namespace": "edge",
        "service": "edge-worker",
    },
    "Cloud": {
        "node_name": "cloud",
        "namespace": "cloud",
        "service": "cloud-worker",
    },
}

DEFAULT_LINK_MAP = {
    frozenset(("iot", "edge")): {"bandwidth": 100, "delay": 5},
    frozenset(("edge", "cloud")): {"bandwidth": 1000, "delay": 1},
    frozenset(("iot", "cloud")): {"bandwidth": 500, "delay": 15},
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_NODES_PATH = PROJECT_ROOT / "scripts" / "nodes.json"
DEFAULT_NETWORK_CHAOS_DIR = PROJECT_ROOT / "manifests" / "06-experiments" / "networking-experiment"


def parse_memory_to_mib(memory_value: str) -> int:
    if memory_value.endswith("Gi"):
        return int(float(memory_value[:-2]) * 1024)
    if memory_value.endswith("Mi"):
        return int(float(memory_value[:-2]))
    raise ValueError(f"Unsupported memory format: {memory_value}")


def parse_rate_to_mbps(rate_value: str) -> int:
    normalized = rate_value.strip().lower()
    if normalized.endswith("gbps"):
        return int(float(normalized[:-4]) * 1000)
    if normalized.endswith("mbps"):
        return int(float(normalized[:-4]))
    raise ValueError(f"Unsupported rate format: {rate_value}")


def parse_latency_to_ms(latency_value: str) -> float:
    normalized = latency_value.strip().lower()
    if normalized.endswith("ms"):
        return float(normalized[:-2])
    raise ValueError(f"Unsupported latency format: {latency_value}")


def build_link_map(network_chaos_dir: Path) -> dict:
    if not network_chaos_dir.exists():
        return DEFAULT_LINK_MAP

    derived_links = {}
    for manifest_path in sorted(network_chaos_dir.glob("*.yaml")):
        with manifest_path.open("r", encoding="utf-8") as in_file:
            for document in yaml.safe_load_all(in_file):
                if not document or document.get("kind") != "NetworkChaos":
                    continue

                spec = document.get("spec", {})
                source_namespaces = spec.get("selector", {}).get("namespaces", [])
                target_namespaces = spec.get("target", {}).get("selector", {}).get("namespaces", [])
                if not source_namespaces or not target_namespaces:
                    continue

                source = source_namespaces[0].lower()
                target = target_namespaces[0].lower()
                key = frozenset((source, target))

                link = {
                    "bandwidth": parse_rate_to_mbps(spec["rate"]["rate"]),
                    "delay": parse_latency_to_ms(spec["delay"]["latency"]),
                }

                existing = derived_links.get(key)
                if existing and existing != link:
                    raise ValueError(
                        f"Conflicting Chaos Mesh link settings for {sorted(key)}: "
                        f"{existing} vs {link}"
                    )
                derived_links[key] = link

    return derived_links or DEFAULT_LINK_MAP


def build_graph(nodes_config_path: Path, network_chaos_dir: Path) -> nx.Graph:
    with nodes_config_path.open("r", encoding="utf-8") as in_file:
        config = json.load(in_file)

    graph = nx.Graph()
    link_map = build_link_map(network_chaos_dir)

    for node in config.get("nodes", []):
        role_name = node.get("name")
        if role_name not in ROLE_MAP or node.get("role") != "worker":
            continue

        role_data = ROLE_MAP[role_name]
        graph.add_node(
            role_data["node_name"],
            memory=parse_memory_to_mib(node["memory"]),
            ip=f'{role_data["service"]}.{role_data["namespace"]}.svc.cluster.local',
        )

    node_names = list(graph.nodes())
    for index, source in enumerate(node_names):
        for target in node_names[index + 1:]:
            link = link_map.get(frozenset((source, target)))
            if link is None:
                continue
            graph.add_edge(source, target, weight=link)

    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nodes",
        default=str(DEFAULT_NODES_PATH),
        help="Path to the Pandora nodes.json file.",
    )
    parser.add_argument(
        "--network-chaos-dir",
        default=str(DEFAULT_NETWORK_CHAOS_DIR),
        help="Directory containing the Chaos Mesh NetworkChaos manifests used as link source-of-truth.",
    )
    parser.add_argument(
        "--output",
        default="cc_graph.gml",
        help="Output GML path.",
    )
    args = parser.parse_args()

    nodes_config_path = Path(args.nodes).resolve()
    network_chaos_dir = Path(args.network_chaos_dir).resolve()
    output_path = Path(args.output).resolve()

    graph = build_graph(nodes_config_path, network_chaos_dir)
    if graph.number_of_nodes() == 0:
        raise ValueError("No worker nodes were found in the provided nodes.json file.")

    nx.write_gml(graph, output_path)
    print(f"Graph written to {output_path} with nodes: {list(graph.nodes())}")


if __name__ == "__main__":
    main()
