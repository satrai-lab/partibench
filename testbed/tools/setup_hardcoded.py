"""
Generates all files needed for the 2-node pipelined VGG16 example:
  - cc_graph.gml               : two-node graph (edge + cloud) with K8s service DNS as IPs
  - hs.txt                     : hardcoded pipelined strategy
  - net_rules/edge_rules.tsv   : delay table for the edge pod (destination = cloud)
  - net_rules/cloud_rules.tsv  : delay table for the cloud pod (destination = edge)

Usage (host-side, before running place.py as a K8s Job):
    python tools/setup_hardcoded.py \\
        --bench  /tmp/benchmark.json \\
        --graph  /tmp/cc_graph.gml \\
        --hs     /tmp/hs.txt \\
        --model  vgg16 \\
        --split-after-block 6
"""

import argparse
import json
from pathlib import Path

import networkx as nx

# Kubernetes service DNS names used as node IPs inside the cluster.
NODE_IPS = {
    "edge":  "edge-worker.edge.svc.cluster.local",
    "cloud": "cloud-worker.cloud.svc.cluster.local",
}


def build_graph(output_path: Path, edge_memory: int, cloud_memory: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("edge",  memory=edge_memory,  ip=NODE_IPS["edge"])
    graph.add_node("cloud", memory=cloud_memory, ip=NODE_IPS["cloud"])
    # 1 Gbps / 1 ms — loopback-class link inside a Kind cluster
    graph.add_edge("edge", "cloud", weight={"bandwidth": 1000, "delay": 1})
    nx.write_gml(graph, output_path)
    print(f"Written graph to {output_path.resolve()}")
    return graph


def build_net_rules(graph: nx.Graph, net_rules_dir: Path) -> None:
    """Write one net_rules TSV per node, keyed by K8s service DNS name.

    core.py reads net_rules.tsv from the working directory at import time and
    builds delays_dict keyed by destination IP.  In the testbed the destination
    IPs are service DNS names, so the TSV rows must use those same names.
    """
    net_rules_dir.mkdir(parents=True, exist_ok=True)

    for node in graph.nodes():
        lines = ["DEST_NAME\tIP\tBANDWIDTH\tDELAY"]
        for neighbor, link_data in graph[node].items():
            bw    = link_data["weight"]["bandwidth"]
            delay = link_data["weight"]["delay"]
            ip    = graph.nodes[neighbor]["ip"]   # K8s service DNS name
            lines.append(f"{neighbor}\t{ip}\t{bw}mbit\t{delay}ms")

        filepath = net_rules_dir / f"{node}_rules.tsv"
        with filepath.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Written net_rules for {node} to {filepath.resolve()}")


def load_num_blocks(bench_path: Path, model_name: str) -> int:
    with bench_path.open("r", encoding="utf-8") as f:
        bench_data = json.load(f)
    model_data = bench_data.get(model_name)
    if model_data is None:
        available = ", ".join(sorted(bench_data))
        raise ValueError(
            f"Model '{model_name}' not found in {bench_path}. "
            f"Available: {available or 'none'}."
        )
    return int(model_data["model_specifics"]["num_blocks"])


def build_strategy(output_path: Path, split_after_block: int, num_blocks: int) -> None:
    max_split = num_blocks - 2
    if split_after_block < 0 or split_after_block > max_split:
        raise ValueError(
            f"split_after_block must be between 0 and {max_split} "
            f"for a {num_blocks}-block model."
        )

    strategy = []
    last_block = num_blocks - 1

    for idx in range(num_blocks):
        node = "edge" if idx <= split_after_block else "cloud"
        last_on_node = split_after_block if node == "edge" else last_block
        merger = node if idx == last_on_node else None
        strategy.append({
            "split_method": "no_split",
            "mapping": {node: 1},
            "merger": merger,
        })

    with output_path.open("w", encoding="utf-8") as f:
        for entry in strategy:
            f.write(json.dumps(entry) + "\n")

    print(f"Written strategy to {output_path.resolve()} "
          f"(edge: blocks 0-{split_after_block}, cloud: blocks {split_after_block+1}-{last_block})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cc_graph.gml and hs.txt for a 2-node K8s deployment.")
    parser.add_argument("--graph",            default="cc_graph.gml",      help="Output graph GML file.")
    parser.add_argument("--hs",               default="hs.txt",            help="Output hardcoded-strategy file.")
    parser.add_argument("--bench",            required=True,               help="Merged benchmark JSON file.")
    parser.add_argument("--model",            default="vgg16",             help="Model name in the benchmark JSON.")
    parser.add_argument("--split-after-block",type=int, default=6,         help="Last block index on the edge node (0-indexed).")
    parser.add_argument("--edge-memory",      type=int, default=6144,      help="Edge node memory in MiB.")
    parser.add_argument("--cloud-memory",     type=int, default=6144,      help="Cloud node memory in MiB.")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    hs_path    = Path(args.hs)
    bench_path = Path(args.bench)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    hs_path.parent.mkdir(parents=True, exist_ok=True)

    graph = build_graph(graph_path, args.edge_memory, args.cloud_memory)
    build_net_rules(graph, graph_path.parent / "net_rules")
    num_blocks = load_num_blocks(bench_path, args.model)
    build_strategy(hs_path, args.split_after_block, num_blocks)


if __name__ == "__main__":
    main()
