import argparse
import json
from pathlib import Path

import networkx as nx


NODE_IPS = {
    "edge": "edge-worker.edge.svc.cluster.local",
    "cloud": "cloud-worker.cloud.svc.cluster.local",
}


def build_graph(output_path: Path, edge_memory: int, cloud_memory: int) -> None:
    graph = nx.Graph()
    graph.add_node("edge", memory=edge_memory, ip=NODE_IPS["edge"])
    graph.add_node("cloud", memory=cloud_memory, ip=NODE_IPS["cloud"])
    graph.add_edge("edge", "cloud", weight={"bandwidth": 1000, "delay": 1})
    nx.write_gml(graph, output_path)


def load_num_blocks(bench_path: Path, model_name: str) -> int:
    with bench_path.open("r", encoding="utf-8") as bench_file:
        bench_data = json.load(bench_file)

    model_data = bench_data.get(model_name)
    if model_data is None:
        available_models = ", ".join(sorted(bench_data))
        raise ValueError(
            f"Model '{model_name}' was not found in {bench_path}. "
            f"Available models: {available_models or 'none'}."
        )

    return int(model_data["model_specifics"]["num_blocks"])


def build_strategy(output_path: Path, split_after_block: int, num_blocks: int) -> None:
    max_split_after_block = num_blocks - 2
    if split_after_block < 0 or split_after_block > max_split_after_block:
        raise ValueError(
            f"split_after_block must be between 0 and {max_split_after_block} "
            f"for a model with {num_blocks} blocks."
        )

    strategy = []
    last_block_index = num_blocks - 1

    for block_index in range(num_blocks):
        node_name = "edge" if block_index <= split_after_block else "cloud"
        last_block_on_node = split_after_block if node_name == "edge" else last_block_index
        merger = node_name if block_index == last_block_on_node else None
        strategy.append({
            "split_method": "no_split",
            "mapping": {node_name: 1},
            "merger": merger,
        })

    with output_path.open("w", encoding="utf-8") as out_file:
        for entry in strategy:
            out_file.write(json.dumps(entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="cc_graph.gml", help="Output graph file.")
    parser.add_argument("--hs", default="hs.txt", help="Output hardcoded-strategy file.")
    parser.add_argument("--bench", required=True, help="Merged benchmark JSON file.")
    parser.add_argument("--model", default="vgg16", help="Model name in the benchmark JSON.")
    parser.add_argument(
        "--split-after-block",
        type=int,
        default=6,
        help="Last model block that stays on the start node.",
    )
    parser.add_argument("--edge-memory", type=int, default=2048, help="Edge memory in MiB.")
    parser.add_argument("--cloud-memory", type=int, default=6144, help="Cloud memory in MiB.")
    args = parser.parse_args()

    build_graph(Path(args.graph), args.edge_memory, args.cloud_memory)
    num_blocks = load_num_blocks(Path(args.bench), args.model)
    build_strategy(Path(args.hs), args.split_after_block, num_blocks)

    print(f"Written graph to {Path(args.graph).resolve()}")
    print(f"Written hardcoded strategy to {Path(args.hs).resolve()}")


if __name__ == "__main__":
    main()
