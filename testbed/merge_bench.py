import json
import os
import argparse
from pathlib import Path


def merge_benchmarks(bench_dir_path):
    bench_dicts = []

    for filename in sorted(os.listdir(bench_dir_path)):
        if not filename.endswith('.json'):
            continue
        if filename == 'benchmark.json':  # skip previously merged output
            continue

        filepath = os.path.join(bench_dir_path, filename)
        with open(filepath, 'r') as in_file:
            bench_dict = json.load(in_file)

        if "node" not in bench_dict or "model" not in bench_dict:
            raise ValueError(f"Unexpected benchmark file format: {filepath}")

        bench_dicts.append(bench_dict)

    if not bench_dicts:
        raise ValueError(f"No node benchmark files found in {bench_dir_path}")

    merged_bench = {}
    for bd in bench_dicts:

        node = bd["node"]
        model = bd["model"]

        if model not in merged_bench:
            merged_bench[model] = {
                "model_specifics": {},
                "node_specifics": {}
            }

        if "inp_sizes" in bd:
            assert all(key in bd for key in ["out_sizes", "num_blocks", "mem_cons", "block_info"])
            merged_bench[model]["model_specifics"]["num_blocks"] = bd["num_blocks"]
            merged_bench[model]["model_specifics"]["inp_sizes"] = bd["inp_sizes"]
            merged_bench[model]["model_specifics"]["out_sizes"] = bd["out_sizes"]
            merged_bench[model]["model_specifics"]["mem_cons"] = bd["mem_cons"]
            merged_bench[model]["model_specifics"]["block_info"] = bd["block_info"]

        merged_bench[model]["node_specifics"][node] = {}
        merged_bench[model]["node_specifics"][node]["exec_times"] = bd["exec_times"]

    return merged_bench


def main():
    parser = argparse.ArgumentParser(description="Merge per-node benchmark JSON files into a single benchmark.json")
    parser.add_argument(
        "--bench-dir",
        default="benchmarks/nodes",
        help="Directory containing per-node benchmark JSON files (default: benchmarks/nodes)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: <bench-dir>/../benchmark.json)"
    )
    args = parser.parse_args()

    bench_dir = os.path.abspath(args.bench_dir)
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_path = os.path.join(os.path.dirname(bench_dir), "benchmark.json")

    merged = merge_benchmarks(bench_dir)

    with open(output_path, 'w') as out_file:
        json.dump(merged, out_file)

    print(f"Merged benchmark written to {output_path}")


if __name__ == "__main__":
    main()
