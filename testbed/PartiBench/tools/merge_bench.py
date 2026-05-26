import argparse
import json
from pathlib import Path


def merge_benchmarks(bench_dir_path: Path) -> dict:
    bench_dicts = []

    for filepath in sorted(bench_dir_path.glob("*.json")):
        if filepath.name == "benchmark.json":
            continue

        with filepath.open("r", encoding="utf-8") as in_file:
            bench_dict = json.load(in_file)

        if "node" not in bench_dict or "model" not in bench_dict:
            raise ValueError(f"Unexpected benchmark file format: {filepath}")

        bench_dicts.append(bench_dict)

    if not bench_dicts:
        raise ValueError(f"No node benchmark files found in {bench_dir_path}")

    merged_bench = {}
    for bench_dict in bench_dicts:
        node = bench_dict["node"]
        model = bench_dict["model"]

        if model not in merged_bench:
            merged_bench[model] = {
                "model_specifics": {},
                "node_specifics": {},
            }

        if "inp_sizes" in bench_dict:
            required_keys = {"out_sizes", "num_blocks", "mem_cons", "block_info"}
            if not required_keys.issubset(bench_dict):
                raise ValueError(f"Incomplete model-specific benchmark data for {node}")

            merged_bench[model]["model_specifics"] = {
                "num_blocks": bench_dict["num_blocks"],
                "inp_sizes": bench_dict["inp_sizes"],
                "out_sizes": bench_dict["out_sizes"],
                "mem_cons": bench_dict["mem_cons"],
                "block_info": bench_dict["block_info"],
            }

        merged_bench[model]["node_specifics"][node] = {
            "exec_times": bench_dict["exec_times"],
        }

    return merged_bench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-dir",
        default="../manifests/workers/benchmark/output",
        help="Directory containing per-node benchmark JSON files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path. Defaults to <bench-dir>/benchmark.json.",
    )
    args = parser.parse_args()

    bench_dir_path = Path(args.bench_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else bench_dir_path / "benchmark.json"

    merged_bench = merge_benchmarks(bench_dir_path)

    with output_path.open("w", encoding="utf-8") as out_file:
        json.dump(merged_bench, out_file)

    print(f"Merged benchmark written to {output_path}")


if __name__ == "__main__":
    main()
