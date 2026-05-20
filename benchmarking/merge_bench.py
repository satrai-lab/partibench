import json
import os

bench_dir_path = "benchmarks"  # path to the benchmarks directory
bench_dicts = []  # store all node benchmarks as dicts

# Iterate over all files in the nodes directory
for filename in os.listdir(bench_dir_path + "/nodes"):
    if filename.endswith('.json'):  # check if the file is a JSON file
        filepath = os.path.join(bench_dir_path + "/nodes", filename)  # full path to the file
        with open(filepath, 'r') as in_file:
            bench_dicts.append(json.load(in_file))  # load benchmark and collect it with others
    else:
        assert False

# Create a unified benchmark dict from the dicts of all nodes
merged_bench = {}
for bd in bench_dicts:

    node = bd["node"]
    model = bd["model"]

    # model seen for the first time -> create base structure for its data
    if model not in merged_bench:
        merged_bench[model] = {
            "model_specifics": {},
            "node_specifics": {}
        }

    # for the node that profiled model specific data
    if "inp_sizes" in bd:
        assert all(key in bd for key in ["out_sizes", "num_blocks", "mem_cons", "block_info"])
        merged_bench[model]["model_specifics"]["num_blocks"] = bd["num_blocks"]
        merged_bench[model]["model_specifics"]["inp_sizes"] = bd["inp_sizes"]
        merged_bench[model]["model_specifics"]["out_sizes"] = bd["out_sizes"]
        merged_bench[model]["model_specifics"]["mem_cons"] = bd["mem_cons"]
        merged_bench[model]["model_specifics"]["block_info"] = bd["block_info"]

    # for each node (execution times)
    merged_bench[model]["node_specifics"][node] = {}
    merged_bench[model]["node_specifics"][node]["exec_times"] = bd["exec_times"]

# Save merged benchmark dict as json file
with open(bench_dir_path + '/benchmark.json', 'w') as out_file:
    json.dump(merged_bench, out_file)
