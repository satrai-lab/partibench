import networkx as nx
import itertools
import json
import socket
import argparse
import core


# Receives a list of possible split configurations and produces a list
# containing all possible combinations of configurations that add up to 1
def produce_possible_splits(configs, target_sum=1, partial=None, results=None):
    if partial is None:
        partial = []
    if results is None:
        results = []

    s = sum(partial)

    # Found combination
    if s == target_sum:

        # Generate all unique permutations of the combination and add to results
        for perm in set(itertools.permutations(partial)):
            results.append(list(perm))
        return

    # Partial sum exceeds target, ignore combination
    if s > target_sum:
        return

    # Recursive call
    for i in range(len(configs)):
        n = configs[i]
        remaining = configs[i:]
        produce_possible_splits(remaining, target_sum, partial + [n], results)

    return results


splits = produce_possible_splits([0.25, 0.5, 0.75, 1])  # all possible ways a layer block can be split
split_methods = ["input_split", "filter_split"]  # to find the data in benchmark json


def _format_ratio_key(prefix, ratio):
    return f"{prefix}_{ratio:g}"


def _extract_block_measurement(value):
    # New benchmark output stores some input-split measurements as nested lists:
    # each block entry contains a list of progressively deeper candidate segments.
    # The automatic placer optimizes one benchmark block at a time, so it should
    # only consume the first entry, which corresponds to the current block alone.
    if isinstance(value, list):
        if not value:
            raise ValueError("Encountered an empty benchmark entry while extracting block data.")
        first = value[0]
        if isinstance(first, list):
            if not first:
                raise ValueError("Encountered an empty nested benchmark entry while extracting block data.")
            return first[0]
        return first
    return value


# Receives the benchmark dict for a model, the node name, split method, split configuration and layer block number,
# and outputs the memory consumption, input size and output size for the given block (model specific) and given split
# config and the block's execution time for the given split config and for the given node (node specific)
def extract_bench_data_helper(bench, node, split_method, config, lb):

    model_specifics = bench["model_specifics"]
    node_specifics = bench["node_specifics"][node]["exec_times"]

    if split_method == "no_split":
        filter_key = _format_ratio_key("fs", 1)
        input_key = _format_ratio_key("is", 1)

        inps = model_specifics["inp_sizes"][filter_key][input_key][lb]
        outs = model_specifics["out_sizes"][filter_key][input_key][lb]
        mem = model_specifics["mem_cons"][filter_key][input_key][lb]
        et = node_specifics[filter_key][input_key][lb]["et_mean"]

    elif split_method == "input_split":
        assert config != 1
        filter_key = _format_ratio_key("fs", 1)
        input_key = _format_ratio_key("is", config)

        inps = _extract_block_measurement(model_specifics["inp_sizes"][filter_key][input_key][lb])
        outs = _extract_block_measurement(model_specifics["out_sizes"][filter_key][input_key][lb])
        mem = _extract_block_measurement(model_specifics["mem_cons"][filter_key][input_key][lb])
        et = _extract_block_measurement(node_specifics[filter_key][input_key][lb])["et_mean"]

    elif split_method == "filter_split":
        assert config != 1
        filter_key = _format_ratio_key("fs", config)

        inps = model_specifics["inp_sizes"][filter_key][lb]
        outs = model_specifics["out_sizes"][filter_key][lb]
        mem = model_specifics["mem_cons"][filter_key][lb]
        et = node_specifics[filter_key][lb]["et_mean"]

    else:
        raise ValueError(f"Unsupported split method: {split_method}")

    return inps, outs, et, mem


# ── Greedy placement ────────────────────────────────────────────────────────
# Optimises one block at a time: picks the locally cheapest placement for
# block i given where block i-1's output currently sits.  Fast (O(B×P)) but
# not globally optimal — a locally cheap choice can force an expensive
# transmission for the next block.
def _place_greedy(g, start_node, bench, model):
    global split_methods
    global splits

    bench = bench[model]
    num_blocks = bench["model_specifics"]["num_blocks"]
    current_node = start_node  # data produced here
    placement_strategy = []
    total_inference_time = 0

    # Optimize each layer block separately
    for lb_index in range(num_blocks):

        best_lb_placement_time = 99999
        best_lb_placement = None

        # Possible ways a lb can split
        for split in splits:

            # All possible node combinations that can be assigned with the execution of the split lb
            node_combinations = itertools.combinations(g.nodes(), len(split))

            lb_placements = []
            for comb in node_combinations:

                mapping = dict(zip(comb, split))  # mapping between nodes and lb split parts

                if len(mapping) == 1:  # no layer-level splitting
                    lb_placement = {
                        "split_method": "no_split",
                        "mapping": mapping,
                        "merger": comb[0]  # node keeps data after processing (no merging)
                    }
                    lb_placements.append(lb_placement)
                else:  # either input or filter splitting
                    for sm in split_methods:
                        for node in g.nodes():  # add merging node also
                            lb_placement = {
                                "split_method": sm,
                                "mapping": mapping,
                                "merger": node
                            }
                            lb_placements.append(lb_placement)

            # After determining all possible lb placements, find the best one
            for lb_placement in lb_placements:

                split_method = lb_placement["split_method"]
                max_node_time = 0  # here, keep the max time cost between nodes

                for node, config in lb_placement["mapping"].items():

                    # extract needed bench data for current node mapping
                    try:
                        inps, outs, et, mem = extract_bench_data_helper(bench, node, split_method, config, lb_index)
                    except (KeyError, IndexError):
                        max_node_time = 99999  # no benchmark data → infeasible
                        break

                    # Check memory constraint
                    if mem > g.nodes[node]["memory"]:
                        max_node_time = 99999
                        break

                    # Compute the cost (in ms) to transmit and process data to node
                    if current_node == node:
                        transmission_cost = 0  # current node performs processing, no need to transmit any data
                    else:
                        transmission_cost = (inps * 8) / (g[current_node][node]["weight"]["bandwidth"] * 1000) + g[current_node][node]["weight"]["delay"]
                    processing_cost = et

                    # Compute the cost to transmit output to the merging node, if there is one
                    merger = lb_placement["merger"]
                    if node == merger:
                        merge_cost = 0
                    else:
                        merge_cost = (outs * 8) / (g[node][merger]["weight"]["bandwidth"] * 1000) + g[node][merger]["weight"]["delay"]

                    # The total time the node takes to compute its assigned split part
                    total_node_time = transmission_cost + processing_cost + merge_cost

                    # Which node is the bottleneck? (since nodes may work in parallel)
                    if total_node_time > max_node_time:
                        max_node_time = total_node_time

                # Which placement has the best execution time for this layer block?
                if max_node_time < best_lb_placement_time:
                    best_lb_placement_time = max_node_time
                    best_lb_placement = lb_placement

        placement_strategy.append(best_lb_placement)
        total_inference_time += best_lb_placement_time
        current_node = best_lb_placement["merger"]  # data is now transferred to merger node

    print(total_inference_time)
    return placement_strategy


# ── Dynamic-programming placement ───────────────────────────────────────────
# DP state: (block_index, merger_node) — "processed blocks 0..i, output sits
# at merger_node".  The only link between consecutive blocks is which node
# holds the output, so this state captures all inter-block dependencies.
#
# Transition:
#   dp_cost[i][v] = min over all placements p of block i that end at v of:
#                   dp_cost[i-1][u] + block_cost(p, starting_from=u)
#
# This is O(B × N × P) vs the greedy's O(B × P), where N is the node count.
# For typical edge/cloud deployments (N = 2–5) the overhead is negligible
# and the result is provably globally optimal.
def _place_dp(g, start_node, bench, model):
    global split_methods
    global splits

    bench = bench[model]
    num_blocks = bench["model_specifics"]["num_blocks"]
    nodes = list(g.nodes())
    INF = float('inf')

    # dp_cost[v] = minimum accumulated time to reach a state where the
    # current block's output sits at node v.
    dp_cost = {v: INF for v in nodes}
    dp_cost[start_node] = 0.0

    # history[lb_index][v] = (best_placement, from_node) used to reach v
    # after processing block lb_index — needed for path reconstruction.
    history = []

    for lb_index in range(num_blocks):

        new_dp_cost = {v: INF for v in nodes}
        new_dp_record = {v: None for v in nodes}

        # Build the full list of candidate placements for this block once,
        # then evaluate each from every reachable previous state (from_node).
        lb_placements = []
        for split in splits:
            for comb in itertools.combinations(nodes, len(split)):
                mapping = dict(zip(comb, split))
                if len(mapping) == 1:
                    lb_placements.append({
                        "split_method": "no_split",
                        "mapping": mapping,
                        "merger": comb[0]
                    })
                else:
                    for sm in split_methods:
                        for merger_node in nodes:
                            lb_placements.append({
                                "split_method": sm,
                                "mapping": mapping,
                                "merger": merger_node
                            })

        for from_node in nodes:
            if dp_cost[from_node] == INF:
                continue  # unreachable state — skip

            for lb_placement in lb_placements:
                split_method = lb_placement["split_method"]
                merger = lb_placement["merger"]
                max_node_time = 0
                feasible = True

                for node, config in lb_placement["mapping"].items():
                    try:
                        inps, outs, et, mem = extract_bench_data_helper(bench, node, split_method, config, lb_index)
                    except (KeyError, IndexError):
                        feasible = False  # no benchmark data → infeasible
                        break

                    # Memory constraint
                    if mem > g.nodes[node]["memory"]:
                        feasible = False
                        break

                    # Transmission from previous output node to this worker node
                    if from_node == node:
                        transmission_cost = 0
                    elif g.has_edge(from_node, node):
                        transmission_cost = (inps * 8) / (g[from_node][node]["weight"]["bandwidth"] * 1000) + g[from_node][node]["weight"]["delay"]
                    else:
                        feasible = False  # no path between these nodes
                        break

                    # Merge cost from worker to merger
                    if node == merger:
                        merge_cost = 0
                    elif g.has_edge(node, merger):
                        merge_cost = (outs * 8) / (g[node][merger]["weight"]["bandwidth"] * 1000) + g[node][merger]["weight"]["delay"]
                    else:
                        feasible = False
                        break

                    total_node_time = transmission_cost + et + merge_cost
                    if total_node_time > max_node_time:
                        max_node_time = total_node_time

                if not feasible:
                    continue

                total_cost = dp_cost[from_node] + max_node_time
                if total_cost < new_dp_cost[merger]:
                    new_dp_cost[merger] = total_cost
                    new_dp_record[merger] = (lb_placement, from_node)

        history.append(new_dp_record)
        dp_cost = new_dp_cost

    # Find the node that holds the final output at minimum total cost
    best_final_node = min(dp_cost, key=lambda v: dp_cost[v])
    if dp_cost[best_final_node] == INF:
        raise RuntimeError(
            "No feasible placement found. "
            "Check memory constraints and graph connectivity."
        )

    print(dp_cost[best_final_node])

    # Backtrack through recorded choices to reconstruct the placement strategy
    placement_strategy = []
    current = best_final_node
    for lb_index in reversed(range(num_blocks)):
        lb_placement, prev_node = history[lb_index][current]
        placement_strategy.append(lb_placement)
        current = prev_node

    placement_strategy.reverse()
    return placement_strategy


# ── Public entry point ───────────────────────────────────────────────────────
# Receives the CC graph, the starting inference node, the benchmarking data,
# the CNN model name and the placement method, and produces a list with the
# best placement solution for each layer block.
#
# method="greedy"  — fast, locally optimal (default)
# method="dp"      — globally optimal, N× slower (N = number of nodes)
def place(g, start_node, bench, model, method="greedy"):
    if method == "greedy":
        return _place_greedy(g, start_node, bench, model)
    elif method == "dp":
        return _place_dp(g, start_node, bench, model)
    else:
        raise ValueError(
            f"Unknown placement method '{method}'. "
            "Choose 'greedy' (fast, locally optimal) or "
            "'dp' (slower, globally optimal)."
        )


port = 5000  # helper global var to set ports in components


# Receives the CC graph, the placement strategy, the next lb index, the components list and the
# merge dimension and creates and assigns a bridge component to the current merger node
def add_bridge_component(g, strategy, lb_index, comp_list, merge_type):

    global port

    # Handle case where bridge component is the first one in the inference (after producer component)
    if lb_index != 0:  # if not the first one
        merger = strategy[lb_index - 1]["merger"]
        inp_units = len(strategy[lb_index - 1]["mapping"])
    else:  # if the first one
        merger = comp_list[0]["node"]  # start node
        assert comp_list[0]["type"] == "producer"
        inp_units = 1

    # Create bridge comp. for merger who merges lb's outputs and prepares (splits, if needed) input for next lb
    bridge_component = {
        "type": "bridge",
        "node": merger,
        "inp_units": inp_units,  # how many inputs to receive?
        "merge_type": merge_type,  # merge received inputs? how?
        "split_ratio": [],  # split data to send? how?
        "dest_ips": [],  # where to send next inputs?
        "dest_port": port + 1,
        "recv_port": port
    }
    port += 1

    # Decide how to split next input and where to send it based on next lb placement
    if lb_index == len(strategy):  # current lb was the last one --> time to send results back to start node
        bridge_component["split_ratio"].append(1)
        if merger == comp_list[0]["node"]:
            bridge_component["dest_ips"].append("127.0.0.1")
        else:
            bridge_component["dest_ips"].append(g.nodes[comp_list[0]["node"]]["ip"])
        bridge_component["send_results"] = "yes"
        comp_list[0]["recv_port"] = bridge_component["dest_port"]

    else:  # what will the bridge component do based on the next layer block placement?
        next_lb_placement = strategy[lb_index]
        split_method = next_lb_placement["split_method"]
        if split_method == "input_split":
            for node, config in next_lb_placement["mapping"].items():
                bridge_component["split_ratio"].append(config)
                bridge_component["dest_ips"].append("127.0.0.1" if node == merger else g.nodes[node]["ip"])
        elif split_method == "filter_split":
            for node, _ in next_lb_placement["mapping"].items():
                bridge_component["split_ratio"].append(1)
                bridge_component["dest_ips"].append("127.0.0.1" if node == merger else g.nodes[node]["ip"])
        elif split_method == "no_split":
            bridge_component["split_ratio"].append(1)
            next_node = next(iter(next_lb_placement["mapping"]))
            bridge_component["dest_ips"].append("127.0.0.1" if next_node == merger else g.nodes[next_node]["ip"])

    comp_list.append(bridge_component)


# TODO COMMENTS
def define_components(g, start_node, bench, model, strategy):

    global port

    block_info = bench[model]["model_specifics"]["block_info"]  # info about layer blocks (from benchmarking)

    # Initiate list that will contain the assigned components to each node
    comp_list = []

    # Add producer component to start_node
    producer_component = {
        "node": start_node,
        "type": "producer",
        "dest_ip": "127.0.0.1",
        "dest_port": port
    }
    comp_list.append(producer_component)

    lb_index = 0

    # Add bridge component in case there's splitting at first layer block
    add_bridge_component(g, strategy, lb_index, comp_list, None)

    while lb_index < len(strategy):

        lb_placement = strategy[lb_index]

        if lb_placement["split_method"] == "filter_split":

            num_filters = block_info[lb_index]["num_filters"]

            current_split_offset = 0
            current_node_position = 0
            for node, config in lb_placement["mapping"].items():

                block_component = {
                    "type": "block",
                    "node": node,
                    "from_layer": block_info[lb_index]["from_layer"],
                    "to_layer": block_info[lb_index]["to_layer"],
                    "from_filter": current_split_offset,
                    "to_filter": current_split_offset + round(num_filters * config),
                    "block_order": current_node_position,  # this helps merger to merge in the correct order
                    "dest_ip": "127.0.0.1" if node == lb_placement["merger"] else g.nodes[lb_placement["merger"]]["ip"],
                    "dest_port": port + 1,
                    "recv_port": port
                }
                current_split_offset = block_component["to_filter"]
                current_node_position += 1
                comp_list.append(block_component)

            lb_index += 1
            port += 1

            add_bridge_component(g, strategy, lb_index, comp_list, "filter")

        elif lb_placement["split_method"] == "input_split":

            num_weights_if_dense = block_info[lb_index]["num_weights_if_dense"]

            # If current component is Dense, inform previous bridge component
            # that it may need to flatten before sending inputs
            # Also, inform next bridge component to merge received data by aggregating them (and not concat)
            if num_weights_if_dense:

                previous_bridge_component = comp_list[-1]  # last component is the previous one

                # Debug checks
                assert lb_index > 0 
                assert previous_bridge_component["node"] == strategy[lb_index-1]["merger"]
                assert previous_bridge_component["type"] == "bridge"

                previous_bridge_component["flatten"] = "yes"
                merge_type = "aggregation"  # for next bridge component
            else:
                merge_type = "width"  # for next bridge component

            current_split_offset = 0
            current_node_position = 0
            for node, config in lb_placement["mapping"].items():

                block_component = {
                    "type": "block",
                    "node": node,
                    "from_layer": block_info[lb_index]["from_layer"],
                    "to_layer": None,
                    "block_order": current_node_position,  # this helps merger to merge in the correct order
                    "dest_ip": None,
                    "dest_port": port + 1,
                    "recv_port": port
                }

                lb_index_copy = lb_index  # we do not want lb_index to change to use it for the other nodes too

                if num_weights_if_dense:  # if layer block contains Dense layer

                    merger = lb_placement["merger"]
                    assert merger  # be sure it has merger info
                    block_component["to_layer"] = block_info[lb_index]["to_layer"]  # configure end-limit, since this is a single-layer block
                    block_component["dest_ip"] = "127.0.0.1" if node == merger else g.nodes[merger]["ip"]  # for the same reason, configure dest ip
                    block_component["from_weight"] = current_split_offset
                    block_component["to_weight"] = current_split_offset + round(num_weights_if_dense * config)
                    current_split_offset = block_component["to_weight"]

                else:  # if layer block is not a Dense layer

                    # Asymmetrical padding makes sense -> configure it
                    if current_node_position == 0:
                        block_component["assym_pad"] = "left"
                    elif current_node_position == len(lb_placement["mapping"]) - 1:
                        block_component["assym_pad"] = "right"
                    else:
                        block_component["assym_pad"] = "center"

                    # Determine block's end-limit (to_layer) and configure its destination ip
                    while True:
                        merger = strategy[lb_index_copy]["merger"]
                        if merger:
                            block_component["to_layer"] = block_info[lb_index_copy]["to_layer"]
                            block_component["dest_ip"] = "127.0.0.1" if node == merger else g.nodes[merger]["ip"]
                            break
                        lb_index_copy += 1

                current_node_position += 1
                comp_list.append(block_component)

            lb_index = lb_index_copy + 1
            port += 1

            add_bridge_component(g, strategy, lb_index, comp_list, merge_type)

        elif lb_placement["split_method"] == "no_split":

            node = next(iter(lb_placement["mapping"]))

            block_component = {
                "type": "block",
                "node": node,
                "from_layer": block_info[lb_index]["from_layer"],
                "to_layer": None,
                "block_order": 0,  # single node deployment
                "dest_ip": "127.0.0.1",  # this node will also deploy the next bridge component
                "dest_port": port + 1,
                "recv_port": port
            }

            # Determine block's end limit (to_layer)
            while True:
                merger = strategy[lb_index]["merger"]
                if merger:
                    block_component["to_layer"] = block_info[lb_index]["to_layer"]
                    break
                lb_index += 1

            lb_index += 1
            port += 1
            comp_list.append(block_component)

            add_bridge_component(g, strategy, lb_index, comp_list, None)

    return comp_list


# Receives the components list and the model name and for the bridge components that split inputs
# for non-Dense blocks, adds split input offsets information
def add_input_split_offsets(c, model_name):

    model = None

    for i in range(len(c)):

        current_component = c[i]

        # If this is a bridge component
        if current_component["type"] == "bridge":
            split_ratio = current_component["split_ratio"]

            # If this bridge component will split the input for a non-Dense block
            if len(split_ratio) > 1 and any(r != 1 for r in split_ratio) and current_component.get("flatten") is None:

                next_component = c[i+1]
                assert next_component["type"] == "block"
                assert all(v is None for v in[next_component.get("from_weight"), next_component.get("to_weight"), next_component.get("from_filter"), next_component.get("to_filter")])

                # Generate model block based on the next block component

                # Load model if not already loaded
                if not model:
                    model = core.load_model(model_name)

                # Generate block
                next_block = core.generate_model_partition(model, from_layer=next_component["from_layer"], to_layer=next_component["to_layer"])

                # Compute exact split offset information and add it to bridge component
                current_component["split_offsets"] = core.split_input_offsets(next_block, split_ratio)


# Receives the CC graph, the list with the components that each node needs to execute and the model name
# and sends this information to each node so that it can actually deploy the components
def distribute_components(g, c, model):

    deploy_recv_port = 6999
    comp_dict = {}

    # Convert component list to dictionary (for easier management)
    for component in c:
        node = component["node"]
        if node not in comp_dict:  # init component data for node
            comp_dict[node] = {
                "model": model,
                "components": []
            }
        comp_dict[node]["components"].append(component)

    # Send assigned components to each node
    for node_name, node_data in comp_dict.items():

        assert len(node_data["components"]) > 0  # make sure node has some components assigned to it

        node_ip = g.nodes[node_name]["ip"]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((node_ip, deploy_recv_port))
            s.sendall(json.dumps(node_data).encode())


# Reads a manually created placement strategy from a file and returns it
def load_hardcoded_strategy(model, filepath='generated/k8s_two_node/hs.txt'):
    strategy = []
    with open(filepath, 'r') as file:
        for line in file:
            strategy.append(json.loads(line.strip()))

    return strategy


# ==== MAIN ====
parser = argparse.ArgumentParser()
parser.add_argument("--start-node", default="cloud", help="Name of the inference start node.")
parser.add_argument("--model", default="vgg16", help="Model name.")
parser.add_argument(
    "--bench",
    default="../manifests/workers/benchmark/output/benchmark.json",
    help="Path to the merged benchmark JSON file.",
)
parser.add_argument(
    "--graph",
    default="generated/k8s_two_node/cc_graph.gml",
    help="Path to the compute-continuum graph.",
)
parser.add_argument(
    "--strategy",
    choices=["auto", "hardcoded"],
    default="auto",
    help="Placement strategy source.",
)
parser.add_argument(
    "--placement-method",
    choices=["greedy", "dp"],
    default="greedy",
    help=(
        "Algorithm used when --strategy=auto. "
        "'greedy' (default): fast, locally optimal, picks the best placement "
        "for each block independently. "
        "'dp': globally optimal via dynamic programming, N× slower where N is "
        "the number of nodes — negligible overhead for small deployments."
    ),
)
parser.add_argument(
    "--hs",
    default="generated/k8s_two_node/hs.txt",
    help="Path to a hardcoded strategy file.",
)
parser.add_argument(
    "--components-out",
    default=None,
    help="Optional path to write the generated deployment components JSON.",
)
parser.add_argument(
    "--strategy-out",
    default=None,
    help="Optional path to write the placement strategy as JSON lines.",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Generate the placement and components without sending them to worker pods.",
)
args = parser.parse_args()

with open(args.bench, "r") as in_file:
    bench_d = json.load(in_file)

G = nx.read_gml(args.graph)
start_node = args.start_node
model_name = args.model

if start_node not in G.nodes:
    raise ValueError(f"Start node '{start_node}' is not present in {args.graph}")

if model_name not in bench_d:
    raise ValueError(f"Model '{model_name}' is not present in {args.bench}")

available_nodes = set(G.nodes())
bench_nodes = set(bench_d[model_name]["node_specifics"].keys())
missing_nodes = bench_nodes - available_nodes
if missing_nodes:
    raise ValueError(
        "Benchmark node names do not match graph node names. "
        f"Missing in graph: {sorted(missing_nodes)}"
    )

if args.strategy == "auto":
    missing_bench_nodes = available_nodes - bench_nodes
    if missing_bench_nodes:
        raise ValueError(
            "Automatic placement requires benchmark data for every graph node. "
            f"Missing benchmark entries: {sorted(missing_bench_nodes)}"
        )

if args.strategy == "auto":
    strategy = place(G, start_node, bench_d, model_name, method=args.placement_method)
else:
    strategy = load_hardcoded_strategy(model_name, filepath=args.hs)

for elem in strategy:
    print(elem)

if args.strategy_out:
    with open(args.strategy_out, "w") as strategy_file:
        for entry in strategy:
            strategy_file.write(json.dumps(entry) + "\n")

C = define_components(G, start_node, bench_d, model_name, strategy)
add_input_split_offsets(C, model_name)

if args.components_out:
    with open(args.components_out, "w") as out_file:
        json.dump(C, out_file)

if not args.dry_run:
    distribute_components(G, C, model_name)

print(json.dumps(C))
