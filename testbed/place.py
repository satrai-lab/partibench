import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

    if s == target_sum:
        for perm in set(itertools.permutations(partial)):
            results.append(list(perm))
        return

    if s > target_sum:
        return

    for i in range(len(configs)):
        n = configs[i]
        remaining = configs[i:]
        produce_possible_splits(remaining, target_sum, partial + [n], results)

    return results


splits = produce_possible_splits([0.25, 0.5, 0.75, 1])
split_methods = ["input_split", "filter_split"]


port = 5000  # global port counter for component assignment


def _node_ip(g, node, local_node):
    """Return 127.0.0.1 when communicating within the same pod, else the K8s service IP."""
    if node == local_node:
        return "127.0.0.1"
    return g.nodes[node]["ip"]


# Creates and appends a bridge component for the merger node at the current lb boundary.
def add_bridge_component(g, strategy, lb_index, comp_list, merge_type):

    global port

    if lb_index != 0:
        merger = strategy[lb_index - 1]["merger"]
        inp_units = len(strategy[lb_index - 1]["mapping"])
    else:
        merger = comp_list[0]["node"]  # start node
        assert comp_list[0]["type"] == "producer"
        inp_units = 1

    bridge_component = {
        "type": "bridge",
        "node": merger,
        "inp_units": inp_units,
        "merge_type": merge_type,
        "split_ratio": [],
        "dest_ips": [],
        "dest_port": port + 1,
        "recv_port": port
    }
    port += 1

    if lb_index == len(strategy):  # last block processed — send results back to start node
        bridge_component["split_ratio"].append(1)
        bridge_component["dest_ips"].append(_node_ip(g, comp_list[0]["node"], merger))
        bridge_component["send_results"] = "yes"
        comp_list[0]["recv_port"] = bridge_component["dest_port"]

    else:
        next_lb_placement = strategy[lb_index]
        split_method = next_lb_placement["split_method"]
        if split_method == "input_split":
            for node, config in next_lb_placement["mapping"].items():
                bridge_component["split_ratio"].append(config)
                bridge_component["dest_ips"].append(_node_ip(g, node, merger))
        elif split_method == "filter_split":
            for node, _ in next_lb_placement["mapping"].items():
                bridge_component["split_ratio"].append(1)
                bridge_component["dest_ips"].append(_node_ip(g, node, merger))
        elif split_method == "no_split":
            bridge_component["split_ratio"].append(1)
            next_node = next(iter(next_lb_placement["mapping"]))
            bridge_component["dest_ips"].append(_node_ip(g, next_node, merger))

    comp_list.append(bridge_component)


def define_components(g, start_node, bench, model, strategy):

    global port

    block_info = bench[model]["model_specifics"]["block_info"]

    comp_list = []

    # Producer is always on start_node; first bridge is also on start_node → 127.0.0.1
    producer_component = {
        "node": start_node,
        "type": "producer",
        "dest_ip": "127.0.0.1",
        "dest_port": port
    }
    comp_list.append(producer_component)

    lb_index = 0
    add_bridge_component(g, strategy, lb_index, comp_list, None)

    while lb_index < len(strategy):

        lb_placement = strategy[lb_index]

        if lb_placement["split_method"] == "filter_split":

            num_filters = block_info[lb_index]["num_filters"]
            merger = lb_placement["merger"]
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
                    "block_order": current_node_position,
                    "dest_ip": _node_ip(g, merger, node),
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

            if num_weights_if_dense:
                previous_bridge_component = comp_list[-1]
                assert lb_index > 0
                assert previous_bridge_component["node"] == strategy[lb_index-1]["merger"]
                assert previous_bridge_component["type"] == "bridge"
                previous_bridge_component["flatten"] = "yes"
                merge_type = "aggregation"
            else:
                merge_type = "width"

            current_split_offset = 0
            current_node_position = 0
            for node, config in lb_placement["mapping"].items():

                block_component = {
                    "type": "block",
                    "node": node,
                    "from_layer": block_info[lb_index]["from_layer"],
                    "to_layer": None,
                    "block_order": current_node_position,
                    "dest_ip": None,
                    "dest_port": port + 1,
                    "recv_port": port
                }

                lb_index_copy = lb_index

                if num_weights_if_dense:
                    merger = lb_placement["merger"]
                    assert merger
                    block_component["to_layer"] = block_info[lb_index]["to_layer"]
                    block_component["dest_ip"] = _node_ip(g, merger, node)
                    block_component["from_weight"] = current_split_offset
                    block_component["to_weight"] = current_split_offset + round(num_weights_if_dense * config)
                    current_split_offset = block_component["to_weight"]

                else:
                    if current_node_position == 0:
                        block_component["assym_pad"] = "left"
                    elif current_node_position == len(lb_placement["mapping"]) - 1:
                        block_component["assym_pad"] = "right"
                    else:
                        block_component["assym_pad"] = "center"

                    while True:
                        merger = strategy[lb_index_copy]["merger"]
                        if merger:
                            block_component["to_layer"] = block_info[lb_index_copy]["to_layer"]
                            block_component["dest_ip"] = _node_ip(g, merger, node)
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
                "block_order": 0,
                "dest_ip": None,  # set below once merger is known
                "dest_port": port + 1,
                "recv_port": port
            }

            # Walk forward until we find the merger for this chain
            while True:
                merger = strategy[lb_index]["merger"]
                if merger:
                    block_component["to_layer"] = block_info[lb_index]["to_layer"]
                    # The BRIDGE lives on the merger node; use 127.0.0.1 if same pod
                    block_component["dest_ip"] = _node_ip(g, merger, node)
                    break
                lb_index += 1

            lb_index += 1
            port += 1
            comp_list.append(block_component)
            add_bridge_component(g, strategy, lb_index, comp_list, None)

    return comp_list


# For bridge components that split inputs for non-Dense blocks, compute exact pixel offsets.
def add_input_split_offsets(c, model_name):

    model = None

    for i in range(len(c)):

        current_component = c[i]

        if current_component["type"] == "bridge":
            split_ratio = current_component["split_ratio"]

            if len(split_ratio) > 1 and any(r != 1 for r in split_ratio) and current_component.get("flatten") is None:

                next_component = c[i + 1]
                assert next_component["type"] == "block"
                assert all(v is None for v in [next_component.get("from_weight"), next_component.get("to_weight"),
                                               next_component.get("from_filter"), next_component.get("to_filter")])

                if not model:
                    model = core.load_model(model_name)

                next_block = core.generate_model_partition(
                    model, from_layer=next_component["from_layer"], to_layer=next_component["to_layer"])

                current_component["split_offsets"] = core.split_input_offsets(next_block, split_ratio)


# Sends component assignments to each node's deployment agent (port 6999).
def distribute_components(g, c, model):

    deploy_recv_port = 6999
    comp_dict = {}

    for component in c:
        node = component["node"]
        if node not in comp_dict:
            comp_dict[node] = {
                "model": model,
                "components": []
            }
        comp_dict[node]["components"].append(component)

    for node_name, node_data in comp_dict.items():

        assert len(node_data["components"]) > 0

        node_ip = g.nodes[node_name]["ip"]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((node_ip, deploy_recv_port))
            s.sendall(json.dumps(node_data).encode())


# Reads the hardcoded placement strategy from a file (one JSON line per block).
def load_hardcoded_strategy(model, filepath='hs.txt'):
    strategy = []
    with open(filepath, 'r') as file:
        for line in file:
            strategy.append(json.loads(line.strip()))
    return strategy


# ==== MAIN ====

parser = argparse.ArgumentParser()
parser.add_argument('--start-node', default='edge', help='Name of the inference start node (default: edge)')
parser.add_argument('--model', default='vgg16', help='Model name (default: vgg16)')
parser.add_argument('--bench', default='/config/benchmark.json', help='Path to merged benchmark JSON')
parser.add_argument('--graph', default='/config/cc_graph.gml', help='Path to CC graph GML file')
parser.add_argument('--hs', default='/config/hs.txt', help='Path to hardcoded placement strategy file')
args = parser.parse_args()

with open(args.bench, 'r') as in_file:
    bench_d = json.load(in_file)

G = nx.read_gml(args.graph)
start_node = args.start_node
model_name = args.model

strategy = load_hardcoded_strategy(model_name, filepath=args.hs)

for elem in strategy:
    print(elem)

C = define_components(G, start_node, bench_d, model_name, strategy)
add_input_split_offsets(C, model_name)
distribute_components(G, C, model_name)
print(json.dumps(C))
