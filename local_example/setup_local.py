"""
Generates the files needed to run a 2-node local simulation:
  - cc_graph.gml        : two nodes on 127.0.0.1 / 127.0.0.2
  - net_rules.tsv       : minimal TSV (no delays, since communication is over loopback)
  - hs.txt              : pipelined strategy — node_A runs blocks 0-6, node_B runs blocks 7-15

Run from the local_example/ directory:
    python setup_local.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

# ── CC graph ─────────────────────────────────────────────────────────────────
G = nx.Graph()

# Two virtual nodes on loopback addresses
# Memory values are just placeholders for the local demo (not enforced by place.py when using hs.txt)
G.add_node('node_A', memory=99999, ip='127.0.0.1')
G.add_node('node_B', memory=99999, ip='127.0.0.2')

# Single link between them (high bandwidth, zero delay — it's loopback)
G.add_edge('node_A', 'node_B', weight={'bandwidth': 10000, 'delay': 0})

# Build fully-connected version (trivial for 2 nodes)
G_fc = nx.Graph()
G_fc.add_nodes_from(G.nodes(data=True))
for source in G.nodes:
    for target in G.nodes:
        if source != target:
            path = nx.shortest_path(G, source=source, target=target)
            total_delay = sum(G[u][v]['weight']['delay'] for u, v in zip(path[:-1], path[1:]))
            min_bandwidth = min(G[u][v]['weight']['bandwidth'] for u, v in zip(path[:-1], path[1:]))
            G_fc.add_edge(source, target, weight={'bandwidth': min_bandwidth, 'delay': total_delay})

nx.write_gml(G_fc, 'cc_graph.gml')
print('Written: cc_graph.gml')

# ── net_rules.tsv for each node ───────────────────────────────────────────────
# core.py reads net_rules.tsv at import time to build the delay lookup dict.
# For loopback traffic the delay will simply be 0 (key not found → None → no sleep).
# We still need the file to exist; a header-only file is enough.
for node in G_fc.nodes():
    lines = ['DEST_NAME\tIP\tBANDWIDTH\tDELAY']
    for neighbour, link in G_fc[node].items():
        lines.append(
            f"{neighbour}\t{G_fc.nodes[neighbour]['ip']}\t"
            f"{link['weight']['bandwidth']}mbit\t{link['weight']['delay']}ms"
        )
    filepath = f'net_rules/{node}_rules.tsv'
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Written: {filepath}')

# ── hs.txt ────────────────────────────────────────────────────────────────────
# Pipelined model partitioning:
#   node_A (127.0.0.1) runs VGG16 blocks 0-6  (early feature extraction)
#   node_B (127.0.0.2) runs VGG16 blocks 7-15 (deep layers + FC)
# Blocks 0-5 use merger:null to chain them into a single BLOCK process on node_A.
# Block 6 finalises the chain with merger:"node_A".
# Blocks 7-14 chain on node_B; block 15 finalises with merger:"node_B".
import json
strategy = []
for i in range(7):
    strategy.append({'split_method': 'no_split', 'mapping': {'node_A': 1},
                     'merger': None if i < 6 else 'node_A'})
for i in range(9):
    strategy.append({'split_method': 'no_split', 'mapping': {'node_B': 1},
                     'merger': None if i < 8 else 'node_B'})

with open('hs.txt', 'w') as f:
    for entry in strategy:
        f.write(json.dumps(entry) + '\n')
print('Written: hs.txt')

print('\nSetup complete. See README.md for next steps.')
