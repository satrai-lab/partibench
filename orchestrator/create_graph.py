import networkx as nx

# Create CC graph with real network connections
G = nx.Graph()

# Add nodes with their memory and ip address info (TODO use REAL available memory)
G.add_node('iot_A1', memory=536, ip="172.31.22.245")
G.add_node('iot_A2', memory=1073, ip="172.31.19.48")
G.add_node('iot_A3', memory=536, ip="172.31.29.241")
G.add_node('iot_B1', memory=536, ip="172.31.31.254")
G.add_node('iot_B2', memory=1073, ip="172.31.31.231")
G.add_node('iot_B3', memory=536, ip="172.31.22.90")
G.add_node('edge_A', memory=2147, ip="172.31.21.155")
G.add_node('edge_B', memory=4294, ip="172.31.20.197")
G.add_node('cloud', memory=17179, ip="172.31.22.227")

# Add edges

G.add_edge('iot_A1', 'iot_A2', weight={"bandwidth": 200, "delay": 8})
G.add_edge('iot_A1', 'iot_A3', weight={"bandwidth": 200, "delay": 8})
G.add_edge('iot_A2', 'iot_A3', weight={"bandwidth": 200, "delay": 8})

G.add_edge('iot_B1', 'iot_B2', weight={"bandwidth": 20, "delay": 15})
G.add_edge('iot_B1', 'iot_B3', weight={"bandwidth": 20, "delay": 15})
G.add_edge('iot_B2', 'iot_B3', weight={"bandwidth": 20, "delay": 15})

G.add_edge('edge_A', 'iot_A1', weight={"bandwidth": 150, "delay": 11})
G.add_edge('edge_A', 'iot_A2', weight={"bandwidth": 150, "delay": 11})
G.add_edge('edge_A', 'iot_A3', weight={"bandwidth": 150, "delay": 11})
G.add_edge('edge_A', 'edge_B', weight={"bandwidth": 100, "delay": 20})
G.add_edge('edge_A', 'cloud', weight={"bandwidth": 100, "delay": 50})

G.add_edge('edge_B', 'iot_B1', weight={"bandwidth": 15, "delay": 20})
G.add_edge('edge_B', 'iot_B2', weight={"bandwidth": 15, "delay": 20})
G.add_edge('edge_B', 'iot_B3', weight={"bandwidth": 15, "delay": 20})
G.add_edge('edge_B', 'cloud', weight={"bandwidth": 100, "delay": 50})

# Create a new fully connected graph from G that also contains virtual node connections
G_fc = nx.Graph()
G_fc.add_nodes_from(G.nodes(data=True))

# Compute the shortest paths (based on hop-count) between each pair of nodes and construct the new graph's edges
for source in G.nodes:
    for target in G.nodes:
        if source != target:

            # Get the shortest path based on hop count
            path = nx.shortest_path(G, source=source, target=target)

            # Calculate the sum of delays along this path
            total_delay = sum(G[u][v]["weight"]["delay"] for u, v in zip(path[:-1], path[1:]))

            # Calculate the minimum bandwidth along this path
            min_bandwidth = min(G[u][v]["weight"]["bandwidth"] for u, v in zip(path[:-1], path[1:]))

            # Add the edge to the new graph
            G_fc.add_edge(source, target, weight={"bandwidth": min_bandwidth, "delay": total_delay})


# Save graph to file
nx.write_gml(G_fc, "cc_graph.gml")

# Create network shaping rules files (for each node)
header = "DEST_NAME\tIP\tBANDWIDTH\tDELAY"
for node in G_fc.nodes():
    content = header
    for connected_node, link_data in G_fc[node].items():
        content += ("\n" + connected_node + "\t" +
                    G_fc.nodes[connected_node]["ip"] + "\t" +
                    str(link_data["weight"]["bandwidth"]) + "mbit\t" +
                    str(link_data["weight"]["delay"]) + "ms")
    net_rules_filepath = "net_rules/" + node + "_rules.tsv"
    with open(net_rules_filepath, 'w') as out_file:
        out_file.write(content)
