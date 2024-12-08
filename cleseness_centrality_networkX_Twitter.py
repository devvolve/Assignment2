import networkx as nx
import numpy as np


def remap_node_ids(file_path):
    """
    Remap node IDs to a contiguous range starting from 0.
    Returns the mapping and the total number of unique nodes.
    """
    nodes = set()
    with open(file_path, "r") as f:
        for line in f:
            node1, node2 = map(int, line.split())
            nodes.add(node1)
            nodes.add(node2)

    # Create a mapping from original IDs to contiguous IDs
    node_map = {node: idx for idx, node in enumerate(sorted(nodes))}
    reverse_node_map = {idx: node for node, idx in node_map.items()}
    return node_map, reverse_node_map, len(nodes)


def load_networkx_graph(file_path, node_map):
    """
    Load an edge list into a NetworkX graph with remapped node IDs.
    """
    G = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            node1, node2 = map(int, line.split())
            G.add_edge(node_map[node1], node_map[node2])  # Add remapped edges
    return G


def compute_and_print_centrality(G, reverse_node_map):
    """
    Compute closeness centrality for each node, displaying the node's rank.
    """
    print("Computing closeness centrality for each node...")
    centrality = {}
    for idx, node in enumerate(G.nodes):
        # Compute closeness centrality for this node
        centrality[node] = nx.closeness_centrality(G, u=node)

        # Map to original node ID and print with the node rank
        print(f"Processing Node Rank {idx + 1}: Closeness Centrality = {centrality[node]:.6f}")

    return centrality


if __name__ == "__main__":
    file_path = "twitter_combined.txt"  # Adjust as needed
    print("Loading graph and remapping nodes...")

    # Remap nodes and load the NetworkX graph
    node_map, reverse_node_map, num_nodes = remap_node_ids(file_path)
    G = load_networkx_graph(file_path, node_map)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Compute and print closeness centrality for every node
    centrality = compute_and_print_centrality(G, reverse_node_map)

    # Sort results by closeness centrality
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_5_nodes = [(reverse_node_map[node], value) for node, value in sorted_centrality[:5]]

    # Display the top 5 nodes
    print("\nTop 5 Nodes by Closeness Centrality (NetworkX):")
    for node, value in top_5_nodes:
        print(f"Node {node}: Closeness Centrality = {value:.6f}")

    # Calculate the average centrality
    avg_centrality = np.mean(list(centrality.values()))
    print(f"\nAverage Centrality (NetworkX): {avg_centrality:.6f}")

    # Save the NetworkX results for comparison
    with open("networkx_closeness_output_verbose.txt", "w") as f:
        for node, centrality_value in centrality.items():
            original_node = reverse_node_map[node]
            f.write(f"{original_node} {centrality_value:.6f}\n")
