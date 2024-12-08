import networkx as nx
import numpy as np


def load_networkx_graph_with_global_ids(file_path):
    """
    Load an edge list into a NetworkX graph using original global node IDs.
    """
    G = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            node1, node2 = map(int, line.split())
            G.add_edge(node1, node2)
    return G


def compute_and_print_centrality_with_global_ids(G):
    """
    Compute closeness centrality for each node using original global node IDs.
    """
    print("Computing closeness centrality for each node...")
    centrality = {}
    for idx, node in enumerate(sorted(G.nodes)):
        # Compute closeness centrality for this node
        centrality[node] = nx.closeness_centrality(G, u=node)

        # Print the node's rank and centrality
        print(f"Processing Node Rank {idx + 1}/{len(G.nodes)} "
              f"(Global Node ID: {node}): Closeness Centrality = {centrality[node]:.6f}")

    return centrality


if __name__ == "__main__":
    file_path = "twitter_combined.txt"  # Adjust as needed
    print("Loading graph with global node IDs...")

    # Load the NetworkX graph
    G = load_networkx_graph_with_global_ids(file_path)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Compute and print closeness centrality for every node
    centrality = compute_and_print_centrality_with_global_ids(G)

    # Sort results by closeness centrality
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_5_nodes = sorted_centrality[:5]

    # Display the top 5 nodes
    print("\nTop 5 Nodes by Closeness Centrality (NetworkX):")
    for node, value in top_5_nodes:
        print(f"Node {node}: Closeness Centrality = {value:.6f}")

    # Calculate the average centrality
    avg_centrality = np.mean(list(centrality.values()))
    print(f"\nAverage Centrality (NetworkX): {avg_centrality:.6f}")

    # Save the NetworkX results for comparison
    with open("networkx_closeness_output_global_ids.txt", "w") as f:
        for node, centrality_value in centrality.items():
            f.write(f"{node} {centrality_value:.6f}\n")
