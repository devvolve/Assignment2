import networkx as nx
import numpy as np

def load_graph(file_path):
    """Load the graph from an edge list file."""
    print("Loading graph...")
    graph = nx.read_edgelist(file_path, nodetype=int)
    print(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

def compute_betweenness_centrality(graph):
    """Compute betweenness centrality for all nodes."""
    print("Computing betweenness centrality for all nodes...")
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    return betweenness

def save_results(filename, centrality):
    """Save the betweenness centrality results to a file."""
    with open(filename, "w") as f:
        for node, value in centrality.items():
            f.write(f"{node} {value:.6f}\n")

def main():
    file_path = "facebook_combined.txt"  # Replace with the path to your Facebook dataset
    output_file = "facebook_betweenness_centrality_networkx.txt"

    # Load the graph
    graph = load_graph(file_path)

    # Compute betweenness centrality
    betweenness = compute_betweenness_centrality(graph)

    # Save results to a file
    save_results(output_file, betweenness)

    # Find the top 5 nodes
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    top_5_nodes = sorted_betweenness[:5]

    # Calculate the average betweenness centrality
    average_centrality = np.mean(list(betweenness.values()))

    # Display results
    print("\nTop 5 Nodes by Betweenness Centrality:")
    for node, centrality in top_5_nodes:
        print(f"Node {node}: Betweenness Centrality = {centrality:.6f}")

    print(f"\nAverage Betweenness Centrality: {average_centrality:.6f}")

if __name__ == "__main__":
    main()
