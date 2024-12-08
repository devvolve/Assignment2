import networkx as nx
import numpy as np

def compute_networkx_closeness(file_path):
    """
    Calculate closeness centrality for all nodes using NetworkX.
    """
    # Load graph using NetworkX
    G = nx.read_edgelist(file_path, nodetype=int)
    
    # Compute closeness centrality for all nodes
    centrality = nx.closeness_centrality(G)
    
    # Sort nodes by centrality (highest to lowest)
    sorted_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
    
    # Get top 5 nodes
    top_5_nodes = sorted_centrality[:5]
    
    # Calculate average centrality
    average_centrality = np.mean(list(centrality.values()))
    
    return centrality, top_5_nodes, average_centrality

if __name__ == "__main__":
    # File path for the graph
    file_path = "facebook_combined.txt"
    
    # Compute closeness centrality
    centrality, top_5_nodes, average_centrality = compute_networkx_closeness(file_path)
    
    # Display results
    print("NetworkX Closeness Centrality:")
    for node, value in centrality.items():
        print(f"Node {node}: Closeness Centrality = {value}")
    
    print("\nTop 5 Nodes:")
    for node, value in top_5_nodes:
        print(f"Node {node}: Closeness Centrality = {value}")
    
    print(f"\nAverage Centrality: {average_centrality}")
    
    # Save results to a file
    with open("networkx_closeness_output.txt", "w") as f:
        for node, value in centrality.items():
            f.write(f"{node} {value}\n")
