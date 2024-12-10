from mpi4py import MPI
import networkx as nx
import numpy as np
from time import time

def load_graph(file_path):
    """Load the graph from an edge list file."""
    print("[Rank 0] Loading graph...")
    graph = nx.read_edgelist(file_path, nodetype=int)
    return graph

def compute_local_betweenness(graph, nodes):
    """Compute betweenness centrality for a subset of nodes."""
    return nx.betweenness_centrality_subset(graph, sources=nodes, targets=list(graph.nodes()), normalized=True)

def save_results(filename, centrality):
    """Save the betweenness centrality results to a file."""
    with open(filename, "w") as f:
        for node, value in centrality.items():
            f.write(f"{node} {value:.6f}\n")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    file_path = "facebook_combined.txt"  # Replace with the correct path to your dataset
    output_file = f"facebook_betweenness_rank_{rank}.txt"

    if rank == 0:
        # Load the graph on the root process
        graph = load_graph(file_path)
        print(f"[Rank 0] Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    else:
        graph = None

    # Broadcast the graph to all processes
    graph = comm.bcast(graph, root=0)

    # Divide the nodes among processes
    nodes = list(graph.nodes())
    local_nodes = np.array_split(nodes, size)[rank]

    # Compute betweenness centrality for the local nodes
    print(f"[Rank {rank}] Computing betweenness centrality for {len(local_nodes)} nodes...")
    start_time = time()
    local_centrality = compute_local_betweenness(graph, local_nodes)
    end_time = time()
    print(f"[Rank {rank}] Computation completed in {end_time - start_time:.2f} seconds.")

    # Print local results as they are calculated
    for node, centrality in local_centrality.items():
        print(f"[Rank {rank}] Node {node}: Betweenness Centrality = {centrality:.6f}")

    # Gather results from all processes
    all_centrality = comm.gather(local_centrality, root=0)

    if rank == 0:
        # Combine the results
        combined_centrality = {}
        for partial_centrality in all_centrality:
            combined_centrality.update(partial_centrality)

        # Find the top 5 nodes with the highest betweenness centrality
        sorted_centrality = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
        top_5_nodes = sorted_centrality[:5]

        # Calculate the average betweenness centrality
        average_centrality = np.mean(list(combined_centrality.values()))

        # Save the combined results to a file
        save_results("facebook_betweenness_centrality_mpi.txt", combined_centrality)

        # Display results
        print("\nTop 5 Nodes by Betweenness Centrality:")
        for node, centrality in top_5_nodes:
            print(f"Node {node}: Betweenness Centrality = {centrality:.6f}")

        print(f"\nAverage Betweenness Centrality: {average_centrality:.6f}")

if __name__ == "__main__":
    main()
