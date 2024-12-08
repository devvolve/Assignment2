from mpi4py import MPI
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
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


def load_adj_matrix(file_path, node_map, num_nodes):
    """
    Convert an edge list into a sparse adjacency matrix.
    """
    rows, cols = [], []
    with open(file_path, "r") as f:
        for line in f:
            node1, node2 = map(int, line.split())
            rows.append(node_map[node1])
            cols.append(node_map[node2])
            rows.append(node_map[node2])  # Undirected graph
            cols.append(node_map[node1])
    data = [1] * len(rows)
    return csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


def compute_closeness_centrality_node(adj_matrix, node):
    """
    Compute closeness centrality for a single node using sparse shortest_path.
    """
    num_nodes = adj_matrix.shape[0]
    distances = shortest_path(adj_matrix, directed=False, indices=node)
    reachable = distances[distances < np.inf]  # Exclude unreachable nodes
    total_reachable = len(reachable) - 1  # Exclude the node itself

    if total_reachable > 0:
        centrality = total_reachable / np.sum(reachable)
        return centrality * (num_nodes - 1) / total_reachable  # Normalize by total nodes
    return 0


if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Remap node IDs and count total nodes
        node_map, reverse_node_map, num_nodes = remap_node_ids("twitter_combined.txt")
        adj_matrix = load_adj_matrix("twitter_combined.txt", node_map, num_nodes)
        print(f"Adjacency matrix dimensions: {adj_matrix.shape}")
    else:
        adj_matrix, num_nodes, reverse_node_map = None, None, None

    # Broadcast adjacency matrix and total number of nodes
    adj_matrix = comm.bcast(adj_matrix, root=0)
    num_nodes = comm.bcast(num_nodes, root=0)
    reverse_node_map = comm.bcast(reverse_node_map, root=0)

    # Divide all nodes among MPI processes
    all_nodes = np.arange(num_nodes)
    local_nodes = np.array_split(all_nodes, size)[rank]
    print(f"[Rank {rank}] Processing {len(local_nodes)} nodes")

    # Compute closeness centrality for local nodes and print results
    local_centrality = np.zeros(num_nodes)
    for idx, node in enumerate(local_nodes):
        centrality = compute_closeness_centrality_node(adj_matrix, node)
        local_centrality[node] = centrality
        print(f"[Rank {rank}] Node {reverse_node_map[node]}: Closeness Centrality = {centrality:.6f}")

    # Gather results from all processes
    global_centrality = None
    if rank == 0:
        global_centrality = np.zeros(num_nodes)
    comm.Allreduce(local_centrality, global_centrality, op=MPI.SUM)

    if rank == 0:
        # Sort results and find the top 5 nodes
        sorted_indices = np.argsort(global_centrality)[-5:][::-1]
        top_5_nodes = [(reverse_node_map[idx], global_centrality[idx]) for idx in sorted_indices]
        average_centrality = np.mean(global_centrality)

        # Display results
        print("\nTop 5 Nodes by Closeness Centrality (MPI):")
        for node, centrality in top_5_nodes:
            print(f"Node {node}: Closeness Centrality = {centrality:.6f}")

        print(f"\nAverage Centrality (MPI): {average_centrality:.6f}")

        # Save results to a file
        with open("mpi_closeness_output_verbose.txt", "w") as f:
            for idx, centrality_value in enumerate(global_centrality):
                f.write(f"{reverse_node_map[idx]} {centrality_value:.6f}\n")

    MPI.Finalize()
