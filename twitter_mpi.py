from mpi4py import MPI
from collections import defaultdict, deque
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


def load_adj_list(file_path, node_map):
    """
    Convert an edge list into an adjacency list using remapped node IDs.
    """
    adj_list = defaultdict(list)
    with open(file_path, "r") as f:
        for line in f:
            node1, node2 = map(int, line.split())
            adj_list[node_map[node1]].append(node_map[node2])
            adj_list[node_map[node2]].append(node_map[node1])  # Undirected graph
    return adj_list


def eppstein_wang_closeness(adj_list, start_node, num_nodes):
    """
    Compute closeness centrality using the Eppstein-Wang algorithm for a single node.
    """
    distances = np.full(num_nodes, np.inf)
    distances[start_node] = 0
    queue = deque([start_node])
    visited_count = 0
    distance_sum = 0

    while queue:
        current_node = queue.popleft()
        current_distance = distances[current_node]

        visited_count += 1
        distance_sum += current_distance

        for neighbor in adj_list[current_node]:
            if distances[neighbor] == np.inf:  # Not visited
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)

    if visited_count > 1:
        closeness = (visited_count - 1) / distance_sum
        return closeness * (num_nodes - 1) / (visited_count - 1)
    return 0


if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Load graph and remap node IDs
        file_path = "twitter_combined.txt"
        node_map, reverse_node_map, num_nodes = remap_node_ids(file_path)
        adj_list = load_adj_list(file_path, node_map)
        print(f"[Rank {rank}] Graph loaded with {num_nodes} nodes.")
    else:
        adj_list, num_nodes, reverse_node_map = None, None, None

    # Broadcast data to all processes
    adj_list = comm.bcast(adj_list, root=0)
    num_nodes = comm.bcast(num_nodes, root=0)
    reverse_node_map = comm.bcast(reverse_node_map, root=0)

    # Divide nodes among MPI processes
    all_nodes = np.arange(num_nodes)
    local_nodes = np.array_split(all_nodes, size)[rank]
    print(f"[Rank {rank}] Assigned {len(local_nodes)} nodes.")

    # Compute closeness centrality for local nodes
    local_centrality = np.zeros(num_nodes)
    for idx, node in enumerate(local_nodes):
        centrality = eppstein_wang_closeness(adj_list, node, num_nodes)
        local_centrality[node] = centrality
        print(f"[Rank {rank}] Node {idx + 1}/{len(local_nodes)} "
              f"(Global Node ID: {reverse_node_map[node]}): "
              f"Closeness Centrality = {centrality:.6f}")

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
