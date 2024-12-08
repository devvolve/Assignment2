from mpi4py import MPI
from collections import deque, defaultdict
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
    return node_map, len(nodes)


def load_adj_list(file_path, node_map):
    """
    Convert an edge list into an adjacency list using remapped node IDs.
    """
    adj_list = defaultdict(list)
    with open(file_path, "r") as f:
        for line in f:
            node1, node2 = map(int, line.split())
            adj_list[node_map[node1]].append(node_map[node2])
            adj_list[node_map[node2]].append(node_map[node1])  # For undirected graph
    return adj_list


def bfs_shortest_paths(adj_list, start_node, num_nodes):
    """
    Perform BFS to find shortest paths from the start_node.
    """
    distances = np.full(num_nodes, np.inf)
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        for neighbor in adj_list[current_node]:
            if distances[neighbor] == np.inf:  # Not visited
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    return distances


def compute_closeness(adj_list, rank, size, num_nodes):
    """
    Compute closeness centrality for all nodes, distributed among MPI processes.
    """
    # Divide nodes among processes
    all_nodes = list(range(num_nodes))
    local_nodes = np.array_split(all_nodes, size)[rank]

    # Compute closeness centrality for local nodes
    local_closeness = np.zeros(num_nodes)
    for idx, node in enumerate(local_nodes):
        distances = bfs_shortest_paths(adj_list, node, num_nodes)
        reachable = distances[distances < np.inf]
        if len(reachable) > 1:
            local_closeness[node] = (len(reachable) - 1) / np.sum(reachable)
        
        # Print progress for each node
        print(f"[Rank {rank}] Processed Node {node + 1}/{num_nodes}: Closeness Centrality = {local_closeness[node]:.6f}")

    # Aggregate results across all processes
    global_closeness = None
    if rank == 0:
        global_closeness = np.zeros(num_nodes)
    MPI.COMM_WORLD.Reduce(local_closeness, global_closeness, op=MPI.SUM, root=0)

    return global_closeness


if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load adjacency list for Facebook dataset
    if rank == 0:
        file_path = "facebook_combined.txt"  # Adjust the file path as necessary
        # Remap node IDs and count total nodes
        node_map, num_nodes = remap_node_ids(file_path)
        adj_list = load_adj_list(file_path, node_map)
        reverse_node_map = {v: k for k, v in node_map.items()}
    else:
        adj_list, num_nodes, reverse_node_map = None, None, None

    # Broadcast data
    adj_list = comm.bcast(adj_list, root=0)
    num_nodes = comm.bcast(num_nodes, root=0)
    reverse_node_map = comm.bcast(reverse_node_map, root=0)

    # Compute closeness centrality
    global_closeness = compute_closeness(adj_list, rank, size, num_nodes)

    if rank == 0:
        # Gather results
        results = [(reverse_node_map[node], global_closeness[node]) for node in range(num_nodes)]

        # Sort nodes by centrality (highest to lowest)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        # Get top 5 nodes
        top_5_nodes = sorted_results[:5]

        # Calculate average centrality
        average_centrality = np.mean([item[1] for item in results])

        # Display results
        print("\nMPI Closeness Centrality:")
        for node, centrality in results:
            print(f"Node {node}: Closeness Centrality = {centrality:.6f}")

        print("\nTop 5 Nodes:")
        for node, centrality in top_5_nodes:
            print(f"Node {node}: Closeness Centrality = {centrality:.6f}")

        print(f"\nAverage Centrality: {average_centrality:.6f}")

        # Save results to a file
        with open("mpi_closeness_facebook_output.txt", "w") as f:
            for node, centrality in results:
                f.write(f"{node} {centrality:.6f}\n")

    MPI.Finalize()
