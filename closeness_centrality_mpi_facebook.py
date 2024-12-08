from mpi4py import MPI
from collections import deque, defaultdict
import numpy as np

def count_nodes(file_path):
    """
    Count the number of unique nodes in an edge list.
    """
    nodes = set()
    with open(file_path, "r") as f:
        for line in f:
            node1, node2 = map(int, line.split())
            nodes.add(node1)
            nodes.add(node2)
    return len(nodes)

def load_adj_list(file_path):
    """
    Convert an edge list into an adjacency list.
    """
    adj_list = defaultdict(list)
    with open(file_path, "r") as f:
        for line in f:
            node1, node2 = map(int, line.split())
            adj_list[node1].append(node2)
            adj_list[node2].append(node1)  # For undirected graph
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
    Compute closeness centrality for a subset of nodes handled by this process.
    """
    nodes_per_proc = num_nodes // size
    extra_nodes = num_nodes % size
    local_start = rank * nodes_per_proc + min(rank, extra_nodes)
    local_end = local_start + nodes_per_proc + (1 if rank < extra_nodes else 0)

    local_closeness = np.zeros(num_nodes)
    for node in range(local_start, local_end):
        distances = bfs_shortest_paths(adj_list, node, num_nodes)
        reachable = distances[distances < np.inf]  # Ignore unreachable nodes
        if len(reachable) > 1:
            local_closeness[node] = (len(reachable) - 1) / np.sum(reachable)

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

    # Load adjacency list
    if rank == 0:
        # Count the number of nodes dynamically
        num_nodes = count_nodes("facebook_combined.txt")
        adj_list = load_adj_list("facebook_combined.txt")
    else:
        adj_list = None
        num_nodes = None

    # Broadcast adjacency list and number of nodes
    adj_list = comm.bcast(adj_list, root=0)
    num_nodes = comm.bcast(num_nodes, root=0)

    # Compute closeness centrality
    global_closeness = compute_closeness(adj_list, rank, size, num_nodes)

    if rank == 0:
        np.savetxt("output_closeness_centrality.txt", global_closeness)
        top_5_nodes = np.argsort(global_closeness)[-5:][::-1]
        avg_closeness = np.mean(global_closeness)
        print("Top 5 nodes:", top_5_nodes)
        print("Average centrality:", avg_closeness)

    MPI.Finalize()
