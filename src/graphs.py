import networkx as nx
import numpy as np
import scipy.sparse as sp
import random

def create_connected_graph(n: int, edge_probability: float):
    """
    Create a random connected graph with n nodes and edge probability edge_probability.

    Args:
        n (int): Number of nodes in the graph.
        edge_probability (float): Probability of an edge between any two nodes.

    Returns:
        tuple[sp.csc_matrix, sp.csc_matrix]: 
            - Sparse adjacency matrix of the graph (float64).
            - Sparse diagonal matrix of the degrees of each node (float64).
    """
    if not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be between 0 and 1")
    if n < 1:
        raise ValueError("n must be greater than 0")

    # Create a graph using networkx's random graph generator (Erdos-Renyi)
    # Note: nx.erdos_renyi_graph might not guarantee connectivity initially
    g = nx.Graph()
    g.add_nodes_from(range(n)) # Start node indices from 0 in Python
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() <= edge_probability:
                g.add_edge(i, j)

    # Ensure the graph is connected
    while not nx.is_connected(g):
        components = list(nx.connected_components(g))
        if len(components) <= 1: # Should not happen if not connected, but safe check
             break 
        # Select two different connected components at random
        comp1_idx, comp2_idx = random.sample(range(len(components)), 2)
        
        # Select a random node from each component
        node1 = random.choice(list(components[comp1_idx]))
        node2 = random.choice(list(components[comp2_idx]))
        
        # Add an edge to connect the components
        g.add_edge(node1, node2)

    # Make sure that the graph is connected (assertion)
    assert nx.is_connected(g), "Graph could not be connected."

    # Create the adjacency and degree matrices
    # networkx returns adjacency matrix with nodes 0 to n-1
    adj_matrix = nx.adjacency_matrix(g).astype(np.float64) 
    # Ensure it's in CSC format for consistency if needed, though CSR is often default/efficient
    adj_matrix = adj_matrix.tocsc() 

    # Calculate degrees
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    deg_matrix = sp.diags(degrees, format='csc', dtype=np.float64)

    return adj_matrix, deg_matrix

def get_neighborhood_indices(indices: list[int], adj_mat: sp.spmatrix) -> list[int]:
    """
    Get the neighborhood indices of a set of nodes in a graph represented by an adjacency matrix.

    Args:
        indices (list[int]): A list containing the indices of nodes (0-based) for which we want to find the neighborhood.
        adj_mat (sp.spmatrix): The sparse adjacency matrix of the graph (e.g., CSC or CSR format).

    Returns:
        list[int]: A sorted list of unique indices (0-based) of the nodes in the neighborhood of the input nodes (including the input nodes themselves).
    """
    num_nodes = adj_mat.shape[0]
    if indices and max(indices) >= num_nodes:
         raise ValueError("An index in 'indices' is out of bounds for the graph size")
    if indices and min(indices) < 0:
         raise ValueError("Indices must be non-negative")

    if not indices: # If indices list is empty, maybe return empty list or handle as needed
        return []

    # Ensure adj_mat is in CSC format for efficient column slicing
    if not isinstance(adj_mat, sp.csc_matrix):
        adj_mat = adj_mat.tocsc()

    neighborhood_indices_set = set(indices) # Start with the initial indices

    for idx in indices:
        # Find row indices for column idx using indptr and indices attributes of CSC matrix
        start_ptr = adj_mat.indptr[idx]
        end_ptr = adj_mat.indptr[idx + 1]
        neighbors = adj_mat.indices[start_ptr:end_ptr]
        neighborhood_indices_set.update(neighbors)

    # Convert set to sorted list
    final_indices = sorted(list(neighborhood_indices_set))
    
    assert all(idx in final_indices for idx in indices), "Input indices are not a subset of the result."
    
    return final_indices

if __name__ == '__main__':
    n_nodes = 10
    prob = 0.3
    adj, deg = create_connected_graph(n_nodes, prob)
    print("Adjacency Matrix (sparse):\n", adj)
    print("\nDegree Matrix (sparse):\n", deg)

    seed_nodes = [0, 3]
    neighborhood = get_neighborhood_indices(seed_nodes, adj)
    print(f"\nNeighborhood of nodes {seed_nodes}: {neighborhood}")

    # Test with empty indices
    neighborhood_empty = get_neighborhood_indices([], adj)
    print(f"\nNeighborhood of empty set: {neighborhood_empty}")
