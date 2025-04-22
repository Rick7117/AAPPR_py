import networkx as nx
import numpy as np
import scipy.sparse as sp
import collections

def loadsnap(filepath: str):
    """
    Load a graph from a SNAP edge list file.

    Assumes the file format is:
    - Lines starting with '#' are comments.
    - Other lines represent edges: 'node1 node2'.
    - Nodes are represented by integers.
    - Creates an undirected graph, ignoring self-loops and duplicate edges.

    Args:
        filepath (str): Path to the SNAP dataset file (edge list format).

    Returns:
        tuple[nx.Graph, sp.csc_matrix, sp.csc_matrix]:
            - graph (networkx.Graph): The loaded graph.
            - adj_matrix (scipy.sparse.csc_matrix): Sparse adjacency matrix (float64).
            - deg_matrix (scipy.sparse.csc_matrix): Sparse diagonal degree matrix (float64).
    """
    g = nx.Graph()
    node_map = {} # Optional: If nodes are not 0-based integers
    next_node_id = 0
    edges_to_add = set()

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split() # Split by whitespace
            if len(parts) < 2:
                print(f"Skipping malformed line: {line}")
                continue

            try:
                # Assume nodes are integers in the file
                u_orig = int(parts[0])
                v_orig = int(parts[1])

                # --- Node Mapping (if needed, e.g., for non-integer or non-0-based nodes) ---
                # This example assumes direct use of integer IDs found in the file.
                # If remapping to 0-based contiguous integers is required:
                # if u_orig not in node_map:
                #     node_map[u_orig] = next_node_id
                #     next_node_id += 1
                # if v_orig not in node_map:
                #     node_map[v_orig] = next_node_id
                #     next_node_id += 1
                # u = node_map[u_orig]
                # v = node_map[v_orig]
                # For simplicity now, we use original IDs directly. Ensure they are suitable.
                u = u_orig
                v = v_orig
                # --- End Node Mapping ---


                # Ensure undirected representation (smaller index first) and avoid self-loops
                if u != v:
                    source = min(u, v)
                    dest = max(u, v)
                    edges_to_add.add((source, dest)) # Use a set to handle duplicates implicitly

            except ValueError:
                print(f"Skipping line with non-integer nodes: {line}")
                continue

    # Add nodes and edges to the graph
    # Add all nodes involved in the edges first
    all_nodes = set()
    for u, v in edges_to_add:
        all_nodes.add(u)
        all_nodes.add(v)
    g.add_nodes_from(sorted(list(all_nodes))) # Add nodes in sorted order for consistency
    
    # Add edges from the set
    g.add_edges_from(list(edges_to_add))

    # Create the adjacency and degree matrices
    # Ensure nodes are sorted for consistent matrix representation if nodes were added out of order
    # node_list = sorted(g.nodes()) # Get nodes in sorted order
    # adj_matrix = nx.adjacency_matrix(g, nodelist=node_list).astype(np.float64).tocsc()
    
    # If nodes added were already sorted (or if order doesn't strictly matter as long as consistent)
    adj_matrix = nx.adjacency_matrix(g).astype(np.float64).tocsc()

    # Calculate degrees
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    deg_matrix = sp.diags(degrees, format='csc', dtype=np.float64)

    return g, adj_matrix, deg_matrix

# Example Usage (optional, requires a sample SNAP file e.g., 'sample_graph.txt')
if __name__ == '__main__':
    # Create a dummy SNAP file for testing
    dummy_file = 'sample_graph.txt'
    with open(dummy_file, 'w') as f:
        f.write("# Sample graph file\n")
        f.write("0 1\n")
        f.write("0 2\n")
        f.write("1 2\n")
        f.write("2 3\n")
        f.write("3 3\n") # Self-loop, should be ignored
        f.write("0 1\n") # Duplicate, should be ignored

    try:
        graph, adj, deg = loadsnap(dummy_file)
        print("Graph loaded:")
        print("Nodes:", graph.nodes())
        print("Edges:", graph.edges())
        print("\nAdjacency Matrix (sparse):\n", adj)
        print("\nDegree Matrix (sparse):\n", deg)
    except FileNotFoundError:
        print(f"Error: Sample file '{dummy_file}' not found. Cannot run example.")
    except Exception as e:
        print(f"An error occurred: {e}")
