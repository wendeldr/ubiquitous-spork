import networkx as nx
import numpy as np

import scipy.io
from tqdm import tqdm

def nan_degree(G, node):
    total = 0
    for nbr, data in G[node].items():
        w = data.get('weight', 1)
        if isinstance(w, float) and np.isnan(w):
            w = 0  # treat NaN as 0;
        total += w
    return total

def undirected_edge_removal(adj_matrix, removal_order="highest"):
    """
    Edge removal on an undirected graph.
    Removal stops when any node becomes isolated (degree==0).

    Parameters:
        adj_matrix (2D array): symmetric matrix of edge weights.
        removal_order (str): "highest" or "lowest" to determine removal order.

    Returns:
        removed_edges (NxN array): adjacency matrix after edges have been removed.
        removal_step (int): number of edges removed when break occurred.
    """
    # make adj_matrix only upper triangular
    adj = adj_matrix.copy()
    adj[np.tril_indices(adj.shape[0], k=-1)] = np.nan
    G = nx.DiGraph()
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if not np.isnan(adj[i, j]):
                G.add_edge(i, j, weight=adj[i, j])

    # Get edges with weights
    edges = [(u, v, data["weight"]) for u, v, data in G.edges(data=True)]

    # Sort edges based on weight
    reverse = True if removal_order == "highest" else False
    sorted_edges = sorted(edges, key=lambda x: x[2], reverse=reverse)
    
    removal_step = 0
    for u, v, w in sorted_edges:
        G.remove_edge(u, v)

        # Check if any node is isolated
        if not nx.is_weakly_connected(G):
            G.add_edge(u, v, weight=w)
            adj = nx.to_numpy_array(G, nonedge=float('nan'))
            # make adj_matrix symmetric
            adj = np.triu(adj)
            adj = adj + adj.T
            return adj, removal_step

        removal_step += 1
    
    adj = nx.to_numpy_array(G, nonedge=float('nan'))
    # make adj_matrix symmetric
    adj = np.triu(adj)
    adj = adj + adj.T
    return adj, removal_step

def simulate_directed_edge_removal(adj_matrix, removal_order="highest"):
    """
    Simulate edge removal on a directed graph.
    Tracks two conditions:
      1) When the graph is no longer strongly connected.
      2) When any node becomes completely isolated (in_degree + out_degree == 0).

    Parameters:
        adj_matrix (2D array): matrix of edge weights (diagonal assumed 0).
        removal_order (str): "highest" or "lowest" removal order.

    Returns:
        removed_edges (list): list of removed edges [(u,v,weight), ...]
        break_traversability (tuple): (step, edge) at which strong connectivity is lost.
        break_isolation (tuple): (step, edge, node) at which a node becomes isolated.
    """
    n = len(adj_matrix)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # Add directed edges (excluding self-loops)
    for i in range(n):
        for j in range(n):
            if i != j:
                weight = adj_matrix[i][j]
                G.add_edge(i, j, weight=weight)
    
    edges = [(u, v, data["weight"]) for u, v, data in G.edges(data=True)]
    reverse = True if removal_order == "highest" else False
    sorted_edges = sorted(edges, key=lambda x: x[2], reverse=reverse)
    
    removed_edges = []
    break_traversability = None   # (step, edge) when strong connectivity is lost
    break_isolation = None        # (step, edge, node) when a node becomes isolated
    removal_step = 0
    
    for u, v, w in sorted_edges:
        G.remove_edge(u, v)
        removal_step += 1
        removed_edges.append((u, v, w))
        
        # Check for strong connectivity break (first occurrence)
        if break_traversability is None:
            if not nx.is_strongly_connected(G):
                break_traversability = (removal_step, (u, v, w))
        
        # Check if any node is completely isolated (no in- and out-edges)
        if break_isolation is None:
            for node in G.nodes():
                if G.in_degree(node) == 0 and G.out_degree(node) == 0:
                    break_isolation = (removal_step, (u, v, w), node)
                    break
        
        # Stop if both break conditions have been encountered
        if break_traversability is not None and break_isolation is not None:
            break
            
    return removed_edges, break_traversability, break_isolation


if __name__ == '__main__':
    path = "/media/dan/Data2/calculations/connectivity/additional_calcs/mats/pdist_euclidean/pdist_euclidean~001.mat"
    mat = scipy.io.loadmat(path)
    undirected_adj = mat["measure"]
    n = undirected_adj.shape[0]

    steps = []
    adj_hats = []
    for i in tqdm(range(n)):
        adj = undirected_adj[i, ...]
        adj_hat, step = undirected_edge_removal(adj, removal_order="highest")
        steps.append(step)
        adj_hats.append(adj_hat)

    steps = np.array(steps)
    adj_hats = np.array(adj_hats)

    np.save("steps.npy", steps)
    np.save("adj_hats.npy", adj_hats)


    # removed_edges, isolated_node, step = simulate_undirected_edge_removal(undirected_adj, removal_order="highest")
    # print("Undirected Graph:")

    # removed_edges, isolated_node, step = simulate_undirected_edge_removal(undirected_adj, removal_order="highest")
    # print("Undirected Graph:")
    # print("Edges removed until a node became isolated:", step)
    # if isolated_node is not None:
    #     print("Node isolated:", isolated_node)
    # else:
    #     print("No node became isolated.")

    # # Example for directed graph:
    # # Generate a random matrix for a directed graph (set diagonal to 0)
    # directed_adj = np.random.rand(n, n)
    # np.fill_diagonal(directed_adj, 0)

    # removed_edges_dir, break_trav, break_iso = simulate_directed_edge_removal(directed_adj, removal_order="highest")
    # print("\nDirected Graph:")
    # if break_trav is not None:
    #     print(f"Strong connectivity lost at removal step {break_trav[0]} (edge removed: {break_trav[1]})")
    # else:
    #     print("Graph remained strongly connected after all removals.")
    # if break_iso is not None:
    #     print(f"A node became isolated at removal step {break_iso[0]} (edge removed: {break_iso[1]}). Node {break_iso[2]} is isolated.")
    # else:
    #     print("No node became completely isolated after all removals.")
