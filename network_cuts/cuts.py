import networkx as nx
import numpy as np
import scipy.io
import os
from tqdm import tqdm
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def find_disconnect_threshold(adj_matrix, removal_order="highest"):
    n = adj_matrix.shape[0]
    # Use only the upper triangular portion (assuming symmetry)
    adj = adj_matrix.copy()
    adj[np.tril_indices(n, k=-1)] = np.nan

    # Build the edge list from the upper triangle
    edges = []
    for i in range(n):
        for j in range(i, n):
            if not np.isnan(adj[i, j]):
                edges.append((i, j, adj[i, j]))

    # Sort edges by weight
    reverse = True if removal_order == "highest" else False
    sorted_edges = sorted(edges, key=lambda x: x[2], reverse=reverse)

    # Binary search boundaries
    low, high = 0, len(sorted_edges)
    threshold_index = len(sorted_edges)  # default if graph never disconnects

    # Helper: Union-Find structure
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Binary search for the minimal index that disconnects the graph
    while low < high:
        mid = (low + high) // 2
        # Reset union-find for this iteration
        parent = list(range(n))
        # Add edges from sorted_edges[mid:] (i.e. the ones not yet removed)
        for i in range(mid, len(sorted_edges)):
            u, v, _ = sorted_edges[i]
            union(u, v)

        # Check if the graph is connected
        rep = find(0)
        connected = all(find(i) == rep for i in range(n))

        if connected:
            # Still connected: need to remove more edges.
            low = mid + 1
        else:
            # Disconnected: record mid as a candidate and try a smaller removal.
            threshold_index = mid - 1
            high = mid

    # build the symmetric thresholded adjacency matrix
    adj_hat = np.empty_like(adj_matrix)
    adj_hat[:] = np.nan
    for i in range(threshold_index, len(sorted_edges)):
        u, v, w = sorted_edges[i]
        adj_hat[u, v] = w
        adj_hat[v, u] = w

    # threshold_index is the minimal number of removals needed
    return  adj_hat, threshold_index

# def undirected_edge_removal(adj_matrix, removal_order="highest"):
#     """
#     Edge removal on an undirected graph.
#     Removal stops when any node becomes isolated (degree==0).

#     Parameters:
#         adj_matrix (2D array): symmetric matrix of edge weights.
#         removal_order (str): "highest" or "lowest" to determine removal order.

#     Returns:
#         removed_edges (NxN array): adjacency matrix after edges have been removed.
#         removal_step (int): number of edges removed when break occurred.
#     """
#     # make adj_matrix only upper triangular
#     adj = adj_matrix.copy()
#     adj[np.tril_indices(adj.shape[0], k=-1)] = np.nan
#     G = nx.DiGraph()
#     n = adj.shape[0]
#     for i in range(n):
#         for j in range(n):
#             if not np.isnan(adj[i, j]):
#                 G.add_edge(i, j, weight=adj[i, j])

#     # Get edges with weights
#     edges = [(u, v, data["weight"]) for u, v, data in G.edges(data=True)]

#     # Sort edges based on weight
#     reverse = True if removal_order == "highest" else False
#     sorted_edges = sorted(edges, key=lambda x: x[2], reverse=reverse)
    
#     removal_step = 0
#     for u, v, w in sorted_edges:
#         G.remove_edge(u, v)

#         # Check if any node is isolated
#         if not nx.is_weakly_connected(G):
#             G.add_edge(u, v, weight=w)
#             adj = nx.to_numpy_array(G, nonedge=float('nan'))
#             # make adj_matrix symmetric
#             adj = np.triu(adj)
#             adj = adj + adj.T
#             return adj, removal_step

#         removal_step += 1
    
#     adj = nx.to_numpy_array(G, nonedge=float('nan'))
#     # make adj_matrix symmetric
#     adj = np.triu(adj)
#     adj = adj + adj.T
#     return adj, removal_step

def process_slice(i, undirected_adj, removal_order, output_subfolder, base_name):
    try:
        adj = undirected_adj[i, ...]
        adj_hat, step = undirected_edge_removal(adj, removal_order=removal_order)
        np.save(os.path.join(output_subfolder, base_name + f"~{i:06}~threshadj~{removal_order}~{step}.npy"), adj_hat)
    except Exception as e:
        print(f"Error processing slice {i}: {e}")


def check_if_exists(i, output_subfolder, base_name, removal_order):
    pattern = os.path.join(output_subfolder, base_name + f"~{i:06}~threshadj~{removal_order}~*.npy")
    return glob.glob(pattern)


if __name__ == '__main__':
    folderpath = "/media/dan/Data2/calculations/connectivity/additional_calcs/mats"

    # output_folder = "/media/dan/Data2/calculations/connectivity/additional_calcs/thresholded_mats"
    output_folder = "/media/dan/Data2/calculations/connectivity/additional_calcs/test"

    undirected_subfolders = {'bary-sq_euclidean_max': 'lowest',
    'bary-sq_euclidean_mean': 'lowest',
    'bary_euclidean_max': 'lowest',
    'ce_gaussian': 'highest',
    'cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195': 'lowest',
    'cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342': 'lowest',
    'cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122': 'lowest',
    'cohmag_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391': 'lowest',
    'cohmag_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586': 'lowest',
    'cohmag_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146': 'lowest',
    'cohmag_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342': 'lowest',
    'cohmag_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732': 'lowest',
    'cohmag_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122': 'lowest',
    'cohmag_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122': 'lowest',
    'cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5': 'lowest',
    'cov-sq_EmpiricalCovariance': 'lowest',
    'cov-sq_GraphicalLassoCV': 'lowest',
    'cov-sq_LedoitWolf': 'lowest',
    'cov-sq_MinCovDet': 'lowest',
    'cov-sq_OAS': 'lowest',
    'cov-sq_ShrunkCovariance': 'lowest',
    'cov_EmpiricalCovariance': 'lowest',
    'cov_GraphicalLassoCV': 'lowest',
    'cov_LedoitWolf': 'lowest',
    'cov_MinCovDet': 'lowest',
    'cov_OAS': 'lowest',
    'cov_ShrunkCovariance': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0_fmax-0-5': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-5': 'lowest',
    'mi_gaussian': 'lowest',
    'pdist_braycurtis': 'highest',
    'pdist_canberra': 'highest',
    'pdist_chebyshev': 'highest',
    'pdist_cityblock': 'highest',
    'pdist_cosine': 'highest',
    'pdist_euclidean': 'highest',
    'pec': 'lowest',
    'pec_log': 'lowest',
    'pec_orth_abs': 'lowest',
    'pec_orth_log': 'lowest',
    'pec_orth_log_abs': 'lowest',
    'dspli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732': 'lowest',
    'dswpli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732': 'lowest',
    'kendalltau-sq': 'lowest',
    'pec_orth': 'lowest',
    'ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195': 'lowest',
    'prec_OAS': 'highest',
    'prec-sq_GraphicalLasso': 'lowest',
    'prec-sq_GraphicalLassoCV': 'lowest',
    'prec-sq_LedoitWolf': 'lowest',
    'prec-sq_OAS': 'lowest',
    'prec-sq_ShrunkCovariance': 'lowest',
    'prec_GraphicalLasso': 'highest',
    'prec_GraphicalLassoCV': 'highest',
    'prec_LedoitWolf': 'highest',
    'prec_ShrunkCovariance': 'highest',
    'spearmanr': 'lowest',
    'spearmanr-sq': 'lowest',
    'xcorr-sq_max_sig-False': 'lowest',
    'xcorr-sq_mean_sig-False': 'lowest',
    'xcorr_max_sig-False': 'lowest',
    'xcorr_mean_sig-False': 'lowest',
    'je_gaussian': 'highest',
    'ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342': 'lowest',
    'ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122': 'lowest',
    'ppc_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391': 'lowest',
    'ppc_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586': 'lowest',
    'ppc_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146': 'lowest',
    'ppc_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342': 'lowest',
    'ppc_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732': 'lowest',
    'ppc_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122': 'lowest',
    'ppc_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122': 'lowest',
    'ppc_multitaper_mean_fs-1_fmin-0_fmax-0-5': 'lowest',
    }

    for subfolder, removal_order in tqdm(undirected_subfolders.items(), desc="Processing subfolders"):

        output_subfolder = os.path.join(output_folder, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        # check if the subfolder is already completed
        if os.path.exists(os.path.join(output_folder, subfolder, f"{subfolder}~completed.txt")):
            # print(f"skipping {subfolder} because it is already completed")
            continue
        
        measure_path = os.path.join(folderpath, subfolder)
        measure_files = os.listdir(measure_path)
        existing_files = list(sorted(os.listdir(os.path.join(output_folder, subfolder))))
        # strip the ~{step}.npy from the existing files
        # bary-sq_euclidean_max~001~000000~threshadj~lowest~5587.npy -> bary-sq_euclidean_max~001~000000~threshadj~lowest
        stripped_files = {f.rsplit('~', 1)[0] for f in existing_files}
        for measure_file in tqdm(measure_files, desc="Processing files"):
            # Remove .mat from filename
            base_name = os.path.basename(measure_file).replace(".mat", "")
            pid = base_name.split("~")[1]
            mat = scipy.io.loadmat(os.path.join(measure_path, measure_file))
            undirected_adj = mat["measure"]
            n = undirected_adj.shape[0]

            # make a list of existing files that match the base_name
            expected_files = {f"{base_name}~{i:06}~threshadj~{removal_order}" for i in range(n)}
            pid_files = stripped_files.intersection(expected_files)
            if len(pid_files) == n:
                # update stripped_files to remove the pid_files
                stripped_files.difference_update(pid_files)
                continue

            output_subfolder = os.path.join(output_folder, subfolder)
            os.makedirs(output_subfolder, exist_ok=True)

            with ProcessPoolExecutor(max_workers=32) as executor:

                func = partial(process_slice,
                        undirected_adj=undirected_adj,
                        removal_order=removal_order,
                        output_subfolder=output_subfolder,
                        base_name=base_name)
                list(tqdm(executor.map(func, range(n)), total=n, desc="Processing slices"))

        # make a "completed" file in the output folder so it's not processed again
        os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)
        with open(os.path.join(output_folder, subfolder, f"{subfolder}~completed.txt"), "w") as f:
            f.write(f"completed")