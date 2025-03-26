
import os
import matplotlib.pyplot as plt
import scipy
import numpy as np


path = "/media/dan/Data2/calculations/connectivity/additional_calcs/mats"
folders = os.listdir(path)


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
    cnt = 0
    while low < high:
        cnt += 1
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
            threshold_index = mid
            high = mid

    # build the symmetric thresholded adjacency matrix
    adj_hat = np.empty_like(adj_matrix)
    adj_hat[:] = np.nan
    for i in range(threshold_index):
        u, v, w = sorted_edges[i]
        adj_hat[u, v] = w
        adj_hat[v, u] = w

    # threshold_index is the minimal number of removals needed
    return  adj_hat, threshold_index

# Example usage:
# adj_matrix = np.array(...)  # your adjacency matrix
# threshold = find_disconnect_threshold(adj_matrix, removal_order="highest")
# print("Graph disconnects after", threshold, "edge removals.")



folderpath = "/media/dan/Data2/calculations/connectivity/additional_calcs/mats"
for subfolder, removal_order in undirected_subfolders.items():
    measure_path = os.path.join(folderpath, subfolder)
    measure_files = os.listdir(measure_path)
    print(subfolder)
    for measure_file in measure_files:
        # Remove .mat from filename
        print(measure_file)
        base_name = os.path.basename(measure_file).replace(".mat", "")
        mat = scipy.io.loadmat(os.path.join(measure_path, measure_file))
        undirected_adj = mat["measure"]
        n = undirected_adj.shape[0]
        break
    break


i = 0
measure= 'bary-sq_euclidean_max'
adj = undirected_adj[i, ...]

adj_hat, threshold = find_disconnect_threshold(adj, removal_order=undirected_subfolders[measure])


path = "/media/dan/Data2/calculations/connectivity/additional_calcs/thresholded_mats/bary-sq_euclidean_max"
files = sorted(os.listdir(path))

data = {}
for z, file in enumerate(files):
    if file.endswith(".npy"):
        # file format: base_name + ~{pid}~{i:06}~threshadj~{removal_order}~{step}.npy 
        # example bary-sq_euclidean_max~001~000000~threshadj~lowest~5587.npy
        # extract pid,  i, step
        pid = int(file.split("~")[1])
        if pid != 1:
            continue
        if z == 0:
            print(file)
        i = int(file.split("~")[2])
        step = int(file.split("~")[5].split(".")[0])
        if pid not in data:
            data[pid] = {'net': [], 'steps': []}
        with open(os.path.join(path, file), "rb") as f:
            net = np.load(f)
        data[pid]['net'].append(net)
        data[pid]['steps'].append(step)




a = data[1]['net'][0]

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(a)
# ax[1].imshow(adj_hat)







