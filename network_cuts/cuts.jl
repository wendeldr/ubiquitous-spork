using MAT
using NPZ
using Printf
using ProgressMeter
using Base.Threads
using FilePathsBase: basename

# --- Function to compute the thresholded adjacency matrix ---
function find_disconnect_threshold(adj_matrix::Matrix{Float64}, removal_order::String = "highest")
    n = size(adj_matrix, 1)
    # Work on a copy and set lower-triangular elements to NaN.
    adj = copy(adj_matrix)
    for i in 2:n
        for j in 1:i-1
            adj[i,j] = NaN
        end
    end

    # Build edge list from the upper triangle (including diagonal)
    edges = Tuple{Int,Int,Float64}[]
    for i in 1:n
        for j in i:n
            if !isnan(adj[i,j])
                push!(edges, (i, j, adj[i,j]))
            end
        end
    end

    # Sort edges by weight; descending if removal_order=="highest", ascending otherwise.
    sorted_edges = sort(edges, by = x -> x[3], rev = (removal_order == "highest"))
    m = length(sorted_edges)

    # Binary search for the minimal number of removals that disconnect the graph.
    low = 0
    high = m
    threshold_index = m  # in 0-indexed terms
    while low < high
        mid = (low + high) ÷ 2
        parent = collect(1:n)
        # Union-find "find" with path compression.
        function find(x)
            while parent[x] != x
                parent[x] = parent[parent[x]]
                x = parent[x]
            end
            return x
        end
        function union(x, y)
            rx = find(x)
            ry = find(y)
            if rx != ry
                parent[ry] = rx
            end
        end

        # Add edges from sorted_edges[mid:] (using 0-indexing: Julia indices mid+1:end)
        if mid < m
            for idx in (mid+1):m
                (u, v, _) = sorted_edges[idx]
                union(u, v)
            end
        end

        rep = find(1)
        connected = all(find(v) == rep for v in 1:n)

        if connected
            low = mid + 1
        else
            threshold_index = mid - 1
            high = mid
        end
    end

    # Build the symmetric thresholded matrix (fill with NaN)
    adj_hat = fill(NaN, size(adj_matrix))
    # In Python code, the kept edges are those with indices >= threshold_index (0-indexed)
    # so here we loop over indices threshold_index+1 to m.
    for idx in (threshold_index+1):m
        (u, v, w) = sorted_edges[idx]
        adj_hat[u,v] = w
        adj_hat[v,u] = w
    end

    return adj_hat, threshold_index
end

# --- Process a single slice ---
function process_slice(i::Int, undirected_adj, removal_order::String, output_subfolder::String, base_name::String)
    try
        # Assume undirected_adj is a 3D array and each slice is a matrix.
        # (Using a view avoids copying; we convert to Array for processing.)
        adj_matrix = Array(@view undirected_adj[i, :, :])
        adj_hat, step = find_disconnect_threshold(adj_matrix, removal_order)
        # File naming: use 0-indexing to format the slice number.
        fname = joinpath(output_subfolder, base_name * "~" *
                         @sprintf("%06d", i-1) * "~threshadj~" * removal_order * "~" *
                         string(step) * ".npy")
        NPZ.npzwrite(fname, adj_hat)
    catch err
        println("Error processing slice $i: ", err)
    end
end

# --- Main processing function ---
function main()
    folderpath = "/media/dan/Data2/calculations/connectivity/additional_calcs/mats"
    output_folder = "/media/dan/Data2/calculations/connectivity/additional_calcs/julia_thresholded_mats"

    # Mapping from subfolder names to removal_order.
    undirected_subfolders = Dict(
        "bary-sq_euclidean_max"                          => "lowest",
        "bary-sq_euclidean_mean"                         => "lowest",
        "bary_euclidean_max"                             => "lowest",
        "ce_gaussian"                                    => "highest",
        "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195" => "lowest",
        "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342"    => "lowest",
        "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122"     => "lowest",
        "cohmag_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391"    => "lowest",
        "cohmag_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586"    => "lowest",
        "cohmag_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146"     => "lowest",
        "cohmag_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342"      => "lowest",
        "cohmag_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732"      => "lowest",
        "cohmag_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122"       => "lowest",
        "cohmag_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122"       => "lowest",
        "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5"              => "lowest",
        "cov-sq_EmpiricalCovariance"                     => "lowest",
        "cov-sq_GraphicalLassoCV"                        => "lowest",
        "cov-sq_LedoitWolf"                              => "lowest",
        "cov-sq_MinCovDet"                               => "lowest",
        "cov-sq_OAS"                                     => "lowest",
        "cov-sq_ShrunkCovariance"                        => "lowest",
        "cov_EmpiricalCovariance"                        => "lowest",
        "cov_GraphicalLassoCV"                           => "lowest",
        "cov_LedoitWolf"                                 => "lowest",
        "cov_MinCovDet"                                  => "lowest",
        "cov_OAS"                                      => "lowest",
        "cov_ShrunkCovariance"                           => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195" => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342"    => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122"     => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391"    => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586"    => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146"     => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342"      => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122"       => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122"       => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0_fmax-0-5"              => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195"  => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342"     => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122"      => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391"     => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586"     => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146"      => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342"       => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122"        => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122"        => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-5"               => "lowest",
        "mi_gaussian"                                   => "lowest",
        "pdist_braycurtis"                              => "highest",
        "pdist_canberra"                               => "highest",
        "pdist_chebyshev"                              => "highest",
        "pdist_cityblock"                              => "highest",
        "pdist_cosine"                                 => "highest",
        "pdist_euclidean"                              => "highest",
        "pec"                                          => "lowest",
        "pec_log"                                      => "lowest",
        "pec_orth_abs"                                 => "lowest",
        "pec_orth_log"                                 => "lowest",
        "pec_orth_log_abs"                             => "lowest",
        "dspli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732" => "lowest",
        "dswpli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732" => "lowest",
        "kendalltau-sq"                                => "lowest",
        "pec_orth"                                     => "lowest",
        "ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195" => "lowest",
        "prec_OAS"                                     => "highest",
        "prec-sq_GraphicalLasso"                       => "lowest",
        "prec-sq_GraphicalLassoCV"                     => "lowest",
        "prec-sq_LedoitWolf"                           => "lowest",
        "prec-sq_OAS"                                  => "lowest",
        "prec-sq_ShrunkCovariance"                     => "lowest",
        "prec_GraphicalLasso"                          => "highest",
        "prec_GraphicalLassoCV"                        => "highest",
        "prec_LedoitWolf"                              => "highest",
        "prec_ShrunkCovariance"                        => "highest",
        "spearmanr"                                    => "lowest",
        "spearmanr-sq"                                 => "lowest",
        "xcorr-sq_max_sig-False"                       => "lowest",
        "xcorr-sq_mean_sig-False"                      => "lowest",
        "xcorr_max_sig-False"                          => "lowest",
        "xcorr_mean_sig-False"                         => "lowest",
        "je_gaussian"                                  => "highest",
        "ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342" => "lowest",
        "ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122"    => "lowest",
        "ppc_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391"   => "lowest",
        "ppc_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586"   => "lowest",
        "ppc_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146"    => "lowest",
        "ppc_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342"     => "lowest",
        "ppc_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732"     => "lowest",
        "ppc_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122"      => "lowest",
        "ppc_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122"      => "lowest",
        "ppc_multitaper_mean_fs-1_fmin-0_fmax-0-5"              => "lowest"
    )

    # Process each subfolder.
    subfolders = collect(keys(undirected_subfolders))
    @showprogress for subfolder in subfolders
        removal_order = undirected_subfolders[subfolder]
        output_subfolder = joinpath(output_folder, subfolder)
        mkpath(output_subfolder)
        completed_file = joinpath(output_subfolder, subfolder * "~completed.txt")
        if isfile(completed_file)
            continue
        end

        measure_path = joinpath(folderpath, subfolder)
        measure_files = readdir(measure_path)
        existing_files = sort(readdir(output_subfolder))
        # Build set of file "stems" (i.e. filename up to the last "~")
        stripped_files = Set{String}()
        for f in existing_files
            idx = findlast(==('~'), f)
            if idx !== nothing
                push!(stripped_files, f[1:idx-1])
            end
        end

        @showprogress for measure_file in measure_files
            base_name = replace(basename(measure_file), ".mat" => "")
            # Load .mat file
            mat_data = matread(joinpath(measure_path, measure_file))
            undirected_adj = mat_data["measure"]
            n = size(undirected_adj, 1)
            # Expected file stems for slices 0:(n-1)
            expected_files = Set{String}(base_name * "~" * @sprintf("%06d", i) * "~threshadj~" * removal_order for i in 0:(n-1))
            if length(intersect(stripped_files, expected_files)) == n
                # Remove these from stripped_files.
                for ef in intersect(stripped_files, expected_files)
                    delete!(stripped_files, ef)
                end
                continue
            end

            # Process each slice in parallel.
            @threads for i in 1:n
                process_slice(i, undirected_adj, removal_order, output_subfolder, base_name)
            end
        end

        # Write a completion marker.
        open(completed_file, "w") do io
            write(io, "completed")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
