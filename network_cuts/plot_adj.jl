using Distributed
using NPZ
using Plots
using Printf
using Base.Threads

# ---------------------------
# Setup Distributed Processing
# ---------------------------
addprocs(32)  # spawn 32 worker processes

@everywhere begin
    using NPZ, Plots
    # The plotting function uses the worker’s global color limits,
    # which will be set later.
    function plot_data(file_tuple)
        # Unpack tuple: (metric, metric_path, file, pid, output_dir, A)
        metric, metric_path, file, pid, output_dir, A = file_tuple

        parts = split(file, "~")
        # Expect format: <metric>~<pid>~<slice>~threshadj~<removal>~<threshold>.npy
        slice_number  = parts[3]
        removal_order = parts[5]
        threshold     = replace(parts[6], ".npy" => "")

        # Prepare output directory and filename.
        out_dir = joinpath(output_dir, metric, pid)
        mkpath(out_dir)
        out_filename = joinpath(out_dir, "$(slice_number)~$(removal_order)~$(threshold).png")

        # Generate heatmap with a square aspect ratio and uniform color limits.
        plt = heatmap(A,
            color        = :jet1,
            clims        = (global_min_worker, global_max_worker),
            aspect_ratio = :equal,   # force square plot
            axis         = nothing,
            border       = :none,
            colorbar     = true
        )
        soz=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # Ensure soz is a matrix of the same dimensions as A.
        soz_matrix = repeat(soz, 1, size(A, 2))

        # Overlay a contour at the level 0.5 (between 0 and 1).
        contour!(plt, soz_matrix, levels=[0.5], linecolor=:black, linewidth=20)
        savefig(plt, out_filename)
    end
end

function main()
    # ---------------------------
    # User-configurable settings
    # ---------------------------
    input_dir  = "/media/dan/Data2/calculations/connectivity/additional_calcs/julia_thresholded_mats"
    output_dir = "/media/dan/Data2/calculations/connectivity/additional_calcs/julia_thresholded_mats_images"

    # Set filters if desired (set to nothing to process all)
    filter_metric = "bary_euclidean_max"  # e.g. "bary_euclidean_max"
    filter_pids   = ["001"]         # e.g. ["001", "002"]

    # ---------------------------
    # Pre-filtering: Collect file list
    # ---------------------------
    metric_dirs = filter(d -> isdir(joinpath(input_dir, d)), readdir(input_dir))
    if filter_metric !== nothing
        metric_dirs = filter(m -> m == filter_metric, metric_dirs)
    end

    file_list = []
    for metric in metric_dirs
        metric_path = joinpath(input_dir, metric)
        for file in readdir(metric_path)
            if endswith(file, ".npy")
                parts = split(file, "~")
                if length(parts) < 6
                    continue  # skip malformed filenames
                end
                pid = parts[2]
                if filter_pids === nothing || pid in filter_pids
                    push!(file_list, (metric, metric_path, file, pid, output_dir))
                end
            end
        end
    end

    # ---------------------------
    # Stage 1: Read Data (Threaded) and compute global color limits
    # ---------------------------
    nfiles = length(file_list)
    # results will be tuples: (metric, metric_path, file, pid, output_dir, A)
    results = Vector{Union{Nothing, Tuple{String, String, String, String, String, Array}}}(undef, nfiles)

    @threads for i in 1:nfiles
        metric, metric_path, file, pid, out_dir = file_list[i]
        file_path = joinpath(metric_path, file)
        A = try
            NPZ.npzread(file_path)
        catch e
            @printf("Error reading file %s: %s\n", file_path, e)
            nothing
        end
        if A === nothing
            results[i] = nothing
        else
            results[i] = (metric, metric_path, file, pid, out_dir, A)
        end
    end

    # Remove any failed reads.
    data_list = filter(!isnothing, results)

    # Compute global min and max over all matrices, ignoring NaN values.
    local_min = Inf
    local_max = -Inf
    for tup in data_list
        A = tup[6]
        valid = A[.!isnan.(A)]
        if isempty(valid)
            continue  # Skip if no valid (non-NaN) values exist.
        end
        local_min = min(local_min, minimum(valid))
        local_max = max(local_max, maximum(valid))
    end
    @printf("Global min: %f, Global max: %f\n", local_min, local_max)

    # ---------------------------
    # Stage 2: Distributed Plotting
    # ---------------------------
    # Set global limits on all workers.
    @everywhere global_min_worker = $local_min
    @everywhere global_max_worker = $local_max

    @time pmap(plot_data, data_list)
end

main()
