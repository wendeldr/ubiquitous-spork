# --- Required Packages ---
using HDF5
using ProgressMeter
using LinearAlgebra # Added for potential use in extract_class_connections
using Plots
using AverageShiftedHistograms
using Statistics
using DataFrames
using CSV
using HypothesisTests
using StatsBase

# --- Define the path ---
const path = "/media/dan/Data/outputs/ubiquitous-spork/pyspi_combined_patient_hdf5s"

# --- Function Definition (Translated from Python) ---
"""
Extracts connections from the upper triangle of adjacency matrices based on SOZ mask classes.
... (full docstring from above) ...
"""
function extract_class_connections(
    adjacency_matrices::AbstractArray{T, 3},
    soz::AbstractVector{S}
    ) where {T<:Real, S<:Union{Bool, Integer}}

    n = size(adjacency_matrices, 2)
    num_time_steps = size(adjacency_matrices, 1)

    if length(soz) != n
        error("Dimension mismatch: length of soz ($(length(soz))) does not match adjacency matrix dimension ($n)")
    end

    # Initialize arrays to store the connections
    non_connections = T[]
    mix_connections = T[]
    soz_connections = T[]

    # Loop through each time step
    for t in 1:num_time_steps
        # Loop through upper triangle of the adjacency matrix
        for i in 1:n
            for j in (i+1):n
                # Get the connection value
                value = adjacency_matrices[t, i, j]
                
                # Determine the connection type based on SOZ values
                if soz[i] == 0 && soz[j] == 0
                    push!(non_connections, value)
                elseif (soz[i] == 1 && soz[j] == 0) || (soz[i] == 0 && soz[j] == 1)
                    push!(mix_connections, value)
                elseif soz[i] == 1 && soz[j] == 1
                    push!(soz_connections, value)
                end
            end
        end
    end

    return non_connections, mix_connections, soz_connections
end

# --- Function to compute statistics for a group ---
function compute_group_statistics(data::Vector{T}) where T<:Real
    if isempty(data)
        return Dict(
            :mean => NaN,
            :median => NaN,
            :std => NaN,
            :min => NaN,
            :max => NaN,
            :count => 0,
            :kurtosis => NaN,
            :skewness => NaN
        )
    end
    
    # Calculate basic statistics
    stats = Dict(
        :mean => mean(data),
        :median => median(data),
        :std => std(data),
        :min => minimum(data),
        :max => maximum(data),
        :count => length(data)
    )
    
    # Add kurtosis and skewness if we have enough data points
    if length(data) >= 4
        stats[:kurtosis] = kurtosis(data)
        stats[:skewness] = skewness(data)
    else
        stats[:kurtosis] = NaN
        stats[:skewness] = NaN
    end
    
    return stats
end

# --- Function to perform statistical tests between groups ---
function compare_groups(group1::Vector{T}, group2::Vector{T}) where T<:Real
    if isempty(group1) || isempty(group2)
        return Dict(
            :p_value => NaN,
            :test_statistic => NaN,
            :test_name => "Insufficient data"
        )
    end
    
    # Perform Mann-Whitney U test (non-parametric)
    test = MannWhitneyUTest(group1, group2)
    return Dict(
        :p_value => pvalue(test),
        :test_statistic => test.U,
        :test_name => "Mann-Whitney U Test"
    )
end

# --- First Pass: Identify files to process and all metrics ---
full_metrics = String[]
files_to_process = String[]

println("Scanning files to identify metrics and valid patients...")
try
    all_files = readdir(path)
    # Filter and sort HDF5 files directly
    h5_files = sort([f for f in all_files if endswith(f, ".h5")])

    @showprogress "Scanning files: " for patient_file in h5_files
        in_path = joinpath(path, patient_file)
        skip = false
        local keys_in_file = String[] # Ensure it's defined even if h5open fails

        try
            h5open(in_path, "r") do f
                # Check necessary paths exist
                if !haskey(f, "metadata/adjacency_matrices") || !haskey(f, "metadata/patient_info/soz")
                    # @warn "Skipping $patient_file: Missing required HDF5 groups/datasets."
                    skip = true
                else
                    local soz_data # Make local to avoid scope issues if read fails
                    try
                        soz_data = read(f["metadata/patient_info/soz"])
                        if !(eltype(soz_data) <: Number)
                            skip = true
                        elseif sum(soz_data) == 0
                            skip = true
                        else
                            # Get keys only if SOZ is valid
                            keys_in_file = keys(f["metadata/adjacency_matrices"])
                        end
                    catch e_read
                        # @error "Error reading SOZ data from $in_path. Skipping." exception=(e_read, catch_backtrace())
                        skip = true
                    end
                end
            end
        catch e_open
            skip = true
        end

        if skip
            continue
        end

        # If we reached here, the file is valid
        push!(files_to_process, in_path)
        append!(full_metrics, keys_in_file)
    end
catch e_readdir
    @error "Error reading directory $path" exception=(e_readdir, catch_backtrace())
end

full_metrics = unique(full_metrics)
println("Found $(length(files_to_process)) valid files.")
println("Found $(length(full_metrics)) unique metrics.")

# --- Second Pass: Process each metric across all selected files ---
println("\nProcessing metrics...")

@showprogress length(full_metrics) "Processing metrics: " for metric in full_metrics
    try
        println("\nProcessing metric: $metric")

        # Determine the data type from the first file/metric if possible, otherwise default to Float64
        # This assumes all adjacency matrices have the same element type.
        DEFAULT_TYPE = Float64
        DATA_TYPE = DEFAULT_TYPE
        if !isempty(files_to_process) && !isempty(full_metrics)
            try
                h5open(files_to_process[1], "r") do f
                    if haskey(f["metadata/adjacency_matrices"], metric)
                        DATA_TYPE = eltype(read(f["metadata/adjacency_matrices"][metric]))
                        #  println("Detected data type as $DATA_TYPE")
                    end
                end
            catch e
                # @warn "Could not determine data type from first file/metric. Defaulting to $DEFAULT_TYPE."
            end
        end

        # Dictionary to store arrays for each ILAE score
        ilae_data = Dict{Int, Dict{Symbol, Vector{DATA_TYPE}}}()

        @showprogress length(files_to_process) "Files for $metric: " for in_path in files_to_process
            try
                h5open(in_path, "r") do f
                    if haskey(f["metadata/adjacency_matrices"], metric)
                        data = read(f["metadata/adjacency_matrices"][metric])
                        soz = read(f["metadata/patient_info/soz"])
                        ilae_score = read(f["metadata/patient_info/ilae"])

                        # Initialize dictionary entry for this ILAE score if it doesn't exist
                        if !haskey(ilae_data, ilae_score)
                            ilae_data[ilae_score] = Dict(
                                :non_nums => Vector{DATA_TYPE}(),
                                :mix_nums => Vector{DATA_TYPE}(),
                                :soz_nums => Vector{DATA_TYPE}()
                            )
                        end

                        if ndims(data) == 3
                            out = extract_class_connections(data, soz)
                            append!(ilae_data[ilae_score][:non_nums], out[1])
                            append!(ilae_data[ilae_score][:mix_nums], out[2])
                            append!(ilae_data[ilae_score][:soz_nums], out[3])
                        elseif ndims(data) == 2
                            @warn "Data for metric '$metric' in file '$in_path' is 2D. Reshaping to 3D with time=1."
                            data_3d = reshape(data, size(data)..., 1)
                            out = extract_class_connections(data_3d, soz)
                            append!(ilae_data[ilae_score][:non_nums], out[1])
                            append!(ilae_data[ilae_score][:mix_nums], out[2])
                            append!(ilae_data[ilae_score][:soz_nums], out[3])
                        else
                            @warn "Data for metric '$metric' in file '$in_path' has unexpected dimensions ($(ndims(data))). Skipping this entry."
                        end
                    else
                        @warn "Metric '$metric' not found in file '$in_path' during second pass. Skipping file for this metric."
                    end
                end
            catch e
                @error "Error processing metric '$metric' in file '$in_path'. Skipping file for this metric." exception=(e, catch_backtrace())
            end
        end

        # Process and plot for each ILAE score
        for (ilae_score, data_dict) in ilae_data
            println("Processing ILAE score $ilae_score for metric: $metric")
            
            # Clean the data
            non_nums = filter(isfinite, data_dict[:non_nums])
            mix_nums = filter(isfinite, data_dict[:mix_nums])
            soz_nums = filter(isfinite, data_dict[:soz_nums])

            # Compute statistics for each group
            non_stats = compute_group_statistics(non_nums)
            mix_stats = compute_group_statistics(mix_nums)
            soz_stats = compute_group_statistics(soz_nums)

            # Perform statistical comparisons
            non_mix_test = compare_groups(non_nums, mix_nums)
            non_soz_test = compare_groups(non_nums, soz_nums)
            mix_soz_test = compare_groups(mix_nums, soz_nums)

            # Create a DataFrame row for this metric and ILAE score
            row = Dict(
                :metric => metric,
                :ilae_score => ilae_score,
                :non_mean => non_stats[:mean],
                :non_median => non_stats[:median],
                :non_std => non_stats[:std],
                :non_count => non_stats[:count],
                :mix_mean => mix_stats[:mean],
                :mix_median => mix_stats[:median],
                :mix_std => mix_stats[:std],
                :mix_count => mix_stats[:count],
                :soz_mean => soz_stats[:mean],
                :soz_median => soz_stats[:median],
                :soz_std => soz_stats[:std],
                :soz_count => soz_stats[:count],
                :non_mix_p_value => non_mix_test[:p_value],
                :non_soz_p_value => non_soz_test[:p_value],
                :mix_soz_p_value => mix_soz_test[:p_value]
            )

            # Append to results DataFrame
            if !@isdefined(results_df)
                results_df = DataFrame(row)
            else
                push!(results_df, row)
            end
        end

    catch e
        @error "Error processing metric '$metric'. Skipping." exception=(e, catch_backtrace())
    end
end

# Save the results to CSV
if @isdefined(results_df)
    output_path = joinpath(path, "statistical_analysis_results.csv")
    CSV.write(output_path, results_df)
    println("\nResults saved to: $output_path")
end

println("\nProcessing complete.")
