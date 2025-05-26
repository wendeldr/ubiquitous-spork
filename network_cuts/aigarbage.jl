# --- Required Packages ---
using HDF5
using ProgressMeter
using LinearAlgebra
using Plots
using AverageShiftedHistograms
using Statistics
using DataFrames
using CSV
using HypothesisTests
using StatsBase
using NaNStatistics

# --- Define the path ---
const path = "/media/dan/Data/outputs/ubiquitous-spork/pyspi_combined_patient_hdf5s"

# --- Function to extract electrode pairs and values ---
function extract_electrode_pairs(
    adjacency_matrix::AbstractArray{T, 2},
    soz::AbstractVector{S}
    ) where {T<:Real, S<:Union{Bool, Integer}}
    
    n = size(adjacency_matrix, 1)
    pairs = DataFrame(
        electrode_pair = String[],
        electrode_a = Int[],
        electrode_b = Int[],
        soz_sum = Int[]
    )
    
    # Loop through upper triangle of the adjacency matrix
    for i in 1:n
        for j in (i+1):n
            push!(pairs, (
                electrode_pair = "$i-$j",
                electrode_a = i,
                electrode_b = j,
                soz_sum = soz[i] + soz[j]
            ))
        end
    end
    
    return pairs
end

# --- First Pass: Identify files to process and all metrics ---
full_metrics = String[]
files_to_process = String[]

println("Scanning files to identify metrics and valid patients...")
try
    all_files = readdir(path)
    h5_files = sort([f for f in all_files if endswith(f, ".h5")])

    for patient_file in h5_files
        in_path = joinpath(path, patient_file)
        skip = false
        local keys_in_file = String[]

        try
            h5open(in_path, "r") do f
                if !haskey(f, "metadata/adjacency_matrices") || !haskey(f, "metadata/patient_info/soz")
                    skip = true
                else
                    local soz_data
                    try
                        soz_data = read(f["metadata/patient_info/soz"])
                        if !(eltype(soz_data) <: Number)
                            skip = true
                        elseif sum(soz_data) == 0
                            skip = true
                        else
                            keys_in_file = keys(f["metadata/adjacency_matrices"])
                        end
                    catch e_read
                        skip = true
                    end
                end
            end
        catch e_open
            skip = true
        end

        if !skip
            push!(files_to_process, in_path)
            append!(full_metrics, keys_in_file)
        end
    end
catch e_readdir
    @error "Error reading directory $path" exception=(e_readdir, catch_backtrace())
end

full_metrics = unique(full_metrics)
println("Found $(length(files_to_process)) valid files.")
println("Found $(length(full_metrics)) unique metrics.")

# --- Second Pass: Process each metric across all selected files ---
println("\nProcessing metrics...")

# First, collect all unique electrode pairs across all patients
println("Collecting unique electrode pairs...")
all_pairs = DataFrame(
    pid = String[],
    electrode_pair = String[],
    electrode_a = Int[],
    electrode_b = Int[],
    soz_sum = Int[],
    ilae = Int[]
)

@showprogress length(files_to_process) "Collecting pairs: " for in_path in files_to_process
    try
        h5open(in_path, "r") do f
            if haskey(f["metadata/adjacency_matrices"], full_metrics[1])
                data = read(f["metadata/adjacency_matrices"][full_metrics[1]])
                soz = read(f["metadata/patient_info/soz"])
                ilae = read(f["metadata/patient_info/ilae"])
                pid = split(basename(in_path), ".")[1]
                
                if ndims(data) == 3
                    # Get the electrode pairs for this patient
                    patient_pairs = extract_electrode_pairs(dropdims(nanmean(data, dims=1), dims=1), soz)
                    patient_pairs[!, :pid] .= pid
                    patient_pairs[!, :ilae] .= ilae
                    append!(all_pairs, patient_pairs)
                end
            end
        end
    catch e
        @error "Error collecting pairs from file '$in_path'" exception=(e, catch_backtrace())
    end
end

# Create the final results DataFrame with all unique pairs
results_df = all_pairs

# Now process each metric and add it as a column
@showprogress length(full_metrics) "Processing metrics: " for metric in full_metrics
    try
        println("\nProcessing metric: $metric")
        
        # Initialize column for this metric
        results_df[!, Symbol(metric)] = Vector{Float64}(undef, nrow(results_df))
        
        # Process each file
        @showprogress length(files_to_process) "Files: " for in_path in files_to_process
            try
                h5open(in_path, "r") do f
                    if haskey(f["metadata/adjacency_matrices"], metric)
                        data = read(f["metadata/adjacency_matrices"][metric])
                        soz = read(f["metadata/patient_info/soz"])
                        ilae = read(f["metadata/patient_info/ilae"])
                        pid = split(basename(in_path), ".")[1]
                        
                        if ndims(data) == 3
                            # Take mean across time dimension, ignoring NaNs
                            mean_data = dropdims(nanmean(data, dims=1), dims=1)
                            
                            # Create temporary DataFrame for this file's data
                            temp_df = extract_electrode_pairs(mean_data, soz)
                            temp_df[!, :pid] .= pid
                            temp_df[!, :ilae] .= ilae
                            temp_df[!, Symbol(metric)] = [mean_data[i,j] for i in 1:size(mean_data,1) for j in (i+1):size(mean_data,2)]
                            
                            # Update the corresponding rows in results_df
                            for row in eachrow(temp_df)
                                mask = (results_df.pid .== row.pid) .& 
                                       (results_df.electrode_a .== row.electrode_a) .& 
                                       (results_df.electrode_b .== row.electrode_b)
                                if any(mask)
                                    results_df[mask, Symbol(metric)] .= row[Symbol(metric)]
                                end
                            end
                        end
                    end
                end
            catch e
                @error "Error processing file '$in_path' for metric '$metric'" exception=(e, catch_backtrace())
            end
        end
    catch e
        @error "Error processing metric '$metric'" exception=(e, catch_backtrace())
    end
end

# Save the results to CSV
if !isempty(results_df)
    output_path = joinpath(path, "electrode_pair_analysis_results.csv")
    CSV.write(output_path, results_df)
    println("\nResults saved to: $output_path")
end

println("\nProcessing complete.")
