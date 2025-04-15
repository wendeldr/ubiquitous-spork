# --- Required Packages ---
using HDF5
using ProgressMeter
using LinearAlgebra # Added for potential use in extract_class_connections
using Plots
using AverageShiftedHistograms

# --- Define the path ---
const path = "/media/dan/Data/outputs/ubiquitous-spork/pyspi_combined_patient_hdf5s"

# --- Function Definition (Translated from Python) ---
"""
Extracts connections from the upper triangle of adjacency matrices based on SOZ mask classes.
Returns three arrays containing the connection values for:
1. Non-EZ to Non-EZ connections (mask value 0)
2. Non-EZ to EZ connections (mask value 1)
3. EZ to EZ connections (mask value 2)
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

@showprogress 1 "Processing metrics: " for metric in full_metrics
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

            # Calculate KDEs
            soz = ash(soz_nums)
            mix = ash(mix_nums)
            non = ash(non_nums)

            # Create plot
            plot(non, hist=false, color=RGBA(0,0,0,.6), label="Non-EZ", 
                 xlabel="Value", ylabel="Density", 
                 title="$metric (ILAE $ilae_score)", legend=false, dpi=300)
            plot!(mix, hist=false, color=RGBA(26/255,133/255,255/255,.6), 
                  label="Non->EZ", legend=false, dpi=300)
            plot!(soz, hist=false, color=RGBA(212/255,17/255,89/255,.6), 
                  label="EZ", legend=false, dpi=300)

            output_path = "/media/dan/Data/git/ubiquitous-spork/plots_for_seminar/by_ilae/all"
            savefig(joinpath(output_path, "$metric~$ilae_score~full.png"))
        end

    catch e
        @error "Error processing metric '$metric'. Skipping." exception=(e, catch_backtrace())
    end
end

println("\nProcessing complete.")
