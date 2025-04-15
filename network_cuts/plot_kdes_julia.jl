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
... (full docstring from above) ...
"""
function extract_class_connections(
    adjacency_matrices::AbstractArray{T, 3},
    soz::AbstractVector{S}
    ) where {T<:Real, S<:Union{Bool, Integer}}

    n = size(adjacency_matrices, 2)
    num_time_steps = size(adjacency_matrices, 1) # for some reason julia reads the first dimension as the number of time steps

    if length(soz) != n
        error("Dimension mismatch: length of soz ($(length(soz))) does not match adjacency matrix dimension ($n)")
    end
    if n <= 1
        # Handle edge case where no upper triangle exists
        return T[], T[], T[]
    end


    mask = soz' .+ soz

    cartesian_indices = CartesianIndices((1:n, 1:n))
    upper_tri_cartesian = filter(ci -> ci[1] < ci[2], vec(cartesian_indices))

    if isempty(upper_tri_cartesian) # Handle case n=0 or n=1 after filtering
         return T[], T[], T[]
    end

    mask_values = mask[upper_tri_cartesian]

    linear_indices_2d = map(ci -> LinearIndices((n, n))[ci], upper_tri_cartesian)
    adj_reshaped = reshape(adjacency_matrices, n*n, num_time_steps)
    connections = adj_reshaped[linear_indices_2d, :]

    # Handle cases where some classes might be empty
    non_indices = findall(==(0), mask_values)
    mix_indices = findall(==(1), mask_values)
    soz_indices = findall(==(2), mask_values)

    non_connections = isempty(non_indices) ? T[] : vec(connections[non_indices, :])
    mix_connections = isempty(mix_indices) ? T[] : vec(connections[mix_indices, :])
    soz_connections = isempty(soz_indices) ? T[] : vec(connections[soz_indices, :])

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
                         # Ensure soz_data is suitable for sum (e.g., handle non-numeric cases if they can occur)
                         if !(eltype(soz_data) <: Number)
                            #  @warn "Skipping $patient_file: SOZ data is not numeric."
                             skip = true
                         elseif sum(soz_data) == 0
                            # println("Skipping $patient_file: SOZ sum is zero.")
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
            end # HDF5 file is automatically closed here
        catch e_open
            # @error "Error opening/processing file $in_path. Skipping." exception=(e_open, catch_backtrace())
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




# Note: The original Python code has a `break`, so it only processes the *first* metric.
# This Julia code replicates that behavior. Remove `break` to process all metrics.
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
                @warn "Could not determine data type from first file/metric. Defaulting to $DEFAULT_TYPE."
            end
        end

        # Initialize arrays with the detected or default data type
        soz_nums = Vector{DATA_TYPE}()
        mix_nums = Vector{DATA_TYPE}()
        non_nums = Vector{DATA_TYPE}()

        @showprogress length(files_to_process) "Files for $metric: " for in_path in files_to_process
            try
                h5open(in_path, "r") do f
                    if haskey(f["metadata/adjacency_matrices"], metric)
                        # Ensure data type consistency when reading
                        data = read(f["metadata/adjacency_matrices"][metric])
                        soz = read(f["metadata/patient_info/soz"]) # Assuming SOZ type is consistent (Bool or Int)

                        # Check if data is 3D as expected by extract_class_connections
                        if ndims(data) == 3
                            # Call the processing function
                            out = extract_class_connections(data, soz)

                            # Append results (Remember Julia is 1-based index)
                            append!(non_nums, out[1]) # Python out[0]
                            append!(mix_nums, out[2]) # Python out[1]
                            append!(soz_nums, out[3]) # Python out[2]
                        elseif ndims(data) == 2 # Handle case where data might be 2D (single time point)
                            @warn "Data for metric '$metric' in file '$in_path' is 2D. Reshaping to 3D with time=1."
                            data_3d = reshape(data, size(data)..., 1)
                            out = extract_class_connections(data_3d, soz)
                            append!(non_nums, out[1])
                            append!(mix_nums, out[2])
                            append!(soz_nums, out[3])
                        else
                            @warn "Data for metric '$metric' in file '$in_path' has unexpected dimensions ($(ndims(data))). Skipping this entry."
                        end
                    else
                        @warn "Metric '$metric' not found in file '$in_path' during second pass. Skipping file for this metric."
                    end
                end # File closed
            catch e
                @error "Error processing metric '$metric' in file '$in_path'. Skipping file for this metric." exception=(e, catch_backtrace())
            end
        end

        # --- Clean the aggregated data for the current metric ---
        println("Cleaning data for metric: $metric")
        initial_non_count = length(non_nums)
        initial_mix_count = length(mix_nums)
        initial_soz_count = length(soz_nums)

        # Use isfinite() which checks for both NaN and Inf
        non_nums = filter(isfinite, non_nums)
        mix_nums = filter(isfinite, mix_nums)
        soz_nums = filter(isfinite, soz_nums)

        # Alternative using boolean indexing (like the previous version):
        # non_nums = non_nums[isfinite.(non_nums)] # isfinite.() is element-wise
        # mix_nums = mix_nums[isfinite.(mix_nums)]
        # soz_nums = soz_nums[isfinite.(soz_nums)]

        # Plot the KDEs
        soz = ash(soz_nums)
        mix = ash(mix_nums)
        non = ash(non_nums)

        plot(non, hist=false, color=RGBA(0,0,0,.6), label="Non-EZ", xlabel="Value", ylabel="Density", title="$metric", legend=false,dpi=300)
        plot!(mix, hist=false, color=RGBA(26/255,133/255,255/255,.6), label="Non->EZ", legend=false,dpi=300)
        plot!(soz, hist=false, color=RGBA(212/255,17/255,89/255,.6), label="EZ", legend=false,dpi=300)

        output_path = "/media/dan/Data/git/ubiquitous-spork/plots_for_seminar/all_columns"
        savefig(joinpath(output_path, "$metric~full.png"))



#     println("  Non-SOZ: Removed $(initial_non_count - length(non_nums)) non-finite values. Final count: $(length(non_nums))")
#     println("  Mixed:   Removed $(initial_mix_count - length(mix_nums)) non-finite values. Final count: $(length(mix_nums))")
#     println("  SOZ:     Removed $(initial_soz_count - length(soz_nums)) non-finite values. Final count: $(length(soz_nums))")

#     # --- IMPORTANT: Replicating Python's break ---
#     println("\n>>> Breaking after processing the first metric ('$metric') as per original code. <<<")
#     println(">>> Remove the 'break' statement to process all metrics. <<<")
#     break

    catch e
        @error "Error processing metric '$metric'. Skipping." exception=(e, catch_backtrace())
    end
end

println("\nProcessing complete.")

# # Display final counts for the processed metric
# if !isempty(full_metrics)
#     println("\nFinal counts for metric '$(full_metrics[1])':")
#     println("  Non-SOZ connections: ", length(non_nums))
#     println("  Mixed connections:   ", length(mix_nums))
#     println("  SOZ connections:     ", length(soz_nums))
#     # You can now use non_nums, mix_nums, soz_nums for further analysis
# end
