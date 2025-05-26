using HDF5
using ProgressMeter
using NaNStatistics
using Statistics
using DataFrames
using CSV
using LinearAlgebra

# --- Define the path ---
const path = "/media/dan/Data/outputs/ubiquitous-spork/pyspi_combined_patient_hdf5s"

# --- First Pass: Identify files to process and all metrics ---
full_metrics = String[]
files_to_process = String[]

println("Scanning files to identify metrics and valid patients...")
try
    all_files = readdir(path)
    # Filter and sort HDF5 files directly
    h5_files = sort([f for f in all_files if endswith(f, ".h5")])

    for patient_file in h5_files
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



# TODO: This didn't work as expected
# println("Finding symmetric and non-symmetric metrics...")
# # read first file and get symmetric and non-symmetric metrics
# f = h5open(files_to_process[1], "r")
# symmetric_metrics = String[]
# non_symmetric_metrics = String[]
# @showprogress for (i, metric) in enumerate(full_metrics)
#     local data
#     data = read(f["metadata/adjacency_matrices"][metric])
#     # Skip if data is not Float64
#     if !(eltype(data) <: Float64)
#         println("Skipping $metric: Data is not Float64")
#         continue
#     end
#     data = data[1,:,:]
#     data[isnan.(data)] .= 0
#     data = abs.(data)

#     # check if data is symmetric
#     if !issymmetric(data)
#         push!(non_symmetric_metrics, metric)
#     else
#         push!(symmetric_metrics, metric)
#     end
# end
# close(f)



measure_csv = "/media/dan/Data/git/ubiquitous-spork/network_cuts/connectivity_metric_directionality.csv"
measure_df = CSV.read(measure_csv, DataFrame)

symmetric_metrics = String[]
non_symmetric_metrics = String[]
for metric in full_metrics
    row = filter(row -> row.estimator == metric, measure_df)
    # check if empty
    if isempty(row)
        @warn "Unknown metric: $metric"
        continue
    end
    if row.directionality[1] == "undirected"
        push!(symmetric_metrics, metric)
    elseif row.directionality[1] == "directed"
        push!(non_symmetric_metrics, metric)
    end
end


println("Found $(length(files_to_process)) valid files.")
println("Found $(length(full_metrics)) unique metrics.")
println("Found $(length(symmetric_metrics)) symmetric metrics.")
println("Found $(length(non_symmetric_metrics)) non-symmetric metrics.")


# read the etiology csv from path
etiology_path = "/media/dan/Data/data/etiologys.csv"
etiology_df = CSV.read(etiology_path, DataFrame)

# # merge etiology_df with results_df on pid
# results_df = merge(results_df, etiology_df, on=:pid => :pid)

# Initialize results DataFrame with all columns
results_df = DataFrame()

# --- Second Pass: Process each metric across all selected files ---
@showprogress "Processing: " for in_path in files_to_process
    # Initialize DataFrame with base columns
    temp_df = DataFrame(
        pid = String[],
        electrode_pair = String[],
        electrode_a = Int[],
        electrode_b = Int[],
        soz_a = Bool[],
        soz_b = Bool[],
        soz_sum = Int[],
        ilae = Int[],
        etiology = String[],
        electrode_pair_names = String[],
        electrode_a_name = String[],
        electrode_b_name = String[],
        miccai_label_a = String[],
        miccai_label_b = String[],
        age_days = Float64[],
        age_years = Float64[]
    )

    # Add columns for each symmetric metric
    for metric in symmetric_metrics
        temp_df[!, Symbol(metric)] = Float64[]
    end

    local soz = Bool[]
    local ilae_score = Int[]
    local pid = String[]
    pid = split(basename(in_path), "_")[1]

    # get the etiology from etiology_df. patient matches pid but is a int not a string
    etiology = etiology_df[etiology_df.patient .== parse(Int, pid), :etiol_grp]
    if isempty(etiology)
        etiology = "NA"
    else
        etiology = etiology[1]
    end

    local electrode_names = String[]
    local miccai_labels = String[]
    local age_days = Float64
    local age_years = Float64

    # read metadata
    f = h5open(in_path, "r")
    soz = read(f["metadata/patient_info/soz"])
    ilae_score = read(f["metadata/patient_info/ilae"])
    electrode_names = read(f["metadata/patient_info/electrode_data/electrode"])
    miccai_labels = read(f["metadata/patient_info/electrode_data/miccai"])
    age_days = read(f["metadata/patient_info/electrode_data/age_days_at_recording"])[1]
    age_years = read(f["metadata/patient_info/electrode_data/age_years_at_recording"])[1]
    close(f)

    # Read all metric data first
    metric_data = Dict{String, Matrix{Float64}}()
    f = h5open(in_path, "r")
    for metric in symmetric_metrics
        data = read(f["metadata/adjacency_matrices"][metric])
        data = dropdims(nanmean(data, dims=1), dims=1)
        data[isnan.(data)] .= 0
        metric_data[metric] = data
    end
    close(f)


    n = size(metric_data[symmetric_metrics[1]], 1)
    for i in 1:n
        for j in (i+1):n
            # Create row with base data
            row = (
                pid = pid,
                electrode_pair = "$i-$j",
                electrode_a = i,
                electrode_b = j,
                soz_a = soz[i],
                soz_b = soz[j],
                soz_sum = soz[i] + soz[j],
                ilae = ilae_score,
                etiology = etiology,
                electrode_pair_names = "$(electrode_names[i])-$(electrode_names[j])",
                electrode_a_name = electrode_names[i],
                electrode_b_name = electrode_names[j],
                miccai_label_a = miccai_labels[i],
                miccai_label_b = miccai_labels[j],
                age_days = age_days,
                age_years = age_years
            )

            # Add metric values to row
            for metric in symmetric_metrics
                row = merge(row, (Symbol(metric) => metric_data[metric][i,j],))
            end

            push!(temp_df, row)
        end
    end

    # Append to results DataFrame
    append!(results_df, temp_df)
end

# Save the results to CSV
output_path = joinpath("/media/dan/Data/outputs/ubiquitous-spork/", "mean_data.csv")
CSV.write(output_path, results_df)
println("\nResults saved to: $output_path")

println("\nProcessing complete.")
