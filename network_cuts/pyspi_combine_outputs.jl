using PyCall
using ProgressMeter
using HDF5
using Dates
using Base.Threads
using Distributed
using DataFrames
using CSV

# Add 15 worker processes
println("Adding worker processes...")
addprocs(15)

# Make PyCall available on all workers
@everywhere using PyCall

@everywhere py"""
import dill as pickle
import numpy as np
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

@everywhere load_pickle = py"load_pickle"

# List of directories to process
dirs_to_scan = [
    "/media/dan/Data/data/connectivity/six_run/",
    "/media/dan/Data/data/connectivity/additional_calculations/"
    # Add more directories here as needed
]

output_dir = "/media/dan/Data/outputs/ubiquitous-spork/pyspi_combined_patient_hdf5s"

OVERWRITE = false

# Function to extract epoch information from directory name
function parse_epoch_info(dirname)
    parts = split(dirname, "_")
    pid = parts[1]
    epoch_parts = split(parts[3], "-")
    begin_idx = parse(Int, epoch_parts[1])
    end_idx = parse(Int, epoch_parts[2])
    return pid, begin_idx, end_idx
end


@everywhere function load_pickle_calc(file_path, info_only=false)
    try
        calc = load_pickle(file_path)

        metrics = unique(calc.columns.get_level_values(0).tolist())
        node_count = calc[metrics[1]].shape[1]

        if info_only
            return node_count, metrics
        end

        # setup a dict of metrics with node_count x node_count arrays
        data = Dict{String, Matrix{Float64}}()
        for metric in metrics
            # Reshape the flattened vector into a node_count × node_count matrix
            data[metric] = reshape(calc[metric].values[:], node_count, node_count)
        end

        return data
    catch
        @warn("Error loading $file_path")
        return nothing
    end
end

# Create output directory if it doesn't exist
mkpath(output_dir)

println("Starting script execution...")

# collect all unique PIDs and their subdirectories across all directories
all_pid_epochs = Dict{String, Dict{String, Set{String}}}()

for dir in dirs_to_scan
    println("Processing directory: $dir")
    # read names of all subdirs in dir
    subdirs = readdir(dir)
    
    # remove any non-dir entries
    subdirs = subdirs[isdir.(joinpath.(dir, subdirs))]
    
    # sort subdirs
    sort!(subdirs)
    println("Found $(length(subdirs)) subdirectories")
    
    # process each subdir
    for subdir in subdirs
        pid, _, _ = parse_epoch_info(subdir)  # Extract PID from subdir name
        
        if !haskey(all_pid_epochs, pid)
            all_pid_epochs[pid] = Dict{String, Set{String}}()
        end

        if !haskey(all_pid_epochs[pid], dir)
            all_pid_epochs[pid][dir] = Set{String}()
        end

        push!(all_pid_epochs[pid][dir], subdir)
    end
end

println("Found $(length(keys(all_pid_epochs))) unique PIDs")

#temporary for testing DELETE
# z = pop!(all_pid_epochs["001"]["/media/dan/Data/data/connectivity/six_run/"])

# loop through and calculate the intersection, and unique elements for each pid.
# pid should have N+1 keys: one "intersection" key and N directory keys.
# the directories contain elements unique to that dir (not in intersection)
to_process = Dict{String, Dict{String, Set{String}}}()

for pid in keys(all_pid_epochs)
    if !haskey(to_process, pid)
        to_process[pid] = Dict{String, Set{String}}()
    end
    
    # Get all directories for this PID
    pid_dirs = collect(keys(all_pid_epochs[pid]))
    
    # Calculate intersection across ALL directories
    intersection = intersect([all_pid_epochs[pid][dir] for dir in pid_dirs]...)
    to_process[pid]["intersection"] = intersection
    
    # Calculate full set (union of all directories)
    full_set = union([all_pid_epochs[pid][dir] for dir in pid_dirs]...)
    to_process[pid]["full"] = full_set
    
    # For each directory, store only the elements unique to it (not in intersection)
    for dir in pid_dirs
        unique_elements = setdiff(all_pid_epochs[pid][dir], intersection)
        to_process[pid][dir] = unique_elements
    end
end

# loop through and convert sets to sorted arrays
temp_process = Dict{String, Dict{String, Vector{String}}}()

for pid in keys(to_process)
    temp_process[pid] = Dict{String, Vector{String}}()
    for dir in keys(to_process[pid])
        temp_process[pid][dir] = sort(collect(to_process[pid][dir]))
    end
end

to_process = temp_process

# Create index lookup for the "full" key for each patient
index_lookup = Dict{String, Dict{String, Int}}()
for pid in keys(to_process)
    # Initialize the lookup dictionary for this patient
    index_lookup[pid] = Dict{String, Int}()
    
    # Create mapping from subdirectory name to its index in the sorted array
    for (idx, subdir) in enumerate(to_process[pid]["full"])
        index_lookup[pid][subdir] = idx
    end
end

# First pass: load all pyspi outputs and get:
# 1) size of adjacency matrices (n,n). This will be the same for all
#    pyspi outputs for a given pid.
# 2) the methods existing across all pyspi outputs
# pyspi outputs are in the form:
# subdir/
#     /calc.pkl
#     ...
patient_metrics = Dict{String, Tuple{Int64, Set{String}}}()

# Create a list of all tasks to process
tasks = []
for pid in keys(all_pid_epochs)
    for dir in keys(all_pid_epochs[pid])
        for subdir in all_pid_epochs[pid][dir]
            calc_path = joinpath(dir, subdir, "calc.pkl")
            push!(tasks, (pid, calc_path))
            break # only one subdir per pid
        end
    end
end

# Process tasks in parallel
results = @distributed (vcat) for (pid, calc_path) in tasks
    try
        node_count, metrics = load_pickle_calc(calc_path, true)
        [(pid, node_count, metrics)]
    catch e
        warn("Error processing $calc_path: $e")
        []
    end
end

# Aggregate results
for (pid, node_count, metrics) in results
    if !haskey(patient_metrics, pid)
        patient_metrics[pid] = (node_count, Set{String}())
    end
    
    existing_node_count, existing_metrics = patient_metrics[pid]
    union!(existing_metrics, metrics)
    patient_metrics[pid] = (node_count, existing_metrics)
end

# Make necessary data structures available to all workers
@everywhere begin
    global patient_metrics = $patient_metrics
    global index_lookup = $index_lookup
    global to_process = $to_process
end

# Function to create tasks for a given pid
@everywhere function create_tasks(pid)
    udirs = collect(keys(to_process[pid]))
    # Filter out "full" and "intersection" keys
    udirs = filter(udir -> udir != "full" && udir != "intersection", udirs)

    tasks = String[]
    for dir in keys(to_process[pid])
        if dir == "full"
            continue
        end

        if dir == "intersection"
            for epoch in to_process[pid][dir]
                for udir in udirs
                    push!(tasks, joinpath(udir, epoch, "calc.pkl"))
                end
            end
            continue
        end

        for epoch in to_process[pid][dir]
            push!(tasks, joinpath(dir, epoch, "calc.pkl"))
        end
    end
    return tasks
end


@everywhere function deep_merge_dicts!(dest::Dict, src::Dict)
    for (key, inner_dict) in src
        if haskey(dest, key)
            # Update the inner dictionary in place.
            merge!(dest[key], inner_dict)
        else
            dest[key] = inner_dict
        end
    end
    return dest
end

@everywhere function process_pid_partial(pid)
    tasks = create_tasks(pid)
    node_count = patient_metrics[pid][1]
    metric_names = collect(patient_metrics[pid][2])

    # Process tasks in parallel and merge results
    partial_results = @distributed (deep_merge_dicts!) for task in tasks
        local_dict = Dict{Int64, Dict{String, Matrix{Float64}}}()
        epoch = splitpath(task)[end-1]
        epoch_idx = index_lookup[pid][epoch]
        
        # Initialize the epoch's dictionary if it doesn't exist
        if !haskey(local_dict, epoch_idx)
            local_dict[epoch_idx] = Dict{String, Matrix{Float64}}()
        end
        
        calc = load_pickle_calc(task, false)
        
        if isnothing(calc)
            # Only add NaN matrices for metrics that don't already exist
            for metric in metric_names
                if !haskey(local_dict[epoch_idx], metric)
                    local_dict[epoch_idx][metric] = fill(NaN, node_count, node_count)
                end
            end
        else
            # Merge the calculation results, preserving existing values
            for (metric, matrix) in calc
                if !haskey(local_dict[epoch_idx], metric)
                    local_dict[epoch_idx][metric] = matrix
                end
            end
        end
        
        local_dict
    end

    return partial_results
end

function process_pid(pid, full_metrics, node_count, epoch_count)
    partial_results = process_pid_partial(pid)
    
    # Initialize result dictionary with NaN arrays for each metric
    result = Dict{String, Array{Float64, 3}}()
    for metric in full_metrics
        result[metric] = fill(NaN, epoch_count, node_count, node_count)
    end
    
    # Loop through partial results and fill in available data
    for (epoch_idx, epoch_data) in partial_results
        for (metric, matrix) in epoch_data
            if haskey(result, metric)
                result[metric][epoch_idx, :, :] = matrix
            end
        end
    end
    
    return result
end


println("Loading patient info...")
# read in the csv file
patient_info = CSV.read("/media/dan/Data/data/FULL_composite_patient_info.csv", DataFrame)

seizure_free = unique(patient_info[:,["pid","seizureFree"]])

ilae = CSV.read("/media/dan/Data/data/ravi_hfo_numbers~N59+v03.csv", DataFrame)
# Select ilae and patient columns
ilae = ilae[:, ["ilae", "patient"]]
# Get unique pairs
unique!(ilae)

# Create a new DataFrame to store the combined results
combined_outcomes = DataFrame(pid = String[], ilae = Int[])

# First add all entries from ilae DataFrame
# Convert patient column to match pid format (3 digits with leading zeros)
for row in eachrow(ilae)
    pid = lpad(string(row.patient), 3, '0')
    push!(combined_outcomes, (pid, row.ilae))
end

# Add entries from seizure_free that aren't in ilae
for row in eachrow(seizure_free)
    pid = row.pid
    pid = lpad(string(pid), 3, '0')
    # Check if this pid is already in combined_outcomes
    if !any(r -> r.pid == pid, eachrow(combined_outcomes))
        # Add new entry with ilae score based on seizureFree status
        ilae_score = row.seizureFree ? -1 : 100
        push!(combined_outcomes, (pid, ilae_score))
    end
end




# # make hdf5s for each pid and fill with metadata.
# # currently only metadata is size of full set.
# # structure of hdf5 is:
# # /pid/
# #     /metadata/
# #         /full_set_size/
# #         /patient_info/
# #     /adjacency_matrices/
# #         /method1/
# #             ...
# #         /method2/
# #             ...
# #         ...
# # store in output_dir. If hdf5 exists and OVERWRITE is true and a existing file
# # for the pid exists, "upsert" the metadata and rename the file to pid_date.h5
# # where date is date of script run in YYYYMMDD format.
# # if OVERWRITE is false and a existing file for the pid exists, skip.
# @showprogress  for pid in keys(to_process)
#     # check if pid.h5 exists in output_dir
#     pattern = "^" * pid * "_\\d{8}\\.h5\$"
#     # Get all files in directory
#     all_files = readdir(output_dir)
#     println("Found $(length(all_files)) files in output_dir")
#     # Filter files matching pattern
#     matching_files = filter(f -> occursin(Regex(pattern), f), all_files)
#     println("Found $(length(matching_files)) matching files for pid $pid")
#     file = nothing
#     if length(matching_files) > 0
#         file = matching_files[1]
#     end

#     date = now()
#     date = Dates.format(date, "yyyymmdd")
#     if file === nothing
#         file = "$(pid)_$(date).h5"
#     end

#     println("File: $file")

#     h5_path = joinpath(output_dir, file)

#     if !OVERWRITE && isfile(h5_path) 
#         println("HDF5 file $h5_path exists and OVERWRITE is false. Skipping...")
#         continue
#     end
    

#     if OVERWRITE
#         # if file exists and date is different, rename to pid_date.h5
#         if isfile(h5_path)
#             # file name is pid_yyyyMMdd.h5
#             file_date = split(split(file, "_")[2], ".")[1]
#             file_date = Date(file_date, "yyyymmdd")
#             file_date = Dates.format(file_date, "yyyymmdd")

#             # check if date is different
#             if date != file_date
#                 file = "$(pid)_$(date).h5"
#                 h5_path_new = joinpath(output_dir, file)
#                 println("Date is different. Renaming $(h5_path) to $(h5_path_new)")

#                 mv(h5_path, h5_path_new, force=true)
#                 h5_path = h5_path_new
#             end
#         end
#     end

#     # create hdf5 file if it doesn't exist
#     if !isfile(h5_path)
#         println("Creating new HDF5 file: $h5_path")

#         # create empty file
#         h5open(h5_path, "w") do file
#         end
#     end

#     h5open(h5_path, "r+") do file  # "r+" mode allows reading and writing
#         # Get or create metadata group
#         metadata = haskey(file, "metadata") ? file["metadata"] : create_group(file, "metadata")
        
#         # Update or create datasets
#         if haskey(metadata, "node_count")
#             delete_object(metadata, "node_count")  # Delete existing dataset
#         end
#         node_count = patient_metrics[pid][1]
#         metadata["node_count"] = node_count
        
#         if haskey(metadata, "metrics")
#             delete_object(metadata, "metrics")
#         end
#         full_metrics = collect(patient_metrics[pid][2])
#         metadata["metrics"] = full_metrics
        
#         if haskey(metadata, "epoch_count")
#             delete_object(metadata, "epoch_count")
#         end
#         epoch_count = length(to_process[pid]["full"])
#         metadata["epoch_count"] = epoch_count
        
#         # Create or get patient_info group
#         if haskey(metadata, "patient_info")
#             delete_object(metadata, "patient_info")
#         end
#         pinfo = create_group(metadata, "patient_info")
        
#         # Add patient information to the group
#         pinfo["pid"] = [pid]
#         pinfo["epoch_indices"] = collect(1:epoch_count)
#         pinfo["epoch_names"] = to_process[pid]["full"]


#         electrode_names = CSV.read("/media/dan/Data/data/electrodes_used/$(pid)_chnames.csv", DataFrame)
#         electrode_names = electrode_names.var"0"

#         my_ilae = nothing
#         try
#             my_ilae = combined_outcomes[combined_outcomes.pid .== pid, :ilae][1]
#         catch
#             my_ilae = -1000
#         end

#         my_info = patient_info[patient_info.pid .== parse(Int, pid), :]

#         # remove any electrodes not in electrode_names
#         my_info = my_info[in.(my_info.electrode, Ref(electrode_names)), :]

#         # sort my_info to match order of electrode_names
#         my_info = my_info[indexin(electrode_names, my_info.electrode), :]

#         pinfo["ilae"] = my_ilae
        
#         # Create electrode_data group and its columns subgroup
#         electrode_data = create_group(pinfo, "electrode_data")
        
#         # Store each column from my_info under the columns group
#         for col in names(my_info)
#             # println("Storing $col")
#             # Delete existing dataset if it exists
#             if haskey(electrode_data, col)
#                 delete_object(electrode_data, col)
#             end
            
#             # Get the column data
#             col_data = my_info[!, col]
            
#             # Handle based on type
#             if eltype(col_data) <: Union{Missing, Number}
#                 # Convert to Float64 and replace missing with NaN
#                 col_data = Float64.(replace(col_data, missing => NaN))
#             elseif eltype(col_data) <: Union{Missing, String} || eltype(col_data) <: Union{Missing, String31}
#                 # Convert PooledArrays to regular string arrays and handle missing values
#                 col_data = String.(replace(col_data, missing => ""))
#             end
            
#             # Write the processed data
#             electrode_data[col] = col_data
#         end
        
#         # Store SOZ information separately
#         pinfo["soz"] = my_info.soz


#         # Create or get adjacency_matrices group
#         if haskey(metadata, "adjacency_matrices")
#             delete_object(metadata, "adjacency_matrices")
#         end
#         adjacency_matrices = create_group(metadata, "adjacency_matrices")  

#         data = process_pid(pid, full_metrics, node_count, epoch_count)

#         for (metric, array) in data
#             adjacency_matrices[metric] = array
#         end
#     end
    
# end

# println("Script completed successfully")


