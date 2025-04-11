using PyCall
using ProgressMeter
using HDF5
using Dates
using Base.Threads

py"""
import dill as pickle
import numpy as np
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

load_pickle = py"load_pickle"

# List of directories to process
dirs_to_scan = [
    "/media/dan/Data/data/connectivity/six_run/",
    "/media/dan/Data/data/connectivity/additional_calculations/"
    # Add more directories here as needed
]

output_dir = "/media/dan/Data/outputs/ubiquitous-spork/pyspi_combined_patient_hdf5s"

OVERWRITE = true

# Function to extract epoch information from directory name
function parse_epoch_info(dirname)
    parts = split(dirname, "_")
    pid = parts[1]
    epoch_parts = split(parts[3], "-")
    begin_idx = parse(Int, epoch_parts[1])
    end_idx = parse(Int, epoch_parts[2])
    return pid, begin_idx, end_idx
end


function load_pickle_calc(file_path, info_only=false)
    try
        calc = load_pickle(file_path)

        metrics = unique(calc.columns.get_level_values(0).tolist())
        node_count = calc[metrics[1]].shape[1]

        if info_only
            return node_count, metrics
        end

        # setup a dict of metrics with node_count x node_count arrays
        data = Dict{String, Array{Float64, 2}}()
        for metric in metrics
            data[metric] = calc[metric].values[:]
        end

        return data
    catch
        warn("Error loading $file_path")
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
pop!(all_pid_epochs["001"]["/media/dan/Data/data/connectivity/six_run/"])

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


# First pass: load all pyspi outputs and get:
# 1) size of adjacency matrices (n,n). This will be the same for all
#    pyspi outputs for a given pid.
# 2) the methods existing across all pyspi outputs
# pyspi outputs are in the form:
# subdir/
#     /calc.pkl
#     ...
patient_metrics = Dict{String, Tuple{Int64, Set{String}}}()
for pid in keys(all_pid_epochs)
    for dir in keys(all_pid_epochs[pid])
        @showprogress for subdir in all_pid_epochs[pid][dir]
            # load calc.pkl
            calc_path = joinpath(dir, subdir, "calc.pkl")
            node_count, metrics = load_pickle_calc(calc_path, true)


            if !haskey(patient_metrics, pid)
                patient_metrics[pid] = (node_count, Set{String}())
            end
            
            # Merge the new metrics with existing ones
            existing_node_count, existing_metrics = patient_metrics[pid]
            # Verify node count is consistent
            # if existing_node_count != node_count
            #     @warn "Inconsistent node count for $pid"
            # end
            # Union the metrics sets
            union!(existing_metrics, metrics)
            patient_metrics[pid] = (node_count, existing_metrics)
        end
    end
end

# make hdf5s for each pid and fill with metadata.
# currently only metadata is size of full set.
# structure of hdf5 is:
# /pid/
#     /metadata/
#         /full_set_size/
#         /patient_info/
#     /adjacency_matrices/
#         /method1/
#             ...
#         /method2/
#             ...
#         ...
# store in output_dir. If hdf5 exists and OVERWRITE is true and a existing file
# for the pid exists, "upsert" the metadata and rename the file to pid_date.h5
# where date is date of script run in YYYYMMDD format.
# if OVERWRITE is false and a existing file for the pid exists, skip.
for pid in keys(to_process)
    # check if pid.h5 exists in output_dir
    pattern = "^" * pid * "_\\d{8}\\.h5\$"
    # Get all files in directory
    all_files = readdir(output_dir)
    println("Found $(length(all_files)) files in output_dir")
    # Filter files matching pattern
    matching_files = filter(f -> occursin(Regex(pattern), f), all_files)
    println("Found $(length(matching_files)) matching files for pid $pid")
    file = nothing
    if length(matching_files) > 0
        file = matching_files[1]
    end

    date = now()
    date = Dates.format(date, "yyyymmdd")
    if file == nothing
        file = "$(pid)_$(date).h5"
    end

    println("File: $file")

    h5_path = joinpath(output_dir, file)

    if !OVERWRITE && isfile(h5_path) 
        # println("HDF5 file $h5_path exists and OVERWRITE is false. Skipping...")
        continue
    end
    

    if OVERWRITE
        # if file exists and date is different, rename to pid_date.h5
        if isfile(h5_path)
            # file name is pid_yyyyMMdd.h5
            file_date = split(split(file, "_")[2], ".")[1]
            file_date = Date(file_date, "yyyymmdd")
            file_date = Dates.format(file_date, "yyyymmdd")

            # check if date is different
            if date != file_date
                file = "$(pid)_$(date).h5"
                h5_path_new = joinpath(output_dir, file)
                println("Date is different. Renaming $(h5_path) to $(h5_path_new)")

                mv(h5_path, h5_path_new, force=true)
                h5_path = h5_path_new
            end
        end
    end

    # create hdf5 file if it doesn't exist
    if !isfile(h5_path)
        println("Creating new HDF5 file: $h5_path")
        h5open(h5_path, "w") do file
            # create /metadata group
            metadata = create_group(file, "metadata")
            
            # create /metadata/full_set_size dataset
            full_set_size = Int64[length(to_process[pid]["full"])]
            metadata["full_set_size"] = full_set_size
            
            # create /metadata/patient_info dataset
            patient_info = [pid]
            metadata["patient_info"] = patient_info
        end
        println("Successfully created HDF5 file")
    end
end

println("Script completed successfully")


