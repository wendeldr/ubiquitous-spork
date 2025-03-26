using CSV
using DataFrames
using Base.Threads

# Generate a normalized key by removing the 4th token in the filename.
function file_key(filename::String)
    parts = split(filename, '~')
    if length(parts) >= 6
        # Exclude the 4th token (index 4)
        return join([parts[1], parts[2], parts[3], parts[5], parts[6]], "~")
    else
        return filename
    end
end

# Scan an immediate subdirectory structure.
# Returns a dictionary mapping subdirectory names to dictionaries of normalized key => filename.
function scan_directory(base_dir::String)
    subdir_files = Dict{String, Dict{String, String}}()
    # List only directories (assumed to be immediate subdirectories)
    for subdir in filter(x -> isdir(joinpath(base_dir, x)), readdir(base_dir))
        path = joinpath(base_dir, subdir)
        files = filter(x -> endswith(x, ".mat"), readdir(path))
        file_dict = Dict{String, String}()
        for file in files
            key = file_key(file)
            file_dict[key] = file
        end
        subdir_files[subdir] = file_dict
    end
    return subdir_files
end

# Define the two base directories
dirA = "/home/dan/Downloads/BCT/outputs/"
dirB = "/media/dan/Data2/calculations/connectivity/additional_calcs/julia_thresholded_mats/"

# Build dictionaries for each directory.
filesA = scan_directory(dirA)
filesB = scan_directory(dirB)

# Get the union of all subdirectory names and convert to an array (to allow indexing)
all_subdirs = collect(union(keys(filesA), keys(filesB)))

# Prepare vectors to hold output rows.
all_same = Vector{NamedTuple}()
all_missing = Vector{NamedTuple}()

# Rename the lock variable to avoid conflict with built-in lock function.
mutex = ReentrantLock()

# Process one subdirectory to compare files.
function process_subdir(subdir)
    local_same = Vector{NamedTuple}()
    local_missing = Vector{NamedTuple}()
    dictA = get(filesA, subdir, Dict{String, String}())
    dictB = get(filesB, subdir, Dict{String, String}())
    # Get union of keys from both dictionaries.
    keys_union = union(keys(dictA), keys(dictB))
    for key in keys_union
        present_a = haskey(dictA, key)
        present_b = haskey(dictB, key)
        present_both = present_a && present_b
        row = (
            subdirectory = subdir,
            filename_a = present_a ? dictA[key] : "",
            present_a = present_a,
            filename_b = present_b ? dictB[key] : "",
            present_b = present_b,
            present_both = present_both
        )
        if present_both
            push!(local_same, row)
        else
            push!(local_missing, row)
        end
    end
    return local_same, local_missing
end

# Process each subdirectory in parallel.
@threads for subdir in all_subdirs
    local_same, local_missing = process_subdir(subdir)
    lock(mutex) do
        append!(all_same, local_same)
        append!(all_missing, local_missing)
    end
end

# Convert collected rows to DataFrames.
df_same = DataFrame(all_same)
df_missing = DataFrame(all_missing)

# Write out the CSV files.
CSV.write("files_same.csv", df_same)
CSV.write("files_missing.csv", df_missing)

# Print a summary.
println("Number of same files: ", nrow(df_same))
println("Number of different/missing files: ", nrow(df_missing))
