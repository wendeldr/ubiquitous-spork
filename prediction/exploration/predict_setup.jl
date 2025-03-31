using DataFrames, CSV, MAT, ProgressMeter
using Base.Threads


# Load metadata
metadata_df = CSV.read("/media/dan/Data/FULL_composite_patient_info.csv", DataFrame)

# Process electrode data files
function process_electrodes(metadata_df)
    new_df = DataFrame()
    
    files = sort(readdir("/media/dan/Data/data/electrodes_used"))
    for file in files
        tmp = CSV.read(joinpath("/media/dan/Data/data/electrodes_used", file), DataFrame)
        pid = parse(Int, split(file, '_')[1])
        electrodes = tmp[:, "0"]
        
        # Match electrodes to metadata_df
        subset = filter(row -> row.pid == pid && row.electrode in electrodes, metadata_df)
        
        # Skip if no seizure onset zone
        sum(subset.soz) == 0 && continue
        
        # Add electrode_idx column
        subset.electrode_idx .= 0
        for (i, e) in enumerate(electrodes)
            subset[subset.electrode .== e, :electrode_idx] .= i-1  # 0-based indexing
        end
        
        append!(new_df, subset)
    end
    
    return new_df
end

# Process calculation files
function process_calculations(new_df, base, subdirs)
        
    all_files = []

    for sub in subdirs
        calcs_path = joinpath(base, sub)
        files = sort(readdir(calcs_path))
        for (z,f) in enumerate(files)
            parts = split(basename(f), '~')
            pid = parse(Int, parts[2])
            time = parse(Int, parts[3])
            
            pid == 111 && continue
            push!(all_files, joinpath(calcs_path, f))
            # if z > 3
            #     break
            # end
        end
    end
    
    results = []
    @showprogress desc="Processing files..." Threads.@threads for i in 1:length(all_files)
        f = all_files[i]
        # println(f)

        # Load MATLAB file
        parts = split(basename(f), '~')
        sub = parts[1]
        pid = parse(Int, parts[2])
        time = parse(Int, parts[3])
        
        pid == 111 && continue
        
        mat_file = matread(f)
        out = mat_file["out"]
        
        # Parse features into a dictionary
        electrodes = Dict{Int, Dict{String, Any}}()
        singular_values = Dict{String, Any}()
        
        for key in keys(out)
            key == "timing" && continue
            
            for subkey in keys(out[key])
                feature_name = "$(sub)~$(key)~$(subkey)"
                
                if isa(out[key][subkey], Array)
                    for (i, val) in enumerate(out[key][subkey])
                        if !haskey(electrodes, i-1)  # 0-based indexing
                            electrodes[i-1] = Dict{String, Any}()
                        end
                        electrodes[i-1][feature_name] = val
                    end
                else
                    singular_values[feature_name] = out[key][subkey]
                end
            end
        end
        
        # Create dataframe rows
        for x in keys(electrodes)
            # Add singular values
            merge!(electrodes[x], singular_values)
            
            # Find matching metadata
            filtered = filter(row -> row.pid == pid && row.electrode_idx == x, new_df)
            isempty(filtered) && continue
            
            # Add location and SOZ info
            row = first(filtered)
            electrodes[x]["x"] = row.x
            electrodes[x]["y"] = row.y
            electrodes[x]["z"] = row.z
            electrodes[x]["soz"] = row.soz
            electrodes[x]["pid"] = pid
            electrodes[x]["time"] = time
            electrodes[x]["electrode_idx"] = x
        end
        
        # Convert to DataFrame
        if !isempty(electrodes)
            rows = [values for (idx, values) in electrodes]
            df = DataFrame(rows)
            push!(results, df)
        end
    end
    
    return results
end

# Merge all results
function merge_results(results)
    if isempty(results)
        return DataFrame()
    end
    
    # Define key columns
    key_cols = [:x, :y, :z, :soz, :pid, :electrode_idx, :time]
    
    # Create a dictionary to store merged data
    # The key is a tuple of the key columns, the value is a dictionary of feature values
    merged_data = Dict{Tuple, Dict{Symbol, Any}}()
    
    # Find all column names
    all_cols = Set{Symbol}(key_cols)
    for df in results
        union!(all_cols, Symbol.(names(df)))
    end
    
    # Fill in the merged data
    for df in results
        for row in eachrow(df)
            # Create key tuple
            key = Tuple(row[col] for col in key_cols)
            
            # Get or create entry for this key
            if !haskey(merged_data, key)
                merged_data[key] = Dict{Symbol, Any}()
                # Initialize key columns
                for (i, col) in enumerate(key_cols)
                    merged_data[key][col] = key[i]
                end
            end
            
            # Add feature values
            for col in names(df)
                if !(col in key_cols)
                    merged_data[key][Symbol(col)] = row[col]
                end
            end
        end
    end
    
    # Convert merged data to DataFrame
    final_data = []
    for (_, row_dict) in merged_data
        # Ensure all columns are present (with missing if not available)
        full_row = Dict{Symbol, Any}()
        for col in all_cols
            full_row[col] = get(row_dict, col, missing)
        end
        push!(final_data, full_row)
    end
    final_df = DataFrame(final_data)

    desired_first_cols = ["x", "y", "z", "soz", "pid", "time", "electrode_idx"]
    other_cols = setdiff(names(final_df), desired_first_cols)
    
    # Create new column order
    new_order = [desired_first_cols; other_cols]
    
    # Reorder the DataFrame
    final_df = final_df[:, new_order]

    return final_df
end

# Main execution
function main()
    # Process electrodes
    new_df = process_electrodes(metadata_df)
    
    # Define base path and subdirectories
    base = "/media/dan/Data/data/connectivity/downloads-dump/BCT/outputs"

    subdirs = ["cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5",
    "cov_EmpiricalCovariance",
    "pdist_cosine",
    "mi_gaussian"]
    
    # Process calculations
    results = process_calculations(new_df, base, subdirs)
    
    # Merge results
    merged_df = merge_results(results)
    CSV.write("predict_df_4NetMets_20250319.csv",merged_df)

    # return merged_df
end

# Run the code
merged_df = main()