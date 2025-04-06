using DataFrames, CSV, MAT, ProgressMeter
using Base.Threads


# Load metadata
metadata_df = CSV.read("/media/dan/Data/data/FULL_composite_patient_info.csv", DataFrame)

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
            
            if pid == 111
                continue
            end
            push!(all_files, joinpath(calcs_path, f))
            # if z > 3
            #     break
            # end
        end
    end
    
    results = []
    # all_files = reverse(all_files)
    # @showprogress desc="Processing files" Threads.@threads for i in 1:length(all_files)
    @showprogress desc="Processing files" Threads.@threads for i in eachindex(all_files)
        try
            f = all_files[i]
            # println(f)

            # Load MATLAB file
            parts = split(basename(f), '~')
            sub = parts[1]
            pid = parse(Int, parts[2])
            time = parse(Int, parts[3])
            
            if pid == 111
                continue
            end

            mat_file = matread(f)
            out = mat_file["out"]
            
            # Parse features into a dictionary
            electrodes = Dict{Int, Dict{String, Any}}()
            singular_values = Dict{String, Any}()
            
            for key in keys(out)
                if key == "timing"
                    continue
                end
                
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

        catch e
            println("Error processing $f at $i: $e")
            break
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
   
    subdirs = [
        "bary-sq_euclidean_max",
        "bary-sq_euclidean_mean",
        "bary_euclidean_max",
        "ce_gaussian",
        "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195",
        "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342",
        "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122",
        "cohmag_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391",
        "cohmag_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586",
        "cohmag_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146",
        "cohmag_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342",
        "cohmag_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732",
        "cohmag_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122",
        "cohmag_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122",
        "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5",
        "cov-sq_EmpiricalCovariance",
        "cov-sq_GraphicalLassoCV",
        "cov-sq_LedoitWolf",
        "cov-sq_MinCovDet",
        "cov-sq_OAS",
        "cov-sq_ShrunkCovariance",
        "cov_EmpiricalCovariance",
        "cov_GraphicalLassoCV",
        "cov_LedoitWolf",
        "cov_MinCovDet",
        "cov_OAS",
        "cov_ShrunkCovariance",
        "dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195",
        "dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342",
        "dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122",
        "dspli_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391",
        "dspli_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586",
        "dspli_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146",
        "dspli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342",
        "dspli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732",
        "dspli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122",
        "dspli_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122",
        "dspli_multitaper_mean_fs-1_fmin-0_fmax-0-5",
        "dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195",
        "dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342",
        "dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122",
        "dswpli_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391",
        "dswpli_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586",
        "dswpli_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146",
        "dswpli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342",
        "dswpli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732",
        "dswpli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122",
        "dswpli_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122",
        "dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-5",
        "je_gaussian",
        "kendalltau-sq",
        "mi_gaussian",
        "pdist_braycurtis",
        "pdist_canberra",
        "pdist_chebyshev",
        "pdist_cityblock",
        "pdist_cosine",
        "pdist_euclidean",
        "pec",
        "pec_log",
        "pec_orth",
        "pec_orth_abs",
        "pec_orth_log",
        "pec_orth_log_abs",
        "ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195",
        "ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342",
        "ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122",
        "ppc_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391",
        "ppc_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586",
        "ppc_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146",
        "ppc_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342",
        "ppc_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732",
        "ppc_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122",
        "ppc_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122",
        "ppc_multitaper_mean_fs-1_fmin-0_fmax-0-5",
        "prec-sq_GraphicalLasso",
        "prec-sq_GraphicalLassoCV",
        "prec-sq_LedoitWolf",
        "prec-sq_OAS",
        "prec-sq_ShrunkCovariance",
        "prec_GraphicalLasso",
        "prec_GraphicalLassoCV",
        "prec_LedoitWolf",
        "prec_OAS",
        "prec_ShrunkCovariance",
        "spearmanr",
        "spearmanr-sq",
        "xcorr-sq_max_sig-False",
        "xcorr-sq_mean_sig-False",
        "xcorr_max_sig-False",
        "xcorr_mean_sig-False",
    ]
    
    @showprogress desc="Subdirectories" for sub in subdirs
        println("Processing $sub")
        # Process calculations
        try
            # if csv exists, skip
            if isfile("NETWORKSTATS~$(sub).csv")
                continue
            end
            results = process_calculations(new_df, base, [sub])
            
            # Merge results
            merged_df = merge_results(results)
            CSV.write("NETWORKSTATS~$(sub).csv", merged_df)
        catch e
            println("Error processing $sub: $e")
        end
    end

    # return merged_df
end

# Run the code
main()