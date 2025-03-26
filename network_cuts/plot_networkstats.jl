using Distributed
using SharedArrays
using StatsBase
using DelimitedFiles
using MAT
using Plots
using HypothesisTests
using ProgressMeter
using Random

# Add processes if not already done
if nprocs() == 1
    addprocs(max(1, 13))
end

@everywhere begin
    using MAT
    using Plots
    using StatsBase
    using HypothesisTests
    using ProgressMeter

    function plot_ecdf(results, m, measure)
            # Create plot with two panels
            p = plot(layout=(3,2), size=(1000, 500), legend=:bottomright)
            
            # Plot Binary data and compute KS test
            try
                data1 = results[m]["BIN_soz_soz"]
                data2 = results[m]["BIN_non_non"]
                
                # Create ECDFs
                ecdf1 = ecdf(data1)
                ecdf2 = ecdf(data2)
                
                # Plot ECDFs
                plot!(p[1], x -> ecdf1(x), minimum(data1), maximum(data1), label="soz", color=:red)
                plot!(p[1], x -> ecdf2(x), minimum(data2), maximum(data2), label="non", color=:blue)
                ylims!(p[1], 0, 1)
                # Two-sample KS test for binary data
                test_result = ApproximateTwoSampleKSTest(data1, data2)
                p_val = pvalue(test_result)
                annotate!(p[1], 0.05, 0.95, text("p = $(round(p_val, digits=6))", :left, :top, 10))
            catch e
                # @warn "Error plotting binary data for $m: $e"
            end


            # Plot Binary data and compute KS test
            try
                data1 = results[m]["BIN_soz_soz"]
                data2 = results[m]["BIN_non_non"]
                
                max_val = max(maximum(data1), maximum(data2))
                min_val = min(minimum(data1), minimum(data2))
                bins = range(min_val, max_val, length=1000)

                histogram!(p[3], data1, bins=bins, normalize=:pdf, color=:red, alpha=0.5)
                histogram!(p[5], data1, bins=bins, normalize=:probability, color=:blue, alpha=0.5)
            catch e
                # @warn "Error plotting binary data for $m: $e"
            end

            # Plot Weighted data and compute KS test
            try
                data1_w = results[m]["WEI_soz_soz"]
                data2_w = results[m]["WEI_non_non"]
                
                # Create ECDFs
                ecdf1_w = ecdf(data1_w)
                ecdf2_w = ecdf(data2_w)
                
                # Plot ECDFs
                plot!(p[2], x -> ecdf1_w(x), minimum(data1_w), maximum(data1_w), label="soz", color=:red)
                plot!(p[2], x -> ecdf2_w(x), minimum(data2_w), maximum(data2_w), label="non", color=:blue)
                ylims!(p[2], 0, 1)
                # Two-sample KS test for weighted data
                test_result = ApproximateTwoSampleKSTest(data1_w, data2_w)
                p_val = pvalue(test_result)
                annotate!(p[2], 0.05, 0.95, text("p = $(round(p_val, digits=6))", :left, :top, 10))
            catch e
                # @warn "Error plotting weighted data for $m: $e"
            end

            # Plot Binary data and compute KS test
            try
                data1 = results[m]["WEI_soz_soz"]
                data2 = results[m]["WEI_non_non"]
                
                max_val = max(maximum(data1), maximum(data2))
                min_val = min(minimum(data1), minimum(data2))
                bins = range(min_val, max_val, length=1000)

                histogram!(p[4], data1, bins=bins, normalize=:pdf, color=:red, alpha=0.5)
                histogram!(p[6], data2, bins=bins, normalize=:pdf, color=:blue, alpha=0.5)
            catch e
                # @warn "Error plotting binary data for $m: $e"
            end

            # Set labels and titles
            plot!(p[1], xlabel="Binary Value", ylabel="ECDF", title="Binary", grid=true)
            plot!(p[2], xlabel="Weighted Value", ylabel="ECDF", title="Weighted", grid=true)
            
            # Set overall title
            plot!(p, plot_title=m)
            
            # Save the plot
            output_path = joinpath(homedir(), "Downloads", "BCT", "images")
            mkpath(output_path)
            savefig(p, joinpath(output_path, "$(measure)~$(m).png"))
            
            # Close the plot
            return nothing

    end
end

function main()
    # Load SOZ data
    soz_path = "/media/dan/Data2/calculations/connectivity/additional_calcs/mats/bary-sq_euclidean_max"
    files = filter(f -> endswith(f, ".mat"), readdir(soz_path, sort=true))
    
    soz = Dict{Int, Dict{String, Vector{Int}}}()
    @showprogress "Loading SOZ data" for file in files
        if endswith(file, ".mat")
            # Extract PID from filename
            base = split(file, ".")[1]
            pid = parse(Int, split(base, "~")[2])
            
            if !haskey(soz, pid)
                soz[pid] = Dict{String, Vector{Int}}()
            end
            
            # Load MAT file
            mat_data = matread(joinpath(soz_path, file))
            soz[pid]["soz"] = convert(Vector{Int}, vec(mat_data["soz"]))
        end
    end

    out_dir = joinpath(homedir(), "Downloads", "BCT", "outputs")
    directories = sort(filter(isdir, map(d -> joinpath(out_dir, d), readdir(out_dir))))
    
    measures = [
        "modularity", "clustering_coefficient", "degrees", "strength", 
        "betweenness", "eigenvector_centrality", "participation_coefficient", 
        "local_efficiency", "rich_club_coefficient", "kcoreness_centrality", 
        "nodal_eccentricity"
    ]
    
    divisions = Dict(
        "BIN_soz_soz" => Float64[],
        "BIN_non_non" => Float64[],
        "WEI_soz_soz" => Float64[],
        "WEI_non_non" => Float64[]
    )

    @showprogress "Processing directories" for d in directories
        dir_name = basename(d)
        
        # Initialize results for this directory
        results = Dict{String, Dict{String, Vector{Float64}}}()
        for m in measures
            results[m] = Dict{String, Vector{Float64}}(
                "BIN_soz_soz" => Float64[],
                "BIN_non_non" => Float64[],
                "WEI_soz_soz" => Float64[],
                "WEI_non_non" => Float64[]
            )
        end
        
        # Process files in parallel
        dir_files = filter(f -> endswith(f, ".mat"), readdir(d))
        
        # Create a shared structure to collect results
        shared_results = Dict{String, Dict{String, Vector{Float64}}}()
        for m in measures
            shared_results[m] = Dict{String, Vector{Float64}}(
                "BIN_soz_soz" => Float64[],
                "BIN_non_non" => Float64[],
                "WEI_soz_soz" => Float64[],
                "WEI_non_non" => Float64[]
            )
        end
        
        # Process files
        @showprogress "Loading results for $dir_name" for f in dir_files
            if endswith(f, ".mat")
                out_data = matread(joinpath(d, f))["out"]
                pid_str = split(f, "~")[2]
                pid = parse(Int, pid_str)
                
                if haskey(soz, pid)
                    soz_data = soz[pid]["soz"]
                    idx = collect(1:length(soz_data))
                    soz_idxs = idx[soz_data .== 1]
                    non_idxs = idx[soz_data .== 0]
                    
                    for m in measures
                        if haskey(out_data, m)
                            measure_data = out_data[m]
                            
                            # Check for binary data
                            if haskey(measure_data, "binary")
                                binary_data = measure_data["binary"]
                                append!(results[m]["BIN_soz_soz"], binary_data[soz_idxs])
                                append!(results[m]["BIN_non_non"], binary_data[non_idxs])
                            end
                            
                            # Check for weighted data
                            if haskey(measure_data, "weighted")
                                weighted_data = measure_data["weighted"]
                                if isreal(weighted_data)
                                    append!(results[m]["WEI_soz_soz"], weighted_data[soz_idxs])
                                    append!(results[m]["WEI_non_non"], weighted_data[non_idxs])
                                end
                            end
                        end
                    end
                end
            end
        end
        
        # Distribute plotting tasks
        @sync begin
            for (i, m) in enumerate(measures)
                @async remotecall_wait(plot_ecdf, workers()[mod1(i, nworkers())], results, m, dir_name)
            end
        end
    end
end

# Run the main function
main()