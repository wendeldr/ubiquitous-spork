ENV["GKSwstype"] = "100"


using CSV
using DataFrames
using StatsBase
using Plots
using Distributed
using ProgressMeter
using KernelDensity

println("Starting ECDF plotting process...")

# Add workers for parallel processing
println("Adding workers for parallel processing...")
addprocs(3)  # Adjust number of workers as needed
println("Added $(nworkers()) workers")

# Define the CSV path and columns to ignore
csv_path = "/media/dan/Data/git/pyspi_testing/prediction/predict_df_4NetMets_20250319.csv"
# csv_path = "/media/dan/Data/git/pyspi_testing/prediction/subsample_df.csv"

ignore_cols = ["soz","pid", "time", "electrode_idx", "x", "y", "z"]

# Read the CSV file
println("Reading CSV file...")
df = CSV.read(csv_path, DataFrame)
println("Loaded $(nrow(df)) rows and $(ncol(df)) columns")

# Get columns to plot (excluding ignore_cols)
plot_cols = setdiff(names(df), ignore_cols)
println("Will plot $(length(plot_cols)) columns")

# Distribute the plotting across workers
println("Setting up distributed environment...")
@everywhere begin
    using CSV
    using DataFrames
    using StatsBase
    using Plots
    using ProgressMeter
    using KernelDensity
    
    # Function to create ECDF and KDE plots for a single column
    function plot_column_ecdf(col_name, df)
        # Filter out missing values and create ECDFs for each group
        data_0 = collect(skipmissing(df[df.soz .== 0, col_name]))
        data_1 = collect(skipmissing(df[df.soz .== 1, col_name]))
        
        # Skip if either group has no data
        if isempty(data_0) || isempty(data_1)
            println("Warning: Skipping $col_name - no valid data for one or both groups")
            return nothing
        end
        
        ecdf_0 = ecdf(data_0)
        ecdf_1 = ecdf(data_1)
        
        # Create x values for plotting
        x_vals = sort(unique(vcat(data_0, data_1)))
        
        # Create KDEs
        kde_0 = kde(data_0)
        kde_1 = kde(data_1)
        
        # Create the subplot layout
        p = plot(layout=(1,2), size=(800,400), title=col_name, title_location=:center)
        
        # ECDF plot (left)
        plot!(p[1], x_vals, ecdf_0.(x_vals), 
             color=:black, 
             label="", 
             title="ECDF",
             xlabel=col_name,
             ylabel="ECDF")
        
        plot!(p[1], x_vals, ecdf_1.(x_vals), 
              color=:red, 
              label="")
        
        # KDE plot (right)
        plot!(p[2], kde_0.x, kde_0.density,
             color=:black,
             label="",
             title="KDE",
             xlabel=col_name,
             ylabel="Density")
        
        plot!(p[2], kde_1.x, kde_1.density,
             color=:red,
             label="")
        
        # Save the plot
        savefig(p, "ecdfs/ecdf_$(col_name).png")
        return p
    end
end

# Create plots in parallel with progress bar
println("Starting parallel plotting...")
@showprogress pmap(col -> plot_column_ecdf(col, df), plot_cols)

# Clean up workers
println("Cleaning up workers...")
rmprocs(workers())
println("Done! Plots saved in 'ecdfs' directory")