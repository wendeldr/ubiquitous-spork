using CairoMakie
using HDF5
using Statistics
using Makie.StructArrays
using ProgressMeter
using Base.Threads


# List of coherence measures to compare
# coherence_keys = [
#     "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195",
#     "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342",
#     "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122",
#     "cohmag_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391",
#     "cohmag_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586",
#     "cohmag_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146",
#     "cohmag_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342",
#     "cohmag_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732",
#     "cohmag_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122",
#     "cohmag_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122",
#     "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5"
# ]
# List of coherence measures to compare
coherence_keys = [
    "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195",
    "cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342",
]



# # Function to create correlation matrix using datashader
# function create_correlation_matrix(main_file_path, alternative_file_path, output_path="coherence_correlation.png")
#     # Dictionary to store data
#     data = Dict()
#     println("Loading data...")
    
#     # Check which keys are in which file
#     h5_structure = Set()
#     h5 = h5open(main_file_path, "r")
#     for key in keys(h5)
#         push!(h5_structure, key)
#         # println("Key $key found in main file")
#     end
#     close(h5)
    
#     # Load data for each key
#     @showprogress for key in coherence_keys
#         file_path = key in h5_structure ? main_file_path : alternative_file_path
#         println("Loading data for key $key from $file_path")
#         h5 = h5open(file_path, "r")
        
#         if !haskey(h5, key)
#             @warn "Key $key not found in any file"
#             close(h5)
#             continue
#         end
        
#         # Extract data similar to Python code
#         non = read(h5["$key/lower/non_non"])
#         soz = read(h5["$key/lower/soz_soz"])
        
#         # Concatenate cross data: soz_non and non_soz
#         soz_non = read(h5["$key/lower/soz_non"])
#         non_soz = read(h5["$key/lower/non_soz"])
#         cross = vcat(soz_non, non_soz)
        
#         # Combine all data
#         data[key] = vcat(cross, non, soz)
        
#         close(h5)
#     end
    
#     # Create a figure for the correlation matrix
#     fig = Figure(size=(1000, 1000))
    
#     # Calculate number of coherence measures for grid
#     n = length(coherence_keys)
    
#     # Create short labels
#     short_labels = [split(k, "_")[end] for k in coherence_keys]
    
#     # Get correlation matrix for text display
#     corr_matrix = ones(n, n)  # Initialize with ones (diagonal elements)
#     for i in 1:n
#         for j in (i+1):n  # Only compute upper triangle
#             # Get the two datasets to compare
#             data1 = data[coherence_keys[i]]
#             data2 = data[coherence_keys[j]]
            
#             # Find indices where either dataset has NaN
#             valid_indices = .!isnan.(data1) .& .!isnan.(data2)
            
#             # Filter both datasets using the same indices to maintain pairing
#             filtered_data1 = data1[valid_indices]
#             filtered_data2 = data2[valid_indices]
            
#             # Calculate correlation only on valid pairs
#             corr_val = cor(filtered_data1, filtered_data2)
            
#             # Store in both upper and lower triangle
#             corr_matrix[i, j] = corr_val
#             corr_matrix[j, i] = corr_val  # Symmetric matrix
#         end
#     end
    
#     # Create datashader plots for all pairwise comparisons
#     for i in 1:n
#         for j in i:n
#             # Create axis - use reversed indices for traditional matrix orientation
#             ax = Axis(fig[i, j], aspect=1)
            
#             if i == j
#                 # Filter NaNs for histogram
#                 valid_data = filter(!isnan, data[coherence_keys[i]])
#                 # Show histogram on diagonal
#                 hist!(ax, valid_data, bins=50, color=:skyblue)
#                 ax.title = short_labels[i]
#                 hidedecorations!(ax, grid=false)
#             else
#                 # Get the two datasets to compare
#                 x = data[coherence_keys[j]]
#                 y = data[coherence_keys[i]]
                
#                 # Find indices where both x and y are valid
#                 valid_indices = .!isnan.(x) .& .!isnan.(y)
#                 x = x[valid_indices]
#                 y = y[valid_indices]
                
#                 # Create StructArray for better performance
#                 points = StructArray{Point2f}((x, y))
                
#                 # Using datashader with threading for better performance
#                 datashader!(ax, points, method=Makie.AggThreads())
                
#                 # Show correlation coefficient in top-right corner
#                 corr_val = round(corr_matrix[i, j], digits=3)
#                 text!(ax, 0.7, 0.9, text="r = $corr_val", space=:relative, 
#                       fontsize=12, color=:white, align=(:center, :center))
                
#                 # Only show axis labels on outer plots
#                 if i < n && j > 1
#                     hidedecorations!(ax, grid=false)
#                 elseif i == n
#                     hidexdecorations!(ax, grid=false, ticks=false)
#                 elseif j == 1
#                     hideydecorations!(ax, grid=false, ticks=false)
#                 else
#                     hidedecorations!(ax, grid=false)
#                 end
#             end
#         end
#     end
    
#     # Add titles for rows and columns
#     for i in 1:n
#         Label(fig[i, 0], short_labels[i], rotation=π/2)
#         Label(fig[0, i], short_labels[i])
#     end
    
#     # Add title
#     Label(fig[0, :], "Coherence Measure Correlations", fontsize=20)
    
#     # Save the figure
#     save(output_path, fig)
    
#     println("Correlation matrix saved to: ", output_path)
    
#     return fig
# end


using HDF5
using Statistics
using Base.Threads
using ProgressMeter


# Example usage - uncomment and update paths
six_calc = "/media/dan/Data2/calculations/connectivity/six_calc/columns.h5"
add_calc = "/media/dan/Data2/calculations/connectivity/additional_calcs/columns.h5"


six_calc = "/media/dan/Data2/calculations/connectivity/six_calc/mean_columns.h5"
add_calc = "/media/dan/Data2/calculations/connectivity/additional_calcs/mean_columns.h5"


# Dictionary to store data
data = Dict()
println("Loading data...")

# Check which keys are in which file
six_structure = Set()
h5 = h5open(six_calc, "r")
for key in keys(h5)
    push!(six_structure, key)
end
close(h5)

add_structure = Set()
h5 = h5open(add_calc, "r")
for key in keys(h5)
    push!(add_structure, key)
end
close(h5)

unique_keys = collect(union(six_structure, add_structure))

correlations = Dict()

h5_six = h5open(six_calc, "r")
h5_add = h5open(add_calc, "r")
# Load data for each key
@showprogress "Outer loop" for key in unique_keys
    h5_a = key in six_structure ? h5_six : h5_add

    # Extract data similar to Python code
    non = read(h5_a["$key/lower/non_non"])
    soz = read(h5_a["$key/lower/soz_soz"])
    
    # Concatenate cross data: soz_non and non_soz
    soz_non = read(h5_a["$key/lower/soz_non"])
    non_soz = read(h5_a["$key/lower/non_soz"])
    cross = vcat(soz_non, non_soz)
        
    # Combine all data
    a = vcat(cross, non, soz) #423250928-element
    @showprogress "Inner loop" for key2 in unique_keys
        if key != key2
            h5_b = key2 in six_structure ? h5_six : h5_add

            non = read(h5_b["$key2/lower/non_non"])
            soz = read(h5_b["$key2/lower/soz_soz"])
            
            # Concatenate cross data: soz_non and non_soz
            soz_non = read(h5_b["$key2/lower/soz_non"])
            non_soz = read(h5_b["$key2/lower/non_soz"])
            cross = vcat(soz_non, non_soz)

            b = vcat(cross, non, soz)

            corr_val = cor(a, b)
            correlations[key, key2] = corr_val
        end
    end    
end
close(h5_six)
close(h5_add)


# Get correlation matrix for text display
corr_matrix = ones(n, n)  # Initialize with ones (diagonal elements)
for i in 1:n
    for j in (i+1):n  # Only compute upper triangle
        # Get the two datasets to compare
        data1 = data[coherence_keys[i]]
        data2 = data[coherence_keys[j]]
        
        # Find indices where either dataset has NaN
        valid_indices = .!isnan.(data1) .& .!isnan.(data2)
        
        # Filter both datasets using the same indices to maintain pairing
        filtered_data1 = data1[valid_indices]
        filtered_data2 = data2[valid_indices]
        
        # Calculate correlation only on valid pairs
        corr_val = cor(filtered_data1, filtered_data2)
        
        # Store in both upper and lower triangle
        corr_matrix[i, j] = corr_val
        corr_matrix[j, i] = corr_val  # Symmetric matrix
    end
end





# create_correlation_matrix(main_file_path, alternative_file_path, "coherence_correlation.png")


