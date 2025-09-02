# ubiquitous-spork
a bunch of garbage

PRimarly to run pyspy calculations. very disorginized

additional_run and sixrun were main runs

timing was used to find things that didn't last too long

network cuts was testing thresholding



Important files:

(may need to change paths in scripts)

network_cuts/pyspi_combine_outputs.jl

- Combines all .pkl outputs from pyspi into hdf5 files for each patient. Bottom needs to be uncommented to correctly run


network_cuts/mean_data.jl

- reads hdf5 and calculates mean for each patient