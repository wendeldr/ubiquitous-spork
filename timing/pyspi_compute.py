# Loads the local calculator from calc.pkl, run compute, and save back to file
import dill
import os
import sys
import random
import pandas as pd

fname=sys.argv[1]
table_only=sys.argv[2]
# os.environ["SPECTRAL_CONNECTIVITY_ENABLE_GPU"] = "true"

# fname = "/home/dan/data/connectivity/pyspi_testing/timing/timing_run_1/ce_gaussian.pkl"
# table_only = True

# # Set the seed manually
# random.seed(127)

print(f'Attempting to open: {fname}')

with open (fname, "rb") as f:
    calc = dill.load(f)
print(f'Done. Computing...')

calc.compute()

print(f'Saving back to {fname}.')
with open(fname, 'wb') as f:
    if table_only:
        SPI_res = calc.table
        dill.dump(SPI_res, f)
    else:
        dill.dump(calc,f)
print('Done.')
