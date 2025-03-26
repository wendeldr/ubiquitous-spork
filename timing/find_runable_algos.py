# Parse command-line arguments
import argparse
import os
from string import Template

from pyspi.calculator import Calculator
from pyspi.data import Data
import numpy as np
import yaml
import dill
from copy import deepcopy
import glob

os.environ["SPECTRAL_CONNECTIVITY_ENABLE_GPU"] = "false"


data_dir = "/home/dan/data/connectivity/pyspi_testing/nonoverlappingepochs_data" 
template_pbs_file = '/home/dan/data/connectivity/pyspi_testing/timing/template.pbs'
user_email = None
pbs_notify = "a"
queue = "workq"
walltime_hrs = 1
walltime_min = "00"
walltime_sec = "00"
cpu = 1
mem = 6
conda_env = "pyspicalc"
overwrite_pkl = True
table_only = True
pyfile = "/home/dan/data/connectivity/pyspi_testing/timing/pyspi_compute.py"
pyspi_config_file = "/home/dan/data/connectivity/pyspi_testing/fast_config.yaml"
npy_file = "/home/dan/data/connectivity/pyspi_testing/nonoverlappingepochs_data/034_epoch_000000.npy"
output_path = "/media/dan/Big/network_mining/calculations/temp"

# Open template file
with open(template_pbs_file,'r') as f:
    _pbs_file_template = f.read()
template = Template(_pbs_file_template)


basecalc = Calculator(configfile=pyspi_config_file)
name = os.path.basename(npy_file).split('.')[0]
data = Data(data=npy_file,dim_order='ps',normalise=True)

for measure in basecalc._spis.keys():
    calc = deepcopy(basecalc)
    # set measure to only the current measure
    calc._spis = {measure:calc._spis[measure]}
    calc.load_dataset(data)
    calc._optional_dependencies = None
    calc.name = name
    calc.labels = [measure]
    calc_file = os.path.join(output_path,f"{measure}.pkl")
    with open(calc_file, 'wb') as f:
        dill.dump(calc, f)
    sample_pbs = os.path.join(output_path,f"{measure}.pbs")

    pbs_file_str = template.substitute(name=measure,data_dir=data_dir,queue=queue,
                                        cpu=cpu,mem=mem,
                                        walltime_hrs=walltime_hrs,
                                        walltime_min=walltime_min,
                                        walltime_sec=walltime_sec,
                                        pbs_notify=pbs_notify,user_email=user_email,
                                        pyfile=pyfile,
                                        table_only=table_only,
                                        sample_pkl_output=calc_file,
                                        conda_env=conda_env,
                                        output_path=output_path)
    with open(sample_pbs, 'w+') as f:
        f.write(pbs_file_str)

    # Submit the job
    print(f"Now submitting {sample_pbs}")
    os.system(f"qsub {sample_pbs}")
    # calc.compute()
