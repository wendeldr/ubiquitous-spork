# Parse command-line arguments
import argparse
import os
from string import Template

parser = argparse.ArgumentParser(description="Distribute pyspi jobs across a cluster.")
parser.add_argument('--data_dir', dest='data_dir',
                    help='Directory where pyspi data is stored.')
parser.add_argument('--calc_file_name', dest='calc_file_name',
                    help='OPTIONAL: output file name for results. Default is calc.pkl.',
                    default = "calc.pkl")                    
parser.add_argument('--compute_file', dest='compute_file',
                    help="OPTIONAL: File path for python script that actually runs pyspi. Default is pyspi_compute.py in the directory where this script is located.",
                    default = './pyspi_compute.py')
parser.add_argument("--pyspi_config", dest="pyspi_config",
                    help = "OPTIONAL: File path to user-generated config file for pyspi.")
parser.add_argument("--sample_yaml", dest="sample_yaml",
                    help = "Name of YAML file containing filepath and metadata about each sample to be processed.",
                    default = "sample.yaml")
parser.add_argument("--template_pbs_file", dest="template_pbs_file",
                    help = "File path to template pbs script. Default is template.pbs in current working directory.",
                    default = "template.pbs")
parser.add_argument("--pbs_notify", dest="pbs_notify",
                    help = "OPTIONAL: When pbs should email user; a=abort, b=begin, e=end. Default is a only.",
                    default = "a")
parser.add_argument("--email", dest="user_email",
                    help = "OPTIONAL: Email address for pbs job status.")
parser.add_argument("--queue", dest="queue", help="Queue to submit jobs to. Default is workq.", 
                    default = "workq")
parser.add_argument("--walltime_hrs", dest="walltime_hrs",
                    help = "OPTIONAL: Maximum walltime allowed for job. Default is 6 hours.",
                    default = "6")
parser.add_argument("--cpu", dest="cpu",
                    help = "OPTIONAL: Number of CPUs to request for each job. Default is 2.",
                    default = "2")
parser.add_argument("--mem", dest="mem",
                    help = "OPTIONAL: Memory to request per job (in GB). Default is 20.",
                    default = "20")
parser.add_argument("--conda_env", dest="conda_env",
                    help = "OPTIONAL: Name of conda environment. Default is base.",
                    default = "base")
parser.add_argument("--overwrite_pkl", dest="overwrite_pkl",
                    help = "OPTIONAL: overwrite all existing .pkl files in data directory? Default is False.",
                    default = False, action="store_true")
parser.add_argument("--table_only", dest="table_only",
                    help = "Only save calc.table to calc.pkl file. Default is False.",
                    action = "store_true", default = False)

# Parse the arguments
args = parser.parse_args()
data_dir = args.data_dir
calc_file_name = args.calc_file_name
sample_yaml = args.sample_yaml
template_pbs_file = args.template_pbs_file
user_email = args.user_email
pbs_notify = args.pbs_notify
queue = args.queue
walltime_hrs = args.walltime_hrs
cpu = args.cpu
mem = args.mem
conda_env = args.conda_env
overwrite_pkl = args.overwrite_pkl
table_only = args.table_only
pyfile = os.path.abspath(args.compute_file)

# Open template file
with open(template_pbs_file,'r') as f:
    _pbs_file_template = f.read()
template = Template(_pbs_file_template)

# Import the rest of the modules
from pyspi.calculator import Calculator
from pyspi.data import Data
import numpy as np
import yaml
import dill
from copy import deepcopy

# Instantiate Calculator
# Use user-generated config file if supplied to subset SPIs
if args.pyspi_config is not None:
    if args.pyspi_config in ["fast", "sonnet", "fabfour"]:
        basecalc = Calculator(subset=args.pyspi_config)
    else:
        pyspi_config_file = os.path.abspath(args.pyspi_config)
        print(f"Custom config file: {pyspi_config_file}")
        basecalc = Calculator(configfile=pyspi_config_file)
else:
	basecalc = Calculator()

sub = 0
# Loop through each .npy file in 'database' as well as the 'sample.yaml' file
print(f"Now walking through data directory: {data_dir}")
for dirpath, _, filenames in os.walk(data_dir):
    for fidx, filename in enumerate(filenames):
        # Look for the user-specified sample YAML file
        if filename == sample_yaml:
            doc = os.path.join(dirpath,filename)
            print(f"Sample YAML found. Loading {doc}")
            with open(doc) as d:
                yf = yaml.load(d,Loader=yaml.FullLoader)
                try:
                    for config in yf:
                        file = config['file']
                        dim_order = config['dim_order']
                        name = str(config['name'])
                        labels = config['labels']
                        try:
                            data = Data(data=file,dim_order=dim_order,name=name,normalise=True)
                        except ValueError as err:
                            print(f'Issue loading dataset: {err}')
                            continue

                        # Create output directory
                        sample_path = data_dir + "/" + name
                        try:
                            os.mkdir(sample_path)
                        except OSError as err:
                            print(f'Creation of the directory {sample_path} failed: {err}')
                        else:
                            print(f'Successfully created the directory {sample_path}')
                        
                        # Create .pkl file in the current sample's folder within the data directory
                        sample_pkl_output = f"{sample_path}/{calc_file_name}"

                        # If the output .pkl file already exists, ask user if they want to overwrite.
                        if os.path.exists(sample_pkl_output) and not overwrite_pkl:
                            print(f'File {sample_pkl_output} already exists. Delete/move if you would like to recompute.')
                            continue

                        print("Now making deepcopy of basecalc")
                        calc = deepcopy(basecalc)
                        calc.load_dataset(data)
                        calc.name = name
                        calc.labels = labels
                        sample_path = data_dir + "/" + name + "/"

                        # Save calculator in directory
                        print(f'Saving object to dill database: "{sample_pkl_output}"')
                        with open(sample_pkl_output, 'wb') as f:
                            dill.dump(calc, f)

                        # Define PBS script and write relevant info to script
                        print("Now writing pbs file")
                        sample_pbs = os.path.join(f"{sample_path}","pyspi_run.pbs")

                        pbs_file_str = template.substitute(name=name,data_dir=data_dir,queue=queue,
                                                            cpu=cpu,mem=mem,walltime_hrs=walltime_hrs,
                                                            pbs_notify=pbs_notify,user_email=user_email,
                                                            pyfile=pyfile,
                                                            table_only=table_only,
                                                            sample_pkl_output=sample_pkl_output,
                                                            conda_env=conda_env)
                        with open(sample_pbs, 'w+') as f:
                            f.write(pbs_file_str)

						# Submit the job
                        print(f"Now submitting {sample_pbs}")
                        os.system(f"qsub {sample_pbs}")
                        # sub +=1
                        # if sub > 3:
                        #     break
                except (yaml.scanner.ScannerError,TypeError) as err:
                    print(f'YAML-file {doc} failed: {err}')
