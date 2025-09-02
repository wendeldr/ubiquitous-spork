from pathlib import Path
from pyspi.calculator import Calculator
from pyspi.data import Data

npy_file = Path('/media/dan/Data/git/ubiquitous-spork/common_average_reref/two_epochs/run_data/test_data.npy')

CONFIG_FILE = Path('/media/dan/Data/git/ubiquitous-spork/common_average_reref/two_epochs/test.yaml')

base_name = Path(npy_file).stem
data = Data(data=str(npy_file), dim_order='ps', name=base_name, normalise=True)

# Create calculator with config file
calc = Calculator(configfile=str(CONFIG_FILE))
calc.load_dataset(data)
calc.compute()


