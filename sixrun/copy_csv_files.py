import os
from tqdm import tqdm
import shutil
calculation_path = "/media/dan/Big/network_mining/calculations/sixrun/calculations/additional_calculations"
chnames_path = "/media/dan/Big/network_mining/calculations/sixrun/calculations/six_run"
output_path = "/media/dan/Big/network_mining/calculations/electrodes_used"
files = list(sorted(os.listdir(calculation_path)))
pids = list(sorted(set([int(f.split("_")[0]) for f in files])))

for pid in tqdm(pids, desc="Patients", leave=True):
    file = f"{pid:03}_chnames.csv"
    shutil.copy(os.path.join(chnames_path, file), os.path.join(output_path, file))