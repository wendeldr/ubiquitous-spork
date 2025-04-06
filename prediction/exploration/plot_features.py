import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path = "/media/dan/Data/databases"

csvs = os.listdir(path)
csvs = [x for x in csvs if x.endswith(".csv")]
csvs.pop(csvs.index("predict_df_4NetMets_20250319.csv"))


output = "/media/dan/Data/outputs/ubiquitous-spork/feature_hists"


# ilae = pd.read_csv("/media/dan/Data/data/data_master1.csv")


def plot_feature(subdf, c, output_dir):
    try:

        # check if column is complex number or string
        if np.iscomplexobj(subdf[c]) or isinstance(subdf[c].values[0],str):
            logger.warning(f"Column {c} is complex, skipping plot")
            return
        
        # Create a copy to avoid modifying the original
        subdf = subdf.copy()
        subdf[c] = pd.to_numeric(subdf[c], errors='coerce')
        
        # Calculate percentage of invalid values (inf or NaN)
        total_rows = len(subdf)
        invalid_mask = np.isinf(subdf[c]) | subdf[c].isna()
        invalid_percentage = (invalid_mask.sum() / total_rows) * 100
        
        if invalid_percentage > 10:
            logger.warning(f"Column {c} has {invalid_percentage:.2f}% invalid values (inf/NaN), skipping plot")
            return
            
        if invalid_percentage > 0:
            logger.warning(f"Column {c} has {invalid_percentage:.2f}% invalid values, removing them")
            subdf = subdf[~invalid_mask]
        
        # Check if we have enough data points
        if len(subdf) <=  2:
            logger.warning(f"Not enough data points for column {c} after cleaning")
            return
            
        # Create the plot
        sns.kdeplot(data=subdf, x=c, hue="soz", common_grid=True, common_norm=False)
        plt.title(f"{c}")
        plt.savefig(os.path.join(output_dir, f"{c}.png"))
        plt.close()
        
    except Exception as e:
        logger.error(f"Error processing column {c}: {str(e)}")
        plt.close()  # Ensure plot is closed even if there's an error

skip = ['x', 'y', 'z', 'soz', 'pid', 'time', 'electrode_idx']
for f in tqdm(csvs, desc="Processing csvs..."):
    # if "cov_GraphicalLassoCV" not in f:
    #     continue
    metric = f.split("~")[1].split(".")[0]
    output_dir = os.path.join(output, metric)
    header = pd.read_csv(os.path.join(path, f), nrows=0, index_col=0).columns.tolist()
    header = [x for x in header if x not in skip]
    # check if images already exist in the output directory

    fn = lambda x: not os.path.exists(os.path.join(output_dir, f"{x}.png"))
    header = list(filter(fn, header))
    
    if len(header) == 0:
        continue
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(path, f))

    # Create sub-dataframes for each feature
    subdfs = [(df[['soz', c]].copy(), c) for c in header]

    del df # free memory

    # for subdf, c in tqdm(subdfs, desc="Processing features..."):
    #     plot_feature(subdf, c, output_dir)

    # Parallelize the plotting
    Parallel(n_jobs=30)(
        delayed(plot_feature)(subdf, c, output_dir)
        for subdf, c in tqdm(subdfs, desc="Processing features...")
    )





