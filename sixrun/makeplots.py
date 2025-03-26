import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm
import gc  # Optional: for explicit garbage collection
import re

from fast_histogram import histogram1d
from textwrap import wrap

# Let's say 200 GB:
MAX_BYTES = 200 * 1024 * 1024 * 1024  # 214748364800 bytes (approx)

def safe_load_dataset(dataset, max_bytes=MAX_BYTES):
    """
    Checks dataset size. If it exceeds max_bytes, return None.
    Otherwise load into memory and return as a NumPy array.
    """
    arr_shape = dataset.shape
    arr_dtype = dataset.dtype
    
    # Estimate memory usage in bytes
    mem_bytes = np.prod(arr_shape) * arr_dtype.itemsize
    if mem_bytes > max_bytes:
        print(f"Skipping large dataset {dataset.name} with size {mem_bytes} bytes.")
        return None
    else:
        return dataset[...]  # Load into memory

def extract_and_clean_data(h5_group):
    """
    Given an h5py group that contains the four sub-datasets
    (soz_soz, soz_non, non_soz, non_non),
    load them (unless they're too big) and return a dict of arrays.
    
    In addition to dropping NaNs, we also drop ±inf so that they
    won't affect our min/max or hist ranges.
    """
    sub_data = {}
    for subname in ["soz_soz", "soz_non", "non_soz", "non_non"]:
        if subname in h5_group:
            ds = safe_load_dataset(h5_group[subname])
            if ds is not None:
                # Drop NaNs AND ±inf
                ds = ds[~np.isnan(ds) & ~np.isinf(ds)]
            sub_data[subname] = ds
        else:
            sub_data[subname] = None
    return sub_data

def plot_ecdf_and_hist(sub_data, title, save_path):
    """
    Plots two subplots side-by-side:
      - Left: ECDF of each dataset (using statsmodels ECDF)
      - Right: normalized histogram of each dataset (using fast_histogram)
    
    sub_data should be a dict with keys ["non_non", "non_soz", "soz_soz", "soz_non"]
    mapped to arrays or None.
    """
    try:
        fig, ax = plt.subplots(1, 2, figsize=(8, 6))  # Adjust figure size as needed
        
        # Plot ECDF curves
        for label in ["non_non", "non_soz", "soz_soz", "soz_non"]:
            arr = sub_data[label]
            if arr is not None and len(arr) > 0:
                ecdf_func = ECDF(arr)
                ax[0].plot(ecdf_func.x, ecdf_func.y, label=f"{label}: N={len(arr)}")

        # Determine global min and max across data sets using np.nanmax/min
        max_val = -np.inf
        min_val = np.inf
        for label in ["non_non", "non_soz", "soz_soz", "soz_non"]:
            arr = sub_data[label]
            if arr is not None and len(arr) > 0:
                local_max = np.nanmax(arr)
                local_min = np.nanmin(arr)
                if local_max > max_val:
                    max_val = local_max
                if local_min < min_val:
                    min_val = local_min

        # If after filtering everything is invalid/empty, handle gracefully
        if max_val == -np.inf:
            max_val = 10
        if min_val == np.inf:
            min_val = -10

        if max_val is None or min_val is None or max_val == np.nan or min_val == np.nan:
            # save empty plot
            plt.savefig(save_path)
            plt.close()
            return

        # Plot histograms
        for label in ["non_non", "non_soz", "soz_soz", "soz_non"]:
            arr = sub_data[label]
            if arr is not None and len(arr) > 0:
                bins = 500
                # Use the same global min_val and max_val for all histograms
                hist = histogram1d(arr, bins=bins, range=(min_val, max_val))
                # Safely normalize the histogram
                max_count = np.nanmax(hist)
                if max_count == 0:
                    max_count = 1
                hist = hist / max_count
                bin_edges = np.linspace(min_val, max_val, bins)

                if "pli_" in title:
                    # get non-zero bins
                    non_zero_bins = np.where(hist > 0)[0]
                    bin_edges = bin_edges[non_zero_bins]
                    hist = hist[non_zero_bins]
                ax[1].plot(bin_edges, hist, label=f"{label}: N={len(arr)}",
                           alpha=0.8, linewidth=0.5)

        # Label axes and titles
        ax[0].set_xlabel("Measured Value")
        ax[0].set_ylabel("ECDF")
        ax[0].set_title("ECDF")

        ax[1].set_xlabel("Measured Value")
        ax[1].set_ylabel("Normalized Counts")
        ax[1].set_title("Histogram Tops")

        # Set consistent y-limits (0 to 1) and adjust x-limits with a buffer
        x_range = max_val - min_val
        buffer = x_range * 0.02 if x_range != 0 else 0.1

        for a in ax:
            a.set_ylim(0, 1)
            a.set_xlim(min_val - buffer, max_val + buffer)
            a.set_box_aspect(1)  # Force square axes box

        # Create a single legend for both plots
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, 
                   loc='lower center',
                   bbox_to_anchor=(0.5, 0.08),
                   ncol=2,
                   borderaxespad=0.)

        # Wrap and set the suptitle
        wrapped_title = '\n'.join(wrap(title, width=40))
        num_lines = len(wrapped_title.split('\n'))
        top_margin = 0.85 + (0.02 * (num_lines - 1))

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.25, top=top_margin, wspace=0.3)
        plt.suptitle(wrapped_title, y=0.98)
        
        # Save figure and close
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        # Create a new figure for the error message
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)}", 
                ha='center', va='center', wrap=True)
        plt.axis('off')
        
        # Add the title
        wrapped_title = '\n'.join(wrap(title, width=40))
        plt.title(wrapped_title)
        
        # Save and close the error figure
        plt.savefig(save_path)
        plt.close()
        print(f"Error in plotting {title}: {str(e)}")

def find_data_groups(h5_file):
    """
    Opens an HDF5 file, finds all groups that have
    ('soz_soz', 'soz_non', 'non_soz', 'non_non') as sub-datasets.
    Returns a list of group names (strings).
    """
    groups_found = []
    with h5py.File(h5_file, "r") as f:
        def is_candidate_group(name, obj):
            if isinstance(obj, h5py.Group):
                sub_keys = set(obj.keys())
                needed = {"soz_soz", "soz_non", "non_soz", "non_non"}
                if needed.issubset(sub_keys):
                    groups_found.append(name)

        f.visititems(is_candidate_group)

    return groups_found


# Regex patterns for fmin and fmax
fmin_pattern = r"fmin-(\d+-\d+|\d+)"
fmax_pattern = r"fmax-(\d+-\d+|\d+)"

# Function to extract fmin and fmax
def extract_fmin_fmax(string):
    fmin_match = re.search(fmin_pattern, string)
    fmax_match = re.search(fmax_pattern, string)

    fmin = (
        float(fmin_match.group(1).replace("-", ".")) if fmin_match and "-" in fmin_match.group(1) else 0
    )
    fmax = (
        float(fmax_match.group(1).replace("-", ".")) if fmax_match and "-" in fmax_match.group(1) else 0
    )

    return get_freq(fmin), get_freq(fmax)

def get_freq(x):
    # frequencies are expressed in fractions of the sampling rate
    # so we need to convert them to Hz
    return round(x*2048)


# Function to modify the string using extracted or provided fmin and fmax
def modify_string_with_fmin_fmax(original_string, fmin, fmax):
    # Convert fmin and fmax to the desired string format
    replacement = f"_fmin-{fmin}_fmax-{fmax}"

    # Replace the original portion of the string
    modified_string = re.sub(r"_fs-1_fmin-\d+-\d+_fmax-\d+-\d+", replacement, original_string)
    return modified_string

def walk_and_plot(h5_file, output_dir):
    """
    1. Find all groups in h5_file that contain the 4 sub-datasets.
    2. For each group, load data, plot, save figure.
    """
    # 1) Gather valid groups:
    data_groups = find_data_groups(h5_file)

    with h5py.File(h5_file, "r") as f:
        for group_name in tqdm(data_groups, desc=f"Groups in {os.path.basename(h5_file)}", leave=False):
            org_group_name = group_name
            if "fmin" in group_name or "fmax" in group_name:
                fmin, fmax = extract_fmin_fmax(group_name)
                group_name = modify_string_with_fmin_fmax(group_name, fmin, fmax)

            # Clean up the group_name for the output file
            safe_group = group_name.strip("/").replace("/", "~")
            out_name = f"{os.path.splitext(os.path.basename(h5_file))[0]}~{safe_group}.png"
            out_path = os.path.join(output_dir, out_name)
            
            # Skip plot creation if file already exists
            if os.path.exists(out_path):
                # print(f"Skipping existing figure {out_path}.")
                continue

            group_node = f[org_group_name]
            
            sub_data = extract_and_clean_data(group_node)
            # Build a nice title
            title = f"{os.path.basename(h5_file)}: {group_name}"
            
            
            plot_ecdf_and_hist(sub_data, title, out_path)
            
            # Optional: Force Python to clean up memory usage
            del sub_data
            try:
                gc.collect()
            except:
                pass

def main():
    files = [
        "patients.h5",
        "columns.h5",
        "ilae.h5",
        "mean_patients.h5",
        "mean_columns.h5",
        "mean_ilae.h5"
    ]
    
    # input_dir = "/media/dan/Data/data/calculations"
    # output_dir = "/home/dan/data/connectivity/pyspi_testing/sixrun/figures"
    input_dir = "/media/dan/Data2/calculations/connectivity/additional_calcs"
    output_dir = "/media/dan/Data2/calculations/connectivity/additional_calcs/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a TQDM progress bar for the top-level iteration of files
    for h5_name in tqdm(files, desc="Processing H5 files"):
        h5_file = os.path.join(input_dir, h5_name)
        if not os.path.exists(h5_file):
            print(f"Warning: {h5_file} does not exist, skipping.")
            continue
        
        # Process the file
        walk_and_plot(h5_file, output_dir)

if __name__ == "__main__":
    main()
