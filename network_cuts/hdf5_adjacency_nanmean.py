#!/usr/bin/env python3
"""
Duplicate HDF5 files while reducing 3D adjacency matrices via np.nanmean(..., axis=-1).

- Scans an input directory for *.h5 / *.hdf5 files (configurable with --pattern).
- For each file, writes a copy to an output directory with the same relative name.
- In the copy, every dataset under "metadata/adjacency_matrices" that is 3D gets
  replaced by its np.nanmean over the last axis (shape (..., ..., k) -> (..., ...)).
- All other groups/datasets/attrs are preserved.
- Safe by default (no in-place editing).

Usage:
    python hdf5_adj_nanmean_dup.py /path/to/input_dir -o /path/to/output_dir
    python hdf5_adj_nanmean_dup.py /path/to/input_dir -o /path/to/output_dir --pattern "*.h5"
    python hdf5_adj_nanmean_dup.py /path/to/input_dir -o /path/to/output_dir --dry-run

Requirements:
    pip install numpy h5py
"""
import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import h5py


full_renames = dict({
    'bary_euclidean_max': 'bary_euclidean_max',
    'bary_euclidean_mean': 'bary_euclidean_mean',
    'ce_gaussian': 'ce_gaussian',
    'cov_EmpiricalCovariance': 'cov_EmpiricalCovariance',
    'cov_GraphicalLassoCV': 'cov_GraphicalLassoCV',
    'cov_LedoitWolf': 'cov_LedoitWolf',
    'cov_MinCovDet': 'cov_MinCovDet',
    'cov_OAS': 'cov_OAS',
    'cov_ShrunkCovariance': 'cov_ShrunkCovariance',
    'cov_sq-EmpiricalCovariance': 'cov_sq-EmpiricalCovariance',
    'je_gaussian': 'je_gaussian',
    'lmfit_BayesianRidge': 'lmfit_BayesianRidge',
    'lmfit_ElasticNet': 'lmfit_ElasticNet',
    'lmfit_Lasso': 'lmfit_Lasso',
    'lmfit_Ridge': 'lmfit_Ridge',
    'lmfit_SGDRegressor': 'lmfit_SGDRegressor',
    'mi_gaussian': 'mi_gaussian',
    'pdist_braycurtis': 'pdist_braycurtis',
    'pdist_canberra': 'pdist_canberra',
    'pdist_chebyshev': 'pdist_chebyshev',
    'pdist_cityblock': 'pdist_cityblock',
    'pdist_cosine': 'pdist_cosine',
    'pdist_euclidean': 'pdist_euclidean',
    'pec': 'pec',
    'pec_log': 'pec_log',
    'pec_orth': 'pec_orth',
    'pec_orth_abs': 'pec_orth_abs',
    'pec_orth_log': 'pec_orth_log',
    'pec_orth_log_abs': 'pec_orth_log_abs',
    'prec_EllipticEnvelope': 'prec_EllipticEnvelope',
    'prec_EmpiricalCovariance': 'prec_EmpiricalCovariance',
    'prec_GraphicalLasso': 'prec_GraphicalLasso',
    'prec_GraphicalLassoCV': 'prec_GraphicalLassoCV',
    'prec_LedoitWolf': 'prec_LedoitWolf',
    'prec_MinCovDet': 'prec_MinCovDet',
    'prec_OAS': 'prec_OAS',
    'prec_ShrunkCovariance': 'prec_ShrunkCovariance',
    'reci': 'reci',
    'spearmanr': 'spearmanr',
    'tlmi_gaussian': 'tlmi_gaussian',

    'gc_gaussian_k-1_kt-1_l-1_lt-1': 'gc_gaussian_k-1_kt-1_l-1_lt-1',
    'te_symbolic_k-1_kt-1_l-1_lt-1': 'te_symbolic_k-1_kt-1_l-1_lt-1',
    'te_symbolic_k-10_kt-1_l-1_lt-1': 'te_symbolic_k-10_kt-1_l-1_lt-1',

'cov-sq_EmpiricalCovariance': "cov_sq-EmpiricalCovariance",
'cov-sq_GraphicalLasso': "cov_sq-GraphicalLasso",
'cov-sq_GraphicalLassoCV': "cov_sq-GraphicalLassoCV",
'cov-sq_LedoitWolf': "cov_sq-LedoitWolf",
'cov-sq_MinCovDet': "cov_sq-MinCovDet",
'cov-sq_OAS': "cov_sq-OAS",
'cov-sq_ShrunkCovariance': "cov_sq-ShrunkCovariance",
'spearmanr-sq': 'spearmanr_sq',
'kendalltau-sq': 'kendalltau_sq',
"xcorr_max_sig-False": "xcorr_max",
"xcorr_mean_sig-False": "xcorr_mean",
"xcorr-sq_max_sig-False": "xcorr_sq-max",
"xcorr-sq_mean_sig-False": "xcorr_sq-mean",
"prec-sq_EmpiricalCovariance": "prec_sq-EmpiricalCovariance",
"prec-sq_EllipticEnvelope": "prec_sq-EllipticEnvelope",
"prec-sq_GraphicalLasso": "prec_sq-GraphicalLasso",
"prec-sq_GraphicalLassoCV": "prec_sq-GraphicalLassoCV",
"prec-sq_LedoitWolf": "prec_sq-LedoitWolf",
"prec-sq_MinCovDet": "prec_sq-MinCovDet",
"prec-sq_OAS": "prec_sq-OAS",
"prec-sq_ShrunkCovariance": "prec_sq-ShrunkCovariance",
"bary-sq_euclidean_mean": "bary_sq-euclidean_mean",
"bary-sq_euclidean_max": "bary_sq-euclidean_max",
"cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195": "CohMag_1-4",
"cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342": "CohMag_1-70",
"cohmag_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122": "CohMag_1-250",
"cohmag_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391": "CohMag_4-8",
"cohmag_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586": "CohMag_8-13",
"cohmag_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146": "CohMag_13-30",
"cohmag_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342": "CohMag_30-70",
"cohmag_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732": "CohMag_70-150",
"cohmag_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122": "CohMag_70-250",
"cohmag_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122": "CohMag_150-250",
"cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5": "CohMag_0-NYQ",
"phase_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195": "Phase_1-4",
"phase_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342": "Phase_1-70",
"phase_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122": "Phase_1-250",
"phase_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391": "Phase_4-8",
"phase_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586": "Phase_8-13",
"phase_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146": "Phase_13-30",
"phase_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342": "Phase_30-70",
"phase_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732": "Phase_70-150",
"phase_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122": "Phase_70-250",
"phase_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122": "Phase_150-250",
"phase_multitaper_mean_fs-1_fmin-0_fmax-0-5": "Phase_0-NYQ",
"gd_multitaper_delay_fs-1_fmin-0-000488_fmax-0-00195": "GrpDel_1-4",
"gd_multitaper_delay_fs-1_fmin-0-000488_fmax-0-0342": "GrpDel_1-70",
"gd_multitaper_delay_fs-1_fmin-0-000488_fmax-0-122": "GrpDel_1-250",
"gd_multitaper_delay_fs-1_fmin-0-00195_fmax-0-00391": "GrpDel_4-8",
"gd_multitaper_delay_fs-1_fmin-0-00391_fmax-0-00586": "GrpDel_8-13",
"gd_multitaper_delay_fs-1_fmin-0-00586_fmax-0-0146": "GrpDel_13-30",
"gd_multitaper_delay_fs-1_fmin-0-0146_fmax-0-0342": "GrpDel_30-70",
"gd_multitaper_delay_fs-1_fmin-0-0342_fmax-0-0732": "GrpDel_70-150",
"gd_multitaper_delay_fs-1_fmin-0-0342_fmax-0-122": "GrpDel_70-250",
"gd_multitaper_delay_fs-1_fmin-0-0732_fmax-0-122": "GrpDel_150-250",
"psi_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195": "PSI_1-4",
"psi_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342": "PSI_1-70",
"psi_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122": "PSI_1-250",
"psi_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391": "PSI_4-8",
"psi_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586": "PSI_8-13",
"psi_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146": "PSI_13-30",
"psi_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342": "PSI_30-70",
"psi_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732": "PSI_70-150",
"psi_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122": "PSI_70-250",
"psi_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122": "PSI_150-250",
"psi_multitaper_mean_fs-1_fmin-0_fmax-0-5": "PSI_0-NYQ",
"icoh_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195": "iCoh_1-4",
"icoh_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342": "iCoh_1-70",
"icoh_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122": "iCoh_1-250",
"icoh_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391": "iCoh_4-8",
"icoh_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586": "iCoh_8-13",
"icoh_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146": "iCoh_13-30",
"icoh_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342": "iCoh_30-70",
"icoh_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732": "iCoh_70-150",
"icoh_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122": "iCoh_70-250",
"icoh_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122": "iCoh_150-250",
"icoh_multitaper_mean_fs-1_fmin-0_fmax-0-5": "iCoh_0-NYQ",
"plv_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195": "PLV_1-4",
"plv_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342": "PLV_1-70",
"plv_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122": "PLV_1-250",
"plv_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391": "PLV_4-8",
"plv_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586": "PLV_8-13",
"plv_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146": "PLV_13-30",
"plv_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342": "PLV_30-70",
"plv_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732": "PLV_70-150",
"plv_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122": "PLV_70-250",
"plv_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122": "PLV_150-250",
"plv_multitaper_mean_fs-1_fmin-0_fmax-0-5": "PLV_0-NYQ",
"ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195": "PPC_1-4",
"ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342": "PPC_1-70",
"ppc_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122": "PPC_1-250",
"ppc_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391": "PPC_4-8",
"ppc_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586": "PPC_8-13",
"ppc_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146": "PPC_13-30",
"ppc_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342": "PPC_30-70",
"ppc_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732": "PPC_70-150",
"ppc_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122": "PPC_70-250",
"ppc_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122": "PPC_150-250",
"ppc_multitaper_mean_fs-1_fmin-0_fmax-0-5": "PPC_0-NYQ",
"pli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195": "PLI_1-4",
"pli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342": "PLI_1-70",
"pli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122": "PLI_1-250",
"pli_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391": "PLI_4-8",
"pli_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586": "PLI_8-13",
"pli_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146": "PLI_13-30",
"pli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342": "PLI_30-70",
"pli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732": "PLI_70-150",
"pli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122": "PLI_70-250",
"pli_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122": "PLI_150-250",
"pli_multitaper_mean_fs-1_fmin-0_fmax-0-5": "PLI_0-NYQ",
"wpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195": "wPLI_1-4",
"wpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342": "wPLI_1-70",
"wpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122": "wPLI_1-250",
"wpli_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391": "wPLI_4-8",
"wpli_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586": "wPLI_8-13",
"wpli_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146": "wPLI_13-30",
"wpli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342": "wPLI_30-70",
"wpli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732": "wPLI_70-150",
"wpli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122": "wPLI_70-250",
"wpli_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122": "wPLI_150-250",
"wpli_multitaper_mean_fs-1_fmin-0_fmax-0-5": "wPLI_0-NYQ",
"dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195": "dsPLI_1-4",
"dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342": "dsPLI_1-70",
"dspli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122": "dsPLI_1-250",
"dspli_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391": "dsPLI_4-8",
"dspli_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586": "dsPLI_8-13",
"dspli_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146": "dsPLI_13-30",
"dspli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342": "dsPLI_30-70",
"dspli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732": "dsPLI_70-150",
"dspli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122": "dsPLI_70-250",
"dspli_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122": "dsPLI_150-250",
"dspli_multitaper_mean_fs-1_fmin-0_fmax-0-5": "dsPLI_0-NYQ",
"dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-00195": "dswPLI_1-4",
"dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-0342": "dswPLI_1-70",
"dswpli_multitaper_mean_fs-1_fmin-0-000488_fmax-0-122": "dswPLI_1-250",
"dswpli_multitaper_mean_fs-1_fmin-0-00195_fmax-0-00391": "dswPLI_4-8",
"dswpli_multitaper_mean_fs-1_fmin-0-00391_fmax-0-00586": "dswPLI_8-13",
"dswpli_multitaper_mean_fs-1_fmin-0-00586_fmax-0-0146": "dswPLI_13-30",
"dswpli_multitaper_mean_fs-1_fmin-0-0146_fmax-0-0342": "dswPLI_30-70",
"dswpli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-0732": "dswPLI_70-150",
"dswpli_multitaper_mean_fs-1_fmin-0-0342_fmax-0-122": "dswPLI_70-250",
"dswpli_multitaper_mean_fs-1_fmin-0-0732_fmax-0-122": "dswPLI_150-250",
"dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-5": "dswPLI_0-NYQ",
})

def parse_args():
    p = argparse.ArgumentParser(description="Duplicate HDF5 files and nanmean adjacency matrices.")
    p.add_argument("input_dir", type=Path, help="Directory containing HDF5 files.")
    p.add_argument("-o", "--output-dir", type=Path, required=True, help="Directory to write modified copies.")
    p.add_argument("--pattern", default="*.h5,*.hdf5",
                   help="Comma-separated glob(s) for files to process. Default: *.h5,*.hdf5")
    p.add_argument("--adj-root", default="metadata/adjacency_matrices",
                   help="Root path of adjacency matrices group inside each file. Default: metadata/adjacency_matrices")
    p.add_argument("--dry-run", action="store_true", help="Don't write files; just print what would change.")
    return p.parse_args()

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        try:
            dst.attrs[k] = v
        except Exception as e:
            print(f"  [warn] Failed copying attr {k!r}: {e}", file=sys.stderr)

def should_transform(dataset: h5py.Dataset, adj_root: str) -> bool:
    # Normalize to no leading slash
    path = dataset.name.lstrip("/")
    return path == adj_root or path.startswith(adj_root + "/")

def check_and_rename_key(key: str, full_renames: dict) -> str:
    """
    Check if a key exists in full_renames and return the renamed version.
    Raises KeyError if the key is missing.
    """
    if key not in full_renames:
        raise KeyError(f"Key '{key}' not found in full_renames dictionary. ")
    return full_renames[key]

def create_dataset_like(out_group: h5py.Group, name: str, data, src_dset: h5py.Dataset):
    # We try to preserve common filters when feasible.
    # If compression was used, keep it; h5py will adapt to new shape.
    create_kwargs = {}
    try:
        if src_dset.compression is not None:
            create_kwargs["compression"] = src_dset.compression
        if src_dset.compression_opts is not None:
            create_kwargs["compression_opts"] = src_dset.compression_opts
        # Let h5py pick suitable chunks for the new shape unless we can infer something better.
        # (Reusing old chunks may be invalid due to shape change.)
        if src_dset.shuffle is not None:
            create_kwargs["shuffle"] = src_dset.shuffle
        if src_dset.fletcher32 is not None:
            create_kwargs["fletcher32"] = src_dset.fletcher32
    except Exception:
        # Some backends / versions may not expose all properties; ignore silently.
        pass

    out_dset = out_group.create_dataset(name, data=data, **create_kwargs)
    copy_attrs(src_dset, out_dset)
    return out_dset

def recursive_copy(in_group: h5py.Group, out_group: h5py.Group, adj_root: str, dry_run: bool, full_renames: dict):
    # Copy group attributes
    copy_attrs(in_group, out_group)

    for name, item in in_group.items():
        if isinstance(item, h5py.Group):
            new_group = out_group.create_group(name) if not dry_run else out_group.require_group(name)
            copy_attrs(item, new_group)
            recursive_copy(item, new_group, adj_root, dry_run, full_renames)
        elif isinstance(item, h5py.Dataset):
            full_path = item.name
            if should_transform(item, adj_root) and item.ndim == 3:
                # Check if the key exists in full_renames and get the renamed version
                try:
                    renamed_name = check_and_rename_key(name, full_renames)
                    print(f"  [transform] {full_path}: 3D -> nanmean(axis=-1) (renamed: {name} -> {renamed_name})")
                    if not dry_run:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            data = np.nanmean(item[()], axis=-1)
                        create_dataset_like(out_group, renamed_name, data, item)
                except KeyError as e:
                    print(f"  [ERROR] {full_path}: {e}", file=sys.stderr)
                    if not dry_run:
                        raise  # Re-raise the error to stop processing
            else:
                action = "copy"
                if should_transform(item, adj_root) and item.ndim != 3:
                    action = f"copy (under adj root but ndim={item.ndim}, leaving untouched)"
                print(f"  [{action}] {full_path}")
                if not dry_run:
                    # Copy by reading data (simple & robust). For huge datasets, consider chunked IO.
                    data = item[()]
                    create_dataset_like(out_group, name, data, item)
        else:
            print(f"  [skip] {item.name}: unsupported object type {type(item)}")

def process_file(src_path: Path, dst_path: Path, adj_root: str, dry_run: bool, full_renames: dict):
    print(f"\nProcessing: {src_path}")
    if not dry_run:
        # Ensure parent directories exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as fin:
        if dry_run:
            print(f"  [dry-run] Would write: {dst_path}")
            # Still create an in-memory "shadow" structure by opening output file in core driver? Not needed.
            # We'll just walk and print the plan.
            # Open a temp in-memory file when not dry-run.
            class DummyOut:
                def __init__(self): pass
                def create_group(self, name): return self
                def require_group(self, name): return self
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def attrs(self): return {}
            dummy = DummyOut()
            recursive_copy(fin, dummy, adj_root, dry_run=True, full_renames=full_renames)
        else:
            with h5py.File(dst_path, "w") as fout:
                # Copy root attrs
                copy_attrs(fin, fout)
                # Recreate full tree
                recursive_copy(fin, fout, adj_root, dry_run=False, full_renames=full_renames)

def main():
    args = parse_args()
    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir
    adj_root: str = args.adj_root.strip("/")
    patterns = [p.strip() for p in args.pattern.split(",") if p.strip()]

    if not in_dir.is_dir():
        print(f"ERROR: input_dir does not exist or is not a directory: {in_dir}", file=sys.stderr)
        sys.exit(1)

    ensure_dir(out_dir)

    files = []
    for pat in patterns:
        files.extend(sorted(in_dir.rglob(pat)))
    files = [p for p in files if p.is_file()]

    if not files:
        print("No files matched the pattern(s). Nothing to do.")
        return

    for src in files:
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        process_file(src, dst, adj_root=adj_root, dry_run=args.dry_run, full_renames=full_renames)

    print("\nDone.")

if __name__ == "__main__":
    main()