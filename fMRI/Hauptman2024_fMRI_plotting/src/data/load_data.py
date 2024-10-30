
import os
import nibabel as nib
import numpy as np
from os.path import join as opj

from src.utils.sulcal_depth import estimate_sulcal_depth
from statsmodels.stats.multitest import multipletests

def load_anatomical_data(fMRI_data_dir, permute_vertices=False, surface_for_plotting_contrast_map='inflated'):
    if surface_for_plotting_contrast_map == 'inflated':
        surface_file = opj(fMRI_data_dir, f"lh.32k_fs_LR.hcp_inflated.surf.gii")
    elif surface_for_plotting_contrast_map == 'midthickness':
        surface_file = opj(fMRI_data_dir, f"lh.32k_fs_LR.midthickness.surf.gii")
    surface = nib.load(surface_file)
    coords, faces = surface.darrays[0].data, surface.darrays[1].data
    
    sulcal_depth, coords_midthickness = estimate_sulcal_depth(fMRI_data_dir)

    if permute_vertices:
        permutation = np.random.permutation(len(coords))
        inverse_permutation = np.zeros_like(permutation)
        inverse_permutation[permutation] = np.arange(len(permutation))
        coords = coords[permutation]
        faces = inverse_permutation[faces]
    else:
        permutation = np.arange(len(coords))

    return coords, faces, sulcal_depth, coords_midthickness, permutation

def load_functional_data(fMRI_data_dir, permute_vertices, permutation, odd_even_all='all', significance_level=0.05, correction='FDR'):
    import xarray as xr

    group_level_data = xr.open_dataset(
        os.path.join(fMRI_data_dir, f'group_level_verb_vs_noun_contrast_lh_{odd_even_all}_runs.nc')
    ).load()
    sub_t_values = group_level_data['t_values']
    sub_p_values = group_level_data['p_values']
    
    mask_verb_sig, mask_noun_sig = create_masks(
        sub_t_values, sub_p_values, significance_level=significance_level, correction=correction
    )
    
    combined_contrast_data = np.zeros_like(sub_t_values.values)

    if permute_vertices:
        combined_contrast_data = combined_contrast_data[permutation]

    combined_contrast_data[mask_verb_sig] = sub_t_values.values[mask_verb_sig]
    combined_contrast_data[mask_noun_sig] = sub_t_values.values[mask_noun_sig]

    n_sig_verb_vertices = np.sum(combined_contrast_data > 0)
    n_sig_noun_vertices = np.sum(combined_contrast_data < 0)
    
    return combined_contrast_data, n_sig_verb_vertices, n_sig_noun_vertices

def create_masks(t_values, p_values, significance_level=0.05, correction='FDR'):
    if correction == 'FDR':
        p_values.values[np.isnan(p_values.values)] = 1
        _, corrected_p_values, _, _ = multipletests(p_values.values, method='fdr_bh')
        mask_significant = corrected_p_values < significance_level
    elif correction == 'uncorrected':
        mask_significant = p_values.values < significance_level
    mask_positive = t_values.values > 0
    mask_negative = t_values.values < 0
    mask_verb_sig = mask_significant & mask_positive
    mask_noun_sig = mask_significant & mask_negative
    return mask_verb_sig, mask_noun_sig
