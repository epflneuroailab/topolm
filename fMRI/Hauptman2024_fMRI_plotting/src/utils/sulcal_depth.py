
import nibabel as nib
import numpy as np
from os.path import join as opj

def estimate_sulcal_depth(fMRI_data_dir, 
                          midthickness_file="lh.32k_fs_LR.hcp_inflated.surf.gii", 
                          inflated_file="lh.32k_fs_LR.midthickness.surf.gii"):
    midthickness = nib.load(opj(fMRI_data_dir, midthickness_file))
    inflated = nib.load(opj(fMRI_data_dir, inflated_file))
    
    coords_midthickness = midthickness.darrays[0].data
    coords_inflated = inflated.darrays[0].data
    
    sulcal_depth = np.linalg.norm(coords_midthickness - coords_inflated, axis=1)
    sulcal_depth = (sulcal_depth - sulcal_depth.min()) / (sulcal_depth.max() - sulcal_depth.min())
    
    return sulcal_depth, coords_midthickness
