# conda activate /work/upschrimpf1/mehrer/code/20240709_faciotopy_GLM/faciotopy_GLM_env
import os
import numpy as np
import pandas as pd
import nibabel as nib
import xarray as xr
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm import OLSModel
from scipy.stats import ttest_1samp
from nilearn.plotting import plot_surf_stat_map
from tqdm import tqdm
import time

main_dir = '/work/upschrimpf1/mehrer/code/20240915_topo_LM_fMRI_data_Hauptman'
output_dir = opj(main_dir, 'outputs')
plots_dir = opj(main_dir, 'plots')

subjects = [f'PTP_{i:02d}' for i in range(22, 44)]  # subs 22-43 are sighted
hemisphere = 'lh'  # Elli 2019 and Hauptman 2024 only analyzed the left hemisphere
# (because responses to linguistic stimuli are typically left-lateralized)

def prepare_events_df(logfile_path, run):
    events_df = pd.read_csv(logfile_path)
    events_df = events_df[events_df['Run'] == run]  

    events = pd.DataFrame({
        'onset': events_df['Onset'],
        'duration': events_df['Duration'],
        'trial_type': events_df['Cat']
    })
    events['trial_type'] = events['trial_type'].fillna('baseline')
    return events

def create_design_matrix(events, n_scans, tr):
    frame_times = np.arange(n_scans) * tr
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model='spm',
        drift_model='cosine',
        high_pass=0.01
    )
    return design_matrix

def compute_contrast_for_vertex(time_series, design_matrix):
    betas = []
    residuals = []
    for vertex_data in time_series:
        glm = OLSModel(design_matrix)
        glm_fit = glm.fit(vertex_data)
        betas.append(glm_fit.theta)
        residuals.append(glm_fit.residuals)
    
    betas = np.array(betas)
    residuals = np.array(residuals)
    return betas, residuals

def compute_contrast_maps(betas, residuals, design_matrix, contrast_vector):
    contrast_map = np.dot(betas, contrast_vector)
    residual_variance = np.var(residuals, axis=1, ddof=len(design_matrix.columns))
    design_matrix_projection = np.dot(
        np.dot(contrast_vector, np.linalg.pinv(design_matrix.T @ design_matrix)),
        contrast_vector.T
    )
    standard_error = np.sqrt(residual_variance * design_matrix_projection)
    standard_error[standard_error == 0] = np.nan
    standard_error = np.nan_to_num(standard_error, nan=np.inf)
    t_map = contrast_map / standard_error
    return t_map

def process_subject(subject):
    subject_contrast_maps = []
    data_dir = '/work/upschrimpf1/mehrer/datasets/Hauptman_2023/DataZip'
    tr = 2.0
    for run in tqdm(range(1, 9), desc=f"Processing Runs for {subject}", leave=False):
        logfile_path = opj(data_dir, f'{subject}/logfile_with_baseline.csv')
        events = prepare_events_df(logfile_path, run)

        hemisphere_func_gii_path = opj(data_dir, \
            f'{subject}/functional/run_{run:02d}/{hemisphere}.32k_fs_LR.surfed_data.func.gii')
        func_data_img = nib.load(hemisphere_func_gii_path)
        n_scans = len(func_data_img.darrays)

        design_matrix = create_design_matrix(events, n_scans, tr)
        time_series = np.array([darray.data for darray in func_data_img.darrays]).T
        betas, residuals = compute_contrast_for_vertex(time_series, design_matrix)
        
        contrast_verb_vs_noun = np.array([0, -1, -1, 1, 1]) # contrast: verb - noun
        contrast_verb_vs_noun = np.concatenate([contrast_verb_vs_noun, np.zeros(10)])
        
        t_map = compute_contrast_maps(betas, residuals, design_matrix, contrast_verb_vs_noun)
        subject_contrast_maps.append(t_map)

    subject_contrast_maps = np.mean(subject_contrast_maps, axis=0)
    return subject_contrast_maps

def group_analysis(all_subjects_contrast_maps):
    all_subjects_contrast_maps = np.array(all_subjects_contrast_maps)
    group_t_values, group_p_values = ttest_1samp(all_subjects_contrast_maps, 0, axis=0)
    return group_t_values, group_p_values

def save_group_results_as_xarray(group_t_values, group_p_values, hemisphere, output_dir):
    coords = np.arange(group_t_values.shape[0])  
    data = xr.Dataset(
        {
            "t_values": (["coords"], group_t_values),
            "p_values": (["coords"], group_p_values),
        },
        coords={"coords": coords}
    )
    output_path = os.path.join(output_dir, f'group_level_verb_vs_noun_contrast_{hemisphere}_all_runs.nc')
    data.to_netcdf(output_path)
    print(f"Group results saved to {output_path}")

def main():
    all_subjects_contrast_maps = []
    for subject in tqdm(subjects, desc="Processing Subjects"):
        subject_contrast_maps = process_subject(subject)
        all_subjects_contrast_maps.append(subject_contrast_maps)

    group_t_values, group_p_values = group_analysis(all_subjects_contrast_maps)
    save_group_results_as_xarray(group_t_values, group_p_values, hemisphere, output_dir)

if __name__ == "__main__":
    main()
