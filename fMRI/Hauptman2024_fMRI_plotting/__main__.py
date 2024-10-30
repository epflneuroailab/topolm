# source /home/mehrer/Documents/projects/topo_language_model/Hauptman_2023/plotting/code/topo_LM_venv/bin/activate

from src.data.load_data import load_anatomical_data, load_functional_data
from src.processing.morans import compute_morans_I
from src.plotting.plots import interactive_plots, static_plots

# params
main_dir = '/home/mehrer/Documents/projects/topo_language_model/Hauptman_2023/plotting'
fMRI_data_dir = f"{main_dir}/data"
plots_dir = f"{main_dir}/plots"
surface_for_plotting_contrast_map = 'inflated'
odd_even_all = 'all'  # 'odd' or 'even' or 'all' runs
p_value_threshold = 0.05  # (1 --> no thresholding)
correction = 'FDR'  # 'FDR' or 'uncorrected'
permute_vertices = False

def main():
    # Load and prepare data
    coords, faces, sulcal_depth, coords_midthickness, permutation = \
        load_anatomical_data(fMRI_data_dir, permute_vertices=permute_vertices,
                             surface_for_plotting_contrast_map=surface_for_plotting_contrast_map)
    combined_contrast_data, n_sig_verb_vertices, n_sig_noun_vertices = \
        load_functional_data(fMRI_data_dir, permute_vertices, permutation, odd_even_all=odd_even_all,
                             significance_level=p_value_threshold, correction=correction)
    moran_I = compute_morans_I(combined_contrast_data, coords, faces)

    # Create args tuple to pass into plot functions
    plot_args = (coords, faces, combined_contrast_data, coords_midthickness, sulcal_depth,
                 plots_dir, odd_even_all, p_value_threshold, correction, permute_vertices,
                 n_sig_verb_vertices, n_sig_noun_vertices, moran_I)

    # Plotting
    interactive_plots(*plot_args) # html --> used for ICLR 2025 submission
    static_plots(*plot_args) # svg, png: not rendered well. ugly. 

if __name__ == "__main__":
    main()
