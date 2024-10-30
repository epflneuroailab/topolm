
import os
import matplotlib.pyplot as plt
from nilearn.plotting import view_surf
from nilearn.plotting import plot_surf
from nilearn.plotting import plot_surf_stat_map

def interactive_plots(coords, faces, combined_contrast_data, coords_midthickness, sulcal_depth,
                      plots_dir, odd_even_all, p_value_threshold, correction, permute_vertices,
                      n_sig_verb_vertices, n_sig_noun_vertices, moran_I):
    # Define the plot title with formatted verb/noun counts and Moran's I
    plot_title = (
        f'Hauptman 2024 - Group effects {odd_even_all} runs, p < {p_value_threshold}, corr.: {correction} | '
        f'<span style="color: red;">Verbs > Nouns: {n_sig_verb_vertices} vert.</span>, '
        f'<span style="color: blue;">Nouns > Verbs: {n_sig_noun_vertices} vert.</span><br>'
        f"Moran's I: {moran_I:.4f}"
    )
    base_filename = f'group_level_combined_contrast_{odd_even_all}_runs_p_threshold_{p_value_threshold}_correction_{correction}'
    permute_suffix = "_permuted_vertices" if permute_vertices else ""

    # Interactive plot without sulcal depth
    view = view_surf(
        (coords, faces),
        surf_map=combined_contrast_data,
        title=plot_title,
        cmap='cold_hot',
        threshold=1e-6,
        symmetric_cmap=True
    )
    view.save_as_html(f"{plots_dir}/{base_filename}{permute_suffix}.html")
    print(f'Interactive plot without sulcal depth saved as HTML.')

    # Interactive plot with sulcal depth
    view = view_surf(
        (coords_midthickness, faces),
        surf_map=combined_contrast_data,
        bg_map=sulcal_depth,
        cmap='cold_hot',
        threshold=1e-6,
        bg_on_data=False,
        darkness=0.7,
        title=plot_title
    )
    view.save_as_html(os.path.join(plots_dir, f"{base_filename}{permute_suffix}_inflated_showing_sulci.html"))
    print(f'Interactive plot with sulcal depth saved as HTML.')


def static_plots(coords, faces, combined_contrast_data, coords_midthickness, sulcal_depth,
                      plots_dir, odd_even_all, p_value_threshold, correction, permute_vertices,
                      n_sig_verb_vertices, n_sig_noun_vertices, moran_I):
    # Titles and file naming base
    static_title = (
        f'Hauptman 2024 - Group effects {odd_even_all} runs, p < {p_value_threshold}, corr.: {correction}\n'
        f'Verbs > Nouns: {n_sig_verb_vertices} vert., '
        f'Nouns > Verbs: {n_sig_noun_vertices} vert.\n'
        f"Moran's I: {moran_I:.4f}"
    )
    base_filename = f'group_level_combined_contrast_{odd_even_all}_runs_p_threshold_{p_value_threshold}_correction_{correction}'
    permute_suffix = "_permuted_vertices" if permute_vertices else ""
    
    # Plot without sulcal depth
    fig = plot_surf(
        surf_mesh=(coords, faces),
        surf_map=combined_contrast_data,
        cmap='cold_hot',
        threshold=1e-6,
        symmetric_cmap=True,
        title=static_title
    )
    for ext in ['svg', 'png']:
        fig.savefig(os.path.join(plots_dir, f"{base_filename}{permute_suffix}.{ext}"), format=ext, dpi=300 if ext == 'png' else None)
    plt.close(fig)
    print(f'Static plot saved in SVG and PNG formats.')

    # Plot with sulcal depth
    fig = plot_surf_stat_map(
        surf_mesh=(coords_midthickness, faces),
        stat_map=combined_contrast_data,
        bg_map=sulcal_depth,
        hemi='left',
        view='lateral',
        colorbar=True,
        cmap='cold_hot',
        threshold=1e-6,
        bg_on_data=True,
        darkness=0.5,
        title=static_title
    )
    for ext in ['svg', 'png']:
        fig.savefig(os.path.join(plots_dir, f"{base_filename}{permute_suffix}_inflated_showing_sulci.{ext}"), format=ext, dpi=300 if ext == 'png' else None)
    print(f'Static plot with sulcal depth saved in SVG and PNG formats.')
