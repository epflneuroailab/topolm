
import numpy as np
from esda.moran import Moran
from libpysal.weights import W
from src.processing.adjacency import compute_adjacency_list

def compute_morans_I(combined_contrast_data, coords, faces):
    # Create adjacency list
    adjacency_list = compute_adjacency_list(faces, len(coords))
    
    # Filter valid data and prepare adjacency list
    valid_mask = ~np.isnan(combined_contrast_data) & (combined_contrast_data != 0)
    valid_indices = np.where(valid_mask)[0]
    valid_data = combined_contrast_data[valid_mask]
    
    # Map valid indices to positions in valid_data for adjacency list
    valid_index_map = {idx: pos for pos, idx in enumerate(valid_indices)}
    valid_adjacency_list = {
        valid_index_map[idx]: [valid_index_map[nbr] for nbr in adjacency_list[idx] if nbr in valid_index_map]
        for idx in valid_indices
    }
    valid_adjacency_list = {idx: nbrs for idx, nbrs in valid_adjacency_list.items() if nbrs}

    # Prepare data for Moran's I computation
    w = W(valid_adjacency_list)
    ordered_data = valid_data[[w.id_order]]

    moran = Moran(ordered_data, w)
    return moran.I
