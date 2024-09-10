import scipy
import numpy as np
import pandas as pd
import xarray as xr
import pickle as pkl

from tqdm import tqdm
from itertools import product

import seaborn as sns
import matplotlib.pyplot as plt

from hrf import NeuronSmoothing

fwhm_mm = 2.0
resolution_mm = 1
smoothing = NeuronSmoothing(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

def compute_selectivity(raw_activations):
    
    t_values_matrix = np.zeros((24, 784))
    p_values_matrix = np.zeros((24, 784))
    
    coordinates = np.array(list(product(np.arange(28), repeat = 2)))
    
    for idx in range(len(raw_activations[0])):
        activations = [raw_activations[0][idx], raw_activations[1][idx]]
        for i in range(2):
            gridx, gridy, activations[i] = smoothing(coordinates, activations[i])
    
        t_values_matrix[idx], p_values_matrix[idx] = scipy.stats.ttest_ind(activations[0], activations[1], axis=0, equal_var=False)
    
    adjusted_p_values = scipy.stats.false_discovery_control(p_values_matrix.flatten())
    adjusted_p_values = adjusted_p_values.reshape((24, activations[0].shape[1]))
    selectivity = t_values_matrix * (adjusted_p_values < 0.05)
    
    return selectivity, t_values_matrix, adjusted_p_values

# function to get neighbors of a given index
def get_neighbors(center_idx, grid_size = 28):
    positions = np.array(list(product(np.arange(grid_size), repeat = 2)))
    
    center_pos = positions[center_idx]
    deltas = np.array([[-1, -1], [-1, 0], [-1, 1],
                       [ 0, -1],          [ 0, 1],
                       [ 1, -1], [ 1, 0], [ 1, 1]])

    neighbor_positions = center_pos + deltas
    valid_neighbors = (neighbor_positions[:, 0] >= 0) & (neighbor_positions[:, 0] < grid_size) & \
                      (neighbor_positions[:, 1] >= 0) & (neighbor_positions[:, 1] < grid_size)

    neighbor_positions = neighbor_positions[valid_neighbors]

    neighbors = np.flatnonzero(
        (positions[:, None] == neighbor_positions).all(-1).any(-1)
    )

    return set(neighbors)

# Clustering algorithm
def grow_cluster(seed_vertex, t_map, threshold=1.0, grid_size=28, flip = False):
    if flip:
        t_map = -t_map
    
    positions = np.array(list(product(np.arange(grid_size), repeat = 2)))
    
    cluster = set(seed_vertex)
    candidates = get_neighbors(seed_vertex, grid_size=grid_size)

    while candidates:
        # get the candidate vertex with the highest t-value
        best_candidate = max(candidates, key=lambda v: t_map[v])
        
        # stop if the best candidate is below the threshold
        if t_map[best_candidate] < threshold:
            break

        # add the best candidate to the cluster
        cluster.add(best_candidate)

        # add new neighbors of the best candidate to the candidate list
        new_neighbors = get_neighbors(best_candidate, grid_size)
        candidates.update([v for v in new_neighbors if v not in cluster])
        candidates.remove(best_candidate)

    return cluster

def mean_cluster_size(t_map, threshold = 1.0):

    sizes = []
    t_map_copy = t_map.copy()

    iter = 0
    
    while iter < 2:

        seed_vertex = np.unravel_index(np.argmax(t_map_copy), t_map_copy.shape)
        
        if abs(t_map_copy[seed_vertex]) < threshold:
            iter += 1

        if iter > 1:
            break
        
        cluster = grow_cluster(seed_vertex, t_map_copy)
        sizes.append(len(cluster))
        t_map_copy[list(cluster)] = 0

    return np.array(sizes).mean()

def mean_cluster_size_all_layers(t_map, n_layers = 24):
    sizes = []
    
    for idx in range(n_layers):
        sizes.append(mean_cluster_size(t_map[idx]))

    return np.array(sizes)

def shuffle_activations(act):
    
    shuffled = [np.empty_like(act[0]), np.empty_like(act[1])]
    
    for i in range(act[0].shape[0]):
        perm = np.random.permutation(act[0].shape[2])
        shuffled[0][i] = act[0][i][:, perm]
        shuffled[1][i] = act[1][i][:, perm]

    return shuffled

def perm_test(raw_activations, n_layers = 24, n_iters = 1000):
    grid_size = int(np.sqrt(raw_activations[0].shape[0]))
    
    s_map, _, _ = compute_selectivity(raw_activations)
    true_areas = mean_cluster_size_all_layers(s_map)
    
    n_success = np.zeros(n_layers)    
    for _ in tqdm(range(n_iters)):
                
        # idx = np.random.rand(*raw_activations[0].shape).argsort(axis = -1)
        
        shuffled = shuffle_activations(raw_activations)        
        shuffled_s_map, _, _ = compute_selectivity(shuffled)
        shuffled_areas = mean_cluster_size_all_layers(shuffled_s_map)
        
        n_success += (shuffled_areas > true_areas).astype(int)

    return n_success / n_iters

contrasts = {
    'moseley' : ['abstract-abstract', 'concrete-concrete', 'noun-verb'],
    'elli' : ['noun_verb']
}

results = []
for stimulus, contrast_list in contrasts.items():
    for contrast in contrast_list:
        results.append([stimulus, contrast] + [None] * 24)  # 24 layers initialized to None

columns = ['stimulus', 'contrasts'] + [f'layer_{i+1}' for i in range(24)]
results = pd.DataFrame(results, columns=columns)

for stimulus in ['moseley', 'elli']:
    for contrast in contrasts[stimulus]:

        print(stimulus, contrast)

        with open(f'data/contrasts/5-5-0-48-mean-0/{stimulus}/{contrast}.pkl', 'rb') as f:
            data = pkl.load(f)

        for key in data:
            if key != 'raw':
                data[key] = data[key].reshape(24, 28, 28)

        x = xr.Dataset(
            {
                key: (("layer", "x", "y"), value)  # Names of dimensions can be modified as needed
                for key, value in data.items() if key != "raw"
            },
            coords={
                "layer": np.arange(24),  # Assuming you have 24 samples
                "x": np.arange(28),    # First spatial dimension
                "y": np.arange(28)     # Second spatial dimension
            }
        )

        n_trials = data['raw'][0].shape[1]

        raw_data = np.stack(data['raw'], axis=0)  # shape becomes (2, 24, n_trials, 784)
        raw_data_reshaped = raw_data.reshape(2, 24, n_trials, 28, 28)  # reshape (2, 24, n_trials, 28, 28)
        x['raw'] = (('condition', 'layer', 'trial', 'x', 'y'), raw_data_reshaped)
        x = x.assign_coords({'condition': ['noun', 'verb']})

        x.to_netcdf(f'data/permtest/nontopo/{stimulus}_{contrast}.nc')

        p_values = perm_test(data['raw'])

        row = pd.Series([stimulus, contrast] + p_values.tolist(), index = results.columns)
        results = pd.concat([results, row.to_frame().T], ignore_index=True)

results.to_csv('nontopo_perm_test_out.csv')