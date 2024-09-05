# conda activate /work/upschrimpf1/mehrer/code/20240709_faciotopy_GLM/faciotopy_GLM_env

import os
import sys

import pickle as pkl
from omegaconf import OmegaConf

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import positions

DATA_PATH = 'data/contrasts/'
SAVE_PATH = 'data/clusters/'
FIG_PATH = '../figures/visualizations/'

layer_names = []
for i in range(12):
    layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

cfg_file = 'cluster_gpt2.yaml'
for i, arg in enumerate(sys.argv):
    if arg[:3] == 'cfg':
        cfg_file = arg.split('=')[1]
        sys.argv.pop(i)

cfg = OmegaConf.load(cfg_file)
cfg.update(OmegaConf.from_cli())

params = [cfg.radius, cfg.neighborhoods, cfg.alpha, cfg.batch_size, cfg.accum, cfg.decay]
params = '-'.join([str(p) for p in params])

with open(os.path.join(DATA_PATH, params, f'{cfg.stimulus}-{cfg.contrast}.pkl'), 'rb') as f:
    t_map = pkl.load(f)[cfg.layer]

position_dir = '../models/gpt2-positions-' + str(cfg.radius) + '-' + str(cfg.neighborhoods)
with open(f'{position_dir}/{layer_names[cfg.layer]}.pkl', 'rb') as f:
    pos = pkl.load(f)

coordinates = pos.coordinates.to(int)

# find the index (vertex) with the highest t-value
# seed_vertex = np.argmax(t_map)
print(np.max(t_map))
seed_vertex = np.argsort((t_map))[-2]
print(f"seed vertex with the highest t-value: {seed_vertex} | t-value: {t_map[seed_vertex]}")

# function to get neighbors of a given index
def get_neighbors(center_idx, positions):

    grid_size = int(np.sqrt(len(positions)))

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

    return list(neighbors)

# Clustering algorithm
def grow_cluster(seed_vertex, t_map, positions, threshold):
    cluster = set([seed_vertex])
    candidates = get_neighbors(seed_vertex, positions)

    while candidates:
        # get the candidate vertex with the highest t-value
        best_candidate = max(candidates, key=lambda v: t_map[v])

        # stop if the best candidate is below the threshold
        if t_map[best_candidate] < threshold:
            break

        # add the best candidate to the cluster
        cluster.add(best_candidate)
        candidates.remove(best_candidate)

        # add new neighbors of the best candidate to the candidate list
        new_neighbors = get_neighbors(best_candidate, positions)
        candidates.extend([v for v in new_neighbors if v not in cluster])

    return cluster

# grow the cluster starting from the seed vertex
cluster = grow_cluster(seed_vertex, t_map, coordinates, cfg.threshold)
print(f"vertices in the cluster: {cluster}")

# create a mask for the cluster
cluster_map = np.zeros(len(t_map))
cluster_map[list(cluster)] = 1

cluster_mask = np.zeros((28, 28))
cluster_mask[coordinates[:, 0], coordinates[:, 1]] = cluster_map

os.makedirs(os.path.join(SAVE_PATH, params, f'{cfg.stimulus}-{cfg.contrast}'), exist_ok=True)
with open(os.path.join(SAVE_PATH, params, f'{cfg.stimulus}-{cfg.contrast}', f'{cfg.layer}.pkl'), 'wb') as f:
    pkl.dump(cluster_mask, f)

# plotting
grid = np.full((28, 28), np.nan)
grid[coordinates[:, 0], coordinates[:, 1]] = t_map

plt.figure(figsize=(8, 8))
sns.heatmap(grid, cmap="gray", cbar=False, square=True)

# transparency mask
mask = cluster_mask.reshape(28, 28).astype(bool)
sns.heatmap(grid, cmap="RdBu_r", mask = ~mask, cbar=True, square=True, center=0)

os.makedirs(os.path.join(FIG_PATH, params, 'clusters', f'{cfg.stimulus}-{cfg.contrast}'), exist_ok=True)
plt.savefig(os.path.join(FIG_PATH, params, 'clusters', f'{cfg.stimulus}-{cfg.contrast}', f'{cfg.layer}.png'))
