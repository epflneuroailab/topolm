import os
import sys

import torch
import numpy as np
import scipy.stats as stats

import pandas as pd
import pickle as pkl

import itertools
from itertools import product
from collections import defaultdict

import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from hrf import NeuronSmoothing

sys.path.insert(1, '../models/')
import positions

cfg = OmegaConf.from_cli()

model_name = cfg.name
pos_dir = '../models/gpt2-positions/gpt2-positions-5-5'

with open(os.path.join(f'data/localizer/{model_name}/lmask.pkl'), 'rb') as f:
    language_mask = pkl.load(f)

### COMPUTE CLUSTERS ###

# function to get neighbors of a given index
def get_neighbors(center_idx, grid_size = 28):
    positions = np.array(list(product(np.arange(grid_size), repeat = 2)))
    
    center_pos = positions[center_idx]
    # deltas = np.array([[-1, -1], [-1, 0], [-1, 1],
    #                    [ 0, -1],          [ 0, 1],
    #                    [ 1, -1], [ 1, 0], [ 1, 1]])
    deltas = np.array([[-1, 0], [ 0, -1], [ 0, 1], [ 1, 0]])

    neighbor_positions = center_pos + deltas
    valid_neighbors = (neighbor_positions[:, 0] >= 0) & (neighbor_positions[:, 0] < grid_size) & \
                      (neighbor_positions[:, 1] >= 0) & (neighbor_positions[:, 1] < grid_size)

    neighbor_positions = neighbor_positions[valid_neighbors]

    neighbors = np.flatnonzero(
        (positions[:, None] == neighbor_positions).all(-1).any(-1)
    )

    return set(neighbors)

# clustering algorithm
def grow_cluster(seed_vertex, t_map, threshold=0.1, grid_size=28, flip = False):
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

def list_of_clusters(mask):
    
    clusters = []
    mask_copy = mask.copy()

    while np.abs(np.sum(mask_copy)) > 0:

        seed_vertex = np.unravel_index(np.argmax(mask_copy), mask_copy.shape)
        if np.abs(mask_copy[seed_vertex]) < 1.0:
            break
        
        cluster = grow_cluster(seed_vertex, mask_copy, 0)

        clusters.append(cluster)
        mask_copy[list(cluster)] = 0
    
    return clusters

clusters = {}
for i, layer in enumerate(language_mask):
    clusters[layer_names[i]] = list_of_clusters(layer)

all_clusters = []

for i, layer in enumerate(clusters):
    for cluster in clusters[layer]:
        mask = np.zeros((24, 784))
        mask[i][list(cluster)] = 1
        all_clusters.append(mask)

with open(f'by-cluster/{model_name}.pkl', 'wb') as f:
    pkl.dump(all_clusters, f)

fwhm_mm = 1.0
resolution_mm = 1
smoothing = NeuronSmoothing(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

all_clusters = [cluster for cluster in all_clusters if np.sum(cluster) >= 25]

layer_names = []
for i in range(12):
    layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

cluster_layers = []

for cluster in all_clusters:
    index = np.argmax(np.sum(cluster != 0, axis=1))
    cluster_layers.append(index)

with open(f'data/responses/{model_name}/fedorenko.pkl', 'rb') as f:
    responses = pkl.load(f)

pos = positions.NetworkPositions.load_from_dir(pos_dir)

all_responses = []

for i, cluster in enumerate(all_clusters):
    
    cluster_response = {
        'layer' : layer_names[cluster_layers[i]],
        'cluster' : i
    }
    
    coordinates = pos.layer_positions[layer_names[cluster_layers[i]]].coordinates.numpy().astype(int)
    
    for condition in responses:
        grid = np.full((160, 28, 28), np.nan)
        gridx, gridy, smoothed_activations = smoothing(coordinates, responses[condition][:, cluster_layers[i], :])
        grid[:, gridx.astype(int).squeeze(), gridy.astype(int).squeeze()] = np.abs(smoothed_activations)
        response = (grid * cluster[np.newaxis, cluster_layers[i], :].reshape(1, 28, 28))
        cluster_response['condition'] = condition

        for j in response:
            cluster_response['mean_act'] = j.mean()
            all_responses.append({
                'layer' : layer_names[cluster_layers[i]],
                'cluster' : i,
                'cluster_size' : cluster.sum(),
                'condition' : condition,
                'mean_act' : j.mean()
            })

df = pd.DataFrame(all_responses)
df.to_csv(f'by-cluster/{model_name}-fedorenko.csv')

### PLOT BY-CLUSTER RESPONSES ###

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

condition_order = ['S', 'W', 'J', 'N']

cmap = plt.get_cmap('Blues_r')
condition_colors = {cond: cmap(i / len(condition_order)) for i, cond in enumerate(condition_order)}
df['condition'] = pd.Categorical(df['condition'], categories=condition_order, ordered=True)

unique_clusters = df['cluster'].unique()

num_clusters = len(unique_clusters)
cols = min(8, num_clusters)
rows = (num_clusters + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 4 * rows))
axes = np.array(axes)
axes = axes.flatten()

for idx, cluster in enumerate(unique_clusters):
    cluster_df = df[df['cluster'] == cluster]
    cluster_size = int(cluster_df['cluster_size'].mean())
    
    grouped = cluster_df.groupby('condition')['mean_act'].agg(['mean', 'count', 'std'])
    ci95_hi = []
    ci95_lo = []
    
    for i in range(len(grouped)):
        m, c, s = grouped.iloc[i]
        ci = stats.t.interval(0.95, c - 1, loc=m, scale=s / np.sqrt(c))
        ci95_hi.append(ci[1] - m)
        ci95_lo.append(m - ci[0])
    
    colors = [condition_colors[cond] for cond in grouped.index]
    
    axes[idx].bar(grouped.index, grouped['mean'], yerr=[ci95_lo, ci95_hi], capsize=0, alpha=0.8, color=colors)
    axes[idx].spines['top'].set_visible(False)
    axes[idx].spines['right'].set_visible(False)
    axes[idx].set_title(f"layer {cluster_layers[idx]} | size {cluster_size}")
    # axes[idx].set_xlabel("Condition")
    axes[idx].set_ylabel("Mean Activation")

plt.tight_layout()
plt.savefig(f'../figures/visualizations/preresid/fedorenko/by-cluster.svg')