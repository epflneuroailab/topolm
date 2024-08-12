"""

this file contains infrastructure for position embeddings
 - partially built on top of spacetorch (github.com/neuroailab/TDANN)
 - infra for loading / saving positions
 - scripts for spatial loss calculation
 - scripts for pre-optimization

"""

import numpy as np
import pickle as pkl
from dataclasses import dataclass

from tqdm import tqdm

from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

### CLASS INFRA - LayerPositions + NetworkPositions ###

@dataclass
class LayerPositions:

    name: str # layer name

    # coordinates is an N x 2 matrix with the x-coordinates of each unit in the first
    # column and the y-coordinates in the second column
    coordinates: torch.Tensor

    # neighborhood_indices is a M x N binary matrix, where there are M neighborhoods
    neighborhood_indices: torch.Tensor

    # number of neighborhoods to compute / average loss over
    neighborhoods_per_batch: int

    def save(self, save_dir: Path):

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok = True, parents = True)
        path = save_dir / f'{self.name}.pkl'

        with path.open('wb') as f:
            pkl.dump(self, f)

    # load positions from disk
    @classmethod
    def load(cls, path: Path) -> 'LayerPositions':

        path = Path(path)

        assert path.exists(), path
        assert path.suffix == '.pkl', 'invalid file, needs to be pickle'

        with path.open('rb') as f:
            return pkl.load(f)

    # helper to put things on gpu
    def to(self, device):
        self.coordinates = self.coordinates.to(device)
        self.neighborhood_indices = self.neighborhood_indices.to(device)

        return self


@dataclass
class NetworkPositions:

    # dictionary of all layer positions
    layer_positions: Dict[str, LayerPositions]
    version: int

    # load ALL positions from disk
    @classmethod
    def load_from_dir(cls, load_dir: Path):

        load_dir = Path(load_dir)

        assert load_dir.is_dir(), load_dir
        layer_files = list(load_dir.glob('*.pkl'))

        d = {}

        for layer_file in layer_files:

            layer_name = layer_file.stem
            d[layer_name] = LayerPositions.load(layer_file)

        version_path = load_dir / 'version.txt'
        version = 1.0

        if version_path.is_file():
            with version_path.open('r') as f:
                version = int(f.readline())

        return cls(version = version, layer_positions = d)

### SPATIAL LOSS ###

# norm implementation for lp-norm
# for some reason this makes things 20x faster than torch.norm????
def p_norm(positions, p='inf'):

    diff = positions[:, None, :] - positions[None, :, :]

    if p == 1:
        return torch.sum(torch.abs(diff), dim=-1)
    if p == 2:
        return torch.sqrt(torch.sum(diff ** 2, dim=-1))
    if p == 'inf':
        return torch.max(torch.abs(diff), dim=-1)[0]

    raise ValueError(f'norm type {p} not supported')

# compute spatial loss for a specific neighborhood (tensors should already be masked)
def local_spatial_loss(activations, positions):

    num_units = activations.shape[1]
    idx = np.tril_indices(num_units, k = -1)

    D = 1 / (1 + p_norm(positions, 2))
    r = torch.corrcoef(activations.T)

    return 0.5 * (1 - torch.corrcoef(torch.stack([r[idx], D[idx]], dim = 0))[0, 1])

# compute spatial loss for an entire batch
def spatial_loss_fn(activations, positions):

    cur_loss = []
    neighborhoods = torch.randperm(len(positions.neighborhood_indices))[:positions.neighborhoods_per_batch]

    for i in neighborhoods:
        mask = positions.neighborhood_indices[i].to(bool)
        cur_loss.append(local_spatial_loss(activations[:, mask], positions.coordinates[mask]))

    return torch.stack(cur_loss).mean()

### PRE OPTIMIZATION ###

# helper function to swap two neurons
def swap_units(positions, swap_ind):
    positions[swap_ind] = positions[torch.flip(swap_ind, [0])]

# run swap optimization on a neighborhood (tensors should already be masked)
def swap_local(activations, positions, neighborhood, steps = 500, p = 'inf', rng=None):
    
    n_swapped = 0
    old_loss = np.inf
    coordinates = positions.coordinates[neighborhood]

    for i in range(steps):

        swap_ind = torch.randperm(coordinates.shape[0], generator = rng)[:2]
        swap_units(coordinates, swap_ind)

        loss = local_spatial_loss(activations[:, neighborhood], coordinates)

        # if the swapping doesn't decrease loss, unswap and move on
        if loss > old_loss:
            swap_units(coordinates, swap_ind)
        else:
            swap_units(neighborhood, swap_ind)
            old_loss = loss
            n_swapped += 1

    return coordinates, n_swapped, neighborhood

# binary mask of points within 'radius' of center
def get_neighborhood(center, positions, radius = 5, p = 'inf'):
    distances = p_norm(positions, p)
    return distances[center] < radius

# randomly select a center point for a neighborhood (avoiding the edges of the layer)
def get_center(positions, radius, rng=None):

    N = int(np.sqrt(positions.shape[0]))

    mask = (positions[:, 0] >= radius - 1) & (positions[:, 0] <= N - radius) & \
           (positions[:, 1] >= radius - 1) & (positions[:, 1] <= N - radius)

    indices = torch.nonzero(torch.Tensor(mask)).flatten()

    return indices[torch.randint(0, len(indices), (1,), generator = rng).item()]

# identify neighborhoods and run local optimization on each
def swap_optimize(activations, positions, num_neighborhoods, local_steps = 500, radius = 5, p = 'inf', rng=None):

    tot_n_swapped = 0

    for i in tqdm(range(num_neighborhoods)):

        # do pre-optimization on predetermined neighborhood
        neighborhood = positions.neighborhood_indices[i].to(bool)
        positions.coordinates[neighborhood], n_swapped, neighborhood = swap_local(activations, positions, neighborhood, local_steps, p, rng=rng)

        # replace the neighborhood with the swapped version
        positions.neighborhood_indices[i] = neighborhood.to(int)

        tot_n_swapped += n_swapped

    return positions, tot_n_swapped

### NUMPY SPATIAL LOSS FUNCTIONS ###
# these are NOT used anywhere, they are just a guide

def p_norm_numpy(positions, p = 'inf'):

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    if p == 1:
        return np.sum(np.abs(diff), axis = -1)
    if p == 2:
        return np.sqrt(np.sum(diff ** 2, axis = -1))
    if p == 'inf':
        return np.max(np.abs(diff), axis = -1)

    raise ValueError(f'norm type {p} not supported')

def local_spatial_loss_numpy(activations, positions):

    num_units = activations.shape[1]
    idx = np.tril_indices(num_units, k = -1)

    D = 1 / (1 + p_norm_numpy(positions, 2))
    r = np.corrcoef(activations.T)

    return 1 - np.corrcoef(r[idx], D[idx])[0][1]

def spatial_loss_numpy(activations, positions):

    cur_loss = []
    neighborhoods = np.random.choice(len(positions.neighborhood_indices), positions.neighborhoods_per_batch, replace = False)

    for i in neighborhoods:
        mask = positions.neighborhood_indices[i].astype(bool)
        cur_loss.append(local_spatial_loss_numpy(activations[:, mask], positions.coordinates[mask]))

    return np.stack(cur_loss).mean()


