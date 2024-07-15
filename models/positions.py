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
    coordinates: np.ndarray

    # neighborhood_indices is a M x N binary matrix, where there are M neighborhoods
    neighborhood_indices: np.ndarray
    neighborhood_radius: float

    def save(self, save_dir: Path):

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok = True, parents = True)
        path = save_dir / f'{self.name}.pkl'

        with path.open('wb') as f:
            pkl.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> 'LayerPositions':

        path = Path(path)

        assert path.exists(), path
        assert path.suffix == '.pkl', 'invalid file, needs to be pickle'

        with path.open('rb') as f:
            return pkl.load(f)


@dataclass
class NetworkPositions:

    # dictionary of all layer positions
    layer_positions: Dict[str, LayerPositions]
    version: int

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

    def to_torch(self):
        """
        Converts each array or float in the original layer positions to be a torch
        Tensor of the appropriate type
        """
        for pos in self.layer_positions.values():

            pos.coordinates = torch.from_numpy(pos.coordinates.astype(np.float32))
            
            pos.neighborhood_indices = torch.from_numpy(
                pos.neighborhood_indices.astype(int)
            )

            pos.neighborhood_width = torch.tensor(pos.neighborhood_width)

### SPATIAL LOSS ###

# compute pairwise lp-norm for an Nx2 array
def p_norm(positions, p = 'inf'):

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    if p == 1:
        return np.sum(np.abs(diff), axis = -1)
    if p == 2:
        return np.sqrt(np.sum(diff ** 2, axis = -1))
    if p == 'inf':
        return np.max(np.abs(diff), axis = -1)

    raise ValueError(f'norm type {p} not supported')

def spatial_loss(activations, positions, p = 'inf'):

    num_units = activations.shape[0]
    idx = np.triu_indices(num_units, k = 1)

    D = 1 / (1 + p_norm(positions, p))
    r = np.corrcoef(activations.T)

    return 1 - np.corrcoef(r[idx], D[idx])[0][1]

### PRE OPTIMIZATION ###

# helper function
def swap_units(positions, swap_ind):
    positions[swap_ind] = positions[np.flip(swap_ind)]

# run swap optimization on a subset of units (e.g. a neighborhood)
def swap_local(activations, positions, steps = 500):
    
    rng = np.random.default_rng(seed = 42)
    old_loss = np.inf

    for _ in tqdm(range(steps)):

        swap_ind = rng.choice(positions.shape[0], size = 2, replace = False)
        swap_units(positions, swap_ind)

        loss = spatial_loss(activations, positions)

        if loss > old_loss:
            swap_units(positions, swap_ind)
        else:
            old_loss = loss

    return positions

# binary mask of points within 'radius' of center
def get_neighborhood(center, positions, radius = 0.3, p = 'inf'):
    distances = p_norm(positions, p)
    return distances[center] < radius

# identify neighborhoods and run local optimization on each
def swap_optimize(activations, positions, steps = 10_000, local_steps = 500, radius = 0.3, p = 'inf'):

    rng = np.random.default_rng(seed = 42)

    for _ in tqdm(range(steps)):

        # center is an index, neighborhood is a binary mask
        center = rng.choice(positions.shape[0])
        neighborhood = get_neighborhood(center, positions, radius, p)

        positions[neighborhood] = swap_local(activations[neighborhood], positions[neighborhood], local_steps, p)




