import math
import torch
from torch.distributions import Normal, Bernoulli
import torch.nn as nn
import scipy.spatial
import numpy as np
import logging

def xy_dists_from_coords(in_coords, out_coords):
    dx = scipy.spatial.distance_matrix(
        np.expand_dims(in_coords[:, 0], 1), np.expand_dims(out_coords[:, 0], 1), p=1
    )
    dy = scipy.spatial.distance_matrix(
        np.expand_dims(in_coords[:, 1], 1), np.expand_dims(out_coords[:, 1], 1), p=1
    )
    return dx, dy


class Constraints:
    def __init__(self, embedding_dim, r_sigma, wrap):
        self.embedding_dim = embedding_dim

        assert (
            math.isqrt(self.embedding_dim) * math.isqrt(self.embedding_dim)
            == self.embedding_dim
        ), "Must be perfect square"

        self.square = math.isqrt(self.embedding_dim)
        self.coordinates = (
            np.indices((self.square, self.square)).reshape(2, self.embedding_dim)
        ).transpose() / (self.square)
        self.dx, self.dy = xy_dists_from_coords(self.coordinates, self.coordinates)
        if wrap:
            self.wrap()
        self.distances = (self.dx**2 + self.dy**2) ** (1 / 2)  # distance formula
        self.r_sigma = r_sigma        

    def wrap(self):
        wrap_x = self.dx > 0.5
        wrap_y = self.dy > 0.5
        self.dx[wrap_x] = 1 - self.dx[wrap_x]
        self.dy[wrap_y] = 1 - self.dy[wrap_y]

    def circular(self):
        """
        helper function to select fraction of units with the smallest distance values
        we assume the total grid side length is 1 and analytically compute the RF radius r
        Connections with distance > r are masked out, and vice versa
        let Ac be area of RF circle, As be area of square, and f be fraction to include for a perfect circle (center of grid)
        we have:
        Ac = f*As
        pi*r**2 = f*s**2 ; s = 1
        r = sqrt(f/pi)
        """
        r = np.sqrt(self.r_sigma / np.pi)
        mask = self.distances <= r

        return mask

    def gaussian(self):
        logits = Normal(torch.Tensor([0.0]), self.r_sigma).log_prob(torch.Tensor(self.distances))
        return self._logits_to_mask(logits)
    
    def full(self):
        assert self.r_sigma == 1, "full connectivity implies r_sigma=1, forcing you to demonstrate you understand this"
        return torch.ones_like(torch.tensor(self.distances), dtype=torch.bool)
    
    def _logits_to_mask(self, logits):
        """
        simple helper function to turn connection logits into samples from a probability distribution where the max logit corresponds to probability of 1
        """
        connection_density = torch.exp(logits)/(1+torch.exp(logits))
        connection_probs = connection_density/torch.max(connection_density)
        mask = Bernoulli(connection_probs).sample()
        return mask