import sys
import os
import pickle as pkl
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import positions

def dist(positions):
    positions = positions.numpy()
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.max(np.abs(diff), axis = -1)

def loss(activations, positions):
    num_units = activations.shape[1]
    idx = np.tril_indices(num_units, k = -1)

    r = np.corrcoef(activations.T)
    d = 1 / (1 + dist(positions))

    return 1 - np.corrcoef(r[idx], d[idx])[0, 1]

# mean_activations = np.array([[0.1, 0.3, 0.45], [-0.2, 0.01, 0.2], [-0.1, -0.2, -0.01]])

# activations = []
# for i in range(5):
#     activations.append(mean_activations + (0.001 ** i)) # + 0.05 * np.random.normal(size = (3, 3)))

# activations = np.array(activations).reshape(5, 9)
# print(activations)
# positions = np.array(list(product(np.arange(3), repeat = 2)))

# print(loss(activations, positions))

# sns.heatmap(mean_activations, cbar=True, cmap='RdBu_r', center=0)
# plt.show()

with open(f'../models/gpt2-positions-5-5/layer.0.attn.pkl', 'rb') as f:
    pos = pkl.load(f)

with open('data/responses/5-5-2.5-24-mean-0/moseley.pkl', 'rb') as f:
    data = pkl.load(f)

# coordinates = pos.coordinates.to(int)
# grid = np.full((28, 28), np.nan)

# print(pos.neighborhood_indices[0].shape)

# grid[coordinates[:, 0], coordinates[:, 1]] = pos.neighborhood_indices[0].numpy()

activations = data['concrete_noun'] # .mean(axis = 0)
# print(pos.neighborhood_indices[2].numpy().reshape(28, 28))

layer_names = []
for i in range(12):
    layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

print(layer_names[0])

coordinates = pos.coordinates.to(int)

# print(activations[:, 20, :])
print(positions.spatial_loss_fn(torch.Tensor(activations[:, 0, :]), pos))

grid = np.full((28, 28), np.nan)
grid[coordinates[:, 0], coordinates[:, 1]] = activations.mean(axis = 0)[0]
sns.heatmap(grid, cbar=True, cmap='RdBu_r', center=0)
plt.show()


