import torch
import numpy as np
from itertools import product

from model import GPTConfig, GPT
from positions import LayerPositions, NetworkPositions, swap_optimize, spatial_loss_fn, get_neighborhood, get_center

import shutil
terminal_width = shutil.get_terminal_size().columns

### HYPERPARAMS ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rng = torch.Generator()
rng.manual_seed(42)

num_neighborhoods = 100
local_steps = 50
neighborhoods_per_batch = 5

batch_size = 12
n_layer = 12
n_head = 1
n_embed = 784
block_size = 4
bias = False
vocab_size = 50304
dropout = 0.0

radius = 10
p = 'inf'

### BATCHIFY DATA ###
def get_batch():

    data = np.memmap('init/train.bin', dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    return x, y

X, Y = get_batch()

### INTIALIZE MODEL ###

layer_names = []
for i in range(n_layer):
    layer_names.append(f'layer.{i}.attn')
    layer_names.append(f'layer.{i}.mlp')

# init layer positions with random neighborhoods
for name in layer_names:
    
    pos = LayerPositions(
        name = name,
        coordinates = torch.Tensor(list(product(np.arange(28), repeat = 2))),
        neighborhood_indices = torch.zeros(size=(num_neighborhoods, n_embed), dtype=int),
        neighborhoods_per_batch = neighborhoods_per_batch)

    for i in range(num_neighborhoods):
        center = get_center(pos.coordinates, radius)
        pos.neighborhood_indices[i] = get_neighborhood(center, pos.coordinates, radius, p)

    pos.save('test')

print('Initialized all layer positions and neighborhoods...')

model_args = dict(
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embed=n_embed,
                    block_size=block_size,
                    bias=bias,
                    vocab_size=vocab_size,
                    dropout=dropout
                )

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

print('Loaded model...')

### INIT POSITIONS ###

model.eval()
activations = {}

with torch.no_grad():
    logits, loss, task_loss, spatial_loss, spatial_outputs = model(X, Y)

for name in layer_names:
    activations[name] = spatial_outputs[name][0]

network_positions = NetworkPositions.load_from_dir('test')

def print_layer(name):
    layer_line = f' LAYER {name} '
    padding = (terminal_width - len(layer_line)) // 2
    print(f'{"=" * padding}{layer_line}{"=" * padding}')

print('Preoptimizing...')

for name in layer_names:

    print_layer(name)

    layer_positions = network_positions.layer_positions[name].to_device(device)
    old_loss = spatial_loss_fn(activations[name], layer_positions)

    layer_positions, n_swapped = swap_optimize(activations[name], layer_positions, num_neighborhoods, local_steps, radius, p, rng)

    new_loss = spatial_loss_fn(activations[name], layer_positions)

    print(f'global loss decreased by {(old_loss - new_loss):.3f} | swapped {n_swapped}/{local_steps * num_neighborhoods} possible pairs')
    layer_positions.save('test')
