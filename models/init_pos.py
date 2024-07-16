import torch
import numpy as np
from itertools import product

from model import GPTConfig, GPT
from positions import LayerPositions, NetworkPositions, swap_optimize, spatial_loss

import shutil
terminal_width = shutil.get_terminal_size().columns

### HYPERPARAMS ###
batch_size = 12
n_layer = 12
n_head = 1
n_embed = 784
block_size = 4
bias = False
vocab_size = 50304
dropout = 0.0

### BATCHIFY DATA ###
def get_batch():

    data = np.memmap('init/train.bin', dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    return x, y

X, Y = get_batch()

### INTIALIZE MODEL ###

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

layer_names = []
for i in range(n_layer):
    layer_names.append(f'layer.{i}.attn')
    layer_names.append(f'layer.{i}.mlp')

### INIT POSITIONS ###

model.eval()
activations = {}

for name in layer_names:
    
    pos = LayerPositions(
        name = name,
        coordinates = np.array(list(product(np.arange(28), repeat = 2))))
    pos.save('test')

with torch.no_grad():
    logits, loss, spatial_outputs = model(X, Y)

for name in layer_names:
    activations[name] = spatial_outputs[name][0]

network_positions = NetworkPositions.load_from_dir('test')

def print_layer(name):
    layer_line = f' LAYER {name} '
    padding = (terminal_width - len(layer_line)) // 2
    print(f'{"=" * padding}{layer_line}{"=" * padding}')

for name in layer_names:
    print_layer(name)
    layer_positions = network_positions.layer_positions[name]
    layer_positions.coordinates = swap_optimize(activations[name], layer_positions.coordinates, steps = 100, local_steps = 25, radius = 10)
    layer_positions.save('test')
