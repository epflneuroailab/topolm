import numpy as np
from itertools import product

from positions import LayerPositions, NetworkPositions, swap_optimize

layer_names = []
for i in range(4):
    layer_names.append(f'layer.{i}.attn')
    layer_names.append(f'layer.{i}.mlp')

activations = {}
for name in layer_names:
    
    pos = LayerPositions(
        name = name,
        coordinates = np.array(list(product(np.arange(28), repeat = 2))))
    pos.save('test')

    # TODO: real activations from nonce inputs
    activations[name] = np.random.uniform(size = (784, 32))

network_positions = NetworkPositions.load_from_dir('test')

for name in layer_names:
    layer_positions = network_positions.layer_positions[name]
    layer_positions.coordinates = swap_optimize(activations[name], layer_positions.coordinates, steps = 100, local_steps = 10)
    layer_positions.save('test')