import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle as pkl
import scipy
from collections import defaultdict

DUMP_PATH = 'data/topobert/responses-fedorenko-attn.pkl'
MASK_FILE = 'data/topobert/lmask.pkl'
SAVE_PATH = 'data/topobert/fedorenko-by-layer-unmasked.csv'

if __name__ == "__main__":

    with open(os.path.expanduser(DUMP_PATH), 'rb') as f:
        responses = pkl.load(f)

    with open(os.path.expanduser(MASK_FILE), 'rb') as f:
        layer_mask = pkl.load(f)
    
    layer_names = [f'encoder.layers.{i}.attn.self_attention.output_projection' for i in range(16)]
    data = {condition : defaultdict(list) for condition in ['J', 'N', 'S', 'W']}

    for condition in responses:
        for activations in responses[condition]:
            # masked = activations * layer_mask
            # data[condition].append(activations[activations != 0].mean())
            for i in range(16):
                data[condition][i].append(activations[i].mean()) #[masked[i] != 0].mean())

    flattened = [(outer_key, inner_key, value) 
                  for outer_key, inner_dict in data.items() 
                  for inner_key, values in inner_dict.items() 
                  for value in values]

    df = pd.DataFrame(flattened, columns=['condition', 'layer', 'activation'])
    df.to_csv(SAVE_PATH)

    # df = pd.melt(pd.DataFrame(data), var_name = 'condition', value_name = 'activation')
    # df.to_csv(SAVE_PATH)