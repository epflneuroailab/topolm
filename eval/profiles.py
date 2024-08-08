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

DUMP_PATH = 'data/responses/fedorenko.pkl'
MASK_FILE = 'data/lmask.pkl'
SAVE_PATH = 'data/responses/fedorenko-profiles.csv'

if __name__ == "__main__":

    with open(os.path.expanduser(DUMP_PATH), 'rb') as f:
        responses = pkl.load(f)

    with open(os.path.expanduser(MASK_FILE), 'rb') as f:
        layer_mask = pkl.load(f)
    
    layer_names = []
    for i in range(12):
        layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

    data = {condition : defaultdict(list) for condition in ['J', 'N', 'S', 'W']}

    for condition in responses:
        for activations in responses[condition]:
            masked = activations * layer_mask # (n_layers, n_embed)
            for i in range(24):
                data[condition][layer_names[i]].append(activations[i][masked[i] != 0].mean())

    flattened = [(outer_key, inner_key, value) 
                  for outer_key, inner_dict in data.items() 
                  for inner_key, values in inner_dict.items() 
                  for value in values]

    df = pd.DataFrame(flattened, columns=['condition', 'layer', 'activation'])
    df.to_csv(SAVE_PATH, index = False)

    # df = pd.melt(pd.DataFrame(data), var_name = 'condition', value_name = 'activation')
    # df.to_csv(SAVE_PATH)