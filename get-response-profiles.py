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

DUMP_PATH = 'data/topobert/responses-fedorenko-'
SAVE_PATH = 'data/topobert/fedorenko-by-layer-by-level.csv'

if __name__ == "__main__":

    # with open(os.path.expanduser(MASK_FILE), 'rb') as f:
    #     layer_mask = pkl.load(f)

    all_responses = {}

    for level in ['query', 'key', 'value', 'attn']:
        
        with open(DUMP_PATH + level + '.pkl', 'rb') as f:
            responses = pkl.load(f)

        if level == 'attn':
            layer_names = [f'encoder.layers.{i}.attn.self_attention.output_projection' for i in range(16)]
        else:
            layer_names = [f'encoder.layers.{i}.attn.self_attention.head.' + level for i in range(16)]
        # data = defaultdict(list)
        data = {condition : defaultdict(list) for condition in ['W', 'N', 'J', 'S']}

        for condition in responses:
            for activations in responses[condition]:
                # data[condition].append(activations[activations != 0].mean())
                for i in range(16):
                    data[condition][i].append(activations[i].mean())

        flattened = [(outer_key, inner_key, value) 
                      for outer_key, inner_dict in data.items() 
                      for inner_key, values in inner_dict.items() 
                      for value in values]

        all_responses[level] = flattened

    all_responses_flattened = [(key, *tup) for key, lst in all_responses.items() for tup in lst]

    df = pd.DataFrame(all_responses_flattened, columns=['level', 'condition', 'layer', 'activation'])
    df['condition'] = df['condition'].replace({'N' : 'S', 'S' : 'N'})
    df.to_csv(SAVE_PATH)

    # df = pd.melt(pd.DataFrame(data), var_name = 'condition', value_name = 'activation')
    # df['condition'] = df['condition'].replace({'N' : 'S', 'S' : 'N'})
    # df.to_csv(SAVE_PATH)