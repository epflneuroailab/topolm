import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle as pkl
import scipy

DUMP_PATH = 'dumps/topotest-extract.pkl'
SAVE_PATH = 'figures/topotest-lmask.png'

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

if __name__ == "__main__":

    with open(DUMP_PATH, 'rb') as f:
        data = pkl.load(f)

    layer_names = [f'encoder.layers.{i}.attn.dense' for i in range(16)]

    num_units = data['sentences'][layer_names[0]].shape[-1]

    language_mask = np.zeros((len(layer_names), num_units))

    p_values_matrix = np.zeros((len(layer_names), num_units))
    t_values_matrix = np.zeros((len(layer_names), num_units))

    for layer_idx, layer_name in enumerate(tqdm(layer_names)):
        
        sentences_actv = data["sentences"][layer_name]
        non_words_actv = data["non-words"][layer_name]

        t_values_matrix[layer_idx], p_values_matrix[layer_idx] = scipy.stats.ttest_ind(sentences_actv, non_words_actv, axis=0, equal_var=False)

    language_mask = is_topk(t_values_matrix, k=num_units)
    
    num_active_units = language_mask.sum()
    total_num_units = np.prod(language_mask.shape)

    desc = f"# of Active Units: {num_active_units:,}/{total_num_units:,} = {(num_active_units/total_num_units)*100:.2f}%"

    print(desc)

    sns.heatmap(language_mask)

    plt.title(desc)
    plt.xlabel('units')
    plt.ylabel('layers')

    plt.savefig(SAVE_PATH)

    plt.clf()
    plt.cla()
    plt.close()
    
    with open(f'dumps/topotest-lmask.pkl', 'wb') as f:
        pkl.dump(language_mask, f)