import os
import sys
import pickle as pkl

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import scipy
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import positions

DUMP_PATH = 'data/extract.pkl'
SAVE_PATH = '../figures/lmask.png'

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

if __name__ == "__main__":

    with open(DUMP_PATH, 'rb') as f:
        data = pkl.load(f)

    layer_names = []
    for i in range(12):
        layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

    n_embed = data['sentences'][layer_names[0]].shape[-1]

    language_mask = np.zeros((len(layer_names), n_embed))
    p_values_matrix = np.zeros((len(layer_names), n_embed))
    t_values_matrix = np.zeros((len(layer_names), n_embed))

    for layer_idx, layer_name in enumerate(layer_names):
        
        # (num_samples, n_embed)
        sentences_actv = data['sentences'][layer_name]
        non_words_actv = data['non-words'][layer_name]

        t_values_matrix[layer_idx], p_values_matrix[layer_idx] = scipy.stats.ttest_ind(sentences_actv, non_words_actv, axis=0, equal_var=False)

    adjusted_p_values = scipy.stats.false_discovery_control(p_values_matrix.flatten())
    adjusted_p_values = adjusted_p_values.reshape((len(layer_names), n_embed))
    
    language_mask = (adjusted_p_values < 0.01) & (t_values_matrix > 0) 
    language_prob_mask = 1 - (adjusted_p_values * (t_values_matrix > 0))

    # language_mask = is_topk(t_values_matrix, k=num_units)
    
    num_active_units = language_mask.sum()
    total_num_units = np.prod(language_mask.shape)

    desc = f"# of Active Units: {num_active_units:,}/{total_num_units:,} = {(num_active_units/total_num_units)*100:.2f}%"

    print(desc)

    fig, axes = plt.subplots(6, 4, figsize=(15, 15))

    for i, ax in enumerate(axes.flatten()):

        with open(f'../models/gpt2-positions/{layer_names[i]}.pkl', 'rb') as f:
            pos = pkl.load(f)

        coordinates = pos.coordinates.to(int)

        grid = np.full((28, 28), np.nan)
        grid[coordinates[:, 0], coordinates[:, 1]] = language_mask[i].astype(int)
        
        sns.heatmap(grid, ax=ax, cbar=False, cmap = sns.color_palette("light:black", as_cmap=True))
        # sns.heatmap(activations[i].reshape(28, 28), ax = ax, cbar = False, cmap = 'RdBu', center = 0)
        ax.set_title(f'{layer_names[i]}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=np.min(language_mask), vmax=np.max(language_mask))

    sm = plt.cm.ScalarMappable(cmap = sns.color_palette("light:black", as_cmap=True), norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    plt.savefig(SAVE_PATH)

    with open(f'data/lmask.pkl', 'wb') as f:
        pkl.dump(language_mask, f)