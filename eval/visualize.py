import os
import sys

import pickle as pkl
from omegaconf import OmegaConf

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

import scipy
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import positions

""" visualize topoformer activations and selectivity for a specific stimulus set """

DATA_PATH = 'data/responses/'
SAVE_PATH = '../figures/visualizations/'

def clip_by_sd(arr, alpha = 2):
    mean = 0
    std = np.std(arr)
    return np.clip(arr, mean - alpha * std, mean + alpha * std)

if __name__ == "__main__":
    cfg = OmegaConf.from_cli()

    print('Loading data...')

    with open(DATA_PATH + cfg.stimulus + '.pkl', 'rb') as f:
        data = pkl.load(f)

    num_units = 784
    layer_names = []
    for i in range(12):
        layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

    if cfg.stimulus == 'fedorenko':
        all_conditions = sorted(['J', 'N', 'S', 'W'])

        contrasts = {
            'sentence-jabberwocky' : [
                data['S'],
                data['J']
            ],
            'words-nonwords' : [
                data['W'],
                data['N']
            ],
            'sentence-words' : [
                data['S'],
                data['W']
            ],
            'jabberwocky-nonwords' : [
                data['J'],
                data['N']
            ],
            'jabberwocky-words' : [
                data['J'],
                data['W']
            ],
            'sentence-nonwords' : [
                data['S'],
                data['N']
            ]
        }
    elif cfg.stimulus == 'moseley':
        all_conditions = sorted(['abstract_noun', 'abstract_verb', 'concrete_noun', 'concrete_verb'])

        contrasts = {
            'concrete-concrete' : [
                data['concrete_noun'],
                data['concrete_verb']
            ],
            'abstract-abstract' : [
                data['abstract_noun'],
                data['abstract_verb']
            ],
            'abstract-concrete' : [
                np.concatenate((data['abstract_noun'], data['abstract_verb']), axis=0),
                np.concatenate((data['concrete_noun'], data['concrete_verb']), axis=0)
            ],
            'noun-verb' : [
                np.concatenate((data['abstract_noun'], data['concrete_noun']), axis=0),
                np.concatenate((data['abstract_verb'], data['concrete_verb']), axis=0)
            ]
        }
    elif cfg.stimulus == 'elli':
        all_conditions = sorted(['birds_noun', 'hand_verb', 'light_verb', 'mammals_noun', 'manmade_noun', 'mouth_verb', 'places_noun', 'sound_verb'])

        contrasts = {
            'noun-verb' : [
                np.concatenate([data[key] for key in all_conditions if key[-4:] == 'noun'], axis=0),
                np.concatenate([data[key] for key in all_conditions if key[-4:] == 'verb'], axis=0),
            ]
        }
    else:
        raise ValueError(f'provided stimulus ({cfg.stimulus}) currently not supported!')

    ### PLOTS FOR ALL CONDITIONS ###
    print('Plotting all conditions...')
    for condition in all_conditions:
        # (n_layers, n_embed)
        activations = data[condition].mean(axis = 0)

        fig, axes = plt.subplots(6, 4, figsize=(15, 15))

        for i, ax in enumerate(axes.flatten()):

            with open(f'../models/gpt2-positions/{layer_names[i]}.pkl', 'rb') as f:
                pos = pkl.load(f)

            coordinates = pos.coordinates.to(int)
            # activations[i] = clip_by_sd(activations[i])

            grid = np.full((28, 28), np.nan)
            grid[coordinates[:, 0], coordinates[:, 1]] = activations[i]
            sns.heatmap(grid, ax=ax, cbar=False, cmap='viridis', center=0)

            # sns.heatmap(activations[i].reshape(28, 28), ax = ax, cbar = False, cmap = 'viridis', center = 0)
            ax.set_title(f'{layer_names[i]}')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = colors.TwoSlopeNorm(vmin=np.min(activations), vcenter = 0, vmax=np.max(activations))

        sm = plt.cm.ScalarMappable(cmap = 'viridis', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

        plt.savefig(SAVE_PATH + cfg.stimulus + '/' + condition + '.png')

    ### CONTRAST PLOTS ###
    print('Plotting all contrasts...')
    for condition in contrasts:

        p_values_matrix = np.zeros((len(layer_names), num_units))
        t_values_matrix = np.zeros((len(layer_names), num_units))

        for layer_idx, layer_name in enumerate(layer_names):
        
            # (num_samples, n_embed)
            activations = (contrasts[condition][0][layer_idx], contrasts[condition][1][layer_idx])
            t_values_matrix[layer_idx], p_values_matrix[layer_idx] = scipy.stats.ttest_ind(activations[0], activations[1], axis=0, equal_var=False)

        adjusted_p_values = scipy.stats.false_discovery_control(p_values_matrix.flatten())
        adjusted_p_values = adjusted_p_values.reshape((len(layer_names), activations[0].shape[1]))
        selectivity = t_values_matrix * (adjusted_p_values < 0.05)

        fig, axes = plt.subplots(6, 4, figsize=(15, 15))

        for i, ax in enumerate(axes.flatten()):

            with open(f'../models/gpt2-positions/{layer_names[i]}.pkl', 'rb') as f:
                pos = pkl.load(f)

            coordinates = pos.coordinates.to(int)

            grid = np.full((28, 28), np.nan)
            grid[coordinates[:, 0], coordinates[:, 1]] = selectivity[i]

            sns.heatmap(grid, ax=ax, cbar=False, cmap='viridis', center=0)
            ax.set_title(f'{layer_names[i]}')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = colors.TwoSlopeNorm(vmin=np.min(selectivity), vcenter = 0, vmax=np.max(selectivity))

        sm = plt.cm.ScalarMappable(cmap = 'viridis', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
        
        plt.savefig(SAVE_PATH + cfg.stimulus + '/' + condition + '.png')