import os
import sys

import pickle as pkl
from omegaconf import OmegaConf

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from pathlib import Path

import scipy
import numpy as np
import torch
from itertools import product
import pandas as pd

from hrf import NeuronSmoothing

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import positions

cold_hot = np.load('../figures/cold_hot_sampled_1000.npy')
cold_hot_cmap = colors.ListedColormap(cold_hot)

DATA_PATH = 'data/responses/'
SAVE_PATH = 'data/contrasts/'
FIG_PATH = '../figures/visualizations/'

def clip_by_sd(arr, alpha = 2):
    mean = 0
    std = np.std(arr)
    return np.clip(arr, mean - alpha * std, mean + alpha * std)

def log_transform(x):
    return (2 / (1 + np.exp(-0.5 * x))) - 1
    # return np.log(np.abs(x) + 1) * np.sign(x)

if __name__ == "__main__":

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    significance = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    cfg_file = 'vis_gpt2.yaml'
    for i, arg in enumerate(sys.argv):
        if arg[:3] == 'cfg':
            cfg_file = arg.split('=')[1]
            sys.argv.pop(i)

    cfg = OmegaConf.load(cfg_file)
    cfg.update(OmegaConf.from_cli())

    position_dir = '../models/gpt2-positions/gpt2-positions-' + str(cfg.radius) + '-' + str(cfg.neighborhoods)
    # position_dir = '../models/gpt2-positions/topoformer'

    fwhm_mm = 2.0
    resolution_mm = 1
    smoothing = NeuronSmoothing(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    print('Loading data...')

    # params = [cfg.radius, cfg.neighborhoods, cfg.alpha, cfg.batch_size, cfg.accum, cfg.decay]
    # params = '-'.join([str(p) for p in params])
    name = cfg.name

    with open(DATA_PATH + name + '/' + cfg.stimulus + '.pkl', 'rb') as f:
        data = pkl.load(f)

    # p_values = pd.read_csv(f'moran_{name}.csv')
    # p_values = p_values.T

    os.makedirs(FIG_PATH + name + '/' + cfg.stimulus, exist_ok = True)

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
        all_conditions = sorted(['bird_noun', 'hand_verb', 'light_verb', 'mammal_noun', 'manmade_noun', 'mouth_verb', 'natural_noun', 'sound_verb'])
        # all_conditions = sorted(['birds_noun', 'hand_verb', 'light_verb', 'mammals_noun', 'manmade_noun', 'mouth_verb', 'places_noun', 'sound_verb'])

        contrasts = {
            'noun_verb' : [
                np.concatenate([data[key] for key in all_conditions if key[-4:] == 'verb'], axis=0),
                np.concatenate([data[key] for key in all_conditions if key[-4:] == 'noun'], axis=0)
            ]
        }
    else:
        raise ValueError(f'provided stimulus ({cfg.stimulus}) currently not supported!')

    ### PLOTS FOR ALL CONDITIONS ###
    print('Plotting all conditions...')
    for condition in all_conditions:
        # (n_layers, n_embed)
        activations = data[condition].mean(axis = 0)

        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        plt.suptitle(f'{cfg.stimulus} | {condition} | {name}',
            ha='center',
            fontsize=24)

        for i, ax in enumerate(axes.flatten()):

            with open(f'{position_dir}/{layer_names[i]}.pkl', 'rb') as f:
                pos = pkl.load(f)
            
            coordinates = pos.coordinates.to(int)

            coordinates = torch.Tensor(list(product(np.arange(28), repeat = 2))).to(int)

            gridx, gridy, smoothed_activations = smoothing(coordinates.numpy(), activations[i])
            # ax.scatter(gridy, -gridx, c=smoothed_activations, cmap='RdBu_r', s=32, marker='s')

            grid = np.full((28, 28), np.nan)
            grid[gridx.astype(int).squeeze(), gridy.astype(int).squeeze()] = smoothed_activations
            sns.heatmap(grid, ax=ax, cbar=False, cmap='viridis', center=0)

            ax.set_title(f'{layer_names[i]}')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 0.9, 0.98])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = colors.TwoSlopeNorm(vmin=np.min(activations), vcenter = 0, vmax=np.max(activations))

        sm = plt.cm.ScalarMappable(cmap = 'viridis', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

        # plt.savefig(FIG_PATH + name + '/' + cfg.stimulus + '-unsmoothed/' + condition + '.png')
        # plt.savefig(FIG_PATH + name + '/' + cfg.stimulus + '-unsmoothed/' + condition + '.svg')
        plt.savefig(FIG_PATH + name + '/' + cfg.stimulus + '/' + condition + '.png')
        plt.savefig(FIG_PATH + name + '/' + cfg.stimulus + '/' + condition + '.svg')

    ### CONTRAST PLOTS ###
    print('Plotting all contrasts...')
    for condition in contrasts:

        num_samples = contrasts[condition][0][:, 0, :].shape[0]

        raw_matrix      = [np.zeros((len(layer_names), num_samples, num_units)) for _ in range(2)]
        p_values_matrix = np.zeros((len(layer_names), num_units))
        t_values_matrix = np.zeros((len(layer_names), num_units))

        grids = dict()

        for layer_idx, layer_name in enumerate(layer_names):

            with open(f'{position_dir}/{layer_name}.pkl', 'rb') as f:
                pos = pkl.load(f)

            coordinates = pos.coordinates.to(int)
        
            # (num_samples, n_embed)
            activations = [contrasts[condition][0][:, layer_idx, :], contrasts[condition][1][:, layer_idx, :]]

            for i, act in enumerate(activations):
                gridx, gridy, activations[i] = smoothing(coordinates.numpy(), act)
            
            grids[layer_idx] = (gridx, gridy)

            t_values_matrix[layer_idx], p_values_matrix[layer_idx] = scipy.stats.ttest_ind(activations[0], activations[1], axis=0, equal_var=False)
            raw_matrix[0][layer_idx] = activations[0]
            raw_matrix[1][layer_idx] = activations[1]

        adjusted_p_values = scipy.stats.false_discovery_control(p_values_matrix.flatten())
        adjusted_p_values = adjusted_p_values.reshape((len(layer_names), activations[0].shape[1]))
        selectivity = t_values_matrix.copy()
        selectivity[adjusted_p_values > 0.05] = np.nan
        print(f'{100 * (adjusted_p_values < 0.05).sum() / t_values_matrix.flatten().shape[0]}% of units significant')

        # fig, axes = plt.subplots(6, 4, figsize=(15, 20))
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        plt.suptitle(f'{cfg.stimulus} | {condition}',
            ha='center',
            fontsize=24)

        for i, ax in enumerate(axes.flatten()):

            # grid = np.full((28, 28), np.nan)
            # grid[coordinates[:, 0], coordinates[:, 1]] = selectivity[i]

            gridx, gridy = grids[i]

            grid = np.full((28, 28), np.nan)
            grid[gridx.astype(int).squeeze(), gridy.astype(int).squeeze()] = selectivity[i]
            sns.heatmap(grid, ax=ax, cbar=False, cmap='RdBu_r', center=0)

            # sns.heatmap(grid, ax=ax, cbar=False, cmap='RdBu_r', center=0)
            ax.set_title(f'{layer_names[i]}')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 0.9, 0.98])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        bound = max(abs(np.nanmin(smoothed_activations)), abs(np.nanmax(smoothed_activations)))
        norm = colors.TwoSlopeNorm(vmin=-bound, vcenter = 0, vmax=bound)

        sm = plt.cm.ScalarMappable(cmap = 'RdBu_r', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

        # plt.savefig(FIG_PATH + name + '/' + cfg.stimulus + '-unsmoothed/' + condition + '-contrast.png')
        # plt.savefig(FIG_PATH + name + '/' + cfg.stimulus + '-unsmoothed/' + condition + '-contrast.svg')
        
        plt.savefig(FIG_PATH + name + '/' + cfg.stimulus + '/' + condition + '_contrast.png')
        plt.savefig(FIG_PATH + name + '/' + cfg.stimulus + '/' + condition + '_contrast.svg')

        os.makedirs(os.path.join(SAVE_PATH, name, f'{cfg.stimulus}'), exist_ok = True)
        with open(os.path.join(SAVE_PATH, name, f'{cfg.stimulus}/', f'{condition}.pkl'), 'wb') as f:
            pkl.dump({
                'raw'         : raw_matrix,
                'selectivity' : selectivity,
                't_values'    : t_values_matrix,
                'p_values'    : adjusted_p_values
                }, f)