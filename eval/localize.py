import os
import sys
import pickle as pkl

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

import scipy
import numpy as np
from omegaconf import OmegaConf

from hrf import NeuronSmoothing

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import positions

DUMP_PATH = 'data/localizer/'
SAVE_PATH = '../figures/visualizations/'

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

if __name__ == "__main__":

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    cfg_file = 'vis_gpt2.yaml'
    for i, arg in enumerate(sys.argv):
        if arg[:3] == 'cfg':
            cfg_file = arg.split('=')[1]
            sys.argv.pop(i)

    cfg = OmegaConf.load(cfg_file)
    cfg.update(OmegaConf.from_cli())

    # position_dir = '../models/gpt2-positions/gpt2-positions-' + str(cfg.radius) + '-' + str(cfg.neighborhoods)
    position_dir = '../models/gpt2-positions/topoformer'

    fwhm_mm = 2.0
    resolution_mm = 1
    smoothing = NeuronSmoothing(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    # params = [cfg.radius, cfg.neighborhoods, cfg.alpha, cfg.batch_size, cfg.accum, cfg.decay]
    # params = '-'.join([str(p) for p in params])
    name = cfg.name

    with open(DUMP_PATH + name + '/extract.pkl', 'rb') as f:
        data = pkl.load(f)

    layer_names = []
    for i in range(16):
        layer_names += [f'layer.{i}.attn']#, f'layer.{i}.mlp']

    saved_layer_names = [f'encoder.layers.{i}.attn.self_attention.head.key' for i in range(16)]

    n_embed = data['sentences'][saved_layer_names[0]].shape[-1]

    language_mask = np.zeros((len(layer_names), n_embed))
    p_values_matrix = np.zeros((len(layer_names), n_embed))
    t_values_matrix = np.zeros((len(layer_names), n_embed))

    for layer_idx, layer_name in enumerate(layer_names):
        
        # (num_samples, n_embed)
        sentences_actv = data['sentences'][saved_layer_names[layer_idx]]
        non_words_actv = data['non-words'][saved_layer_names[layer_idx]]

        with open(f'{position_dir}/{layer_name}.pkl', 'rb') as f:
            pos = pkl.load(f)

        coordinates = pos.coordinates.to(int)

        smoothed = []
        smoothed.append(smoothing(coordinates.numpy(), sentences_actv)[2])
        smoothed.append(smoothing(coordinates.numpy(), non_words_actv)[2])

        t_values_matrix[layer_idx], p_values_matrix[layer_idx] = scipy.stats.ttest_ind(smoothed[0], smoothed[1], axis=0, equal_var=False)
        # t_values_matrix[layer_idx], p_values_matrix[layer_idx] = scipy.stats.ttest_ind(sentences_actv, non_words_actv, axis=0, equal_var=False)

    adjusted_p_values = scipy.stats.false_discovery_control(p_values_matrix.flatten())
    adjusted_p_values = adjusted_p_values.reshape((len(layer_names), n_embed))

    if cfg.topk == 0:
        language_mask = t_values_matrix.copy() # * (adjusted_p_values < 0.05) # t_values_matrix # (adjusted_p_values < 0.05) & (t_values_matrix > 0)
        language_mask[adjusted_p_values > 0.05] = np.nan
        language_prob_mask = 1 - (adjusted_p_values * (t_values_matrix > 0))
    else:
        language_mask = is_topk(t_values_matrix, k=cfg.topk)
    
    num_active_units = language_mask.sum()
    total_num_units = np.prod(language_mask.shape)

    desc = f"# of Active Units: {num_active_units:,}/{total_num_units:,} = {(num_active_units/total_num_units)*100:.2f}%"

    print(desc)

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    # plt.suptitle(f'{name}',
    #     ha='center',
    #     fontsize=24)

    for i, ax in enumerate(axes.flatten()):

        with open(f'{position_dir}/{layer_names[i]}.pkl', 'rb') as f:
            pos = pkl.load(f)

        coordinates = pos.coordinates.to(int)

        # grid = np.full((28, 28), np.nan)
        # grid[coordinates[:, 0], coordinates[:, 1]] = language_mask[i]
        # ax.contour((language_mask[i] * (language_mask[i] > 0)).reshape(28, 28), levels=[1], colors='black')
        # ax.invert_yaxis()

        grid = (language_mask[i]).reshape(28, 28) # .astype(int)

        # coordinates = pos.coordinates.to(int)
        # gridx, gridy, smoothed_activations = smoothing(coordinates.numpy(), language_mask[i])
        # ax.scatter(gridy, -gridx, c=smoothed_activations, cmap='RdBu_r', s=32, marker='s')

        # grid = np.full((28, 28), np.nan)
        # grid[gridx.astype(int).squeeze(), gridy.astype(int).squeeze()] = smoothed_activations
        # sns.heatmap(grid, ax=ax, cbar=False, cmap='RdBu_r', center=0)
        
        sns.heatmap(grid, ax=ax, cbar=False, cmap = 'inferno')
        # sns.heatmap(activations[i].reshape(28, 28), ax = ax, cbar = False, cmap = 'RdBu', center = 0)
        ax.set_title(f'{layer_names[i]}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = colors.TwoSlopeNorm(vmin=np.nanmin(language_mask), vcenter = 0, vmax=np.nanmax(language_mask))

    sm = plt.cm.ScalarMappable(cmap = 'inferno', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    plt.savefig(SAVE_PATH + name + '/lmask.svg')
    plt.savefig(SAVE_PATH + name + '/lmask.png')
    plt.savefig(SAVE_PATH + name + '/lmask.pdf')

    with open(f'data/localizer/{name}/lmask.pkl', 'wb') as f:
        pkl.dump(language_mask, f)