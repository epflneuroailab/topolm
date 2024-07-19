import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm 
import pickle as pkl
import scipy

""" visualize topoformer activations """

DUMP_PATH = 'data/topobert/responses-elli-key.pkl'
MASK_PATH = 'data/topobert/lmask.pkl'
SAVE_PATH = 'figures/topobert/elli-keys-t-values-adjust_all_layers_unmasked.png'

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

if __name__ == "__main__":

    with open(DUMP_PATH, 'rb') as f:
        data = pkl.load(f)

    with open(MASK_PATH, 'rb') as f:
        mask = pkl.load(f)

    layer_names = [f'encoder.layers.{i}.attn.self_attention.head.key' for i in range(16)]
    all_conditions = ['birds_noun', 'hand_verb', 'light_verb', 'mammals_noun', 'manmade_noun', 'mouth_verb', 'places_noun', 'sound_verb']
    num_units = 784

    p_values_matrix = np.zeros((len(layer_names), num_units))
    t_values_matrix = np.zeros((len(layer_names), num_units))
    selectivity = np.zeros((len(layer_names), num_units))

    all_nouns = np.concatenate((data['birds_noun'], data['mammals_noun'], data['manmade_noun'], data['places_noun']), axis=0)
    all_verbs = np.concatenate((data['hand_verb'], data['light_verb'], data['mouth_verb'], data['sound_verb']), axis=0)

    noun_activations = np.stack(all_nouns, axis = 0) # * mask # .mean(axis = 0)
    verb_activations = np.stack(all_verbs, axis = 0) # * mask # .mean(axis = 0)

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))

    norm = mcolors.Normalize(-3, 3)

    for i, ax in enumerate(axes.flatten()):

        for j in range(noun_activations.shape[2]):
            t_values_matrix[i, j], p_values_matrix[i, j] = scipy.stats.ttest_ind(noun_activations[:, i, j], verb_activations[:, i, j], axis = 0)
            if np.isnan(p_values_matrix[i, j]):
                p_values_matrix[i, j] = 1
                t_values_matrix[i, j] = 0
        # t_values_matrix[i] = noun_activations[i] - verb_activations[i]
    
    adjusted_p_values = scipy.stats.false_discovery_control(p_values_matrix.flatten()).reshape((len(layer_names), num_units))

    for i, ax in enumerate(axes.flatten()):

        selectivity[i] = (adjusted_p_values[i] < 0.05) * t_values_matrix[i]

        sns.heatmap(selectivity[i].reshape(28, 28), vmin = -3, vmax = 3, cbar = False, cmap = 'viridis', ax = ax, center = 0)

        ax.set_title(f'layer {i}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position of the colorbar
    norm = plt.Normalize(vmin=np.min(selectivity), vmax=np.max(selectivity))
    sm = plt.cm.ScalarMappable(cmap = 'viridis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.savefig(SAVE_PATH)


    # for c, activations in enumerate([noun_activations, verb_activations]):
    #     activations = np.stack(activations, axis = 0).mean(axis = 0)

    #     fig, axes = plt.subplots(4, 4, figsize=(15, 15))

    #     for i, ax in enumerate(axes.flatten()):
    #         sns.heatmap(activations[i].reshape(28, 28), vmin = -5, vmax = 5, ax = ax, cbar = False, center = 0)
    #         ax.set_title(f'layer {i}')
    #         ax.axis('off')

    #     plt.tight_layout()
    #     if c == 0:
    #         condition = 'noun'
    #     else:
    #         condition = 'verb'
    #     plt.savefig(SAVE_PATH + condition + '.png')

    # activations = np.array([data['sentences'][layer_name].mean(axis = 0) for layer_name in layer_names])

    # fig, axes = plt.subplots(4, 4, figsize=(15, 15))

    # for i, ax in enumerate(axes.flatten()):
    #     sns.heatmap(activations[i].reshape(28, 28), ax = ax, cbar = False, cmap = 'viridis')
    #     ax.set_title(f'layer {i}')
    #     ax.axis('off')

    # plt.tight_layout()
    # plt.savefig(SAVE_PATH)