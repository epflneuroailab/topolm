import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle as pkl
import scipy

""" visualize topoformer activations """

DUMP_PATH = 'data/topobert/responses-moseley-attn.pkl'
MASK_PATH = 'data/topobert/lmask.pkl'
SAVE_PATH = 'figures/topobert/moseley-attn-t-values-adjust_all_layers_masked.png'

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

if __name__ == "__main__":

    with open(DUMP_PATH, 'rb') as f:
        data = pkl.load(f)

    with open(MASK_PATH, 'rb') as f:
        mask = pkl.load(f)

    layer_names = [f'encoder.layers.{i}.attn.dense' for i in range(16)]
    all_conditions = ['abstract_noun', 'abstract_verb', 'concrete_noun', 'concrete_verb']
    num_units = 784

    p_values_matrix = np.zeros((len(layer_names), num_units))
    t_values_matrix = np.zeros((len(layer_names), num_units))
    selectivity = np.zeros((len(layer_names), num_units))

    noun_activations = np.stack(data['concrete_noun'], axis = 0) * mask # .mean(axis = 0)
    verb_activations = np.stack(data['concrete_verb'], axis = 0) * mask # .mean(axis = 0)

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))

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

        sns.heatmap(selectivity[i].reshape(28, 28), cbar = False, cmap = 'viridis', ax = ax, center = 0)

        ax.set_title(f'layer {i}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position of the colorbar
    norm = plt.Normalize(vmin=np.min(selectivity), vmax=np.max(selectivity))
    sm = plt.cm.ScalarMappable(cmap = 'viridis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.savefig(SAVE_PATH)


    # for condition in all_conditions:
    #     activations = np.stack(data[condition], axis = 0).mean(axis = 0)

    #     fig, axes = plt.subplots(4, 4, figsize=(15, 15))

    #     for i, ax in enumerate(axes.flatten()):
    #         sns.heatmap(activations[i].reshape(28, 28), ax = ax, cbar = False, center = 0)
    #         ax.set_title(f'layer {i}')
    #         ax.axis('off')

    #     plt.tight_layout()
    #     plt.savefig(SAVE_PATH + condition + '.png')

    # activations = np.array([data['sentences'][layer_name].mean(axis = 0) for layer_name in layer_names])

    # fig, axes = plt.subplots(4, 4, figsize=(15, 15))

    # for i, ax in enumerate(axes.flatten()):
    #     sns.heatmap(activations[i].reshape(28, 28), ax = ax, cbar = False, cmap = 'viridis')
    #     ax.set_title(f'layer {i}')
    #     ax.axis('off')

    # plt.tight_layout()
    # plt.savefig(SAVE_PATH)