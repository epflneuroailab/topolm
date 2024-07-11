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

DUMP_PATH = 'data/topobert/responses-moseley.pkl'
SAVE_PATH = 'figures/topobert/moseley-'

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

if __name__ == "__main__":

    with open(DUMP_PATH, 'rb') as f:
        data = pkl.load(f)

    layer_names = [f'encoder.layers.{i}.attn.dense' for i in range(16)]
    all_conditions = ['abstract_noun', 'abstract_verb', 'concrete_noun', 'concrete_verb']
    
    for condition in all_conditions:
        activations = np.stack(data[condition], axis = 0).mean(axis = 0)

        fig, axes = plt.subplots(4, 4, figsize=(15, 15))

        for i, ax in enumerate(axes.flatten()):
            sns.heatmap(activations[i].reshape(28, 28), ax = ax, cbar = False, center = 0)
            ax.set_title(f'layer {i}')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(SAVE_PATH + condition + '.png')

    # activations = np.array([data['sentences'][layer_name].mean(axis = 0) for layer_name in layer_names])

    # fig, axes = plt.subplots(4, 4, figsize=(15, 15))

    # for i, ax in enumerate(axes.flatten()):
    #     sns.heatmap(activations[i].reshape(28, 28), ax = ax, cbar = False, cmap = 'viridis')
    #     ax.set_title(f'layer {i}')
    #     ax.axis('off')

    # plt.tight_layout()
    # plt.savefig(SAVE_PATH)