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

DUMP_PATH = 'data/topotest-extract.pkl'
SAVE_PATH = 'figures/topotest-vis-all.png'

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

if __name__ == "__main__":

    with open(DUMP_PATH, 'rb') as f:
        data = pkl.load(f)

    layer_names = [f'encoder.layers.{i}.attn.dense' for i in range(16)]
    activations = np.array([data['sentences'][layer_name].mean(axis = 0) for layer_name in layer_names])

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    
    for i, ax in enumerate(axes.flatten()):
        sns.heatmap(activations[i].reshape(28, 28), ax = ax, cbar=i == 15, cbar_ax=None if i != 15 else cbar_ax, center = 0)
        ax.set_title(f'layer {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(SAVE_PATH)