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

from omegaconf import OmegaConf

cfg = OmegaConf.from_cli()

DUMP_PATH = f'data/responses/{cfg.name}/fedorenko.pkl'
MASK_FILE = f'data/localizer/{cfg.name}/lmask.pkl'
SAVE_PATH = f'data/responses/{cfg.name}/fedorenko-profiles.csv'

with open(os.path.expanduser(DUMP_PATH), 'rb') as f:
    responses = pkl.load(f)

with open(os.path.expanduser(MASK_FILE), 'rb') as f:
    layer_mask = pkl.load(f)

layer_names = []
for i in range(12):
    layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

data = {condition : defaultdict(list) for condition in ['J', 'N', 'S', 'W']}

for condition in responses:
    for activations in responses[condition]:
        masked = activations * layer_mask # (n_layers, n_embed)
        for i in range(24):
            data[condition][layer_names[i]].append(np.abs(activations[i][masked[i] != 0]).mean())

flattened = [(outer_key, inner_key, value) 
              for outer_key, inner_dict in data.items() 
              for inner_key, values in inner_dict.items() 
              for value in values]

df = pd.DataFrame(flattened, columns=['condition', 'layer', 'activation'])
df.to_csv(SAVE_PATH, index = False)

### PLOT WHOLE NETWORK RESPONSE ###
df['condition'] = pd.Categorical(df['condition'], categories=condition_order, ordered=True)

grouped = df.groupby('condition')['activation'].agg(['mean', 'count', 'std'])
ci95_hi = []
ci95_lo = []

for i in range(len(grouped)):
    m, c, s = grouped.iloc[i]
    ci = stats.t.interval(0.95, c - 1, loc=m, scale=s / np.sqrt(c))
    ci95_hi.append(ci[1] - m)
    ci95_lo.append(m - ci[0])

colors = [condition_colors[cond] for cond in grouped.index]

plt.bar(grouped.index, grouped['mean'], yerr=[ci95_lo, ci95_hi], capsize=0, alpha=0.8, color=colors)
ax = plt.gca()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("Condition")
ax.set_ylabel("Mean Activation")

plt.tight_layout()
plt.savefig(f'../figures/visualizations/{cfg.name}/fedorenko/whole-network.svg')