import os
import sys

from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F

import tiktoken
import itertools

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

sys.path.insert(1, '../../models/')
from model import GPT, GPTConfig

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model_name = 'preresid'
pos_dir = '../../models/gpt2-positions-5-5'

all_dfs = []

for filename in os.listdir('.'):
    
    if not filename.endswith('.jsonl'):
        continue

    cur_df = pd.read_json(filename, lines=True)
    all_dfs.append(cur_df)

df = pd.concat(all_dfs).reset_index(drop=True)

checkpoint = torch.load(f'../../models/out/{model_name}.pt', map_location=device)
model_args = checkpoint['model_args']
model_args['position_dir'] = pos_dir

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict, strict=False)
model.eval()

tokenizer = tiktoken.get_encoding('gpt2')
pad_token = tokenizer.encode('<|endoftext|>', allowed_special="all")[0]

@torch.no_grad()
def surprisal(model, tokens):
    """
    compute surprisal of a batch of tokens
    """
    B, L = tokens.shape
    context = tokens[:, 0].unsqueeze(1)
    surp = torch.zeros(B)
    
    for i in range(L - 1):
        
        logits, _, _, _, _ = model(context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        
        next_token = tokens[:, i + 1]
        surp += -torch.log(probs[range(B), next_token])
        
        # append sampled index to the running sequence and continue
        context = torch.cat((context, tokens[:, i + 1].unsqueeze(1)), dim=1)

    return surp

def tokenize_batch(sents):
    tokens = tokenizer.encode_batch(sents, allowed_special = 'all')
    padded = list(zip(*itertools.zip_longest(*tokens, fillvalue=pad_token)))
    return torch.from_numpy(np.array(padded))

good_surps = []
bad_surps = []

batch_size = 48
for i in tqdm(range(0, len(df), batch_size)):
    batch_df = df.iloc[i:i + batch_size]
    
    good_surps += list(surprisal(model, tokenize_batch(batch_df['sentence_good'])))
    bad_surps += list(surprisal(model, tokenize_batch(batch_df['sentence_bad'])))

df['good_surps'] = good_surps
df['bad_surps'] = bad_surps
df['correct'] = df['good_surps'] < df['bad_surps']

df.to_csv('blimp_results.csv')