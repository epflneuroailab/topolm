"""
example script for running language mask ablations on blimp
"""

import os
import sys

import torch
import torch.nn.functional as F
from contextlib import nullcontext

import tiktoken
import itertools

import numpy as np
import pandas as pd
import pickle as pkl

from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import positions
from model import GPT, GPTConfig

cfg = OmegaConf.from_cli()
model_name = cfg.name

all_dfs = []

for filename in os.listdir('blimp/data/'):
    
    if not filename.endswith('.jsonl'):
        continue

    f = open(f'blimp/data/{filename}', 'r')
    cur_df = pd.read_json(f, lines=True)
    all_dfs.append(cur_df)

blimp = pd.concat(all_dfs).reset_index(drop=True)
blimp = blimp[['sentence_good', 'sentence_bad', 'field', 'linguistics_term', 'UID']]

num_samples    = 1
max_new_tokens = 50
temperature    = 1.0
seed           = 42

model_name = 'preresid'

if model_name == 'preresid':
    model_file = f'../models/out/preresid-run-5-5-2.5-48-mean-0/ckpt-311.pt'
elif model_name == 'nontopo':
    model_file = f'../models/out/run-5-5-0-48-mean-0/ckpt-399.pt'
else:
    raise Exception("Invalid model name.")

device = 'cuda'
dtype  = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

checkpoint = torch.load(model_file, map_location=device)
model_args = checkpoint['model_args']
model_args['position_dir'] = '../models/gpt2-positions-5-5'

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

layer_names = []
for i in range(12):
    layer_names += [f'transformer.h.{i}.attn.c_attn.weight', f'transformer.h.{i}.mlp.c_fc.weight']

original_params = {}

for name, param in model.named_parameters():
    if name in layer_names:
        original_params[name] = param.data

with open(f'data/localizer/{model_name}/lmask.pkl', 'rb') as f:
    selectivity = pkl.load(f)

enc = tiktoken.get_encoding("gpt2")
pad_token = enc.encode('<|endoftext|>', allowed_special="all")[0]

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

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
        logits = logits[:, -1, :].to('cpu')
        probs = F.softmax(logits, dim=-1)
        
        next_token = tokens[:, i + 1].to('cpu')
        surp += -torch.log(probs[range(B), next_token])
        
        # append sampled index to the running sequence and continue
        context = torch.cat((context, tokens[:, i + 1].unsqueeze(1)), dim=1)

    return surp

def tokenize_batch(sents):
    tokens = enc.encode_batch(sents, allowed_special = 'all')
    padded = list(zip(*itertools.zip_longest(*tokens, fillvalue=pad_token)))
    return torch.from_numpy(np.array(padded)).to(device)

def blimp_score(model):
    
    good_surps = []
    bad_surps = []
    
    batch_size = 48

    for i in range(0, len(blimp), batch_size):
        batch_df = blimp.iloc[i:i + batch_size]
        
        good_surps += list(surprisal(model, tokenize_batch(batch_df['sentence_good'])))
        bad_surps += list(surprisal(model, tokenize_batch(batch_df['sentence_bad'])))
    
    return good_surps, bad_surps

for k in [1, 28, 196, 392, 784, 1568, 3136, 12544]:

    language_mask = torch.tensor(1 - is_topk(selectivity, k), dtype=torch.float32).to(device)

    i = 0
    for name, param in model.named_parameters():
        if name in layer_names:
            param.data = original_params[name] * language_mask[i]
            i += 1

    good_surps, bad_surps = blimp_score(model)
    blimp[f'correct_{k}'] = np.array(good_surps) < np.array(bad_surps)

    blimp.to_csv(f'ablation/{model_name}_blimp_results.csv')