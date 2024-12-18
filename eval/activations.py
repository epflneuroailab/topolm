"""

example script for getting all activations from a model

- requires an input txt file (cfg.input_text)
- output will be a dict with 24 keys
    * each key corresponds to a list of length len(input_text)
    * each list item is a list of length 784, i.e. the activations for word i

"""

import os
import sys

import torch
import torch.nn.functional as F
from contextlib import nullcontext

import tiktoken
import itertools
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

import numpy as np
import pandas as pd
import pickle as pkl

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from model import GPT, GPTConfig

MODEL_DIR = '../models/out/'
SAVE_PATH = 'data/localizer/'
STIMULI_DIR = 'stimuli/fedorenko10_stimuli'

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

cfg_file = 'activations.yaml'
for i, arg in enumerate(sys.argv):
    if arg[:3] == 'cfg':
        cfg_file = arg.split('=')[1]
        sys.argv.pop(i)

cfg = OmegaConf.load(cfg_file)
cfg.update(OmegaConf.from_cli())

num_samples    = cfg.num_samples
max_new_tokens = cfg.max_new_tokens
temperature    = cfg.temperature
top_k          = cfg.top_k
seed           = cfg.seed

model_name = cfg.model_name
pos_dir    = cfg.pos_dir
input_text = cfg.input_text
output_dir = cfg.output_dir

model_file = f'out/{model_name}.pt'
device     = 'cpu'
dtype      = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

checkpoint = torch.load(model_file, map_location=device)
model_args = checkpoint['model_args']
model_args['position_dir'] = pos_dir

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
    layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

enc = tiktoken.get_encoding("gpt2")
pad_token = enc.encode('<|endoftext|>', allowed_special="all")[0]

with open(input_text, 'r') as f:
    txt = f.read().split(' ')

def tokenize_batch(sents):
    tokens = enc.encode_batch(sents, allowed_special = 'all')
    padded = list(zip(*itertools.zip_longest(*tokens, fillvalue=pad_token)))
    return torch.from_numpy(np.array(padded))

x = tokenize_batch(txt)
batch_size, batch_len = x.shape

activations = {layer : [] for layer in layer_names}

with torch.no_grad():
    _, _, _, _, spatial_outputs = model(x)
    for layer in layer_names:
        
        reshaped = spatial_outputs[layer][0].view(batch_size, batch_len, -1).mean(axis=1).detach().cpu()
        
        for j in range(batch_size):
            activations[layer].append(reshaped[j].to(torch.float16).tolist())

with open(f'{output_dir}/{input_text.split(".")[0]}_activations.pkl', 'wb') as f:
    pkl.dump(activations, f)