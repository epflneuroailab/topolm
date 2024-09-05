"""
Sample from a trained model
"""
import os
import sys
import pickle as pkl
from contextlib import nullcontext
import torch
import tiktoken

import numpy as np 
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
import positions
from model import GPTConfig, GPT

num_units = 784

# -----------------------------------------------------------------------------
model_file = '../models/out/ckpt-5-5-0-48-mean-0.pt'
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
num_samples = 1 # number of samples to draw
max_new_tokens = 50 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 42
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# -----------------------------------------------------------------------------

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

with open(f'data/localizer/5-5-0-48-mean-0/lmask.pkl', 'rb') as f:
    language_mask = pkl.load(f)

language_mask = torch.tensor(1 - language_mask, dtype=torch.float32)

for name, param in model.named_parameters():
    if name in layer_names:
        if param.shape[-1] == 784:
            param.data *= language_mask[i]
        elif param.shape[0] == 784:
            param.data *= language_mask[i].view(-1, 1)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

prompt = 'the'

start_ids = encode(prompt)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print('TOP-K MASK:')
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')

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

random_mask = torch.zeros(24 * 784, dtype=torch.float32)
random_indices = torch.randperm(24 * 784)[:num_units]
random_mask[random_indices] = 1
random_mask = 1 - random_mask.view(24, 784)

for name, param in model.named_parameters():
    if name in layer_names:
        if param.shape[-1] == 784:
            param.data *= random_mask[i]
        elif param.shape[0] == 784:
            param.data *= random_mask[i].view(-1, 1)

prompt = 'the'

start_ids = encode(prompt)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print('RANDOM MASK:')
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))