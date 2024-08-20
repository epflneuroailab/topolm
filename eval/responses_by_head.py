import os
import sys

from glob import glob
import pickle as pkl

import pandas as pd
import numpy as np

import tiktoken
import itertools

from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from model import GPT, GPTConfig, CausalSelfAttention

MODEL_DIR = '../models/out/'

FEDORENKO = 'stimuli/fedorenko_stimuli.csv'
MOSELEY= 'stimuli/moseley_stimuli.csv'
ELLI = 'stimuli/elli_pairs.csv'

class Fedorenko_Dataset(Dataset):
    def __init__(self, is_pretrained):
        data = pd.read_csv(os.path.expanduser(FEDORENKO))
        vocab = set(' '.join(data['sentence']).split())

        self.is_pretrained = is_pretrained

        self.vocab = sorted(list(vocab))
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        items = list(zip(data['sentence'], data['condition']))
        self.items = sorted(items, key = lambda x: x[1])

        self.all_conditions = sorted(set([i[1] for i in self.items]))
        self.num_samples = len(self.items) // len(self.all_conditions)
        self.batch_size = 32

        # self.sentences = data[data["stim14"]=="S"]["sent"]
        # self.non_words = data[data["stim14"]=="N"]["sent"]

    def tokenize(self, sent):
        return torch.tensor([self.w2idx[w]+20_000 for w in sent.split()])

    def __getitem__(self, idx):
        if self.is_pretrained:
            return self.items[idx][0].strip(), self.items[idx][1]
        else:
            return self.tokenize(self.items[idx][0].strip()), self.items[idx][1]

    def __len__(self):
        return len(self.items)
    
    def vocab_size(self):
        return len(self.vocab) + 20_000

class Moseley_Dataset(Dataset):
    def __init__(self, is_pretrained):
        data = pd.read_csv(os.path.expanduser(MOSELEY))
        data['condition'] = data[['category', 'class']].agg('_'.join, axis=1)

        vocab = set(' '.join(data['word']).split())

        self.is_pretrained = is_pretrained

        self.vocab = sorted(list(vocab))
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        items = list(zip(data['word'], data['condition']))
        self.items = sorted(items, key = lambda x: x[1])

        self.all_conditions = sorted(set([i[1] for i in self.items]))
        self.num_samples = len(self.items) // len(self.all_conditions)
        self.batch_size = 10

    def tokenize(self, sent):
        return torch.tensor([self.w2idx[w]+20_000 for w in sent.split()])

    def __getitem__(self, idx):
        if self.is_pretrained:
            return self.items[idx][0].strip(), self.items[idx][1]
        else:
            return self.tokenize(self.items[idx][0].strip()), self.items[idx][1]

    def __len__(self):
        return len(self.items)
    
    def vocab_size(self):
        return len(self.vocab) + 20_000

class Elli_Dataset(Dataset):
    def __init__(self, is_pretrained):
        data = pd.read_csv(os.path.expanduser(ELLI))
        data['condition'] = data[['category', 'class']].agg('_'.join, axis=1)

        vocab = set(' '.join(data['pair']).split())

        self.is_pretrained = is_pretrained

        self.vocab = sorted(list(vocab))
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        items = list(zip(data['pair'], data['condition']))
        self.items = sorted(items, key = lambda x: x[1])
        
        self.all_conditions = sorted(set([i[1] for i in self.items]))
        self.num_samples = len(self.items) // len(self.all_conditions)
        self.batch_size = 18

    def tokenize(self, sent):
        return torch.tensor([self.w2idx[w]+20_000 for w in sent.split()])

    def __getitem__(self, idx):
        if self.is_pretrained:
            return self.items[idx][0].strip(), self.items[idx][1]
        else:
            return self.tokenize(self.items[idx][0].strip()), self.items[idx][1]

    def __len__(self):
        return len(self.items)
    
    def vocab_size(self):
        return len(self.vocab) + 20_000

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.from_cli()

    params = [cfg.radius, cfg.neighborhoods, cfg.alpha, cfg.batch_size, cfg.accum, cfg.decay]
    params = '-'.join([str(p) for p in params])

    checkpoint = torch.load(MODEL_DIR + 'ckpt-' + params + '.pt', map_location=device)
    model_args = checkpoint['model_args']
    model_args['position_dir'] = '../models/gpt2-positions-' + str(cfg.radius) + '-' + str(cfg.neighborhoods) + '/'

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()

    layer_names = []
    for i in range(12):
        layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

    activations = []

    def attention_hook(module, input, output):
        # output is the tensor before the output projection (self.c_proj(y))
        activations.append(output.detach().cpu().numpy())

    def register_hooks(model):
        for name, module in model.named_modules():
            if isinstance(module, CausalSelfAttention):
                module.register_forward_hook(attention_hook)

    tokenizer = tiktoken.get_encoding('gpt2')
    pad_token = tokenizer.encode('<|endoftext|>', allowed_special="all")[0]

    if cfg.stimulus == 'moseley':
        dataset = Moseley_Dataset(tokenizer)
    elif cfg.stimulus == 'fedorenko':
        dataset = Fedorenko_Dataset(tokenizer)
    elif cfg.stimulus == 'elli':
        dataset = Elli_Dataset(tokenizer)
    else:
        raise ValueError(f'provided stimulus ({cfg.stimulus}) currently not supported!')

    dataloader = DataLoader(dataset, batch_size=dataset.batch_size)

    register_hooks(model)

    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        sents, input_type = batch_data

        tokens = tokenizer.encode_batch(sents, allowed_special = 'all')
        padded = list(zip(*itertools.zip_longest(*tokens, fillvalue=pad_token)))

        X = np.array(padded)
        Y = np.zeros_like(X)
        Y[:, :-1] = X[:, 1:]
        Y[:, -1] = pad_token

        _, _, _, _, spatial_outputs = model(torch.from_numpy(X), torch.from_numpy(Y))

    print(activations[0].shape)

