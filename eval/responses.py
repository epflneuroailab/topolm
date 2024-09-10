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
from model import GPT, GPTConfig

MODEL_DIR = '../models/out/'
SAVE_PATH = 'data/responses/'

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

    cfg_file = 'vis_gpt2.yaml'
    for i, arg in enumerate(sys.argv):
        if arg[:3] == 'cfg':
            cfg_file = arg.split('=')[1]
            sys.argv.pop(i)

    cfg = OmegaConf.load(cfg_file)
    cfg.update(OmegaConf.from_cli())

    params = [cfg.radius, cfg.neighborhoods, cfg.alpha, cfg.batch_size, cfg.accum, cfg.decay]
    params = '-'.join([str(p) for p in params])

    checkpoint = torch.load(MODEL_DIR + 'ckpt-' + params + '.pt', map_location=device)
    model_args = checkpoint['model_args']
    model_args['position_dir'] = '../models/gpt2-positions-' + str(cfg.radius) + '-' + str(cfg.neighborhoods)

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

    layer_names = []
    for i in range(12):
        layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

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

    # print(dataset.items) len = 8 * 72 = 576
    # print(dataset.all_conditions) len = 8
    # print(dataset.num_samples) 72
    # print(dataset.batch_size) 18

    dataloader = DataLoader(dataset, batch_size=dataset.batch_size)

    activations = defaultdict(list)
    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        sents, input_type = batch_data

        tokens = tokenizer.encode_batch(sents, allowed_special = 'all')
        padded = list(zip(*itertools.zip_longest(*tokens, fillvalue=pad_token)))

        X = np.array(padded)
        Y = np.zeros_like(X)
        Y[:, :-1] = X[:, 1:]
        Y[:, -1] = pad_token

        _, _, _, _, spatial_outputs = model(torch.from_numpy(X), torch.from_numpy(Y))
        batch_size, batch_len = X.shape

        for layer in layer_names:

            reshaped = spatial_outputs[layer][0].view(batch_size, batch_len, -1).mean(axis=1).detach().cpu()

            for i in range(batch_size):
                activations[layer].append((reshaped[i], input_type[i]))

    final_responses = defaultdict(list)
    for i in range(len(activations[layer_names[0]])):

        condition = activations[layer_names[0]][i][1] # dataset.all_conditions[i // dataset.num_samples]

        # activations[layer][i] is (n_embed,) so tot_activations is (n_layers, n_embed)
        tot_activations = np.array([activations[layer][i][0] for layer in activations])
        final_responses[condition].append(tot_activations)

    for condition in final_responses:
        # (num_samples, num_layers, n_embed)
        final_responses[condition] = np.stack(final_responses[condition], axis = 0)

    savedir = SAVE_PATH + params
    os.makedirs(savedir, exist_ok = True)
    
    with open(os.path.expanduser(savedir + '/' + cfg.stimulus + '.pkl'), 'wb') as f:
        pkl.dump(final_responses, f)

