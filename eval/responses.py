import os
from glob import glob
import pickle as pkl

import pandas as pd
import numpy as np

from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

MODEL_FILE = '../out/ckpt.pt'
SAVE_PATH = 'data/responses/'

FEDORENKO = 'stimuli/fedorenko_stimuli.csv'
MOSELEY= 'stimuli/moseley_stimuli.csv'
ELLI = 'stimuli/elli_stimuli.csv'

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

        vocab = set(' '.join(data['word']).split())

        self.is_pretrained = is_pretrained

        self.vocab = sorted(list(vocab))
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        items = list(zip(data['word'], data['condition']))
        self.items = sorted(items, key = lambda x: x[1])
        
        self.all_conditions = sorted(set([i[1] for i in self.items]))
        self.num_samples = len(self.items) // len(self.all_conditions)

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
    cfg = OmegaConf.from_cli()

    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model_args = checkpoint['model_args']

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
    for i in range(16):
        layer_names += [f'layer.{i}.attn', f'layer.{i}.mlp']

    tokenizer = tiktoken.get_encoding('gpt2')

    if cfg.stimulus == 'moseley':
        dataset = Moseley_Dataset(tokenizer)
    elif cfg.stimulus == 'fedorenko':
        dataset = Fedorenko_Dataset(tokenizer)
    elif cfg.stimulus == 'elli':
        dataset = Elli_Dataset(tokenizer)
    else:
        raise ValueError(f'provided stimulus ({cfg.stimulus}) currently not supported!')

    dataloader = DataLoader(dataset, batch_size=1)

    activations = defaultdict(list)
    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        sent, input_type = batch_data
        tokens = tokenizer.encode_ordinary(sent)
        _, _, _, _, spatial_outputs = model(**tokens)

        for layer in layer_names:
            # spatial_outputs[layer][0] has shape (1, n_embed) so we squeeze to (n_embed,)
            activations[layer].append(spatial_outputs[layer][0].squeeze(0).detach().cpu())

    final_responses = defaultdict(list)
    for i in range(len(activations[layer_names[0]])):

        condition = dataset.all_conditions[i // dataset.num_samples]

        # activations[layer][i] is (n_embed,) so tot_activations is (n_layers, n_embed)
        tot_activations = np.array([activations[layer][i] for layer in activations])
        final_responses[condition].append(tot_activations)

    for condition in final_responses:
        # (num_samples, num_layers, n_embed)
        final_responses[condition] = np.stack(final_responses[condition], axis = 0)

    with open(os.path.expanduser(SAVEPATH + stimuli + '.pkl'), 'wb') as f:
        pkl.dump(final_responses, f)

