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

# MODEL_FILE = '../models/out/ckpt.pt'
MODEL_DIR = '../models/out/'
SAVE_PATH = 'data/localizer/'
STIMULI_DIR = 'stimuli/fedorenko10_stimuli'

class Fed10_LocLangDataset(Dataset):
    def __init__(self, is_pretrained):
        dirpath = os.path.expanduser(STIMULI_DIR)
        paths = glob(f'{dirpath}/*.csv')
        vocab = set()
        self.is_pretrained = is_pretrained

        data = pd.read_csv(paths[0])
        for path in paths[1:]:
            run_data = pd.read_csv(path)
            data = pd.concat([data, run_data])

        data["sent"] = data["stim2"].apply(str.lower)

        vocab.update(data["stim2"].apply(str.lower).tolist())
        for stimuli_idx in range(3, 14):
            data["sent"] += " " + data[f"stim{stimuli_idx}"].apply(str.lower)
            vocab.update(data[f"stim{stimuli_idx}"].apply(str.lower).tolist())

        self.vocab = sorted(list(vocab))
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        items = list(zip(data['sent'], data['stim14']))
        self.items = sorted(items, key = lambda x: x[1])

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

    n_embed = 784
    num_samples = 240

    final_layer_representations = {
        "sentences": {layer_name: np.zeros((num_samples, n_embed)) for layer_name in layer_names},
        "non-words": {layer_name: np.zeros((num_samples, n_embed)) for layer_name in layer_names}
    }

    tokenizer = tiktoken.get_encoding('gpt2')
    pad_token = tokenizer.encode('<|endoftext|>', allowed_special="all")[0]

    dataset = Fed10_LocLangDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=12)

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
                activations[layer].append(reshaped[i])

    for layer in layer_names:

        # (num_samples, n_embed)
        final_layer_representations['non-words'][layer] = torch.stack(activations[layer][:num_samples])
        final_layer_representations['sentences'][layer] = torch.stack(activations[layer][num_samples:])

    savedir = SAVE_PATH + params
    os.makedirs(savedir, exist_ok = True)

    with open(os.path.expanduser(savedir + '/extract.pkl'), 'wb') as f:
        pkl.dump(final_layer_representations, f)