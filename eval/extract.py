import os
from glob import glob
import pickle as pkl

import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

MODEL_FILE = '../out/ckpt.pt'
SAVE_PATH = 'data/extract.pkl'
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

    n_embed = 784
    num_samples = 240

    final_layer_representations = {
        "sentences": {layer_name: np.zeros((num_samples, n_embed)) for layer_name in layer_names},
        "non-words": {layer_name: np.zeros((num_samples, n_embed)) for layer_name in layer_names}
    }

    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = Fed10_LocLangDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    activations = defaultdict(list)
    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        sent, input_type = batch_data
        tokens = tokenizer.encode_ordinary(sent)
        _, _, _, _, spatial_outputs = model(**tokens)

        for layer in layer_names:
            # spatial_outputs[layer][0] has shape (1, n_embed) so we squeeze
            activations[layer].append(spatial_outputs[layer][0].squeeze(0).detach().cpu())

    for layer in layer_names:

        # (num_samples, n_embed)
        final_layer_representations['non-words'][layer] = torch.stack(activations[layer][:num_samples])
        final_layer_representations['sentences'][layer] = torch.stack(activations[layer][num_samples:])

    with open(os.path.expanduser(SAVE_PATH), 'wb') as f:
        pkl.dump(final_layer_representations, f)