import torch
import logging
import time
import matplotlib.pyplot as plt
import cramming
# import evaluate
from tqdm import tqdm
import pickle as pkl
from glob import glob
import os
import pandas as pd
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

MODEL_FILE = '~/projects/topo-eval/outputs/topobert/checkpoints/ScriptableMaskedLM_2023-09-23_1.7895/model.pth'
SAVEPATH = '~/projects/topo-eval/data/topobert/extract.pkl'
FEDORENKO_DIR = '~/projects/topo-eval/fedorenko10_stimuli'

attentions = defaultdict(lambda: {'sents': [], 'non_words': []})

class Fed10_LocLangDataset(Dataset):
    def __init__(self, is_pretrained):
        dirpath = os.path.expanduser(FEDORENKO_DIR)
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

def hook_fn(layer_name, input_type, module, inp, out):
    attentions[layer_name][input_type].append(out.mean(dim = 1).detach().cpu())

def _register_hook(model, layer_name, input_type):
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer.register_forward_hook(lambda module, inp, out: hook_fn(layer_name, input_type, module, inp, out))



@torch.no_grad()
def main_process(cfg, setup):
    layer_names = [f'encoder.layers.{i}.attn.self_attention.output_projection' for i in range(16)]

    hidden_dim = 784
    num_samples = 240

    final_layer_representations = {
        "sentences": {layer_name: np.zeros((num_samples, hidden_dim)) for layer_name in layer_names},
        "non-words": {layer_name: np.zeros((num_samples, hidden_dim)) for layer_name in layer_names}
    }

    cfg.impl['microbatch_size'] = 4

    tokenizer = AutoTokenizer.from_pretrained("JonasGeiping/crammed-bert")

    dataset = Fed10_LocLangDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    model = cramming.construct_model(cfg.arch, tokenizer.vocab_size)
    
    model_engine, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.train, cfg.impl, setup=setup)
    model_path = os.path.expanduser(MODEL_FILE)
    model_engine.load_checkpoint(cfg.arch, model_path)

    model_engine.eval()

    for i in range(16):
        print(f'Evaluating layer {layer_names[i]}...')
        nonwords_hook = _register_hook(model, layer_names[i], 'non_words')

        for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            sent, input_type = batch_data
            tokens = tokenizer(sent, truncation=True, max_length=12, return_attention_mask = False, return_tensors='pt')

            if input_type[0] == 'S':
                prev_idx = batch_idx
                break

            # assert sent_tokens.input_ids.size(1) == non_words_tokens.input_ids.size(1)

            _, _ = model_engine.forward_inference(**tokens)

        nonwords_hook.remove()
        sents_hook = _register_hook(model, layer_names[i], 'sents')

        for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            if batch_idx < prev_idx:
                continue

            sent, input_type = batch_data
            tokens = tokenizer(sent, truncation=True, max_length=12, return_attention_mask = False, return_tensors='pt')

            # assert sent_tokens.input_ids.size(1) == non_words_tokens.input_ids.size(1)

            _, _ = model_engine.forward_inference(**tokens)

        sents_hook.remove()

        sents_attns = []
        nonwords_attns = []

        for sample_idx in range(num_samples):
            sents_attns.append(attentions[layer_names[i]]['sents'][sample_idx].mean(dim=0).cpu())
            nonwords_attns.append(attentions[layer_names[i]]['non_words'][sample_idx].mean(dim=0).cpu())

        sents_attns = torch.stack(sents_attns)#.squeeze(1).reshape(-1, 784)
        nonwords_attns = torch.stack(nonwords_attns)#.squeeze(1).reshape(-1, 784)

        final_layer_representations['sentences'][layer_names[i]] = sents_attns.numpy()
        final_layer_representations['non-words'][layer_names[i]] = nonwords_attns.numpy()

    with open(os.path.expanduser(SAVEPATH), 'wb') as f:
        pkl.dump(final_layer_representations, f)

if __name__ == "__main__":
    # layer_names = [f'encoder.layers.{i}.attn.self_attention.output_projection' for i in range(16)]

    config.arch["attention"]["type"] = "pytorch" 
    
    tokenizer = AutoTokenizer.from_pretrained("JonasGeiping/crammed-bert")
    model  = AutoModelForMaskedLM.from_pretrained("JonasGeiping/crammed-bert-legacy")
    model.eval()

    print(model.named_modules())

    # text = "Replace me by any text you'd like."
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)


