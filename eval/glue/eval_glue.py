import os
import sys

import numpy as np
import pandas as pd
import pickle as pkl

import tiktoken
import itertools

from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'models'))
from model import GPT, GPTConfig

device = 'cuda'
models = [('nontopo', 1), ('topo', 0), ('topo', 0.1), ('topo', 1)]
scores = defaultdict(list)

tokenizer = tiktoken.get_encoding('gpt2')
pad_token = tokenizer.encode('<|endoftext|>', allowed_special="all")[0]
def tokenize_batch(sents):

    tokens = tokenizer.encode_batch(sents, allowed_special='all')
    
    padded = list(zip(*itertools.zip_longest(*tokens, fillvalue=pad_token)))
    padded = np.array(padded)

    attn_mask = (padded != pad_token).astype(bool)
        
    return torch.from_numpy(padded), torch.from_numpy(attn_mask)

def matthews_corrcoef(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    TP = torch.sum((y_true == 1) & (y_pred == 1)).float()
    TN = torch.sum((y_true == 0) & (y_pred == 0)).float()
    FP = torch.sum((y_true == 0) & (y_pred == 1)).float()
    FN = torch.sum((y_true == 1) & (y_pred == 0)).float()

    numerator = (TP * TN) - (FP * FN)
    denominator = torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    return (numerator / denominator).item() if denominator != 0 else 0.0

def get_f1(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    tp = torch.sum((y_true == 1) & (y_pred == 1)).item()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).item()
    fn = torch.sum((y_true == 1) & (y_pred == 0)).item()

    accuracy = torch.mean((y_true == y_pred).float()).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def glue_metric(labels, predictions, task):
    if task == 'cola':
        return matthews_corrcoef(labels, predictions)
    if task == 'stsb':
        stacked = torch.stack((torch.tensor(predictions), torch.tensor(labels)))
        return torch.corrcoef(stacked)[0, 1].item()
    if task == 'qqp' or task == 'mrpc':
        return get_f1(labels, predictions)
    else:
        return sum(predictions == labels) / len(predictions == labels)

# need to add alpha scale 1
for model_name, alpha_scale in models:
    print(f'MODEL: {model_name}-{alpha_scale}')

    scores['model'].append(f'{model_name}-{alpha_scale}')
    
    # need to add 'wnli', 'mnli_matched', 'mnli_mismatched'
    for task in ['mrpc', 'stsb', 'rte', 'sst2', 'cola', 'qqp', 'qnli', 'wnli', 'mnli_matched', 'mnli_mismatched']:
        
        if model_name == 'nontopo' and task == 'wnli':
            scores[task].append(0)
            continue

        print(f'TASK: {task}')

        if task == 'cola' or task == 'sst2':
            s_names = ['sentence']
        elif task == 'mnli_matched' or task == 'mnli_mismatched':
            s_names = ['premise', 'hypothesis']
        elif task == 'qqp':
            s_names = ['question1', 'question2']
        elif task == 'qnli':
            s_names = ['question', 'sentence']
        else:
            s_names = ['sentence1', 'sentence2']

        batch_size = 64
        if 'mnli' in task:
            dataset = load_dataset("nyu-mll/glue", 'mnli')
        else:
            dataset = load_dataset("nyu-mll/glue", task)

        if task == 'stsb':
            num_labels = 1
            dataloader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle = False)
        elif task == 'mnli_matched':
            num_labels = len(dataset['validation_matched'].features['label'].names)
            dataloader = DataLoader(dataset['validation_matched'], batch_size=batch_size, shuffle = False)
        elif task == 'mnli_mismatched':
            num_labels = len(dataset['validation_mismatched'].features['label'].names)
            dataloader = DataLoader(dataset['validation_mismatched'], batch_size=batch_size, shuffle = False)
        else:
            num_labels = len(dataset['validation'].features['label'].names)
            dataloader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle = False)

        ckpt = f'{model_name}-scale-{alpha_scale}.pt'
        if 'mnli' in task:
            ckpt_path = os.path.join('../../models/finetuned', 'mnli', ckpt)
        else:
            ckpt_path = os.path.join('../../models/finetuned', task, ckpt)
        checkpoint = torch.load(ckpt_path, map_location=device)

        model_args = checkpoint['model_args']
        model_args['position_dir'] = '../../models/gpt2-positions-5-5'

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.lm_head = torch.nn.Linear(in_features = model.lm_head.in_features,
                                        out_features = num_labels,
                                        bias = model.lm_head.bias)

        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.to(device)

        model.eval()

        results = defaultdict(list)

        with torch.no_grad():

            for i, batch in enumerate(dataloader):

                if len(s_names) == 2:
                    sents = [' '.join(x) for x in zip(batch[s_names[0]], batch[s_names[1]])]
                else:
                    sents = batch[s_names[0]]

                X, attn_mask = tokenize_batch(sents)
                X = X.to(device)
                attn_mask = attn_mask.to(device)

                logits, _, _, _, _ = model(X, attn_mask=attn_mask)

                if task == 'stsb':
                    predictions = logits[:, -1, :].squeeze(-1)
                else:
                    predictions = torch.nn.functional.softmax(logits[:, -1, :], dim=-1).argmax(dim=-1)

                results['prediction'].extend(predictions.cpu().tolist())
                results['label'].extend(batch['label'].tolist())
                results[s_names[0]].extend(batch[s_names[0]])

                if len(s_names) == 2:
                    results[s_names[1]].extend(batch[s_names[1]])

        df = pd.DataFrame(results)
        score = glue_metric(df['label'], df['prediction'], task=task)
        print(f'model: {model_name}-{alpha_scale} | task: {task} | score: {score}')
        
        scores[task].append(score)

        os.makedirs(task, exist_ok=True)
        df.to_csv(f'{task}/{model_name}-scale-{alpha_scale}.csv')

df = pd.DataFrame(scores)
df.to_csv('scores.csv', index=False)

