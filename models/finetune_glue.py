# python3 train.py train_gpt2.yaml device=cpu eval_interval=5 eval_iters=2 log_interval=1 block_size=64 batch_size=12 n_head=1 max_iters=10 lr_decay_iters=1

"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
# torchrun --standalone --nproc_per_node=4 train.py train_gpt2.yaml eval_interval=5 eval_iters=2 log_interval=1 block_size=64 batch_size=12 n_head=1 max_iters=10 lr_decay_iters=1

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

# python3 finetune_glue.py finetune_glue.yaml device=cpu eval_interval=5 eval_iters=2 log_interval=1 batch_size=12 max_iters=10 lr_decay_iters=1

import os
import sys
import time
import math
import pickle
import logging
import tiktoken
import itertools

from contextlib import nullcontext
from omegaconf import OmegaConf
from datasets import load_dataset

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader

from model import GPTConfig, GPT

# shuffle = True, pin_memory=True

# evil config magic
cfg_file = 'config/finetune_glue.yaml'
for i, arg in enumerate(sys.argv):
    if arg[:3] == 'cfg':
        cfg_file = arg.split('=')[1]
        sys.argv.pop(i)

cfg = OmegaConf.load(cfg_file)
cfg.update(OmegaConf.from_cli())

for key in cfg:
    try:
        exec(key + '=' + str(cfg[key]))
    except (NameError, SyntaxError) as e:
        exec(key + '="' + cfg[key] + '"')
    
    # if key not in important_cfg_keys:
    #     del cfg[key]

cfg = OmegaConf.to_container(cfg)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# huggingface dataset
dataset = load_dataset("nyu-mll/glue", task)
num_labels = len(dataset['train'].features['label'].names)

if task == 'mnli':
    dataloader = {
        'train' : DataLoader(dataset['train'], batch_size=batch_size, shuffle = True),
        'val'   : DataLoader(dataset['validation_matched'], batch_size=batch_size, shuffle = True)
    }
else:
    dataloader = {
        'train' : DataLoader(dataset['train'], batch_size=batch_size, shuffle = True),
        'val'   : DataLoader(dataset['validation'], batch_size=batch_size, shuffle = True)
    }

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init from checkpoint
ckpt_path = os.path.join('out', ckpt)
checkpoint = torch.load(ckpt_path, map_location=device)

# we can change dropout, nothing else
model_args = checkpoint['model_args']
model_args['dropout'] = dropout
model_args['alpha'] = 0.1 * model_args['alpha']
model_args['finetune'] = True

if master_process:
    os.makedirs(os.path.join('finetuned', task, f'a{model_args["alpha"]}-l{weight_decay}-lr{learning_rate}-d{dropout}-b{batch_size}'), exist_ok=True)

    # logging
    logging.basicConfig(filename=os.path.join('finetuned', task, f'a{model_args["alpha"]}-l{weight_decay}-lr{learning_rate}-d{dropout}-b{batch_size}', 'logs.txt'),
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        filemode='w')

    logger = logging.getLogger(__name__)

def quick_log(s):
    if master_process:
        logging.info(s)

# create the model
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
quick_log(gptconf)

state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.lm_head = torch.nn.Linear(in_features = model.lm_head.in_features,
                                out_features = num_labels,
                                bias = model.lm_head.bias)
torch.nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
model.to(device)

tokenizer = tiktoken.get_encoding('gpt2')
pad_token = tokenizer.encode('<|endoftext|>', allowed_special="all")[0]

def tokenize_batch(sents):
    
    tokens = tokenizer.encode_batch(sents, allowed_special='all')
    
    padded = list(zip(*itertools.zip_longest(*tokens, fillvalue=pad_token)))
    padded = np.array(padded)
    
    attn_mask = (padded != pad_token).astype(bool)
        
    return torch.from_numpy(padded), torch.from_numpy(attn_mask)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
checkpoint = None # free up memory

# compile the model
if compile:
    quick_log("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:

        losses = torch.zeros(n_batches)
        task_losses = torch.zeros(n_batches)
        accuracies = torch.zeros(n_batches)
        spatial_losses = torch.zeros(n_batches)

        for i, batch in enumerate(dataloader[split]):

            if task == 'mnli':
                sents = ['<|endoftext|>'.join(x) for x in zip(batch['premise'], batch['hypothesis'])]
            else:
                sents = batch['sentence']
            
            X, attn_mask = tokenize_batch(sents)
            X = X.to(device)
            attn_mask = attn_mask.to(device)
            
            Y = batch['label'].unsqueeze(1).to(device) # torch.cat((batch['label'].unsqueeze(1), X), dim=1)[:, 1:]

            with ctx:
                logits, loss, task_loss, spatial_loss, _ = model(X, Y, attn_mask)

            losses[i] = loss.item()
            task_losses[i] = task_loss.item()
            spatial_losses[i] = spatial_loss.item()
            predictions = (torch.nn.functional.softmax(logits[:, -1, :], dim=-1).argmax(dim=-1) == Y.squeeze())
            accuracies[i] = sum(predictions) / len(predictions)

            if i >= n_batches - 1:
                break

        out[split] = [losses.mean(), task_losses.mean(), spatial_losses.mean(), accuracies.mean()]

    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# pretty print iteration
def iterlog(iter_num, lossf, dt, running_mfu):
    return ' | '.join([f'Iter {iter_num}', f'Loss: {lossf:.4f}', f'Time: {dt*1000:.2f}ms', f'MFU: {running_mfu*100:.2f}%'])

# logging
if wandb_log and master_process:
    import wandb
    wandb_run_name += f'-a{model_args["alpha"]}-l{weight_decay}-lr{learning_rate}-d{dropout}-b{batch_size}'
    wandb.init(project=wandb_project, name=wandb_run_name, config=cfg)

# training loop
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process

raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

early_stop_counter = 0
prev_val_loss = 1e9
for iter_num in range(max_iters):

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()

        logging.info('-' * 50)
        logging.info(f'EVALUATING: ITERATION {iter_num}')
        logging.info('Train | ' + ' | '.join([f'{comp} Loss: {loss:.4f}' for comp, loss in zip(['Total', 'Task', 'Spatial', 'Acc'], losses['train'])]))
        logging.info('Valid | ' + ' | '.join([f'{comp} Loss: {loss:.4f}' for comp, loss in zip(['Total', 'Task', 'Spatial', 'Acc'], losses['val'])]))

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss-total": losses['train'][0],
                "train/loss-task": losses['train'][1],
                "train/loss-spatial": losses['train'][2],
                "train/accuracy": losses['train'][3],
                "val/loss-total": losses['val'][0],
                "val/loss-task": losses['val'][1],
                "val/loss-spatial": losses['train'][2],
                "val/accuracy": losses['val'][3],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'][0] < best_val_loss:
            best_val_loss = losses['val'][0]
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': cfg,
                }

                # if iter_num != eval_interval:
                #     os.rename(os.path.join(out_dir, 'ckpt.pt'), os.path.join(out_dir, f'ckpt-{(iter_num // eval_interval) - 1}.pt'))

                if iter_num % save_interval == 0:
                    logging.info(f"... saving checkpoint to finetuned/{task}/ckpt.pt")
                    os.makedirs(os.path.join('finetuned', task, f'a{model_args["alpha"]}-l{weight_decay}-lr{learning_rate}-d{dropout}-b{batch_size}'), exist_ok=True)
                    torch.save(checkpoint, os.path.join('finetuned', task, f'a{model_args["alpha"]}-l{weight_decay}-lr{learning_rate}-d{dropout}-b{batch_size}', 'ckpt.pt'))

            logging.info('-' * 50)

        if losses['val'][0] > prev_val_loss:
            early_stop_counter += 1
        else:
            early_stop_counter = 0

        prev_val_loss = losses['val'][0]
        
        if early_stop_counter > 3:
            break

    for micro_step in range(gradient_accumulation_steps):

        loss = 0
        for i, batch in enumerate(dataloader['train']):

            if task == 'mnli':
                sents = [' '.join(x) for x in zip(batch['premise'], batch['hypothesis'])]
            else:
                sents = batch['sentence']

            X, attn_mask = tokenize_batch(sents)
            X = X.to(device)
            attn_mask = attn_mask.to(device)

            Y = batch['label'].to(device)

            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

            with ctx:
                _, batch_loss, _, _, _ = model(X, Y, attn_mask)
                loss += batch_loss.to('cpu') / gradient_accumulation_steps # scale the loss to account for gradient accumulation

            if i >= n_batches - 1:
                break

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        quick_log(iterlog(iter_num, lossf, dt, running_mfu))
    
    local_iter_num += 1

if ddp:
    destroy_process_group()
