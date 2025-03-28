"""

this file contains definitions for a spatial gpt model
 - mostly based on nanoGPT (github.com/karpathy/nanoGPT)
 - partially built on top of spacetorch (github.com/neuroailab/TDANN)

"""

import math
import inspect
from dataclasses import dataclass

import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
from torch.nn import functional as F

from positions import LayerPositions, NetworkPositions, spatial_loss_fn

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embed % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        
        # output projection
        if not config.head_loss:
            self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        else:
            self.c_proj = ReshapeAttention(config.n_head)

        # regularization
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x, attn_mask = None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # flash attention
        if attn_mask is not None:

            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn_mask = attn_mask.expand(-1, self.n_head, -1, -1)  # (B, nh, T, T)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)

        else:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class ReshapeAttention(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        head_size = C // self.n_head
        
        n_heads_dim = int(np.sqrt(self.n_head))
        head_size_dim = int(np.sqrt(head_size))

        x = x.view(B, T, self.n_head, head_size_dim, head_size_dim) \
                   .view(B, T, n_heads_dim, n_heads_dim, head_size_dim, head_size_dim) \
                   .permute(0, 1, 2, 4, 3, 5) \
                   .reshape(B, T, -1)

        return x

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn_proj = config.attn_proj
        self.n_embed = config.n_embed
        self.bias = config.bias
        self.with_resid = config.with_resid

        self.ln_1 = LayerNorm(self.n_embed, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(self.n_embed, bias=config.bias)
        self.mlp = MLP(config)

        if config.attn_proj:
            self.attn_proj = nn.Linear(self.n_embed, self.n_embed, bias=self.bias)
        else:
            self.attn_proj = nn.Identity(self.n_embed, self.n_embed)

    def forward(self, x, attn_mask=None):

        if self.with_resid:

            attn_out = self.attn(self.ln_1(x), attn_mask)
            attn_out = x + self.attn_proj(attn_out)
            mlp_out = attn_out + self.mlp(self.ln_2(attn_out))

            return mlp_out, attn_out, mlp_out 

        else:

            attn_out = self.attn_proj(self.attn(self.ln_1(x), attn_mask))
            x = x + attn_out
            mlp_out = self.mlp(self.ln_2(x))
            x = x + mlp_out

            return x, attn_out, mlp_out

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 784
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    position_dir: str = 'gpt2-positions'
    alpha: float = 0.25
    accum: str = 'mean'
    activation_decay: float = 0.0
    head_loss: bool = False
    attn_proj: bool = False
    finetune: bool = False
    is_regression: bool = False
    with_resid: bool = True

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # pre-optimized unit positions
        positions = NetworkPositions.load_from_dir(config.position_dir)
        self.positions = positions.layer_positions

        self.alphas = [config.alpha for _ in range(2 * config.n_layer)]

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embed, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, attn_mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embed)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embed)

        spatial_outputs = {}
        x = self.transformer.drop(tok_emb + pos_emb)

        for i, block in enumerate(self.transformer.h):
            x, attn_out, mlp_out = block(x, attn_mask)
            out_shape = attn_out.shape

            spatial_outputs[f'layer.{i}.attn'] = (attn_out.view(out_shape[0] * out_shape[1], out_shape[2]), self.positions[f'layer.{i}.attn'].to(device))
            spatial_outputs[f'layer.{i}.mlp'] = (mlp_out.view(out_shape[0] * out_shape[1], out_shape[2]), self.positions[f'layer.{i}.mlp'].to(device))

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)

            if self.config.finetune:
                    if self.config.is_regression:
                        task_loss = F.mse_loss(logits[:, -1, :].squeeze(-1), targets.to(torch.float32).view(-1))
                    else:
                        task_loss = F.cross_entropy(logits[:, -1, :], targets.view(-1), ignore_index=-1)
            else:
                task_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            spatial_loss = 0
            reg_loss = 0

            if self.config.alpha != 0:

                for i, name in enumerate(spatial_outputs):
                    spatial_loss += self.alphas[i] * spatial_loss_fn(*spatial_outputs[name], self.config.accum)
                    reg_loss += (spatial_outputs[name][0] ** 2).sum(dim = 1).mean()

            else:
                spatial_loss = torch.Tensor([0]).to(device)

            loss = task_loss + spatial_loss + self.config.activation_decay * reg_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            task_loss = None
            spatial_loss = None
            loss = None

        return logits, loss, task_loss, spatial_loss, spatial_outputs

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embed//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
