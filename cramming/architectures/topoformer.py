import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constraints import Constraints

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LocalityMask(nn.Module):
    def __init__(self,
                 hidden_size=400,
                 width=0.3,
                 wrap=True,
                 local_querying=True,
                 learned_spatial_querying=True,
                 mask_type="circular",
                 normalize_mask=True,
                 ):
        super().__init__()
        
        self.local_querying = local_querying
        self.learned_spatial_querying = learned_spatial_querying
        self.normalize_queries = normalize_mask
        self.width = width
        
        if not local_querying:
            logging.info(f"Using spatial querying with width: {width}")
            self.constraint = Constraints(hidden_size, width, wrap)
            locality_mask = getattr(self.constraint, mask_type)()
            self.register_buffer('locality_mask', torch.Tensor(locality_mask).to(DEVICE))
            if self.normalize_queries:
                print('Normalizing queries...')
                self.locality_mask = self.locality_mask/torch.norm(self.locality_mask)

            if learned_spatial_querying:
                self.locality_weight = nn.Parameter(torch.randn(hidden_size))
            else:
                self.locality_weight = 1

    def forward(self, queries):
        # If width -1 then return only queries(same as multiplying with identity) If local querying, then no locality operation is performed on queries.
        if self.width == -1 or self.local_querying:
            return queries
        
        # Locality operations.
        locality_mask = self.locality_mask
        if self.learned_spatial_querying:
            locality_mask = locality_mask * self.locality_weight
    
        queries = torch.matmul(queries, locality_mask)
        return queries
    

class LocallyConnected(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        width=0.3,
        wrap=True,
        bias=True,
        feedforward=False,
        mask_type='circular'
    ):
        super().__init__(in_features, out_features, bias=bias)

        if width != -1:
            self.weight.data = torch.abs(self.weight.data) * 10
            logging.info(f"using mask_type: {mask_type} in locally connected...")

            self.locality_constraint = getattr(Constraints(in_features, width, wrap), mask_type)()
            self.locality_constraint = torch.Tensor(self.locality_constraint).to(DEVICE)
        else:
            self.locality_constraint = None

    def forward(self, input):
        if self.locality_constraint is not None:
            self.weight.data = torch.abs(
                self.weight.data * self.locality_constraint.transpose(0, 1)
            )
        return F.linear(input, self.weight, self.bias)

class AttentionHead(nn.Module):
    """
    Topoformer Attention Head used by all Topoformer models
    """
    def __init__(self,
                 hidden_size,
                 attention_head_size,
                 dropout,
                 sq,
                 bias=True,
                 learned_spatial_querying=True,
                 local_querying=True,
                 mask_type='circular',
                 relu=False,
                 transpose=False,
                 ):
        
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.local_querying = local_querying
        self.sq = sq
        self.mask_type = mask_type
        self.transpose = transpose
        self.relu = relu
        
        assert hidden_size == attention_head_size, "q, k, v are not square."
        
        self.query, self.key, self.value = self.create_projection_layers(hidden_size, attention_head_size, self.local_querying, bias, self.mask_type)
        self.dropout = nn.Dropout(dropout)
        self.use_weighted_locality = learned_spatial_querying
        self.locality_mask = LocalityMask(hidden_size=self.hidden_size,
                                          width=self.sq,
                                          wrap=True,
                                          local_querying=self.local_querying,
                                          learned_spatial_querying=self.use_weighted_locality,
                                          mask_type=self.mask_type
                                          )
    
    def create_projection_layers(self, hidden_size, attention_head_size, use_locally_connected, bias, mask_type):
        if use_locally_connected:
            return (
                LocallyConnected(hidden_size, attention_head_size, self.sq, mask_type=mask_type),
                LocallyConnected(hidden_size, attention_head_size, self.sq, mask_type=mask_type),
                LocallyConnected(hidden_size, attention_head_size, self.sq, mask_type=mask_type)
            )
        else:
            return (
                nn.Linear(hidden_size, attention_head_size, bias=bias),
                nn.Linear(hidden_size, attention_head_size, bias=bias),
                nn.Linear(hidden_size, attention_head_size, bias=bias)
            )
    
    def forward(self, x):
        # Transpose the input to (batch, seq, hidden)
        if self.transpose:
            x = x.transpose(0, 1)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        if self.relu:
            query = nn.functional.relu(query)
            key = nn.functional.relu(key)
            value = nn.functional.relu(value)
        
        # Apply locality
        query = self.locality_mask(query) 
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        if self.relu:
            attention_output = nn.functional.relu(attention_output)
        if self.transpose:
            attention_output = attention_output.transpose(0, 1) # (B, S, H) -> (S, B, H)
        return (attention_output, attention_probs)

class AttentionConfig:
    """
    dataclass used for non-BERT Topoformer
    """
    def __init__(self, hidden_size, sq, sr, mask_type='circular', learned_spatial_querying=False, local_querying=False, qkv_bias=False, relu=False, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1):
        self.hidden_size = hidden_size
        self.sq = sq
        self.sr = sr
        self.mask_type = mask_type
        self.learned_spatial_querying = learned_spatial_querying
        self.local_querying = local_querying
        self.qkv_bias = qkv_bias
        self.relu = relu
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob

        assert (not (self.local_querying and self.learned_spatial_querying)), "can't use local querying and spatial querying..."

    @property
    def attention_head_size(self):
        return self.hidden_size


class AttentionOutput(nn.Module):
    """
    wrapper for non-BERT Topoformer
    """
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([])
        self.init_attention_head()
        self.init_output_projection()

    def init_attention_head(self):
        self.head = AttentionHead(
            self.config['hidden_size'],
            self.config['hidden_size'],  # attention_head_size is equal to hidden_size
            self.config["attention_probs_dropout_prob"],
            self.config['sq'],
            self.config["qkv_bias"],
            self.config['learned_spatial_querying'],
            self.config['local_querying'],
            self.config['mask_type'],
            relu=self.config['relu'],
        )


    def init_output_projection(self):
        self.output_projection = LocallyConnected(
            self.config.hidden_size, self.config.hidden_size, self.config.sr, mask_type=self.config.mask_type
        )  # This is our fc_out
        self.output_dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, x, output_attentions=False):
        assert not output_attentions, "Not implemented yet"
        attention_output, _ = self.head(x)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        if self.config.relu:
            attention_output = nn.functional.relu(attention_output)

        return (attention_output, None)

