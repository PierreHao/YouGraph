"""
Reproduce of Linear Multihead Attention class introduced in Linformer paper (https://arxiv.org/abs/2006.04768)

Copy-paste from torch.nn.MultiheadAttention and F.multi_head_attention_forward with modifications:
    * E and F projection from seq_len to k
    * layerwise parameters sharing

"""
import warnings

import torch
from torch import nn

from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.functional import linear, softmax, dropout

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, encode_dim=32, quick_mode=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.encode_dim = encode_dim
        self.quick_mode = quick_mode
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        if self.quick_mode:
            self.in_proj_weight = Parameter(torch.empty(3 * self.encode_dim, embed_dim))
            self.in_proj_bias = Parameter(torch.empty(3 * self.encode_dim))
            self.out_proj = Linear(self.encode_dim, embed_dim, bias=bias)
        else:
            self.in_proj_weight = Parameter(torch.empty(2 * self.encode_dim, embed_dim))
            self.in_proj_bias = Parameter(torch.empty(2 * self.encode_dim))
            self.v_proj_weight = Parameter(torch.empty(self.embed_dim, embed_dim))
            self.v_proj_bias = Parameter(torch.empty(self.embed_dim))
            self.out_proj = Linear(embed_dim, embed_dim, bias=bias)


        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if not self.quick_mode:
            xavier_uniform_(self.v_proj_weight)
            constant_(self.v_proj_bias, 0)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key=None, value=None, need_weights=True):
        num_heads, in_proj_weight, in_proj_bias = self.num_heads, self.in_proj_weight, self.in_proj_bias
        
        if self.quick_mode:
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
            tgt_len, bsz, encode_dim = q.size()
            head_dim = encode_dim // num_heads
            assert head_dim * num_heads == encode_dim, "embed_dim must be divisible by num_heads"
            scaling = float(head_dim) ** -0.5
            q = q * scaling

            q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

            src_len = k.size(1)
            
            attn_output_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

            attn_output_weights = softmax(
                attn_output_weights, dim=-1)
            attn_output_weights = dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.bmm(attn_output_weights, v)
            assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, encode_dim)
            attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        else:
            q, k = linear(query, in_proj_weight, in_proj_bias).chunk(2, dim=-1)
            v = linear(query, self.v_proj_weight, self.v_proj_bias)
            tgt_len, bsz, encode_dim = q.size()
            head_dim = encode_dim // num_heads
            assert head_dim * num_heads == encode_dim, "embed_dim must be divisible by num_heads"
            scaling = float(head_dim) ** -0.5
            q = q * scaling

            q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz * num_heads, self.embed_dim // num_heads).transpose(0, 1)

            src_len = k.size(1)
            
            attn_output_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

            attn_output_weights = softmax(
                attn_output_weights, dim=-1)
            attn_output_weights = dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.bmm(attn_output_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
            attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, -1, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None
