# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers import DynamicCache
from transformers.pytorch_utils import (find_pruneable_heads_and_indices,
                                        prune_linear_layer)

try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor




# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        if hasattr(config, "qkv_bias"):
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()
        
        # Pruned layers for inference
        self.q_proj_prune = None
        self.k_proj_prune = None
        self.v_proj_prune = None
        self.o_proj_prune = None

        # Head selection variables
        self.head_mask = None
        self.num_active_heads = None
        self.active_head_indices = None
        self.heads_to_prune = None

        self.use_loss_change_criterion = True

        # Track pruned heads
        self.pruned_heads = set()

        # Cache for head importance scores (computed once)
        self._head_importance_cache = None
        self._sorted_head_indices = None

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings,
                                                       base=self.config.rope_theta)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _compute_head_importance_loss_change(self):
        """
        Compute importance score for each head based on training loss change.
        Uses |gradient × weight| as pruning criterion from the slides.
        """
        if not self.use_loss_change_criterion:
            # Fall back to weight magnitude during inference
            return self._compute_head_importance_weight_magnitude()

        if not self.training and self._head_importance_cache is not None:
            return self._sorted_head_indices

        head_importance_scores = []
        
        for head_idx in range(self.num_heads):
            start_idx = head_idx * self.head_dim
            end_idx = start_idx + self.head_dim
            
            total_score = 0.0
            
            # Q projection: gradient × weight for this head
            if self.q_proj.weight.grad is not None:
                q_weights = self.q_proj.weight[start_idx:end_idx, :]
                q_grads = self.q_proj.weight.grad[start_idx:end_idx, :]
                q_score = torch.sum(torch.abs(q_grads * q_weights)).item()
                total_score += q_score
            
            # K and V projections (handle GQA case)
            kv_head_idx = head_idx // self.num_key_value_groups
            kv_start_idx = kv_head_idx * self.head_dim
            kv_end_idx = kv_start_idx + self.head_dim
            
            if self.k_proj.weight.grad is not None:
                k_weights = self.k_proj.weight[kv_start_idx:kv_end_idx, :]
                k_grads = self.k_proj.weight.grad[kv_start_idx:kv_end_idx, :]
                k_score = torch.sum(torch.abs(k_grads * k_weights)).item()
                total_score += k_score
            
            if self.v_proj.weight.grad is not None:
                v_weights = self.v_proj.weight[kv_start_idx:kv_end_idx, :]
                v_grads = self.v_proj.weight.grad[kv_start_idx:kv_end_idx, :]
                v_score = torch.sum(torch.abs(v_grads * v_weights)).item()
                total_score += v_score
            
            # O projection: gradient × weight for this head
            if self.o_proj.weight.grad is not None:
                o_weights = self.o_proj.weight[:, start_idx:end_idx]
                o_grads = self.o_proj.weight.grad[:, start_idx:end_idx]
                o_score = torch.sum(torch.abs(o_grads * o_weights)).item()
                total_score += o_score
            
            head_importance_scores.append((head_idx, total_score))
        
        # Sort by importance (ascending order - lower scores = more prunable)
        head_importance_scores.sort(key=lambda x: x[1])
        
        # Cache the results
        self._sorted_head_indices = [head_idx for head_idx, _ in head_importance_scores]
        self._head_importance_cache = [importance for _, importance in head_importance_scores]
        
        return self._sorted_head_indices

    def _compute_head_importance_weight_magnitude(self):
        """
        Compute importance score for each head based on weight magnitude.
        Returns sorted indices of heads from most important to least important.
        """
        if not self.training and self._head_importance_cache is not None:
            return self._sorted_head_indices
            
        head_importance_scores = []
        
        # For each head, compute the sum of weight magnitudes across Q, K, V projections
        for head_idx in range(self.num_heads):
            start_idx = head_idx * self.head_dim
            end_idx = start_idx + self.head_dim
            
            # Q projection weights for this head
            q_weights = self.q_proj.weight[start_idx:end_idx, :]
            q_magnitude = torch.sum(torch.abs(q_weights)).item()
            
            # K projection weights for this head (handle GQA case)
            kv_head_idx = head_idx // self.num_key_value_groups
            kv_start_idx = kv_head_idx * self.head_dim
            kv_end_idx = kv_start_idx + self.head_dim
            
            k_weights = self.k_proj.weight[kv_start_idx:kv_end_idx, :]
            v_weights = self.v_proj.weight[kv_start_idx:kv_end_idx, :]
            k_magnitude = torch.sum(torch.abs(k_weights)).item()
            v_magnitude = torch.sum(torch.abs(v_weights)).item()
            
            # O projection weights for this head
            o_weights = self.o_proj.weight[:, start_idx:end_idx]
            o_magnitude = torch.sum(torch.abs(o_weights)).item()
            
            # Total importance for this head
            total_importance = q_magnitude + k_magnitude + v_magnitude + o_magnitude
            head_importance_scores.append((head_idx, total_importance))
        
        # Sort by importance (descending order)
        head_importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Cache the results
        self._sorted_head_indices = [head_idx for head_idx, _ in head_importance_scores]
        self._head_importance_cache = [importance for _, importance in head_importance_scores]
        
        return self._sorted_head_indices

    def select_heads(self, hidden_states, head_ratio):
        """
        Select heads based on their importance (weight magnitude).
        Returns mask and indices for the most important heads.
        """
        if not self.training and self.active_head_indices is not None:
            return self.head_mask, self.num_active_heads, self.active_head_indices, self.heads_to_prune

        batch_size, seq_len, _ = hidden_states.shape
        
        # Get sorted head indices by importance
        sorted_head_indices = self._compute_head_importance_loss_change()
        
        # Select top heads based on ratio
        num_active_heads = max(1, int(self.num_heads * head_ratio))
        
        # Take the most important heads
        active_head_indices = sorted(sorted_head_indices[:num_active_heads])
        heads_to_prune = sorted(sorted_head_indices[num_active_heads:])
        
        # Create mask
        head_mask = torch.zeros(self.num_heads, dtype=torch.bool, device=hidden_states.device)
        head_mask[active_head_indices] = True

        self.head_mask = head_mask
        self.num_active_heads = num_active_heads
        self.active_head_indices = active_head_indices
        self.heads_to_prune = heads_to_prune
        
        return head_mask, num_active_heads, active_head_indices, heads_to_prune

    def prune_attention_heads(self, heads_to_keep, heads_to_prune):
        """Hard pruning for inference - only called when not training"""
        if self.training or self.q_proj_prune is not None:
            return

        heads, param_indices = find_pruneable_heads_and_indices(
            heads_to_prune, self.num_heads, self.head_dim, set()
        )

        original_dtype = next(self.q_proj.parameters()).dtype

        self.q_proj_prune = prune_linear_layer(self.q_proj, param_indices)
        self.k_proj_prune = prune_linear_layer(self.k_proj, param_indices)
        self.v_proj_prune = prune_linear_layer(self.v_proj, param_indices)
        self.o_proj_prune = prune_linear_layer(self.o_proj, param_indices, dim=1)

        if original_dtype in [torch.float16, torch.bfloat16]:
            self.q_proj_prune = self.q_proj_prune.to(original_dtype)
            self.k_proj_prune = self.k_proj_prune.to(original_dtype)
            self.v_proj_prune = self.v_proj_prune.to(original_dtype)
            self.o_proj_prune = self.o_proj_prune.to(original_dtype)

        self.num_heads_prune = self.num_heads - len(heads)
        self.hidden_size_prune = self.head_dim * self.num_heads_prune 
        self.pruned_heads = self.pruned_heads.union(heads)

    def _apply_soft_head_mask(self, tensor, head_mask, tensor_type="qkv"):
        """
        Apply soft masking to Q, K, V tensors during training.
        
        Args:
            tensor: Input tensor of shape [batch, seq_len, num_heads * head_dim]
            head_mask: Boolean mask of shape [num_heads]
            tensor_type: "qkv" for Q/K/V tensors, "output" for attention output
        """

        original_dtype = next(self.q_proj.parameters()).dtype

        if tensor_type == "qkv":
            # Reshape to [batch, seq_len, num_heads, head_dim]
            batch_size, seq_len, _ = tensor.shape
            tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Apply mask: zero out heads that should be pruned
            head_mask_expanded = head_mask.view(1, 1, self.num_heads, 1)
            tensor = tensor * head_mask_expanded.to(original_dtype)
            
            # Reshape back to [batch, seq_len, num_heads * head_dim]
            tensor = tensor.view(batch_size, seq_len, -1)
            
        elif tensor_type == "attn_weights":
            # tensor shape: [batch, num_heads, seq_len, seq_len]
            head_mask_expanded = head_mask.view(1, self.num_heads, 1, 1)
            tensor = tensor * head_mask_expanded.to(original_dtype)
            
        elif tensor_type == "attn_output":
            # tensor shape: [batch, seq_len, num_heads, head_dim] 
            head_mask_expanded = head_mask.view(1, 1, self.num_heads, 1)
            tensor = tensor * head_mask_expanded.to(original_dtype)
            
        return tensor

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            head_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        apply_head_masking = head_ratio is not None and head_ratio < 1.0

        if self.config.pretraining_tp > 1:
            # Handle tensor parallel case (same as original)
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            if apply_head_masking:
                head_mask, num_active_heads, active_head_indices, heads_to_prune = self.select_heads(
                    hidden_states, 
                    head_ratio=head_ratio
                )
                
                if self.training:
                    # SOFT MASKING: Use original projections but apply masks
                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)
                    
                    # Apply soft masking to Q, K, V
                    query_states = self._apply_soft_head_mask(query_states, head_mask, "qkv")
                    key_states = self._apply_soft_head_mask(key_states, head_mask, "qkv")
                    value_states = self._apply_soft_head_mask(value_states, head_mask, "qkv")
                    
                    cur_num_heads = self.num_heads  # Keep original number for reshaping
                    cur_hidden_size = self.hidden_size
                else:
                    # HARD PRUNING: Use pruned projections for inference
                    self.prune_attention_heads(active_head_indices, heads_to_prune)
                    
                    query_states = self.q_proj_prune(hidden_states)
                    key_states = self.k_proj_prune(hidden_states)
                    value_states = self.v_proj_prune(hidden_states)
                    
                    cur_num_heads = self.num_heads_prune
                    cur_hidden_size = self.hidden_size_prune
            else:
                # No masking applied
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
                
                cur_num_heads = self.num_heads
                cur_hidden_size = self.hidden_size

        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, cur_num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, cur_num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, cur_num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if n_kv_heads < n_heads (GQA)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, cur_num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, cur_num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # Apply soft masking to attention weights during training
        if apply_head_masking and self.training:
            attn_weights = self._apply_soft_head_mask(attn_weights, head_mask, "attn_weights")

        # Softmax and attention output
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, cur_num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, cur_num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Apply soft masking to attention output during training
        if apply_head_masking and self.training:
            attn_output = self._apply_soft_head_mask(attn_output, head_mask, "attn_output")
            
        attn_output = attn_output.reshape(bsz, q_len, cur_hidden_size)

        # Final output projection
        if apply_head_masking and not self.training:
            # Use pruned projection for inference
            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(cur_hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj_prune.weight.split(cur_hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
            else:
                attn_output = self.o_proj_prune(attn_output)
        else:
            # Use original projection (for training or no masking)
            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(cur_hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(cur_hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
            else:
                attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value




class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def compute_attention_scores(self, self_attn_weights: torch.Tensor, original_prompt_length: int, image_start: Optional[int] = None,  image_end: Optional[int] = None,  text_start: Optional[int] = None,  text_end: Optional[int] = None) -> torch.Tensor:
        scores = self_attn_weights.mean(dim=1)
        scores = scores.mean(dim=1)
        scores = scores[:, image_start:text_end]

        return scores

    def forward(
            self,
            hidden_states: torch.Tensor,
            last_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            output_attention_scores: Optional[bool] = False,
            original_prompt_length: Optional[int] = None,
            image_start: Optional[int] = None, 
            image_end: Optional[int] = None, 
            text_start: Optional[int] = None, 
            text_end: Optional[int] = None,
            use_cache: Optional[bool] = False,
            head_ratio: Optional[float] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if head_ratio is not None and int(head_ratio) != 1:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions or output_attention_scores,
                use_cache=use_cache,
                head_ratio=head_ratio,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions or output_attention_scores,
                use_cache=use_cache,
            )
        hidden_states = residual + hidden_states

        attention_scores = None
        if output_attention_scores and self_attn_weights is not None:
            attention_scores = self.compute_attention_scores(self_attn_weights, original_prompt_length, image_start=image_start, image_end=image_end,  text_start=text_start, text_end=text_end)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)


        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        
        if output_attention_scores:
            outputs += (attention_scores,)

        return outputs


class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        return x + self.dummy - self.dummy  # (also tried x+self.dummy)


def len_list(x, n):
    return [i for i in x if len(i) <= n]


class LlamaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.dropout = config.dropout if hasattr(config, "dropout") else 0.0
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ============ HEAD PRUNING ADDITIONS ============
        # Pruned layers for inference
        self.q_proj_prune = None
        self.k_proj_prune = None
        self.v_proj_prune = None
        self.o_proj_prune = None

        # Head selection variables
        self.head_mask = None
        self.num_active_heads = None
        self.active_head_indices = None
        self.heads_to_prune = None

        self.use_loss_change_criterion = True

        # Track pruned heads
        self.pruned_heads = set()

        # Cache for head importance scores (computed once)
        self._head_importance_cache = None
        self._sorted_head_indices = None
        # ===============================================

    def _compute_head_importance_loss_change(self):
        """
        Compute importance score for each head based on training loss change.
        Uses |gradient × weight| as pruning criterion.
        """
        if not self.training and self._head_importance_cache is not None:
            return self._sorted_head_indices

        head_importance_scores = []
        
        for head_idx in range(self.num_heads):
            start_idx = head_idx * self.head_dim
            end_idx = start_idx + self.head_dim
            
            total_score = 0.0
            
            # Q projection: gradient × weight for this head
            if self.q_proj.weight.grad is not None:
                q_weights = self.q_proj.weight[start_idx:end_idx, :]
                q_grads = self.q_proj.weight.grad[start_idx:end_idx, :]
                q_score = torch.sum(torch.abs(q_grads * q_weights)).item()
                total_score += q_score
            
            # K and V projections (handle GQA case)
            kv_head_idx = head_idx // self.num_key_value_groups
            kv_start_idx = kv_head_idx * self.head_dim
            kv_end_idx = kv_start_idx + self.head_dim
            
            if self.k_proj.weight.grad is not None:
                k_weights = self.k_proj.weight[kv_start_idx:kv_end_idx, :]
                k_grads = self.k_proj.weight.grad[kv_start_idx:kv_end_idx, :]
                k_score = torch.sum(torch.abs(k_grads * k_weights)).item()
                total_score += k_score
            
            if self.v_proj.weight.grad is not None:
                v_weights = self.v_proj.weight[kv_start_idx:kv_end_idx, :]
                v_grads = self.v_proj.weight.grad[kv_start_idx:kv_end_idx, :]
                v_score = torch.sum(torch.abs(v_grads * v_weights)).item()
                total_score += v_score
            
            # O projection: gradient × weight for this head
            if self.o_proj.weight.grad is not None:
                o_weights = self.o_proj.weight[:, start_idx:end_idx]
                o_grads = self.o_proj.weight.grad[:, start_idx:end_idx]
                o_score = torch.sum(torch.abs(o_grads * o_weights)).item()
                total_score += o_score
            
            head_importance_scores.append((head_idx, total_score))
        
        # Sort by importance (ascending order - lower scores = more prunable)
        head_importance_scores.sort(key=lambda x: x[1])
        
        # Cache the results
        self._sorted_head_indices = [head_idx for head_idx, _ in head_importance_scores]
        self._head_importance_cache = [importance for _, importance in head_importance_scores]
        
        return self._sorted_head_indices

    def prune_attention_heads(self, heads_to_keep, heads_to_prune):
        """Hard pruning for inference - only called when not training"""
        if self.training or self.q_proj_prune is not None:
            return

        heads, param_indices = find_pruneable_heads_and_indices(
            heads_to_prune, self.num_heads, self.head_dim, set()
        )

        original_dtype = next(self.q_proj.parameters()).dtype

        self.q_proj_prune = prune_linear_layer(self.q_proj, param_indices)
        self.k_proj_prune = prune_linear_layer(self.k_proj, param_indices)
        self.v_proj_prune = prune_linear_layer(self.v_proj, param_indices)
        self.o_proj_prune = prune_linear_layer(self.o_proj, param_indices, dim=1)

        if original_dtype in [torch.float16, torch.bfloat16]:
            self.q_proj_prune = self.q_proj_prune.to(original_dtype)
            self.k_proj_prune = self.k_proj_prune.to(original_dtype)
            self.v_proj_prune = self.v_proj_prune.to(original_dtype)
            self.o_proj_prune = self.o_proj_prune.to(original_dtype)

        self.num_heads_prune = self.num_heads - len(heads)
        self.hidden_size_prune = self.head_dim * self.num_heads_prune 
        self.pruned_heads = self.pruned_heads.union(heads)

    def _apply_soft_head_mask(self, tensor, head_mask, tensor_type="qkv"):
        """
        Apply soft masking to Q, K, V tensors during training.
        
        Args:
            tensor: Input tensor of shape [batch, seq_len, num_heads * head_dim]
            head_mask: Boolean mask of shape [num_heads]
            tensor_type: "qkv" for Q/K/V tensors, "output" for attention output
        """
        original_dtype = next(self.q_proj.parameters()).dtype

        if tensor_type == "qkv":
            # Reshape to [batch, seq_len, num_heads, head_dim]
            batch_size, seq_len, _ = tensor.shape
            tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Apply mask: zero out heads that should be pruned
            head_mask_expanded = head_mask.view(1, 1, self.num_heads, 1)
            tensor = tensor * head_mask_expanded.to(original_dtype)
            
            # Reshape back to [batch, seq_len, num_heads * head_dim]
            tensor = tensor.view(batch_size, seq_len, -1)
            
        elif tensor_type == "attn_weights":
            # tensor shape: [batch, num_heads, seq_len_q, seq_len_k]
            head_mask_expanded = head_mask.view(1, self.num_heads, 1, 1)
            tensor = tensor * head_mask_expanded.to(original_dtype)
            
        elif tensor_type == "attn_output":
            # tensor shape: [batch, seq_len, num_heads, head_dim] 
            head_mask_expanded = head_mask.view(1, 1, self.num_heads, 1)
            tensor = tensor * head_mask_expanded.to(original_dtype)
            
        return tensor

    def make_causal_mask_for_cross_attention(self, q, k):
        batch, heads, q_len, _ = q.shape
        _, _, k_len, _ = k.shape
        dtype, device = torch.float32, q.device
        neg_inf = torch.finfo(dtype).min
        tri = torch.tril(torch.ones((q_len, q_len), device=device))

        if k_len >= q_len:
            past_len = k_len - q_len
            left = torch.ones((q_len, past_len), device=device)
            full = torch.cat([left, tri], dim=1)  # (q_len, k_len)
        else:
            full = tri[:, -k_len:] 

        mask2d = torch.where(
            full.bool(),
            torch.zeros_like(full, dtype=dtype, device=device),
            torch.full_like(full, neg_inf, dtype=dtype, device=device)
        )

        mask = mask2d.unsqueeze(0).unsqueeze(0).expand(batch, heads, q_len, k_len)

        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return mask

    def compute_attention_scores(self, cross_attn_weights: torch.Tensor, original_prompt_length: int,  image_start: Optional[int] = None,  image_end: Optional[int] = None,  text_start: Optional[int] = None,  text_end: Optional[int] = None) -> torch.Tensor:
        """
        cross_attn_weights: [B, H, Q_draft, K_target]
        
        Returns:
        scores: [B, Q_draft] - attention "spread" for each draft token
        """
        # Average across heads -> [B, Q_draft, K_target]
        scores = cross_attn_weights.mean(dim=1)
        scores = scores.mean(dim=1)

        # Sum over target keys (total influence received) -> [B, Q_draft]
        scores = scores[:, image_start:text_end]
            
        return scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
        head_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()
        
        # ============ HEAD PRUNING LOGIC ============
        apply_head_masking = head_ratio is not None and head_ratio < 1.0
        
        if apply_head_masking:
            head_mask, num_active_heads, active_head_indices, heads_to_prune = self.select_heads(
                hidden_states, 
                head_ratio=head_ratio
            )
            
            if self.training:
                # SOFT MASKING: Use original projections but apply masks
                query_states = self.q_proj(hidden_states)
                query_states = self._apply_soft_head_mask(query_states, head_mask, "qkv")
                cur_num_heads = self.num_heads  # Keep original number for reshaping
                cur_hidden_size = self.hidden_size
            else:
                # HARD PRUNING: Use pruned projections for inference
                self.prune_attention_heads(active_head_indices, heads_to_prune)
                query_states = self.q_proj_prune(hidden_states)
                cur_num_heads = self.num_heads_prune
                cur_hidden_size = self.hidden_size_prune
        else:
            # No masking applied
            query_states = self.q_proj(hidden_states)
            cur_num_heads = self.num_heads
            cur_hidden_size = self.hidden_size
        # ==========================================
        
        query_states = query_states.view(bsz, q_len, cur_num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)
        
        if cross_attention_states is not None:
            # Apply head masking to K and V projections as well
            if apply_head_masking:
                if self.training:
                    key_states = self.k_proj(cross_attention_states)
                    value_states = self.v_proj(cross_attention_states)
                    key_states = self._apply_soft_head_mask(key_states, head_mask, "qkv")
                    value_states = self._apply_soft_head_mask(value_states, head_mask, "qkv")
                else:
                    key_states = self.k_proj_prune(cross_attention_states)
                    value_states = self.v_proj_prune(cross_attention_states)
            else:
                key_states = self.k_proj(cross_attention_states)
                value_states = self.v_proj(cross_attention_states)
                
            key_states = key_states.view(bsz, -1, cur_num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, cur_num_heads, self.head_dim).transpose(1, 2)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            key_states = self.k_norm(key_states)
            
            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None
        else:
            key_states, value_states = past_key_value 

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = self.make_causal_mask_for_cross_attention(query_states, key_states)
            attn_weights = attn_weights + causal_mask

        # Apply soft masking to attention weights during training
        if apply_head_masking and self.training:
            attn_weights = self._apply_soft_head_mask(attn_weights, head_mask, "attn_weights")

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Apply soft masking to attention output during training
        if apply_head_masking and self.training:
            attn_output = self._apply_soft_head_mask(attn_output, head_mask, "attn_output")
            
        attn_output = attn_output.reshape(bsz, q_len, cur_hidden_size)

        # Final output projection with head masking
        if apply_head_masking and not self.training:
            # Use pruned projection for inference
            attn_output = self.o_proj_prune(attn_output)
        else:
            # Use original projection (for training or no masking)
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaCrossAttentionDecoderLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.cross_attn = LlamaCrossAttention(config, layer_idx=layer_idx)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))

        self.mlp = LlamaMLP(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_attention_scores: Optional[bool] = False, 
        original_prompt_length: Optional[int] = None,
        image_start: Optional[int] = None, 
        image_end: Optional[int] = None, 
        text_start: Optional[int] = None, 
        text_end: Optional[int] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        head_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights, past_key_value = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions or output_attention_scores,
            cache_position=position_ids,
            use_cache=use_cache,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        
        attention_scores = None
        if output_attention_scores and attn_weights is not None:
            attention_scores = self.cross_attn.compute_attention_scores(attn_weights, original_prompt_length, image_start=image_start, image_end=image_end,  text_start=text_start, text_end=text_end)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        
        if output_attention_scores:
            outputs += (attention_scores,)

        return outputs

class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)

        self.layers = nn.ModuleList([LlamaDecoderLayer(config, 0), LlamaCrossAttentionDecoderLayer(config, 1), LlamaDecoderLayer(config, 2)])
        self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.last_head_ratio = None
        self.dynamic_head = True

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.norm.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.norm.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.norm.weight.device)

    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask


    
    def forward(
            self,
            last_hidden_states,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_attention_scores: Optional[bool] = False,
            original_prompt_length: Optional[int] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            image_start: Optional[int] = None, 
            image_end: Optional[int] = None, 
            text_start: Optional[int] = None, 
            text_end: Optional[int] = None,
            std=None,
            last_head_ratio: Optional[float] = None,
    ):
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
        
        if last_head_ratio is not None:
            self.last_head_ratio = last_head_ratio
        
        seq_length_with_past = seq_length
        past_key_values_length = 0
        
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        hidden_states = inputs_embeds
        
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None
        all_attention_scores = () if output_attention_scores else None

        layer_head_ratio = self.last_head_ratio if self.dynamic_head else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if idx == 2: 
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, past_key_value, output_attentions, head_ratio=layer_head_ratio)

                        return custom_forward
                else:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, past_key_value, output_attentions)

                        return custom_forward
                
                    
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    last_hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    last_hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    output_attention_scores=output_attention_scores,
                    original_prompt_length=original_prompt_length,
                    image_start=image_start, 
                    image_end=image_end, 
                    text_start=text_start, 
                    text_end=text_end,
                    use_cache=use_cache,
                    head_ratio=layer_head_ratio,
                )

            hidden_states = layer_outputs[0]

            if output_attention_scores:
                all_attention_scores += (layer_outputs[-1],)

            if use_cache:
                cache_item = layer_outputs[2 if output_attentions else 1]
                next_decoder_cache += (cache_item,)

        hidden_states = self.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        

        if output_attention_scores:
            if use_cache and not output_hidden_states:
                return hidden_states, next_decoder_cache, all_attention_scores
            elif use_cache and output_hidden_states:
                return hidden_states, next_decoder_cache, all_hidden_states, all_attention_scores
            elif not use_cache and output_hidden_states:
                return hidden_states, all_hidden_states, all_attention_scores

        else:
            # Original return logic
            if use_cache and not output_hidden_states:
                return hidden_states, next_decoder_cache
            elif use_cache and output_hidden_states:
                return hidden_states, next_decoder_cache, all_hidden_states
            elif not use_cache and output_hidden_states:
                return hidden_states, all_hidden_states

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor, input_embeds=None, output_draft_attention_scores=False, original_prompt_length=None, image_start=None, image_end=None, text_start=None, text_end=None, use_prune_head=None, head_ratio=None):
        head = self.head_weight
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []
        all_attention_scores = [] if output_draft_attention_scores else None
            
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]

            inputs_embeds = self.embed_model(input_ids[:, kv_len:])
            if output_draft_attention_scores:
                out_hidden, past_key_values, step_attention_scores = self(
                    hidden_states, 
                    inputs_embeds=inputs_embeds,
                    past_key_values=self.stable_kv, 
                    use_cache=True,
                    output_attention_scores=True,
                    original_prompt_length=original_prompt_length,
                    image_start=image_start, 
                    image_end=image_end, 
                    text_start=text_start, 
                    text_end=text_end,
                )
            else:
                out_hidden, past_key_values = self(
                    hidden_states, 
                    inputs_embeds=inputs_embeds,
                    past_key_values=self.stable_kv, 
                    use_cache=True
                )
        else:
            self.dynamic_head = use_prune_head
            if output_draft_attention_scores:
                out_hidden, past_key_values, step_attention_scores = self(
                    hidden_states, 
                    inputs_embeds=input_embeds, 
                    use_cache=True,
                    output_attention_scores=True,
                    original_prompt_length=original_prompt_length,
                    image_start=image_start, 
                    image_end=image_end, 
                    text_start=text_start, 
                    text_end=text_end,
                    last_head_ratio=head_ratio,
                )
                all_attention_scores.append(step_attention_scores)
            else:
                out_hidden, past_key_values = self(
                    hidden_states, 
                    inputs_embeds=input_embeds, 
                    use_cache=True
                )
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = head(last_hidden)

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.norm.weight.device)

        # 4
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            inputs_embeds = self.embed_model(input_ids)
            if output_draft_attention_scores:
                out_hidden, past_key_values, step_attention_scores = self(
                    input_hidden, 
                    inputs_embeds=inputs_embeds, 
                    past_key_values=past_key_values,
                    position_ids=position_ids, 
                    use_cache=True,
                    output_attention_scores=True,
                    original_prompt_length=original_prompt_length,
                    image_start=image_start, 
                    image_end=image_end, 
                    text_start=text_start, 
                    text_end=text_end,
                )
            else:
                out_hidden, past_key_values = self(
                    input_hidden, 
                    inputs_embeds=inputs_embeds, 
                    past_key_values=past_key_values,
                    position_ids=position_ids, 
                    use_cache=True
                )

            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]

            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)


        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        
        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents


        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        if output_draft_attention_scores:
            return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, all_attention_scores
        else:
            return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


    @torch.no_grad()
    def acc(self, data, head, max_length=5):
        hidden_states = data["hidden_states"]
        input_ids = data["input_ids"]
        # attention_mask=data["attention_mask"]
        loss_mask = data["loss_mask"]
        sample_mask = data["sample_mask"]
        target = data["target"]
        total = [0 for _ in range(max_length)]
        correct = [0 for _ in range(max_length)]
        bs, sl = hidden_states.shape[0], hidden_states.shape[1]
        target_headout = head(target)
        hidden_states_headout = head(hidden_states)

        for i in range(bs):
            for j in range(sl):
                if loss_mask[i, j] == 0:
                    continue
                single_hidden_states = hidden_states[i, :j]
                single_input_ids = input_ids[i, :j]

                single_hidden_states = single_hidden_states[None, :, :]
                single_input_ids = single_input_ids[None, :]
                for k in range(max_length):
                    tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                    tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                    target_in_token = torch.argmax(tmp_in_target_headout)
                    target_out_token = torch.argmax(tmp_out_target_headout)
                    tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                    tmp_sample_mask = sample_mask[i, single_hidden_states.shape[1] - 1]
                    if not (target_in_token == tmp_token):
                        break
                    out_hidden = self(single_hidden_states, input_ids=single_input_ids)
                    last_hidden = out_hidden[:, -1]
                    last_headout = head(last_hidden)
                    token = torch.argmax(last_headout)
                    total[k] += 1
                    if token == target_out_token:
                        correct[k] += 1
                    else:
                        for kk in range(k, max_length):
                            total[kk] += 1
                        break

                    single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                    single_input_ids = torch.cat(
                        (single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)

        acc = [correct[i] / total[i] for i in range(len(correct))]
        return acc


class Vhead(nn.Module):
    def __init__(self, ins=6566, outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins, outs, bias=False)

    def forward(self, x):
        return self.fc(x)


import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = EConfig.from_pretrained('config.json')
    model = Model(config, load_emb=False)
    print(model)