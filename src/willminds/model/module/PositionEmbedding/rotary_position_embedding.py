import math
from typing import Optional

import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0), rope_scaling.get("beta_slow", 1.0)
        )
        if end / orig_max > 1.0:
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            # λ = (β·α - β + 1)/(β·α) YaRN标准公式
            scale = torch.where(torch.arange(dim // 2, device=freqs.device) < corr_dim, (beta * factor - beta + 1) / (beta * factor), 1.0 / factor)
            freqs = freqs * scale

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Apply rotary positional embedding to queries and keys.

    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine frequency tensor
        sin: Sine frequency tensor
        position_ids: Optional position ids
        unsqueeze_dim: Dimension to unsqueeze for broadcasting

    Returns:
        Tuple of (q_embed, k_embed) with rotary positional encoding applied
    """
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Encoding module that handles precomputation and application
    of rotary positional embeddings.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 32768, rope_base: float = 1e6,
                 rope_scaling: Optional[dict] = None):
        """
        Initialize the rotary positional encoding module.

        Args:
            dim: Dimension of the positional encoding
            max_position_embeddings: Maximum sequence length
            rope_base: Base value for rotary positional encoding
            rope_scaling: Optional rope scaling configuration
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling

        # Precompute frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=dim,
            end=max_position_embeddings,
            rope_base=rope_base,
            rope_scaling=rope_scaling
        )

        # Register as buffers (non-persistent)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0, seq_len: int = None):
        """
        Apply rotary positional encoding to queries and keys.

        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
            start_pos: Starting position for positional embeddings
            seq_len: Sequence length (if None, inferred from q)

        Returns:
            Tuple of (q_embed, k_embed) with rotary positional encoding applied
        """
        if seq_len is None:
            seq_len = q.shape[1]

        # cos = self.freqs_cos[start_pos:start_pos + seq_len]
        # sin = self.freqs_sin[start_pos:start_pos + seq_len]
        cos, sin = self.get_position_embeddings(start_pos, seq_len)

        return apply_rotary_pos_emb(q, k, cos, sin)

    def get_position_embeddings(self, start_pos: int = 0, seq_len: int = None):
        """
        Get the positional embeddings for a given position range.

        Args:
            start_pos: Starting position
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) tensors for the specified position range
        """
        if seq_len is None:
            seq_len = self.max_position_embeddings - start_pos

        cos = self.freqs_cos[start_pos:start_pos + seq_len]
        sin = self.freqs_sin[start_pos:start_pos + seq_len]

        return cos, sin