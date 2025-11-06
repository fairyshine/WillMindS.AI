import torch
import torch.nn as nn

from transformers.activations import ACT2FN

class GatedFeedForward(nn.Module):
    '''
    Config:
        hidden_size
        intermediate_size
        dropout
        hidden_act
    如果 hidden_act = "silu"，那就是 SwiGLU；
    如果 hidden_act = "gelu"，那就是 GEGLU；
    如果 hidden_act = "relu"，那就是 ReGLU。
    '''
    def __init__(self, config):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)))
        return hidden_states