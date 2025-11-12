import torch
import torch.nn as nn

from transformers.activations import ACT2FN

class GatedFeedForward(nn.Module):
    r'''门控前馈网络。 

    Parameters:
        hidden_size, int, 输入变量的末维度
        intermediate_size, int, 中间层变量的末维度
        dropout, float, dropout超参
        hidden_act, str, 使用的激活函数

    Settings:
      - 如果 hidden_act = "silu"，那就是 SwiGLU；
      - 如果 hidden_act = "gelu"，那就是 GEGLU；
      - 如果 hidden_act = "relu"，那就是 ReGLU。
    '''
    def __init__(
        self, 
        hidden_size: int = None,        
        intermediate_size: int = None,
        dropout: float = None,
        hidden_act: str = None, 
        **kwargs,
    ):
        super().__init__()
        # 如果中间层变量未设置，则自动设置
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout   = nn.Dropout(p=dropout)
        self.act_fn    = ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)))
        return hidden_states