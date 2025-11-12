import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):
    r'''Root Mean Square Layer Normalization 均方根层归一化
                x
    y = γ ---------------
            sqrt(x^2)+eps

    Parameters:
        hidden_size, int, 输入变量的末维度
        eps, float

    '''
    def __init__(
        self, 
        hidden_size: int, 
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) #scale参数， 长度为 hidden_size的全1的一维张量

    def _norm(self, x):
        #反平方根 rsqrt: 倒数 reciprocal of the square-root
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
