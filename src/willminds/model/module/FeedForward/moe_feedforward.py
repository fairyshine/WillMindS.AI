import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, **kwargs):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear =nn.Linear(n_embed, num_experts)
    
    def forward(self, hidden_states):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(hidden_states)

        #Noise logits
        noise_logits = self.noise_linear(hidden_states)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class MoEFeedForward(nn.Module):
    def __init__(self, 
            FeedForward,
            hidden_size: int,
            intermediate_size: int,
            dropout: float,
            hidden_act: str,
            n_experts,
            num_experts_per_tok,
            **kwargs
        ):
        super(MoEFeedForward, self).__init__()
        self.params = {k:v for k,v in locals().items() if k not in ['self', 'kwargs']}
        for k in locals()['kwargs']:
            self.params[k] = locals()['kwargs'][k]
        self.router = NoisyTopkRouter(hidden_size, n_experts, num_experts_per_tok)
        self.experts = nn.ModuleList([FeedForward(**self.params) for _ in range(n_experts)])
        self.top_k = num_experts_per_tok

    def forward(self, x):
        # 1. 输入进入router得到两个输出
        gating_output, indices = self.router(x)
        # 2.初始化全零矩阵，后续叠加为最终结果
        final_output = torch.zeros_like(x)

        # 3.展平，即把每个batch拼接到一起，这里对输入x和router后的结果都进行了展平
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # 以每个专家为单位进行操作，即把当前专家处理的所有token都进行加权
        for i, expert in enumerate(self.experts):
            # 4. 对当前的专家(例如专家0)来说，查看其对所有tokens中哪些在前top2
            expert_mask = (indices == i).any(dim=-1)
            # 5. 展平操作
            flat_mask = expert_mask.view(-1)
            # 如果当前专家是任意一个token的前top2
            if flat_mask.any():
                # 6. 得到该专家对哪几个token起作用后，选取token的维度表示
                expert_input = flat_x[flat_mask]
                # 7. 将token输入expert得到输出
                expert_output = expert(expert_input)

                # 8. 计算当前专家对于有作用的token的权重分数
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                # 9. 将expert输出乘上权重分数
                weighted_output = expert_output * gating_scores

                # 10. 循环进行做种的结果叠加
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


# class MoEGate(nn.Module):
#     def __init__(self,
#         num_experts_per_tok,
#         n_routed_experts,
#         scoring_func,
#         aux_loss_alpha,
#         seq_aux,
#         norm_topk_prob,
#         hidden_size,
#         **kwargs):
#         super().__init__()
#         self.top_k = num_experts_per_tok
#         self.n_routed_experts = n_routed_experts

#         self.scoring_func = scoring_func
#         self.alpha = aux_loss_alpha
#         self.seq_aux = seq_aux

#         self.norm_topk_prob = norm_topk_prob
#         self.gating_dim = hidden_size
#         self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))

#     def forward(self, hidden_states):
#         bsz, seq_len, h = hidden_states.shape
#         hidden_states = hidden_states.view(-1, h)
#         logits = F.linear(hidden_states, self.weight, None)
#         if self.scoring_func == 'softmax':
#             scores = logits.softmax(dim=-1)
#         else:
#             raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

#         topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

#         if self.top_k > 1 and self.norm_topk_prob:
#             denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
#             topk_weight = topk_weight / denominator

#         if self.training and self.alpha > 0.0:
#             scores_for_aux = scores
#             aux_topk = self.top_k
#             topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
#             if self.seq_aux:
#                 scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
#                 ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
#                 ce.scatter_add_(1, topk_idx_for_aux_loss,
#                                 torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
#                     seq_len * aux_topk / self.n_routed_experts)
#                 aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
#             else:
#                 mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
#                 ce = mask_ce.float().mean(0)
#                 Pi = scores_for_aux.mean(0)
#                 fi = ce * self.n_routed_experts
#                 aux_loss = (Pi * fi).sum() * self.alpha
#         else:
#             aux_loss = 0
#         return topk_idx, topk_weight, aux_loss


# class MoEFeedForward(nn.Module):
#     def __init__(self,
#         FeedForward,
#         hidden_size: int,        
#         intermediate_size: int,
#         dropout: float,
#         hidden_act: str, 
#         num_experts_per_tok,
#         n_routed_experts,
#         n_shared_experts,
#         scoring_func,
#         aux_loss_alpha,
#         seq_aux,
#         norm_topk_prob,
#         **kwargs):
#         super().__init__()
#         self.params = {k:v for k,v in locals().items() if k not in ['self', 'kwargs']}
#         for k in locals()['kwargs']:
#             self.params[k] = locals()['kwargs'][k]
#         self.experts = nn.ModuleList([
#             FeedForward(**self.params)
#             for _ in range(n_routed_experts)
#         ])
#         self.gate = MoEGate(num_experts_per_tok,
#             n_routed_experts,
#             scoring_func,
#             aux_loss_alpha,
#             seq_aux,
#             norm_topk_prob,
#             hidden_size
#         )
#         if n_shared_experts > 0:
#             self.shared_experts = nn.ModuleList([
#                 FeedForward(**self.params)
#                 for _ in range(n_shared_experts)
#             ])

#     def forward(self, x):
#         identity = x
#         orig_shape = x.shape
#         bsz, seq_len, _ = x.shape
#         # 使用门控机制选择专家
#         topk_idx, topk_weight, aux_loss = self.gate(x)
#         x = x.view(-1, x.shape[-1])
#         flat_topk_idx = topk_idx.view(-1)
#         if self.training:
#             x = x.repeat_interleave(self.params['num_experts_per_tok'], dim=0)
#             y = torch.empty_like(x, dtype=torch.float16)
#             for i, expert in enumerate(self.experts):
#                 y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
#             y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
#             y = y.view(*orig_shape)
#         else:
#             y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
#         if self.params['n_shared_experts'] > 0:
#             for expert in self.shared_experts:
#                 y = y + expert(identity)
#         self.aux_loss = aux_loss
#         return y

#     @torch.no_grad()
#     def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
#         expert_cache = torch.zeros_like(x)
#         idxs = flat_expert_indices.argsort()
#         tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
#         token_idxs = idxs // self.params['num_experts_per_tok']
#         # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
#         # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
#         # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
#         # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
#         for i, end_idx in enumerate(tokens_per_expert):
#             start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
#             if start_idx == end_idx:
#                 continue
#             expert = self.experts[i]
#             exp_token_idx = token_idxs[start_idx:end_idx]
#             expert_tokens = x[exp_token_idx]
#             expert_out = expert(expert_tokens).to(expert_cache.dtype)
#             expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
#             expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

#         return expert_cache