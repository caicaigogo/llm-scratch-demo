from transformers import PretrainedConfig, PreTrainedModel
import torch
from torch import nn
import torch.nn.functional as F
from llm_scratch.k_model import RMSNorm, apply_rotary_pos_emb, eager_attention_forward


class ModelConfig(PretrainedConfig):
    model_type = "Tiny-MOE"

    def __init__(
            self,
            hidden_size: int = 768,
            q_lora_rank: int = 192,
            kv_lora_rank: int = 64,
            qk_nope_head_dim: int = 128,
            qk_rope_head_dim: int = 64,
            v_head_dim: int = 128,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 16,
            num_key_value_heads: int = 8,
            num_key_value_groups: int = 1,
            vocab_size: int = 6144,
            intermediate_size: int = 3072,
            rms_norm_eps: float = 1e-5,
            max_position_embeddings: int = 512,
            rope_theta: float = 10000.0,
            n_routed_experts: int = 32,
            n_groups: int = 8,
            n_limited_groups: int = 3,
            score_func: str = 'sigmoid',
            n_activated_experts: int = 6,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.n_routed_experts = n_routed_experts
        self.n_groups = n_groups
        self.n_limited_groups = n_limited_groups
        self.n_activated_experts = n_activated_experts
        self.score_func = score_func

        super().__init__(**kwargs)


class MLA(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.rope_theta = config.rope_theta
        self.num_key_value_groups = config.num_key_value_groups

        self.is_causal = True

        self.wq_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.rope_theta)
        self.wq_up_proj = nn.Linear(self.q_lora_rank, self.num_attention_heads * self.qk_head_dim)

        self.wkv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank, self.rope_theta)
        self.wkv_up_proj = nn.Linear(self.kv_lora_rank, self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo_proj = nn.Linear(self.num_attention_heads * self.v_head_dim, self.hidden_size)

        self.scaling = self.qk_head_dim ** -0.5

        # 实际缓存的kv, 只做参考了解
        # if attn_impl == "naive":
        #     self.register_buffer("k_cache", torch.zeros(max_batch_size, max_seq_len, self.n_heads, self.qk_head_dim),
        #                          persistent=False)
        #     self.register_buffer("v_cache", torch.zeros(max_batch_size, max_seq_len, self.n_heads, self.v_head_dim),
        #                          persistent=False)
        # else:
        #     self.register_buffer("kv_cache", torch.zeros(max_batch_size, max_seq_len, self.kv_lora_rank),
        #                          persistent=False)
        #     self.register_buffer("pe_cache", torch.zeros(max_batch_size, max_seq_len, self.qk_rope_head_dim),
        #                          persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        casual_mask: torch.Tensor
    ):
        # 获取批次大小和序列长度，[batch_size, seq_len, hidden_size]
        input_shape = hidden_states.shape[:-1]
        bsz, seqlen = input_shape

        # query
        # (batch_size, seq_len, hidden_size)
        # -> (batch_size, seq_len, q_lora_rank)
        # -> (batch_size, seq_len, num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim))
        q = self.wq_up_proj(self.q_norm(self.wq_down_proj(hidden_states)))

        # (batch_size, seq_len, num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim))
        # -> (batch_size, seq_len, num_attention_heads, (qk_nope_head_dim + qk_rope_head_dim))
        # -> (batch_size, num_attention_heads, seq_len,  (qk_nope_head_dim + qk_rope_head_dim))
        q = q.view(bsz, seqlen, self.num_attention_heads, self.qk_head_dim).transpose(1, 2)

        # (batch_size, num_attention_heads, seq_len, (qk_nope_head_dim + qk_rope_head_dim))
        # -> (batch_size, num_attention_heads, seq_len, qk_nope_head_dim)
        # + (batch_size, num_attention_heads, seq_len, qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # # -> (seq_len, qk_rope_head_dim //2)
        cos_emb = freqs_cos[:seqlen]
        sin_emb = freqs_sin[:seqlen]

        q_pe = apply_rotary_pos_emb(q_pe, cos_emb, sin_emb)

        # (batch_size, num_attention_heads, seqlen, (qk_nope_head_dim + qk_rope_head_dim))
        q = torch.cat([q_nope, q_pe], dim=-1)

        # k-value
        # (batch_size, seq_len, hidden_size)
        # -> (batch_size, seq_len, kv_lora_rank + qk_rope_head_dim)
        kv = self.wkv_down_proj(hidden_states)

        # (batch_size, seq_len, kv_lora_rank + qk_rope_head_dim)
        # -> (batch_size, seq_len, kv_lora_rank)
        # + (batch_size, seq_len, qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # (batch_size, seq_len, qk_rope_head_dim)
        # ->(batch_size, 1, seq_len, qk_rope_head_dim)
        k_pe = k_pe[:, None, :, :]

        # (batch_size, 1, seq_len, qk_rope_head_dim)
        k_pe = apply_rotary_pos_emb(k_pe, cos_emb, sin_emb)

        # (batch_size, seq_len, kv_lora_rank)
        # -> (batch_size, seq_len, num_attention_heads * (qk_nope_head_dim + v_head_dim))
        kv = self.wkv_up_proj(self.kv_norm(kv))

        # (batch_size, seq_len, num_attention_heads * (qk_nope_head_dim + v_head_dim))
        # -> (batch_size, seq_len, num_attention_heads, (qk_nope_head_dim + v_head_dim))
        # -> (batch_size, num_attention_heads, seq_len, (qk_nope_head_dim + v_head_dim))
        kv = kv.view(bsz, seqlen, self.num_attention_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)

        # (batch_size, num_attention_heads, seq_len, (qk_nope_head_dim + v_head_dim))
        # -> (batch_size, num_attention_heads, seq_len, qk_nope_head_dim)
        # + (batch_size, num_attention_heads, seq_len, v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # (batch_size, 1, seq_len, qk_rope_head_dim)
        # -> (batch_size, num_attention_heads, seq_len, qk_rope_head_dim)
        k_pe = k_pe.expand(-1, self.num_attention_heads, -1, -1)

        # (batch_size, num_attention_heads, seqlen, (qk_nope_head_dim + qk_rope_head_dim))
        k = torch.cat([k_nope, k_pe], dim=-1)

        # q: (batch_size, num_attention_heads, seqlen, (qk_nope_head_dim + qk_rope_head_dim))
        # k: (batch_size, num_attention_heads, seqlen, (qk_nope_head_dim + qk_rope_head_dim))
        # v: (batch_size, num_attention_heads, seqlen, v_head_dim)
        # -> (batch_size, seqlen, num_attention_heads, v_head_dim)
        mla_output = eager_attention_forward(
            self, q, k, v, scaling=self.scaling, casual_mask=casual_mask
        )

        # (batch_size, seqlen, num_attention_heads, v_head_dim)
        # -> (batch_size, seq_len, num_attention_heads * v_head_dim)
        mla_output = mla_output.reshape(*input_shape, -1).contiguous()

        # ((batch_size, seq_len, num_attention_heads * v_head_dim)
        # -> (batch_size, seq_len, hidden_size)
        mla_output = self.wo_proj(mla_output)

        return mla_output


class Gate(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.score_func = config.score_func
        self.n_groups = config.n_groups
        self.topk_groups = config.n_limited_groups
        self.topk = config.n_activated_experts


    def forward(self, x: torch.Tensor):

        # (batch_size * seq_len, hidden_size)
        # -> (batch_size * seq_len, n_routed_experts)
        # -> (all_tokens, n_routed_experts)
        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            # (all_tokens, n_routed_experts)
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            # (all_tokens, n_routed_experts)
            scores = scores.sigmoid()
        original_scores = scores

        if self.n_groups > 1:
            # (all_tokens, n_routed_experts)
            # -> (all_tokens, n_groups, group_experts)
            # 将专家分成n_groups 组， 每组有group_experts 个专家
            scores = scores.view(x.size(0), self.n_groups, -1)

            # 结果是算出 token 中每个组的分数，每个组的分数取该组的前两名作为该组的分数
            # 选group，可以降低通信成本。 这里每个token 取每组前两个得分的分数作为组得分
            # topk 返回的 是tuple，第一个元素为 values， 第二个元素为index, 两个元素的size 都是 (all_tokens, n_groups, 2)
            # input: (all_tokens, n_groups, group_experts)
            # topk()[0] -> (all_tokens, n_groups, 2)
            # sum -> (all_tokens, n_groups)
            group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)

            # 相当于n_groups中选topk_groups， 举例来说，即基于每组分数，从8组里选3组。
            # 结果是算出每个token 所选择的 topk_groups 所对应的 group indices
            # (all_tokens, n_groups) ->
            # (all_tokens, topk_groups)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]

            # 创建(all_tokens, n_groups)的zeros
            # 沿着 n_groups维度， 将n_groups对应 的indices，置True
            # 即创造(all_tokens, n_groups)，将每个token 选择的topk_groups置True(相当于置1)，其他置0
            # mask类型为float32
            # (all_tokens, n_groups, group_experts) -> (all_tokens, n_groups)
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)

            # 在非 topk groups中的scores被清零， 并展开除all_tokens的所有维度, 相当于回到了(all_tokens, n_routed_experts)，
            # 只是会对不再 topk group 中的 experts的 scores进行清零
            # (all_tokens, n_groups, group_experts)
            # -> (all_tokens, n_routed_experts)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)

        # 选取每个token 对应n_activated_experts的index,
        # 即从n_routed_experts挑选了对应n_activated_experts的index
        # (all_tokens, n_routed_experts)
        # -> (all_tokens, n_activated_experts)
        indices = torch.topk(scores, self.topk, dim=-1)[1]

        # 选取每个token 对应n_activated_experts的scores weight
        # 该scores weights是经过 softmax 或 sigmoid 转换
        # 即从n_routed_experts挑选了对应n_activated_experts的index
        # (all_tokens, n_routed_experts)
        # -> (all_tokens, n_activated_experts)
        weights = original_scores.gather(1, indices)

        # 如果score_func为sigmoid，还需要对权重进行归一化处理， 其实softmax应该也要进行归一化才对
        if self.score_func == "sigmoid":
            # 注意这时候是在 选择的n_activated_experts 进行归一化，而不是所有的n_routed_experts
            weights /= weights.sum(dim=-1, keepdim=True)

        return weights.type_as(x), indices

