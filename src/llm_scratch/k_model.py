from transformers import PretrainedConfig
import torch
from torch import nn


class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"

    def __init__(
            self,
            hidden_size: int = 768,
            n_layers: int = 12,
            num_attention_heads: int = 16,
            num_key_value_heads: int = 8,
            vocab_size: int = 6144,
            hidden_dim: int = None,
            multiple_of: int = 64,
            rms_norm_eps: float = 1e-5,
            max_position_embeddings: int = 512,
            dropout: float = 0.0,
            flash_attn: bool = True,
            rope_theta: float = 10000.0,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.rope_theta = rope_theta
        super().__init__(**kwargs)


# 获得旋转嵌入的实部和虚部
# 注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freq_cos_sin(head_dim: int, max_seq_len: int, base: float = 10000.0):

    # torch.Size([head_dim/2])
    # [0., 2., 4.,  ..., head_dim -2]
    exponent_numerator = torch.arange(0, head_dim, 2).float()
    exponent_denominator = head_dim
    # torch.Size([head_dim/2])
    # [0.0000,  0.0317, 0.0635, ..., 0.9841]
    exponent = exponent_numerator / exponent_denominator
    # torch.Size([head_dim/2])
    # [1.0000e+00, 7.4648e-01, ..., 1.5505e-04, 1.1574e-04]
    freq = 1.0 / (base ** exponent)

    # 生成一个从0到end的max_seq_len，长度为max_seq_len
    # torch.Size([max_seq_len])
    pos_ids = torch.arange(max_seq_len)
    # 计算外积，得到一个二维矩阵，每一行是的position_ids元素乘以freq的元素
    # torch.Size([max_seq_len, [head_dim/2])
    pos_embedding = torch.outer(pos_ids, freq).float()
    # 计算频率的余弦值，得到实部
    # torch.Size([max_seq_len, [head_dim/2])
    pos_cos = torch.cos(pos_embedding)
    # 计算频率的正弦值，得到虚部
    # torch.Size([max_seq_len, [head_dim/2])
    pos_sin = torch.sin(pos_embedding)

    return pos_cos, pos_sin


def apply_rotary_pos_emb(h, cos, sin):

    # hidden_states shape (batch_size, heads, seq_len, head_dim)
    # cos shape (seq_len, head_dim//2)
    # sin shape (seq_len, head_dim//2)

    head_dim = h.shape[-1]
    r_i_split_dim = head_dim//2

    # shape (1, 1, seq_len, head_dim//2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    #
    # 应用旋转，分别计算旋转后的实部和虚部
    # xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    # xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    # 向量被分成实部和虚部两部分
    # 1. 新实部和新虚部都需要原本旧实部和旧虚部乘以cos，所以直接使用 q * cos
    # 2.1 新实部还需要旧虚部，新虚部还需要旧实部。 所以需要对原有向量虚实调换。
    #             # 2.2 都用到的是sin，新实部是乘以 -sin， 新虚部是 *sin， 所以还需要注意个符号的变换。
    # 2.1，2.2 这部分放在 rotate_half 实现。

    # shape (batch_size, heads, seq_len, head_dim//2)
    h_r = h[..., :r_i_split_dim]
    h_i = h[..., r_i_split_dim:]

    h_out_r = h_r * cos - h_i * sin
    h_out_i = h_r * sin + h_i * cos

    # shape (batch_size, heads, seq_len, head_dim)
    rotary_emb_out = torch.cat([h_out_r, h_out_i], dim=-1)

    return rotary_emb_out


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


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # eps是为了防止除以0的情况
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # 计算了输入hidden_states的平方的均值
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)

        return hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Attention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        # key_value_head 需要repeat多少次
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True


        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)


    def forward(self, hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # 获取批次大小和序列长度，[batch_size, seq_len, dim]

        # (batch_size, seq_len)
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_len = input_shape

        # (batch_size, seq_len, -1, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)
        # query
        # (batch_size, seq_len, hidden_size)
        # -> (batch_size, seq_len, hidden_size)
        # -> (batch_size, seq_len, num_attention_heads, head_dim)
        # -> (batch_size, num_attention_heads, seq_len, head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # k-value
        # (batch_size, seq_len, hidden_size)
        # -> (batch_size, seq_len, num_key_value_heads * head_dim)
        # -> (batch_size, seq_len, num_key_value_heads, head_dim)
        # -> (batch_size, num_key_value_heads, seq_len, head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # -> (seq_len, head_dim //2)
        cos_emb = freqs_cos[:seq_len]
        sin_emb = freqs_sin[:seq_len]
        print('cos_emb shape', cos_emb.shape)
        print('sin_emb shape', sin_emb.shape)

        query_states = apply_rotary_pos_emb(query_states, cos_emb, sin_emb)
        key_states = apply_rotary_pos_emb(key_states, cos_emb, sin_emb)

        print('rotatry query_states shape', query_states.shape)
        print('rotatry key_states shape', key_states.shape)


