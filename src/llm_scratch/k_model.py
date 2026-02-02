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
            n_kv_heads: int = 8,
            vocab_size: int = 6144,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            flash_attn: bool = True,
            rope_theta: float = 10000.0,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_attention_heads = num_attention_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
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


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # eps是为了防止除以0的情况
        self.variance_epsilon = eps

    def forward(self, hidden_states):

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

