from transformers import PretrainedConfig, PreTrainedModel
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models import Qwen2Model, Qwen2ForCausalLM

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"

    def __init__(
            self,
            hidden_size: int = 768,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 16,
            num_key_value_heads: int = 8,
            vocab_size: int = 6144,
            intermediate_size: int = 3072,
            rms_norm_eps: float = 1e-5,
            max_position_embeddings: int = 512,
            rope_theta: float = 10000.0,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_position_embeddings
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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    casual_mask,
    scaling: float,
):
    seq_len = query.shape[2]

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # (batch_size, num_attention_heads, seq_len, head_dim)
    # -> (batch_size, num_attention_heads, seq_len, seq_len)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # 右上角部分为很大的负数，中间线及左下角为正常值
    attn_weights = attn_weights + casual_mask[:, :, :seq_len, :seq_len]

    # (batch_size, num_attention_heads, seq_len, seq_len)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # (batch_size, num_attention_heads, seq_len, seq_len)
    # -> (batch_size, num_attention_heads, seq_len, head_dim)
    attn_output = torch.matmul(attn_weights, value_states)
    # (batch_size, seq_len, num_attention_heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


def precompute_causal_mask(max_seq_len: int, dtype=None):

    if dtype is None:
        dtype = torch.float16

    # 创建一个上三角矩阵，用于遮蔽未来信息。
    # torch.Size([1, 1, 32768, 32768])
    mask = torch.full((1, 1, max_seq_len, max_seq_len), torch.finfo(dtype).min)
    # 实心部分是0， 空心部分是 最小负数
    # 0 ■ ⬚ ⬚ ⬚ ⬚
    # 1 ■ ■ ⬚ ⬚ ⬚
    # 2 ■ ■ ■ ⬚ ⬚
    # 3 ■ ■ ■ ■ ⬚
    # 4 ■ ■ ■ ■ ■
    causal_mask = torch.triu(mask, diagonal=1)

    return causal_mask

def init_weights(
    module: nn.Module
):
    # 初始化权重的函数
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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


    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        casual_mask: torch.Tensor
    ):
        # 获取批次大小和序列长度，[batch_size, seq_len, hidden_size]

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

        query_states = apply_rotary_pos_emb(query_states, cos_emb, sin_emb)
        key_states = apply_rotary_pos_emb(key_states, cos_emb, sin_emb)

        # (batch_size, seq_len, num_attention_heads, head_dim)
        attn_output = eager_attention_forward(
            self, query_states, key_states, value_states, scaling=self.scaling, casual_mask=casual_mask
        )

        # (batch_size, seq_len, hidden_size)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # (batch_size, seq_len, hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class MLP(nn.Module):
    def __init__(self, config=None, hidden_size: int =None, intermediate_size: int = None):
        super().__init__()
        self.config = config
        if hidden_size:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = config.hidden_size
        if intermediate_size:
            self.intermediate_size = intermediate_size
        else:
            self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        casual_mask: torch.Tensor
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            casual_mask=casual_mask
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LLaMAModel(PreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)

        self.embed_tokens = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freq_cos_sin(head_dim, self.config.max_position_embeddings)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        causal_mask = precompute_causal_mask(self.config.max_position_embeddings)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # 初始化所有权重
        self.apply(init_weights)

    def forward(
        self,
        input_ids
    ):

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(hidden_states, self.freqs_cos, self.freqs_sin, self.causal_mask)

        hidden_states = self.norm(hidden_states)

        return hidden_states

class LLaMAForCausalLM(PreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.model = LLaMAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化所有权重
        self.apply(init_weights)

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # 输出容器
        self._no_split_modules = [name for name, _ in self.named_modules()]  # 不分割的模块列表

    def forward(
        self,
        input_ids,
        targets = None
    ):
        # (batch_size, seq_len)
        # -> (batch_size, seq_len, hidden_size)
        hidden_states = self.model(input_ids)

        if targets is not None:
            # 如果给定了目标，计算损失
            # (batch_size, seq_len, vocab_size)
            # -> (batch_size, seq_len, vocab_size)
            vocab_logits = self.lm_head(hidden_states)
            self.last_loss = None

            # cross_entropy 的input 的第二个维度（或第一个维度，当无 batch 时）必须是类别数 C， 即vocab_SIZE
            # input ： (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            # target : (batch_size, seq_len) -> (batch_size * seq_len)
            # last_loss: (batch_size * seq_len)
            self.last_loss = nn.functional.cross_entropy(vocab_logits.view(-1, vocab_logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')

        else:
            # (batch_size, seq_len, hidden_size)
            # -> (batch_size, 1, vocab_size)
            # -> (batch_size, 1, vocab_size)
            # 推理时的小优化：只对最后一个位置的输出进行前向传播
            vocab_logits = self.lm_head(hidden_states[:, [-1], :])
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('logits', vocab_logits)
        self.OUT.__setitem__('last_loss', self.last_loss)

        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=2):
        """
        给定输入序列 input_ids（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """

        # (batch_size, seq_len)
        # -> input_seq_len: int
        input_seq_len = input_ids.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度 , 确保  input_seq_len <= max_seq_len, 要么原来，要么从后往前截max_seq_len
            # (batch_size, seq_len) or (batch_size, max_seq_len)
            input_ids_cond = \
                input_ids \
                if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            # (batch_size, seq_len)
            # -> # (batch_size, 1, vocab_size)
            logits = self(input_ids_cond).logits

            # (batch_size, seq_len)
            # -> # (batch_size, vocab_size)
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出
            if temperature == 0.0:
                # 选择最有可能的索引
                # (batch_size, vocab_size)
                # -> # (batch_size, 1)
                _, next_token_id = torch.topk(logits, k=1, dim=-1)

            else:
                # 缩放 logits 并应用 softmax
                # 温度越小，倍数越大，概率分布更两极化
                # (batch_size, vocab_size)
                logits = logits / temperature

                # print(logits)
                if top_k is not None:
                    # (batch_size, vocab_size)
                    # -> (batch_size, top_k)
                    # 取topk的logits
                    topk_logits_value, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # logits: (batch_size, vocab_size)
                    # topk_logits_value: (batch_size, top_k)
                    # topk_logits_value[:, [-1]]: 取各个batch_size能筛选的最小概率值 -> (batch_size, 1)
                    # logits < topk_logits_value[:, [-1]]]: 广播后取得 bool的Tensor： 小于的需要mask的为True，需要保留的为原值
                    # ->  (batch_size, vocab_size)
                    # logits[logits < topk_logits_value[:, [-1]]] : 取cond 为True，即需要mask的 logits的值
                    # 这里的话，得到的1维度的向量，长度为 (batch_size * (vocab_size - top_k))
                    # mask, 将不再topk的概率值赋 负无穷
                    logits[logits < topk_logits_value[:, [-1]]] = -float('Inf')

                    # (batch_size, vocab_size)
                    probs = nn.functional.softmax(logits, dim=-1)
                    # 抽样最有可能的索引
                    # (batch_size, vocab_size)
                    # -> (batch_size, 1)
                    next_token_id = torch.multinomial(probs, num_samples=1)

            if next_token_id == stop_id:
                break
            # 将采样的索引添加到序列中并继续
            # (batch_size, seq_len)
            # -> (batch_size, seq_len + 1)
            input_ids = torch.cat((input_ids, next_token_id), dim=1)

        return input_ids[:, input_seq_len:]  # 只返回生成的token
