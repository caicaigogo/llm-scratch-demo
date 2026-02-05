import os
import torch
from torch import nn
import unittest
from transformers import AutoTokenizer, AutoConfig

from llm_scratch.k_model import (
    precompute_freq_cos_sin,
    apply_rotary_pos_emb,
    repeat_kv,
    precompute_causal_mask,
    ModelConfig,
    RMSNorm,
    Attention,
    MLP,
    DecoderLayer,
    LLaMAModel,
    LLaMAForCausalLM
)
from utils.path import find_project_root_with_tests


class TestTransformer(unittest.TestCase):

    def setUp(self):
        project_root = find_project_root_with_tests()
        os.chdir(project_root)
        model_type = 'Tiny-K'
        # model_type = 'qwen2'
        if model_type == 'Tiny-K':
            self.config = ModelConfig()
            tokenizer = AutoTokenizer.from_pretrained("tokenizer_k")
        elif model_type == 'qwen2':
            self.config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        else:
            raise(Exception('no support model_type {}'.format(model_type)))

        self.embed_tokens = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size
        )

        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.rms_norm_eps = getattr(self.config, "rms_norm_eps", None)

        cos_emb, sin_emb = precompute_freq_cos_sin(self.head_dim, self.config.max_position_embeddings)
        self.cos_emb = cos_emb
        self.sin_emb = sin_emb

        self.causal_mask = precompute_causal_mask(self.config.max_position_embeddings)

        # 测试聊天模板
        messages = [
            {"role": "system", "content": "你是一个AI助手。"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm fine, thank you. and you?"},
            {"role": "user", "content": "I'm good too."},
        ]

        # print("\n=== 聊天模板测试 ===")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        # print("Generated prompt:\n", prompt, sep="")

        # len(input_ids) -> [44]
        input_ids = tokenizer(prompt).data['input_ids']
        self.input_ids = torch.tensor(input_ids[:-1]).unsqueeze(0)
        # torch.Size([1, 43])
        # print('input_ids shape ', self.input_ids.shape)

        self.inputs_embeds = self.embed_tokens(self.input_ids)
        # print('inputs_embeds shape ', self.inputs_embeds.shape)

    def test_embed_tokens(self):
        # Embedding(151936, 896)
        print('embed_tokens ', self.embed_tokens)

        inputs_embeds = self.embed_tokens(self.input_ids)
        # torch.Size([1, 43, 896])
        print('inputs_embeds shape ', inputs_embeds.shape)

    def test_precompute_freq_cos_sin(self):

        cos_emb, sin_emb = precompute_freq_cos_sin(self.head_dim, self.config.max_position_embeddings)
        # torch.Size([32768, 32])
        print('cos_emb shape ', cos_emb.shape)
        # torch.Size([32768, 32])
        print('sin_emb shape ', sin_emb.shape)

    def test_RMSNorm(self):

        if self.rms_norm_eps:
            # RMSNorm((896,), eps=1e-06)
            norm = RMSNorm(self.config.hidden_size, self.rms_norm_eps)
        else:
            norm = RMSNorm(self.config.hidden_size)
        print(norm)

        input_hidden_states = self.inputs_embeds
        # input_hidden_states shape  torch.Size([1,43, 896])
        print('input_hidden_states shape ', input_hidden_states.shape)
        # output_hidden_states shape  torch.Size([1, 43, 896])
        output_hidden_states = norm(input_hidden_states)
        print('output_hidden_states shape ', output_hidden_states.shape)

    def test_apply_rotary_pos_emb(self):

        # input_hidden_states shape  torch.Size([1,43, 896])
        input_hidden_states = self.inputs_embeds
        input_shape = input_hidden_states.shape[:-1]
        batch_size, seq_len = input_shape
        # (batch_size, seq_len, -1, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)

        # (batch_size, heads, seq_len, head_dim)
        # transpose_states shape  torch.Size([1, 14, 43, 64])
        transpose_states = input_hidden_states.view(hidden_shape).transpose(1, 2)
        print('transpose_states shape ', transpose_states.shape)

        # cos_emb shape  torch.Size([43, 32])
        cos_emb = self.cos_emb[:seq_len]
        print('cos_emb shape ', cos_emb.shape)

        # sin_emb shape  torch.Size([43, 32])
        sin_emb = self.sin_emb[:seq_len]
        print('sin_emb shape ', sin_emb.shape)

        # rotary_hidden_states shape  torch.Size([1, 14, 43, 64])
        rotary_hidden_states = apply_rotary_pos_emb(transpose_states, cos_emb, sin_emb)
        print('rotary_hidden_states shape ', rotary_hidden_states.shape)

    def test_repeat_kv(self):

        # input_hidden_states shape  torch.Size([1,43, 896])
        input_hidden_states = self.inputs_embeds
        input_shape = input_hidden_states.shape[:-1]
        # (batch_size, seq_len, -1, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)

        key_states = input_hidden_states[:, :, :self.head_dim * self.config.num_key_value_heads]

        # (batch_size, num_key_value_heads, seq_len, head_dim)
        # transpose_states shape  torch.Size([1, 2, 43, 64])
        transpose_states = key_states.view(hidden_shape).transpose(1, 2)
        print('transpose_states shape ', transpose_states.shape)

        # num_key_value_groups is  7
        num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        print('num_key_value_groups ', num_key_value_groups)

        # expand_key_states shape  torch.Size([1, 14, 43, 64])
        expand_key_states = repeat_kv(transpose_states, n_rep=num_key_value_groups)
        print('expand_key_states shape ', expand_key_states.shape)

    def test_precompute_causal_mask(self):

        # float32
        # precompute_causal_mask(self.config.max_position_embeddings, self.inputs_embeds.dtype)
        precompute_causal_mask(5, self.inputs_embeds.dtype)

    def test_Attention(self):
        # Attention(
        #   (q_proj): Linear(in_features=896, out_features=896, bias=False)
        #   (k_proj): Linear(in_features=896, out_features=128, bias=False)
        #   (v_proj): Linear(in_features=896, out_features=128, bias=False)
        #   (o_proj): Linear(in_features=896, out_features=896, bias=False)
        # )
        self_attn = Attention(config=self.config)

        # input_hidden_states shape  torch.Size([1, 43, 896])
        input_hidden_states = self.inputs_embeds

        # cos_emb shape  torch.Size([32768, 32])
        cos_emb = self.cos_emb
        print('register cos_emb shape ', cos_emb.shape)

        # sin_emb shape  torch.Size([32768, 32])
        sin_emb = self.sin_emb
        print('register sin_emb shape ', sin_emb.shape)

        # causal_mask shape  torch.Size([1, 1, 32768, 32768])
        causal_mask = self.causal_mask
        print('register causal_mask shape ', causal_mask.shape)

        # attn_output shape  torch.Size([1, 43, 896])
        attn_output = self_attn(input_hidden_states, cos_emb, sin_emb, casual_mask=causal_mask)
        print('attn_output shape ', attn_output.shape)

    def test_MLP(self):
        # MLP(
        #   (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
        #   (up_proj): Linear(in_features=896, out_features=4864, bias=False)
        #   (down_proj): Linear(in_features=4864, out_features=896, bias=False)
        # )

        mlp = MLP(config=self.config)
        # input_hidden_states shape  torch.Size([1,43, 896])
        input_hidden_states = self.inputs_embeds

        # mlp_output shape  torch.Size([1, 43, 896])
        mlp_output = mlp(input_hidden_states)
        print('mlp_output shape ', mlp_output.shape)

    def test_DecoderLayer(self):
        # DecoderLayer(
        #   (self_attn): Attention(
        #     (q_proj): Linear(in_features=896, out_features=896, bias=False)
        #     (k_proj): Linear(in_features=896, out_features=128, bias=False)
        #     (v_proj): Linear(in_features=896, out_features=128, bias=False)
        #     (o_proj): Linear(in_features=896, out_features=896, bias=False)
        #   )
        #   (mlp): MLP(
        #     (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
        #     (up_proj): Linear(in_features=896, out_features=4864, bias=False)
        #     (down_proj): Linear(in_features=4864, out_features=896, bias=False)
        #     (act_fn): SiLU()
        #   )
        #   (input_layernorm): RMSNorm((896,), eps=1e-06)
        #   (post_attention_layernorm): RMSNorm((896,), eps=1e-06)
        # )

        decoder_layer = DecoderLayer(config=self.config, layer_idx=0)

        # input_hidden_states shape  torch.Size([1,43, 896])
        input_hidden_states = self.inputs_embeds

        # cos_emb shape  torch.Size([32768, 32])
        cos_emb = self.cos_emb

        # sin_emb shape  torch.Size([32768, 32])
        sin_emb = self.sin_emb

        # causal_mask shape  torch.Size([1, 1, 32768, 32768])
        causal_mask = self.causal_mask

        # decoder_layer_output shape  torch.Size([1,43, 896]
        decoder_layer_output = decoder_layer(input_hidden_states, cos_emb, sin_emb, causal_mask)
        print('decoder_layer_output shape ', decoder_layer_output.shape)

    def test_LLaMAModel(self):

        # LLaMAModel(
        #   (embed_tokens): Embedding(151936, 896)
        #   (layers): ModuleList(
        #     (0-23): 24 x DecoderLayer(
        #       (self_attn): Attention(
        #         (q_proj): Linear(in_features=896, out_features=896, bias=False)
        #         (k_proj): Linear(in_features=896, out_features=128, bias=False)
        #         (v_proj): Linear(in_features=896, out_features=128, bias=False)
        #         (o_proj): Linear(in_features=896, out_features=896, bias=False)
        #       )
        #       (mlp): MLP(
        #         (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
        #         (up_proj): Linear(in_features=896, out_features=4864, bias=False)
        #         (down_proj): Linear(in_features=4864, out_features=896, bias=False)
        #         (act_fn): SiLU()
        #       )
        #       (input_layernorm): RMSNorm((896,), eps=1e-06)
        #       (post_attention_layernorm): RMSNorm((896,), eps=1e-06)
        #     )
        #   )
        #   (norm): RMSNorm((896,), eps=1e-06)
        # )

        # LLaMAModel(
        #   (embed_tokens): Embedding(6144, 768)
        #   (layers): ModuleList(
        #     (0-11): 12 x DecoderLayer(
        #       (self_attn): Attention(
        #         (q_proj): Linear(in_features=768, out_features=768, bias=False)
        #         (k_proj): Linear(in_features=768, out_features=384, bias=False)
        #         (v_proj): Linear(in_features=768, out_features=384, bias=False)
        #         (o_proj): Linear(in_features=768, out_features=768, bias=False)
        #       )
        #       (mlp): MLP(
        #         (gate_proj): Linear(in_features=768, out_features=3072, bias=False)
        #         (up_proj): Linear(in_features=768, out_features=3072, bias=False)
        #         (down_proj): Linear(in_features=3072, out_features=768, bias=False)
        #         (act_fn): SiLU()
        #       )
        #       (input_layernorm): RMSNorm((768,), eps=1e-05)
        #       (post_attention_layernorm): RMSNorm((768,), eps=1e-05)
        #     )
        #   )
        #   (norm): RMSNorm((768,), eps=1e-05)
        # )

        llama_model = LLaMAModel(config=self.config)
        print(llama_model)
        # input_ids shape  torch.Size([1, 99])
        input_ids = self.input_ids
        print('input_ids shape ', input_ids.shape)

        # model_output shape  torch.Size([1, 99, 768])
        model_output = llama_model(input_ids)
        print('model_output shape ', model_output.shape)

    def test_LLaMAForCausalLM(self):

        # LLaMAForCausalLM(
        #   (model): LLaMAModel(
        #     (embed_tokens): Embedding(6144, 768)
        #     (layers): ModuleList(
        #       (0-11): 12 x DecoderLayer(
        #         (self_attn): Attention(
        #           (q_proj): Linear(in_features=768, out_features=768, bias=False)
        #           (k_proj): Linear(in_features=768, out_features=384, bias=False)
        #           (v_proj): Linear(in_features=768, out_features=384, bias=False)
        #           (o_proj): Linear(in_features=768, out_features=768, bias=False)
        #         )
        #         (mlp): MLP(
        #           (gate_proj): Linear(in_features=768, out_features=3072, bias=False)
        #           (up_proj): Linear(in_features=768, out_features=3072, bias=False)
        #           (down_proj): Linear(in_features=3072, out_features=768, bias=False)
        #           (act_fn): SiLU()
        #         )
        #         (input_layernorm): RMSNorm((768,), eps=1e-05)
        #         (post_attention_layernorm): RMSNorm((768,), eps=1e-05)
        #       )
        #     )
        #     (norm): RMSNorm((768,), eps=1e-05)
        #   )
        #   (lm_head): Linear(in_features=768, out_features=6144, bias=False)
        # )

        casual_lm = LLaMAForCausalLM(config=self.config)
        print(casual_lm)
        # input_ids shape  torch.Size([1, 99])
        input_ids = self.input_ids
        print('input_ids shape ', input_ids.shape)

        # model_output shape  torch.Size([1, 1, 6144])
        casual_lm_output = casual_lm(input_ids)
        print('casual_lm_output shape ', casual_lm_output.shape)
