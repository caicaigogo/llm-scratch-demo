import os
import torch
from torch import nn
import unittest
from transformers import AutoTokenizer, AutoConfig

from llm_scratch.k_model import (
    ModelConfig,
    precompute_freq_cos_sin,
    RMSNorm,
    Attention,
    apply_rotary_pos_emb,
    repeat_kv
)
from utils.path import find_project_root_with_tests


class TestTransformer(unittest.TestCase):

    def setUp(self):
        project_root = find_project_root_with_tests()
        os.chdir(project_root)
        # model_type = 'Tiny-K'
        model_type = 'qwen2'
        if model_type == 'Tiny-K':
            self.config = ModelConfig()
            self.embed_tokens = nn.Embedding(
                num_embeddings=self.config.vocab_size,
                embedding_dim=self.config.hidden_size
            )
            tokenizer = AutoTokenizer.from_pretrained("tokenizer_k")

        elif model_type == 'qwen2':
            self.config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            self.embed_tokens = nn.Embedding(
                num_embeddings=self.config.vocab_size,
                embedding_dim=self.config.hidden_size,
                padding_idx=self.config.pad_token_id  # null
            )

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        else:
            raise(Exception('no support model_type {}'.format(model_type)))

        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        self.rms_norm_eps = getattr(self.config, "rms_norm_eps", None)

        cos_emb, sin_emb = precompute_freq_cos_sin(self.head_dim, self.config.max_position_embeddings)
        self.cos_emb = cos_emb
        self.sin_emb = sin_emb

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

    def test_Attention(self):
        # Attention(
        #   (q_proj): Linear(in_features=896, out_features=896, bias=False)
        #   (k_proj): Linear(in_features=896, out_features=128, bias=False)
        #   (v_proj): Linear(in_features=896, out_features=128, bias=False)
        #   (o_proj): Linear(in_features=896, out_features=896, bias=False)
        # )
        self_attn = Attention(config=self.config)

        # input_hidden_states shape  torch.Size([1,43, 896])
        input_hidden_states = self.inputs_embeds

        # cos_emb shape  torch.Size([32768, 32])
        cos_emb = self.cos_emb
        print('register cos_emb shape ', cos_emb.shape)

        # sin_emb shape  torch.Size([32768, 32])
        sin_emb = self.sin_emb
        print('register sin_emb shape ', sin_emb.shape)

        self_attn(input_hidden_states, cos_emb, sin_emb)
