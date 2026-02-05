import os
import torch
from torch import nn
import unittest
from transformers import AutoTokenizer

from llm_scratch.k_model import (
    precompute_freq_cos_sin,
    precompute_causal_mask,
)

from llm_scratch.moe_model import (
    ModelConfig,
    MLA,
    Gate,
    MoE
)
from utils.path import find_project_root_with_tests


class TestDeepSeek(unittest.TestCase):

    def setUp(self):
        project_root = find_project_root_with_tests()
        os.chdir(project_root)

        self.config = ModelConfig()
        tokenizer = AutoTokenizer.from_pretrained("tokenizer_k")

        self.embed_tokens = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size
        )

        cos_emb, sin_emb = precompute_freq_cos_sin(self.config.qk_rope_head_dim, self.config.max_position_embeddings)
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

    def test_MLA(self):

        # MLA(
        #   (wq_down_proj): Linear(in_features=768, out_features=192, bias=True)
        #   (q_norm): RMSNorm((192,), eps=10000.0)
        #   (wq_up_proj): Linear(in_features=192, out_features=3072, bias=True)
        #   (wkv_down_proj): Linear(in_features=768, out_features=128, bias=True)
        #   (kv_norm): RMSNorm((64,), eps=10000.0)
        #   (wkv_up_proj): Linear(in_features=64, out_features=4096, bias=True)
        #   (wo_proj): Linear(in_features=2048, out_features=768, bias=True)
        # )
        mla = MLA(config=self.config)
        print(mla)
        # input_hidden_states shape  torch.Size([1, 99, 768])
        input_hidden_states = self.inputs_embeds
        print('input_hidden_states shape ', input_hidden_states.shape)

        # cos_emb shape  torch.Size([512, 32])
        cos_emb = self.cos_emb
        print('register cos_emb shape ', cos_emb.shape)

        # sin_emb shape  torch.Size([512, 32])
        sin_emb = self.sin_emb
        print('register sin_emb shape ', sin_emb.shape)

        # causal_mask shape  torch.Size([1, 1, 512, 512])
        causal_mask = self.causal_mask
        print('register causal_mask shape ', causal_mask.shape)

        # mla_output shape  torch.Size([1, 99, 768])
        mla_output = mla(input_hidden_states, cos_emb, sin_emb, casual_mask=causal_mask)
        print('mla_output shape ', mla_output.shape)

    def test_Gate(self):

        # Gate()
        gate = Gate(config=self.config)
        print(gate)

        # all_token_emb shape  torch.Size([99, 768])
        all_token_emb = self.inputs_embeds.view(-1, self.config.hidden_size)
        print('all_token_emb shape ', all_token_emb.shape)

        # top_k_expert_weights shape  torch.Size([99, 6])
        # top_k_expert_indices shape  torch.Size([99, 6])
        top_k_expert_weights, top_k_expert_indices = gate(all_token_emb)
        print('top_k_expert_weights shape ', top_k_expert_weights.shape)
        print('top_k_expert_indices shape ', top_k_expert_indices.shape)

    def test_MoE(self):
        # MoE(
        #   (gate): Gate()
        #   (experts): ModuleList(
        #     (0-31): 32 x MLP(
        #       (gate_proj): Linear(in_features=768, out_features=192, bias=False)
        #       (up_proj): Linear(in_features=768, out_features=192, bias=False)
        #       (down_proj): Linear(in_features=192, out_features=768, bias=False)
        #       (act_fn): SiLU()
        #     )
        #   )
        #   (shared_experts): MLP(
        #     (gate_proj): Linear(in_features=768, out_features=384, bias=False)
        #     (up_proj): Linear(in_features=768, out_features=384, bias=False)
        #     (down_proj): Linear(in_features=384, out_features=768, bias=False)
        #     (act_fn): SiLU()
        #   )
        # )
        moe = MoE(config=self.config)
        print(moe)
        # input_hidden_states shape  torch.Size([1, 99, 768])
        input_hidden_states = self.inputs_embeds
        print('input_hidden_states shape ', input_hidden_states.shape)

        # moe_output shape  torch.Size([1, 99, 768])
        moe_output = moe(input_hidden_states)
        print('moe_output shape ', moe_output.shape)
