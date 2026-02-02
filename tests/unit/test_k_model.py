import os
import unittest
from llm_scratch.k_model import ModelConfig, precompute_freq_cos_sin
from torch import nn
from transformers import AutoTokenizer, AutoConfig
import torch

from utils.path import find_project_root_with_tests


class TestTransformer(unittest.TestCase):

    def setUp(self):
        project_root = find_project_root_with_tests()
        os.chdir(project_root)
        model_type = 'Tiny-K'
        # model_type = 'qwen2'
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

        # 测试聊天模板
        messages = [
            {"role": "system", "content": "你是一个AI助手。"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm fine, thank you. and you?"},
            {"role": "user", "content": "I'm good too."},
        ]

        print("\n=== 聊天模板测试 ===")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        print("Generated prompt:\n", prompt, sep="")

        input_ids = tokenizer(prompt).data['input_ids']
        # torch.Size([1, 43])
        self.input_ids = torch.tensor(input_ids[:-1]).unsqueeze(0)
        # torch.Size([1, 43])
        print('input_ids shape ', self.input_ids.shape)

    def test_embed_tokens(self):
        # Embedding(151936, 896)
        print('embed_tokens ', self.embed_tokens)

        inputs_embeds = self.embed_tokens(self.input_ids)
        # torch.Size([1, 43, 896])
        print('inputs_embeds shape ', inputs_embeds.shape)

    def test_precompute_freq_cos_sin(self):

        cos_emb, sin_emb = precompute_freq_cos_sin(self.head_dim, self.input_ids.shape[1])
        # torch.Size([43, 32])
        print('cos_emb shape ', cos_emb.shape)
        # torch.Size([43, 32])
        print('sin_emb shape ', sin_emb.shape)

