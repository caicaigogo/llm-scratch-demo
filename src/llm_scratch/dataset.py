import json
import random
import re

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        #  截长
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        # 补段
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = torch.tensor(input_id, dtype = torch.long)
        X = input_id[:-1]
        Y = input_id[1:]

        loss_mask = torch.tensor(loss_mask[1:], dtype = torch.long)

        # 基于item来看，返回的都是torch.Size([self.max_length -1])
        # 首个token 会被忽略掉预测，所以长度为self.max_length -1
        # X为input_ids
        # Y为移位后作为labels的output_ids
        # loss_mask 1或0， 0代表不会计算损失。 [0] * padding_len， 0的数量与padding_len一致
        return X, Y, loss_mask
