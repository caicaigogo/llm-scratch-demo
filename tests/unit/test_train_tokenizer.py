import unittest
import os
import json
from transformers import AutoTokenizer

from utils.path import find_project_root_with_tests


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        project_root = find_project_root_with_tests()
        os.chdir(project_root)
        data_path = 'data/sft_train_50_CN.jsonl'
        with open(data_path, 'rb') as f:
            line = f.readline().decode('utf-8')
        self.sample = json.loads(line)

        self.tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    def test_apply_chat_template(self):
        sample = self.sample

        # [{"role": "system", "content": "你是一个AI助手"}, {"role": "user", "content": "针对健身房的新手，设计一套适合他们的健身器械使用指南，包括安全应用、正确姿势等方面。"}, {"role": "assistant", "content": "健身器械使用指南\n1. 开始前，请先进行热身运动。这会帮助你的身体适应运动，并减少受伤的风险。\n2. 在使用健身器械前，确保你已经了解了其使用方法。请阅读说明书或咨询教练以获得正确的使用技巧。\n3. 谨防过度使用或过度挑战你的身体。 如果你觉得有些动作太难或太重，请添加锻炼计划，以逐步提高动作难度。\n4. 使用合适的装备。 确保你拥有合适的运动鞋和舒适的运动服。 不要在裸露的脚或短裤上进行重量训练。\n5. 在健身器械上使用安全装置。 这些通常用于保护你的身体免受不当操作造成的损伤。 例如，重量训练中，你需要使用杠铃和负重时，一定要使用卡子来防止重量滑落。\n6. 注意正确的姿势。 如果你的姿势是错误的，那么你的身体很容易被伤害到，你也可能无法获得最佳的锻炼效果。 至关重要的是，保持直立的身体，保持头部和颈部的稳定，并使用合适的重量。\n7. 保持合理的呼吸方式。 无论何时进行训练，都必须保持正常呼吸。 当你需要用力时，呼气； 当你放松时，吸气。\n8. 安全存放器械。 在使用健身器械后，你需要把它们归还给适当的位置，以便其他人可以使用它们。\n总之，健身器械的正确使用是关键之一，如果不健康和不安全，它们将无法帮助您达到您所需的健康成果。 选择适当的训练计划，并为训练提供足够的时间，以备逐渐适应新方法。 对于任何问题，请向教练咨询。"}]
        print(sample)

        # <|im_start|>system
        # 你是一个AI助手<|im_end|>
        # <|im_start|>user
        # 针对健身房的新手，设计一套适合他们的健身器械使用指南，包括安全应用、正确姿势等方面。<|im_end|>
        # <|im_start|>assistant
        # 健身器械使用指南
        # 1. 开始前，请先进行热身运动。这会帮助你的身体适应运动，并减少受伤的风险。
        # 2. 在使用健身器械前，确保你已经了解了其使用方法。请阅读说明书或咨询教练以获得正确的使用技巧。
        # 3. 谨防过度使用或过度挑战你的身体。 如果你觉得有些动作太难或太重，请添加锻炼计划，以逐步提高动作难度。
        # 4. 使用合适的装备。 确保你拥有合适的运动鞋和舒适的运动服。 不要在裸露的脚或短裤上进行重量训练。
        # 5. 在健身器械上使用安全装置。 这些通常用于保护你的身体免受不当操作造成的损伤。 例如，重量训练中，你需要使用杠铃和负重时，一定要使用卡子来防止重量滑落。
        # 6. 注意正确的姿势。 如果你的姿势是错误的，那么你的身体很容易被伤害到，你也可能无法获得最佳的锻炼效果。 至关重要的是，保持直立的身体，保持头部和颈部的稳定，并使用合适的重量。
        # 7. 保持合理的呼吸方式。 无论何时进行训练，都必须保持正常呼吸。 当你需要用力时，呼气； 当你放松时，吸气。
        # 8. 安全存放器械。 在使用健身器械后，你需要把它们归还给适当的位置，以便其他人可以使用它们。
        # 总之，健身器械的正确使用是关键之一，如果不健康和不安全，它们将无法帮助您达到您所需的健康成果。 选择适当的训练计划，并为训练提供足够的时间，以备逐渐适应新方法。 对于任何问题，请向教练咨询。<|im_end|>
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        print(text)

    def test_token_id(self):

        eos_token_id = self.tokenizer.eos_token_id
        # 4
        print(eos_token_id)

        # <|im_end|>
        print(self.tokenizer.decode(eos_token_id))

        pad_token_id = self.tokenizer.pad_token_id
        # 4
        print(pad_token_id)

        # <|im_end|>
        print(self.tokenizer.decode(pad_token_id))

    def test_call(self):

        source = ['source_a', 'source_b']

        # {'input_ids': [[87, 83, 89, 86, 71, 73, 67, 69], [87, 83, 89, 86, 71, 73, 67, 70]],
        # 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]}
        output = self.tokenizer(source)
        print(output)
