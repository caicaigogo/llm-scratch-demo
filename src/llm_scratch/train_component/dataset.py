import json
from loguru import logger
from torch.utils.data import Dataset


class UnifiedSFTDataset(Dataset):
    """
    统一的数据处理dataset
    """
    def __init__(self, file, tokenizer, max_seq_len, template):
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.system = template.system

        self.max_seq_len = max_seq_len
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据拼接格式为: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        if data[0]['role'] == 'system':
            system_content = data[0]['content']
            conversations_messages = data[1:]
        else:
            system_content = self.system
            conversations_messages = data

        # setting system information
        if self.system_format is not None:
            system_text = self.system_format.format(content=system_content)
            input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
            target_mask = [0] * len(input_ids)

        # 拼接多轮对话
        for i, message in enumerate(conversations_messages):
            role = message['role']
            message_content =  message['content']

            if role == 'user':
                user_text = self.user_format.format(content=message_content)
                input_tokens = self.tokenizer.encode(user_text, add_special_tokens=False)
                input_ids += input_tokens
                target_mask += [0] * len(input_tokens)
            elif role == 'assistant':
                assistant_text = self.assistant_format.format(content=message_content)
                output_tokens = self.tokenizer.encode(assistant_text, add_special_tokens=False)
                input_ids += output_tokens
                target_mask += [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_len]
        target_mask = target_mask[:self.max_seq_len]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs
