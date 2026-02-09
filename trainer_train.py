import argparse
import os
import json
from loguru import logger

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoTokenizer,
    AutoConfig
)

from llm_scratch.train_component.argument import CustomizedArguments


def setup_everything():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument(
        "--train_args_file", type=str, default='train_args/pretrain/full/scratch-100m-pretrain-full.json', help=""
    )
    cmd_args = cmd_parser.parse_args()
    train_args_file = cmd_args.train_args_file
    print(train_args_file)

    # 读取训练的参数配置
    hf_parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    cus_args, training_args = hf_parser.parse_json_file(json_file=train_args_file)
    # CustomizedArguments(max_seq_len=512)
    print('cus_args \n', cus_args)
    print('training_args \n', training_args)

    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(os.path.join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = json.load(f)
    # 保存训练参数到输出目录
    with open(os.path.join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    # 设置随机种子
    set_seed(training_args.seed)

    return cus_args, training_args


def load_tokenizer(args):

    # ModelConfig {
    #   "architectures": [
    #     "LLaMAForCausalLM"
    #   ],
    #   "auto_map": {
    #     "AutoConfig": "k_model.ModelConfig",
    #     "AutoModelForCausalLM": "k_model.LLaMAForCausalLM"
    #   },
    #   "dtype": "float32",
    #   "hidden_size": 512,
    #   "intermediate_size": 3072,
    #   "max_position_embeddings": 512,
    #   "max_seq_len": 512,
    #   "model_type": "Tiny-K",
    #   "num_attention_heads": 16,
    #   "num_hidden_layers": 2,
    #   "num_key_value_heads": 8,
    #   "rms_norm_eps": 1e-05,
    #   "rope_theta": 10000.0,
    #   "transformers_version": "4.57.6",
    #   "vocab_size": 6144
    # }
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    print(config)
    # 加载tokenzier
    # 从本地路径加载预训练的分词器
    # PreTrainedTokenizerFast(name_or_path='k-model-215M', vocab_size=6144, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|im_start|>', '
    # eos_token': '<|im_end|>', 'unk_token': '<unk>', 'pad_token': '<|im_end|>', 'additional_special_tokens': ['<s>', '</s>']}, clean_up_tokenization_spaces=False, added_tokens_decoder={
    #         0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    #         1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    #         2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    #         3: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    #         4: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    # }
    # )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print(tokenizer)
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')

    return tokenizer


def init_components(cus_args, training_args):
    """
    初始化各个组件
    """
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    # 加载tokenizer
    tokenizer = load_tokenizer(cus_args)

def main():
    # 进行一些配置和检查
    cus_args, training_args = setup_everything()

    # 加载各种组件
    trainer = init_components(cus_args, training_args)

if __name__ == "__main__":
    main()
