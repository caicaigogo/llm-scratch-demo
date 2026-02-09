import argparse
import os
import json
from loguru import logger

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed
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


def main():
    # 进行一些配置和检查
    cus_args, training_args = setup_everything()


if __name__ == "__main__":
    main()
