import argparse
import os
import json
from loguru import logger
import torch
from tqdm import tqdm
import datasets
from datasets import load_dataset, concatenate_datasets
from itertools import chain

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer
)

from llm_scratch.train_component.argument import CustomizedArguments
from llm_scratch.train_component.collator import PretrainCollator


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


def load_model(args, training_args):
    """
    加载模型
    """
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    # logger.info(f'Train model with {args.train_mode}')

    # init model kwargs
    # todo add flash attention
    # attn_implementation = None
    torch_dtype = torch.float32

    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None
    )

    # LLaMAForCausalLM(
    #   (model): LLaMAModel(
    #     (embed_tokens): Embedding(6144, 512)
    #     (layers): ModuleList(
    #       (0-1): 2 x DecoderLayer(
    #         (self_attn): Attention(
    #           (q_proj): Linear(in_features=512, out_features=512, bias=False)
    #           (k_proj): Linear(in_features=512, out_features=256, bias=False)
    #           (v_proj): Linear(in_features=512, out_features=256, bias=False)
    #           (o_proj): Linear(in_features=512, out_features=512, bias=False)
    #         )
    #         (mlp): MLP(
    #           (gate_proj): Linear(in_features=512, out_features=3072, bias=False)
    #           (up_proj): Linear(in_features=512, out_features=3072, bias=False)
    #           (down_proj): Linear(in_features=3072, out_features=512, bias=False)
    #           (act_fn): SiLU()
    #         )
    #         (input_layernorm): RMSNorm((512,), eps=1e-05)
    #         (post_attention_layernorm): RMSNorm((512,), eps=1e-05)
    #       )
    #     )
    #     (norm): RMSNorm((512,), eps=1e-05)
    #   )
    #   (lm_head): Linear(in_features=512, out_features=6144, bias=False)
    # )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    print(model)

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model
    }


def load_pretrain_dataset(training_args, cus_args, tokenizer):
    """
    多线程预处理预训练数据
    """
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        output = {'input_ids': output.input_ids}
        return output

    def group_texts(examples):
        # Concatenate all texts.
        # k -> input_ids
        # examples 是 lazy Batch 类型， 循环取
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        # 截尾
        if total_length >= max_seq_len:
            total_length = (total_length // max_seq_len) * max_seq_len
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_len] for i in range(0, total_length, max_seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_path = cus_args.train_file
    max_seq_len = cus_args.max_seq_len
    # 创建缓存路径
    cache_dir = os.path.join(data_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    logger.info('Pretraining data path: {}'.format(data_path))

    # 扫描所有jsonl文件
    logger.info('Scanning all the training file...')
    files = []
    for root, dir_names, file_names in os.walk(data_path):
        for file_name in file_names:
            file = os.path.join(root, file_name)
            if file_name.endswith('.jsonl'):
                files.append(file)
    logger.info(f'Total num of training file: {len(files)}')

    # 预处理所有文本，将其id化，并且进行packing操作
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        pretrain_dataset = []  # 汇总所有dataset
        for idx, file in enumerate(tqdm(files)):
            logger.info(f'Loading file: {file}')
            file_name = os.path.basename(file)
            file_name = file_name.replace('.jsonl', '')
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_path, exist_ok=True)

            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'Finished loading datasets-{file_name} from cache')
            except Exception:
                tmp_cache_path = os.path.join(cache_path, 'tmp')    # 临时缓存目录，会被自动删除
                logger.info(f'There is no cache of file {file_name}, start preprocessing...')
                raw_dataset = load_dataset("json", data_files=file, cache_dir=tmp_cache_path, keep_in_memory=False)

                # DatasetDict({
                #     train: Dataset({
                #         features: ['text'],
                #         num_rows: 50
                #     })
                # })
                print(raw_dataset)

                # dict类型， {'text': 'xxxx'}
                print('out', raw_dataset['train'][0])

                #  When using `batched=True`, make sure provided `function` returns a `dict` of types like
                #  `(<class 'list'>, <class 'numpy.ndarray'>, <class 'pandas.core.series.Series'>,
                #  <class 'torch.Tensor'>)`.
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=cus_args.tokenize_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                # DatasetDict({
                #     train: Dataset({
                #         features: ['input_ids'],
                #         num_rows: 50
                #     })
                # })
                print(tokenized_dataset)

                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=cus_args.tokenize_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {max_seq_len}",
                )

                # num_row, 每行长度都是 max_seq_len
                # DatasetDict({
                #     train: Dataset({
                #         features: ['input_ids'],
                #         num_rows: 38
                #     })
                # })
                print(grouped_datasets)

                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
                # 删除临时目录
                # shutil.rmtree(tmp_cache_path)

            logger.info(f"Training number of {file_name}: {len(processed_dataset['train'])}")
            if idx == 0:
                pretrain_dataset = processed_dataset['train']
            else:
                assert pretrain_dataset.features.type == processed_dataset["train"].features.type
                pretrain_dataset = concatenate_datasets([pretrain_dataset, processed_dataset["train"]])
    logger.info(f"Total training number: {len(pretrain_dataset)}")
    return pretrain_dataset


def init_components(cus_args, training_args):
    """
    初始化各个组件
    """
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    # 加载tokenizer
    tokenizer = load_tokenizer(cus_args)
    # 加载model

    components = load_model(cus_args, training_args)
    model = components['model']

    # 初始化dataset和collator
    if cus_args.task_type == 'pretrain':
        logger.info('Train model with pretrain task')
        train_dataset = load_pretrain_dataset(training_args, cus_args, tokenizer)
        data_collator = PretrainCollator(tokenizer, cus_args.max_seq_len)

    else:
        raise Exception('no support task_type')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    return trainer


def main():
    # 进行一些配置和检查
    cus_args, training_args = setup_everything()

    # 加载各种组件
    trainer = init_components(cus_args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = os.path.join(training_args.output_dir)
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
