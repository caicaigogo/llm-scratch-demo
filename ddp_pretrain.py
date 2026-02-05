# -*- coding: utf-8 -*-
import os
import argparse
import warnings
import torch
from contextlib import nullcontext

from transformers import AutoTokenizer

from llm_scratch.k_model import ModelConfig, LLaMAForCausalLM


# 忽略警告信息
warnings.filterwarnings('ignore')


def Logger(content):
    """
    简单的日志记录函数

    Args:
        content (str): 要打印的内容
    """
    print(content)


def init_model():
    """
    初始化模型和分词器

    功能包括：
    1. 加载预训练的分词器
    2. 创建Transformer模型
    3. 设置多GPU并行训练（如果可用）
    4. 将模型移动到指定设备
    5. 统计并打印模型参数量

    Returns:
        tuple: (model, tokenizer) 初始化后的模型和分词器
    """

    def count_parameters(model):
        """
        统计模型中可训练参数的数量

        Args:
            model: PyTorch模型

        Returns:
            int: 可训练参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 从本地路径加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # 根据配置创建Transformer模型
    model = LLaMAForCausalLM(lm_config)

    # 多卡初始化：检查可用GPU数量并设置DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        # 使用DataParallel包装模型以支持多GPU训练
        model = torch.nn.DataParallel(model)

    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(args.device)

    # 计算并打印模型参数量（以百万为单位）
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")

    # 基础训练参数
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="float32", help="数据类型")

    # 实验跟踪和数据加载参数
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, default="./seq_monkey_datawhale.jsonl", help="训练数据路径")

    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代次数")

    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # 多GPU训练参数
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="使用的GPU ID，用逗号分隔 (例如: '0,1,2')")

    # Namespace(out_dir='base_model_215M', epochs=1, batch_size=64, learning_rate=0.0002, device='cpu', dtype='float32',
    #           use_swanlab=False, num_workers=8, data_path='./seq_monkey_datawhale.jsonl', accumulation_steps=8,
    #           grad_clip=1.0, warmup_iters=0, log_interval=100, save_interval=1000, gpus='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    print(args)

    # ==================== GPU环境设置 ====================
    # 设置可见的GPU设备
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # 自动设置主设备为第一个可用GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # ==================== 实验跟踪初始化 ====================
    if args.use_swanlab:
        import swanlab
        # 注意：使用前需要先登录 swanlab.login(api_key='your key')
        run = swanlab.init(
            project="Happy-LLM",  # 项目名称
            experiment_name="Pretrain-215M",  # 实验名称
            config=args,  # 保存所有超参数
        )

    # ==================== 模型配置 ====================
    # 定义语言模型的配置参数
    lm_config = ModelConfig(
        hidden_size=1024,  # 模型维度
        num_hidden_layers=18,  # Transformer层数
    )

    # ==================== 训练环境设置 ====================
    max_seq_len = lm_config.max_seq_len  # 最大序列长度
    args.save_dir = os.path.join(args.out_dir)  # 模型保存目录

    # 创建必要的目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 确定设备类型（用于选择合适的上下文管理器）
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置混合精度训练的上下文管理器
    # CPU训练时使用nullcontext，GPU训练时使用autocast
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # ==================== 模型和数据初始化 ====================
    # 初始化模型和分词器
    model, tokenizer = init_model()
