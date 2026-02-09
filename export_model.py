import torch
import warnings
from transformers import AutoTokenizer
from llm_scratch.k_model import ModelConfig, LLaMAForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_model(tokenizer_path, model_config, model_ckpt_path, save_directory):
    # 注册自定义类和配置
    ModelConfig.register_for_auto_class()
    # 注册到 AutoModelForCausalLM
    LLaMAForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    # 初始化模型
    model = LLaMAForCausalLM(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型权重
    state_dict = torch.load(model_ckpt_path, map_location=device)
    # 移除可能存在的多余前缀
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # 加载权重到模型
    model.load_state_dict(state_dict, strict=False)
    print(f'模型参数: {count_parameters(model)/1e6:.2f}M = {count_parameters(model)/1e9:.2f}B')

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    # 保存完整模型和tokenizer
    model.save_pretrained(save_directory, safe_serialization=False)
    tokenizer.save_pretrained(save_directory)
    print(f'模型和tokenizer已保存至: {save_directory}')


if __name__ == '__main__':
    # 示例用法
    config = ModelConfig(
        hidden_size=512,  # 模型维度
        num_hidden_layers=2,  # Transformer层数
    )
    export_model(
        tokenizer_path='./tokenizer_k/',
        model_config=config,
        model_ckpt_path='./sft_model_215M/512_2_6144.pth',
        save_directory="k-model-215M"
    )
