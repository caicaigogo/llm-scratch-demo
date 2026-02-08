import argparse
import json
from tqdm import tqdm
from pathlib import Path


def split_text(text, chunk_size=512):
    """将文本按指定长度切分成块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def deal_pretain_data(input_path):

    input_path = Path(input_path).resolve()
    output_name = 'pretrain_' + input_path.name

    parent = input_path.parent

    output_pretrain_path = parent / output_name

    with open(output_pretrain_path, 'w', encoding='utf-8') as pretrain:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in tqdm(data, desc=f"Processing lines in {input_path}", leave=False):  # 添加行级别的进度条
                line = json.loads(line)
                text = line['text']
                chunks = split_text(text)
                for chunk in chunks:
                    pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="deal dataset")

    # 基础训练参数
    parser.add_argument("--data_path", type=str, help="数据路径")
    parser.add_argument("--train_mode", help="训练方式")
    args = parser.parse_args()

    if args.deal_type == 'pretrain':
        deal_pretain_data(args.data_path)

