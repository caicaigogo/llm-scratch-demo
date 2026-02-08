# dataset dir 下载到本地目录
dataset_dir="data"

#hf download \
#  --repo-type dataset \
#  YeungNLP/firefly-pretrain-dataset \
#  --local-dir ${dataset_dir}

#hf download \
#  --repo-type dataset \
#  YeungNLP/firefly-pretrain-dataset \
#  jinyong.jsonl \
#  --local-dir ${dataset_dir}

# 下载SFT数据集
hf download \
  --repo-type dataset \
  BelleGroup/train_3.5M_CN \
  --local-dir ${dataset_dir}
