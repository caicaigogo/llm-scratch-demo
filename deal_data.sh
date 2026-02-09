#python deal_dataset.py --data_path ./data/jinyong.jsonl --train_mode pretrain
#head -n 50 ./data/pretrain_jinyong.jsonl > ./data/pretrain_50_jinyong.jsonl
# mkdir ./data/pretrain
# cp ./data/pretrain_50_jinyong.jsonl ./data/pretrain

#head -n 50 ./data/train_3.5M_CN.json > ./data/train_50_CN.jsonl
python deal_dataset.py --data_path ./data/train_50_CN.jsonl --train_mode sft
