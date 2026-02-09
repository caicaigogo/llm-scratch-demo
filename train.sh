# pip install -e ".[train]"
python ddp_train.py --data_path ./data/pretrain_50_jinyong.jsonl --train_mode pretrain
python ddp_train.py --data_path ./data/sft_train_50_CN.jsonl --out_dir sft_model_215M --train_mode sft
python export_model.py
python trainer_train.py
