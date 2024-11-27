# torchrun --nnodes 1 --nproc-per-node 4

# deepspeed --include localhost:0,1,2,3
CUDA_VISIBLE_DEVICES=1 python hz_run_self.py \
    --output_dir modeloutput/1127 \
    --embedding_model_name bert \
    --model_name_or_path model/bge_base_en_v1.5 \
    --data_dir data/shopee \
    --cache_dir_data cache_data \
    --learning_rate 2e-5 \
    --fp16 False \
    --num_train_epochs 2 \
    --per_device_train_batch_size 24 \
    --query_max_len 64 \
    --passage_max_len 256 \
    --remove_unused_columns False \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --temperature 0.05 \
    --logging_steps 5 #
