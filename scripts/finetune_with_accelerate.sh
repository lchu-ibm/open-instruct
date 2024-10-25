export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training Mamba model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path /gpfs/users/lchu/ckpt/mamba2_9b_dolma_2t/hf/step_1280000_ckp \
    --use_flash_attn \
    --tokenizer_name /gpfs/users/lchu/ckpt/mamba2_9b_dolma_2t/hf/step_1280000_ckp \
    --train_file data/processed/tulu_v2/daring_anteater_data.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir output/mamba2_9b_daring_anteater_fine_tuned \
    --report_to tensorboard \
    --logging_steps 1 \
    --try_launch_beaker_eval_jobs False \
    --push_to_hub False \
    # --max_train_samples 1000000 \