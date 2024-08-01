export MODEL_NAME="CompVis/stable-diffusion-v1-4"

accelerate launch --main_process_port=29503 \
  --gpu_ids=0 \
  train_style_gal_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir="./snapshot/GAL_style/cat/" \
  --instance_data_dir="./data/style/cat" \
  --total_round=4 \
  --start_round=1 \
  --resolution=512 \
  --feedback="uncertainty" \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --checkpointing_steps=500 \
  --balance --openness_lambda=0.005 \
  --seed=0