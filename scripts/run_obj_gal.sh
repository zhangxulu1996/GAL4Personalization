export MODEL_NAME="CompVis/stable-diffusion-v1-4"

accelerate launch --main_process_port=29503 \
  --gpu_ids=0 \
  train_obj_gal_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir="./snapshot/GAL_object/cat/" \
  --instance_data_dir="./data/object/cat" \
  --class_data_dir="./data/reg_data/cat/images" \
  --class_prompt="a photo of a cat" \
  --foreground_prompt="a <cute-cat> cat" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --total_round=4 \
  --start_round=1 \
  --resolution=512 \
  --feedback="uncertainty" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --checkpointing_steps=800 \
  --balance --openness_lambda=0.005 \
  --seed=0