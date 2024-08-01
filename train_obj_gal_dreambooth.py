#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import sys
import argparse
import random
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
import json
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from clip_eval import CLIPEvaluator
import clip


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--balance",
        default=False,
        action="store_true",
        help="Whether to balance the new addtional samples.",
    )
    parser.add_argument(
        "--openness_lambda",
        type=float,
        default=0.0035,
        help="The weight of openness loss.",
    )
    parser.add_argument(
        "--foreground_prompt",
        type=str,
        default=None,
        help="The prompt to specify foreground images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--log",
        default=False,
        action="store_true",
        help="Whether to generate images during the training phrase.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        required=False,
        help="The number of anchor directions selected in each iteration.",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        default="random",  # clip, random, or human
        required=True,
        help="The feedback type for the training. If not set, the training will be performed 'random'.",
    )
    parser.add_argument(
        "--start_round",
        type=int,
        default=1,
        required=True,
        help="The start round of training.",
    )
    parser.add_argument(
        "--total_round",
        type=int,
        default=4,
        required=True,
        help="The total round of training.",
    )
    parser.add_argument(
        "--current_round",
        type=int,
        default=1,
        required=False,
        help="The current round of training.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="a bag {}",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        size=512,
        center_crop=False,
        add=False,
        feedback="random",
        class_data_root=None,
        class_prompt=None,
        class_num=None
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.feedback = feedback

        self.instance_data_root = instance_data_root
        if not os.path.exists(self.instance_data_root):
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        if add:
            instance_init_info = json.load(open(os.path.join(self.instance_data_root, "init.json"))) # list
            instance_add_info = json.load(open(os.path.join(self.instance_data_root, f"{self.feedback}_add.json"))) # list
            info = instance_init_info + instance_add_info
        else:
            instance_init_info = json.load(open(os.path.join(self.instance_data_root, "init.json"))) # list
            info = instance_init_info

        self.instance_images_path = [i["instance_data_path"] for i in info]
        self.instance_prompt = [i["instance_prompt"] for i in info]
        self.sample_weights = [i["weight"] for i in info]

        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.inverted_prompts = [detail["plain_prompt"] for detail in instance_init_info]
        self.inverted_prompts = set(self.inverted_prompts)

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
    
    def get_inverted_prompt(self):
        return self.inverted_prompts

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_prompt = self.instance_prompt[index % self.num_instance_images]
        instance_weight = self.sample_weights[index % self.num_instance_images]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["instance_prompt"] = instance_prompt
        example["instance_weights"] = instance_weight

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
            example["class_img_weights"] = instance_weight

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    weights = [example["instance_weights"] for example in examples]
    
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        weights += [example["class_img_weights"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    weights = torch.tensor(weights)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "sample_weights": weights,
    }
    return batch



class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def start_new_train(args):
    current_round = args.current_round
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit,logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        for model in models:
            sub_dir = "unet" if isinstance(model, type(accelerator.unwrap_model(unet))) else "text_encoder"
            model.save_pretrained(os.path.join(output_dir, sub_dir))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                # load transformers style into model
                load_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
                model.config = load_model.config
            else:
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        add=False,
        feedback=args.feedback,
        class_data_root=args.class_data_dir,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images
    )

    if current_round > 1:        
        if args.feedback == "human":   
            logger.info("write feedback to json...")
            write_feedback_to_json(args, train_dataset.get_foreground_prompt(), args.feedback)
        
        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
            add=True,
            feedback=args.feedback,
            class_data_root=args.class_data_dir,
            class_prompt=args.class_prompt,
            class_num=args.num_class_images
        )
    
    evaluator = CLIPEvaluator(accelerator.device)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Current round = {current_round}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  TopK = {args.topk}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(0, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                if args.offset_noise:
                    noise = torch.randn_like(latents) + 0.1 * torch.randn(
                        latents.shape[0], latents.shape[1], 1, 1, device=latents.device
                    )
                else:
                    noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual    
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    pred_weights, prior_pred_weight = batch['sample_weights'].unsqueeze(1).unsqueeze(1).unsqueeze(1) 
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = pred_weights * F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Compute prior loss
                    prior_loss = prior_pred_weight * F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    # [2] -> [2,1,1,1]
                    weights = batch['sample_weights'].unsqueeze(1).unsqueeze(1).unsqueeze(1) 
                    loss = weights * F.mse_loss(model_pred.to(dtype=weight_dtype), target.to(dtype=weight_dtype), reduction="none")
                    loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % 100 == 0 and args.log == True:
                        log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, args.foreground_prompt, global_step)

                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            # max_round_reached = False
            max_round_reached = True
            if global_step >= args.max_train_steps: #  or global_step in round_steps
                if current_round != args.total_round:
                    max_round_reached = generate_new_images(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, evaluator, foreground_prompt=args.foreground_prompt, inverted_prompts=train_dataset.get_inverted_prompt())
                    if max_round_reached:
                        return max_round_reached
                    
                if args.feedback == "human":
                    print("Waiting for human feedback...")
                    return max_round_reached
                break
        
    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    del unet
    del vae
    del text_encoder
    del optimizer
    del lr_scheduler
    torch.cuda.empty_cache()
    return max_round_reached


prompts_all = ["{} in a workshop", "{} in a museum", "{} on a plane", "{} on a boat", "{} in a laboratory", "{} at an aquarium",
                "{} on a subway", "{} in a pet store", "{} at a cafe", "{} in a gym", "{} at a bus stop", "{} in a garden",
                "{} at a train station", "{} in a junkyard", "{} in an alleyway", "{} in a schoolyard", "{} in a bank", "{} under a tree"]


def check_mk_file_dir(file_name):
    check_mkdir(file_name[:file_name.rindex("/")])
    
def check_mkdir(dir_name):
    """
    check if the folder exists, if not exists, the func will create the new named folder.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, foreground_prompt, step):
    logger.info(
        f"Running validation... \n Generating 2 images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    prompt = args.validation_prompt.format(foreground_prompt)

    with torch.autocast("cuda"):
        for kk in range(2):
            image = pipeline(prompt, num_inference_steps=25, generator=generator).images[0]
            save_path_init = "log/{}/{}_{}.jpg".format(args.feedback, step, kk)
            check_mk_file_dir(save_path_init)
            image.save(save_path_init)

    del pipeline
    torch.cuda.empty_cache()


def generate_new_images(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, evaluator, foreground_prompt, inverted_prompts):
    max_iter_reach = False
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    prompts = prompts_all
    
    gen_num = 1 if args.feedback == "random" else 10

    src_txt_features = [evaluator.get_text_features(inverted_prompt) for inverted_prompt in inverted_prompts]

    # walk through all images under instance_data_dir
    src_img_features = []
    for img_name in os.listdir(os.path.join(args.instance_data_dir, "imgs")):
        src_img = Image.open(os.path.join(args.instance_data_dir, "imgs", img_name))
        src_img_features.append(evaluator.get_image_features(src_img))

    info = []
    best_imgs = []
    best_sims = []
    entropy_list = []

    for i, prompt in enumerate(prompts):
        prompt = prompt.format(foreground_prompt)
        logger.info(
            f"Running validation... \n Generating {gen_num} image with prompt:"
            f" {prompt}."
        )
        text_feature = evaluator.get_text_features(prompt)
        imgs = []
        sims = []
        overfit_flag = []
        with torch.autocast("cuda"):
            for kk in range(gen_num):
                save_path_init = "{}/{}_all/R{}_p{}_{}.jpg".format(args.instance_data_dir, args.feedback, str(args.current_round), str(i), str(kk))

                image = pipeline(prompt, num_inference_steps=50, generator=generator).images[0]
                check_mk_file_dir(save_path_init)
                image.save(save_path_init)
                
                sim = evaluator.txt_to_img_similarity(image, text_features=text_feature).cpu().numpy()
                sim_img_src_txt = 0.
                for src_txt_feat in src_txt_features:
                    sim_img_src_txt = max(sim_img_src_txt, evaluator.txt_to_img_similarity(image, text_features=src_txt_feat).cpu().numpy())
                if sim > sim_img_src_txt: 
                    overfit_flag.append(0)
                else:
                    overfit_flag.append(1)

                imgs.append(image)
                sims.append(sim)
        
        # calculate uncertainty for each prompt
        overfitting_rate = np.array(overfit_flag).mean()
        if overfitting_rate == 0. or overfitting_rate == 1.:
            entropy = 0.
        else:
            entropy = - overfitting_rate * np.log(overfitting_rate) - (1 - overfitting_rate) * np.log(1 - overfitting_rate)
        entropy_list.append(entropy)

        if args.feedback == "random":
            idx = 0
            best_sims.append(sims[idx])
            best_imgs.append(imgs[idx])
        elif args.feedback == "uncertainty":
            idx = np.argmax(sims)
            best_sims.append(sims[idx])
            best_imgs.append(imgs[idx])
        elif args.feedback == "human":
            continue
        else:
            raise ValueError("Unknown feedback type: {}".format(args.feedback))
        
        save_path = "{}/{}_pos/R{}_p{}_{}.jpg".format(args.instance_data_dir, args.feedback, str(args.current_round), str(i), str(idx))
        info.append({"instance_data_path": save_path, "instance_prompt": prompt, "weight":1.0, "max_sim": sims[idx].item(), "subject": prompts[i]})

    # calculate the openness score of current iteration, and update the info list
    entropy_list = np.array(entropy_list)
    openness = entropy_list.mean() * args.openness_lambda
    if args.current_round == 1:
        openness_info = {}
    else:
        openness_info = json.load(open(os.path.join(args.openness_dir, "openness.json"), "r"))
    key = "openness_r{}_feedback_{}".format(args.current_round, args.feedback)
    openness_info[key] = openness
    json.dump(openness_info, open(os.path.join(args.openness_dir, "openness.json"), "w"))
    if args.balance:
        for i in range(len(info)):
            info[i]["weight"] = openness


    # if random, random select 3 values from info
    if args.feedback == "human":
        return True

    if args.feedback == "random":
        random.shuffle(info)
        info = info[:args.topk]
        best_imgs = best_imgs[:args.topk]

    # select top-k anchor direcctions based on the entropy
    elif args.feedback == "uncertainty":
        if sum(entropy_list > 0) < args.topk:
            print("Max iteration reaches! No enough uncertainty images for the next training, the entropy is: {}, only {} larger than 0".format(entropy_list, sum(entropy_list > 0)))
            max_iter_reach = True
            return max_iter_reach

        best_sort_idx = np.argsort(entropy_list)[::-1]
        info = [info[x] for x in best_sort_idx]
        best_imgs = [best_imgs[x] for x in best_sort_idx]
        info = info[:args.topk]
        best_imgs = best_imgs[:args.topk]

    for kk, gen_detail in enumerate(info):
        save_path = gen_detail["instance_data_path"]
        check_mk_file_dir(save_path)
        # find the image
        best_imgs[kk].save(save_path)
        if args.feedback == "uncertainty" or args.feedback == "random":
            prompts_all.remove(gen_detail["subject"])

    if args.current_round == 1:
        concated_info = info    
    else:
        addtional_info = json.load(open(os.path.join(args.instance_data_dir, f"{args.feedback}_add.json"), "r"))
        concated_info = addtional_info + info
       
    json.dump(concated_info, open(os.path.join(args.instance_data_dir, f"{args.feedback}_add.json"), "w"))
    del pipeline
    torch.cuda.empty_cache()
    return max_iter_reach


def write_feedback_to_json(args, descriptor, feedback):
    add_data_dir = os.path.join(args.instance_data_dir, f"{feedback}_pos")
    if not os.path.exists(add_data_dir):
        raise ValueError("{} not found! Please complete the Human Feedback".format(add_data_dir))
    openness_info = json.load(open(os.path.join(args.openness_dir, "openness.json"), "r"))
    selected_info = []
    print(add_data_dir)
    for r in range(1, args.current_round):
        openness = openness_info["openness_r{}_feedback_{}".format(r, feedback)]
        removed_prompts = []
        for file_path in os.listdir(add_data_dir):
            # walk through the directory and get the path of images
            img_r = int(file_path.split('.')[0].split('_')[0][1:])
            if r != img_r:
                continue

            prompts = prompts_all
            print(file_path)
            print(int(file_path.split('.')[0].split('_')[-2][1:]))
            prompt = prompts[int(file_path.split('.')[0].split('_')[-2][1:])]
            print(prompt)
            img_path = os.path.join(add_data_dir, file_path)
            if args.balance:
                selected_info.append({"instance_data_path": img_path, "instance_prompt": prompt.format(descriptor), "weight":openness})
            else:
                selected_info.append({"instance_data_path": img_path, "instance_prompt": prompt.format(descriptor), "weight":1.0})
            removed_prompts.append(prompt)
        for prompt in removed_prompts:
            prompts_all.remove(prompt)

    json.dump(selected_info, open(os.path.join(args.instance_data_dir, f"{args.feedback}_add.json"), "w"))


def main(args):
    if args.start_round < 1:
        assert False, "start_round must be greater than 0."
    if args.start_round > args.total_round:
        assert False, "start_round must be less than total_round."

    output_dir = args.output_dir
    args.openness_dir = output_dir
    for current_round in range(args.start_round, args.total_round+1):
        args.current_round = current_round
        args.output_dir = os.path.join(output_dir, str(args.current_round))
        if os.path.exists(os.path.join(args.output_dir, f"checkpoint-{args.max_train_steps}", "unet")): #
            print("Round {} has been finished, please find the checkpoint in {}!".format(args.current_round, args.output_dir))
            continue
        max_iter_reach = start_new_train(args)
        gc.collect()
        torch.cuda.empty_cache()
        if args.feedback == "human" and max_iter_reach:
            print("Waiting for human feedback...")
            return
        if args.feedback == "uncertainty" and max_iter_reach:
            print("Max iteration reaches! No enough potential prompts for the next training!")
            return

if __name__ == "__main__":
    args = parse_args()
    main(args)
