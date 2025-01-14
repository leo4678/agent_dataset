import argparse
import datasets
import random
import math
import pickle
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from mlx.utils import tree_flatten
from models import LoRALinear
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from stable_diffusion import StableDiffusion
from dataloader import DataLoader
from lora import (
    build_parser as build_lora_parser,
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_dataset(sd, args):
    print("Loading datasets")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        data_dir=args.train_data_dir
    )

    # 获取图像和文本描述
    column_names = dataset["train"].column_names
    image_column = column_names[0]
    caption_column = column_names[1]

    # 文本数据预处理流程
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption)
                                if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = sd.tokenizer(
            captions, max_length=sd.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # 图像数据预处理流程
    def resize_image(image, size, interpolation=Image.Resampling.BILINEAR):
        return image.resize(size, interpolation)

    def random_crop(image, size):
        width, height = image.size
        crop_width, crop_height = size
        if width < crop_width or height < crop_height:
            raise ValueError("Image size is smaller than the crop size")
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height
        return image.crop((left, top, right, bottom))

    def to_tensor(image):
        return np.array(image) / 255.0

    def normalize(tensor, mean, std):
        return (tensor - mean) / std

    def preprocess_image(image_path, resolution, mean=0.5, std=0.5):
        image = Image.open(image_path)
        image = resize_image(image, (resolution, resolution))
        image = random_crop(image, (resolution, resolution))
        tensor = to_tensor(image)
        tensor = normalize(tensor, mean, std)
        return tensor

    # 预处理
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [
            preprocess_image(image, args.resolution) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    return train_dataloader

def setup_model(args):
    sd = StableDiffusion(args.model, float16=args.float16)
    # 所有预训练模型不需要计算梯度，冻结模型参数
    sd.unet.requires_grad_(False)
    sd.autoencoder.requires_grad_(False)
    sd.text_encoder.requires_grad_(False)
    for param in sd.unet.parameters():
        param.requires_grad_(False)

    # lora插入unet
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    sd.unet.add_adapter(unet_lora_config)

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    return sd, optimizer

def train_model(train_dataloader, sd, optimizer, args):
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.n_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )

    for epoch in range(first_epoch, num_train_epochs):
        sd.unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # 图像编码成潜在向量
            latents = sd.autoencoder.encode(batch["pixel_values"].to(dtype=mx.float32)).latent_dist.sample()
            latents = latents * sd.autoencoder.config.scaling_factor

            # 生成与潜在向量表示相同的高斯噪声
            noise = np.random.randn(latents)
            if args.noise_offset:
                noise += args.noise_offset * np.random.randn(latents.shape[0], latents.shape[1], 1, 1)

            bsz = latents.shape[0]
            # 生成时间戳
            timesteps = np.random.randint(0, sd.sampler.config.num_train_timesteps, (bsz,))
            timesteps = timesteps.astype(np.int64)

            # 时间戳添加噪声
            noisy_latents = sd.sampler.add_noise(latents, noise, timesteps)

            # 获取文本输入
            encoder_hidden_states = sd.text_encoder(batch["input_ids"])[0]

            if sd.sampler.config.prediction_type == "epsilon":
                target = noise
            elif sd.sampler.config.prediction_type == "v_prediction":
                target = sd.sampler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {sd.sampler.config.prediction_type}")

            # 预测噪声残差和均方误差损失
            model_pred = sd.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = np.mean((model_pred - target) ** 2)

            # 反向传播损失
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() / args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
                global_step += 1
            # if global_step % args.checkpointing_steps == 0:
            #     save_path = os.path.join(args.output, f"checkpoint-{global_step}")
            #     os.makedirs(save_path, exist_ok=True)
            #     torch.save(sd.unet.state_dict(), os.path.join(save_path, "unet.pth"))
            #     print(f"Saved state to {save_path}")

        if global_step >= max_train_steps:
            break
def save_lora_layers(sd, args):
    mx.savez(args.adapter_file, **dict(tree_flatten(sd.unet.trainable_parameters())))

def main():
    parser = build_lora_parser()
    parser.add_argument("--sd_specific_param", type=str, default="default_value", help="Specific param for SD finetuning")
    parser.add_argument("prompt")
    parser.add_argument("--model", choices=["sd", "sdxl"], default="sdxl")
    parser.add_argument("--n_images", type=int, default=4)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--n_rows", type=int, default=1)
    parser.add_argument("--decoding_batch_size", type=int, default=1)
    parser.add_argument("--no-float16", dest="float16", action="store_false")
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--preload-models", action="store_true")
    parser.add_argument("--output", default="out.png")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    try:
        sd, optimizer = setup_model(args)
        train_dataloader = prepare_dataset(sd, args)
        train_model(train_dataloader, sd, optimizer, args)
        save_lora_layers(sd, args)
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
